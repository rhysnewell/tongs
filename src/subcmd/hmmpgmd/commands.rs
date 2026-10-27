use super::*;

pub(super) fn parse_c_client_command(request: &str) -> Result<ClientCommand, String> {
    let mut lines = request.lines();
    let options = lines
        .next()
        .ok_or_else(|| "Missing options string".to_string())?
        .trim();
    let db_mode = if options.contains("--seqdb") {
        DbMode::Seq
    } else if options.contains("--hmmdb") {
        DbMode::Hmm
    } else {
        return Err("No search database specified, --seqdb or --hmmdb.".to_string());
    };

    let query_text = lines.collect::<Vec<_>>().join("\n");
    let query = parse_client_query(&query_text, db_mode)?;
    let search_options = parse_hmmpgmd_search_options(options)?;
    let seqdb_ranges = parse_seqdb_ranges_option(options)?;
    Ok(ClientCommand {
        db_mode,
        query,
        options: search_options,
        seqdb_ranges,
        db_slice: None,
    })
}

pub(super) fn parse_worker_command_body(body: &[u8]) -> Result<ClientCommand, String> {
    if body.starts_with(b"@") {
        let request = std::str::from_utf8(body)
            .map_err(|e| format!("worker search request is not valid UTF-8: {e}"))?;
        return parse_c_client_command(request);
    }

    if body.len() < HMMD_SEARCH_CMD_FIXED_SIZE {
        return Err("short HMMD_SEARCH_CMD body".to_string());
    }

    let db_type = read_native_u32(body, 4)?;
    let inx = read_native_u32(body, 8)? as usize;
    let cnt = read_native_u32(body, 12)? as usize;
    let query_type = read_native_u32(body, 16)?;
    let query_length = read_native_u32(body, 20)? as usize;
    let opts_length = read_native_u32(body, 24)? as usize;
    let data = &body[HMMD_SEARCH_CMD_FIXED_SIZE..];
    if opts_length == 0 || data.len() < opts_length {
        return Err("invalid HMMD_SEARCH_CMD options length".to_string());
    }
    let opts_bytes = &data[..opts_length];
    if opts_bytes.last().copied() != Some(0) {
        return Err("HMMD_SEARCH_CMD options string is not NUL-terminated".to_string());
    }
    let options = std::str::from_utf8(&opts_bytes[..opts_bytes.len() - 1])
        .map_err(|e| format!("HMMD_SEARCH_CMD options are not valid UTF-8: {e}"))?;
    let query_bytes = &data[opts_length..];
    let db_mode = if options.contains("--seqdb") {
        DbMode::Seq
    } else if options.contains("--hmmdb") {
        DbMode::Hmm
    } else {
        match db_type {
            1 => DbMode::Seq,
            2 => DbMode::Hmm,
            _ => return Err("No search database specified, --seqdb or --hmmdb.".to_string()),
        }
    };
    let query = match query_type {
        HMMD_SEQUENCE => parse_hmmd_sequence_query(query_bytes, query_length)?,
        HMMD_HMM => {
            if query_bytes.len() < query_length {
                return Err("short HMMD_SEARCH_CMD query data".to_string());
            }
            if query_bytes.starts_with(b"HMM") {
                let query_text = std::str::from_utf8(query_bytes)
                    .map_err(|e| format!("HMMD_SEARCH_CMD query is not valid UTF-8: {e}"))?;
                parse_client_query(query_text, db_mode)?
            } else {
                parse_hmmd_hmm_query(query_bytes, query_length)?
            }
        }
        _ => return Err(format!("unknown HMMD_SEARCH_CMD query type {query_type}")),
    };
    let search_options = parse_hmmpgmd_search_options(options)?;
    let mut seqdb_ranges = parse_seqdb_ranges_option(options)?;
    if db_mode == DbMode::Seq && cnt > 0 {
        let slice_range = DbRange {
            start: inx.saturating_add(1),
            end: inx.saturating_add(cnt),
        };
        seqdb_ranges = Some(match seqdb_ranges {
            Some(ranges) => intersect_db_ranges(&ranges, slice_range),
            None => vec![slice_range],
        });
    }
    Ok(ClientCommand {
        db_mode,
        query,
        options: search_options,
        seqdb_ranges,
        db_slice: Some(DbSlice {
            start: inx,
            count: cnt,
        }),
    })
}

fn intersect_db_ranges(ranges: &[DbRange], slice: DbRange) -> Vec<DbRange> {
    ranges
        .iter()
        .filter_map(|range| {
            let start = range.start.max(slice.start);
            let end = range.end.min(slice.end);
            (start <= end).then_some(DbRange { start, end })
        })
        .collect()
}

fn parse_hmmd_sequence_query(
    query_bytes: &[u8],
    query_length: usize,
) -> Result<ClientQuery, String> {
    let (name, mut pos) = take_hmmd_query_string(query_bytes, 0, "sequence name")?;
    let (desc, next) = take_hmmd_query_string(query_bytes, pos, "sequence description")?;
    pos = next;
    let end = pos
        .checked_add(query_length)
        .ok_or_else(|| "HMMD_SEARCH_CMD query length overflow".to_string())?;
    if query_length < 2 || end > query_bytes.len() {
        return Err("short HMMD_SEARCH_CMD query data".to_string());
    }
    let dsq = &query_bytes[pos..end];
    if dsq.first().copied() != Some(tongs::alphabet::DSQ_SENTINEL)
        || dsq.last().copied() != Some(tongs::alphabet::DSQ_SENTINEL)
    {
        return Err("HMMD_SEARCH_CMD sequence query missing digital sentinels".to_string());
    }
    let abc = Alphabet::amino();
    let seq = abc.textize(dsq, query_length - 2);
    Ok(ClientQuery::Sequence { name, desc, seq })
}

pub(super) fn parse_hmmd_hmm_query(
    query_bytes: &[u8],
    query_length: usize,
) -> Result<ClientQuery, String> {
    if query_bytes.len() < C_P7_HMM_SHELL_SIZE {
        return Err("short HMMD_SEARCH_CMD HMM query shell".to_string());
    }
    let m = query_length;
    if !(1..=MAX_HMMD_HMM_QUERY_LENGTH).contains(&m) {
        return Err("HMMD_SEARCH_CMD HMM query length out of range".to_string());
    }
    let rows = m
        .checked_add(1)
        .ok_or_else(|| "HMMD_SEARCH_CMD HMM query length overflow".to_string())?;
    let values_per_row =
        tongs::hmm::NTRANSITIONS
            .checked_add(2usize.checked_mul(Alphabet::amino().k).ok_or_else(|| {
                "HMMD_SEARCH_CMD HMM query serialized length overflow".to_string()
            })?)
            .ok_or_else(|| "HMMD_SEARCH_CMD HMM query serialized length overflow".to_string())?;
    let required_floats = rows
        .checked_mul(values_per_row)
        .ok_or_else(|| "HMMD_SEARCH_CMD HMM query serialized length overflow".to_string())?;
    let required_bytes = required_floats
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|n| C_P7_HMM_SHELL_SIZE.checked_add(n))
        .ok_or_else(|| "HMMD_SEARCH_CMD HMM query serialized length overflow".to_string())?;
    if query_bytes.len() < required_bytes {
        return Err("short HMMD_SEARCH_CMD HMM query data".to_string());
    }
    let mut hmm = Hmm::new(m, AlphabetType::Amino, Alphabet::amino().k);
    hmm.flags = read_native_i32(query_bytes, 288)? as u32;
    hmm.nseq = read_native_i32(query_bytes, 104)?;
    hmm.eff_nseq = read_native_f32(query_bytes, 108)?;
    hmm.max_length = read_native_i32(query_bytes, 112)?;
    hmm.checksum = read_native_u32(query_bytes, 136)?;
    for idx in 0..hmm.evparam.len() {
        hmm.evparam[idx] = read_native_f32(query_bytes, 140 + idx * 4)?;
    }
    for idx in 0..hmm.cutoff.len() {
        hmm.cutoff[idx] = read_native_f32(query_bytes, 164 + idx * 4)?;
    }
    for idx in 0..hmm.compo.len() {
        hmm.compo[idx] = read_native_f32(query_bytes, 188 + idx * 4)?;
    }

    let mut pos = C_P7_HMM_SHELL_SIZE;
    for node in 0..=m {
        for trans in 0..tongs::hmm::NTRANSITIONS {
            hmm.t[node][trans] = take_native_f32(query_bytes, &mut pos, "HMM transitions")?;
        }
    }
    for node in 0..=m {
        for sym in 0..hmm.abc_k {
            hmm.mat[node][sym] = take_native_f32(query_bytes, &mut pos, "HMM match emissions")?;
        }
    }
    for node in 0..=m {
        for sym in 0..hmm.abc_k {
            hmm.ins[node][sym] = take_native_f32(query_bytes, &mut pos, "HMM insert emissions")?;
        }
    }

    if read_native_usize(query_bytes, 32)? != 0 {
        let (name, next) = take_hmmd_query_string(query_bytes, pos, "HMM name")?;
        hmm.name = name;
        pos = next;
    }
    if read_native_usize(query_bytes, 40)? != 0 {
        let (acc, next) = take_hmmd_query_string(query_bytes, pos, "HMM accession")?;
        hmm.acc = Some(acc);
        pos = next;
    }
    if read_native_usize(query_bytes, 48)? != 0 {
        let (desc, next) = take_hmmd_query_string(query_bytes, pos, "HMM description")?;
        hmm.desc = Some(desc);
        pos = next;
    }
    if hmm.flags & P7H_RF != 0 {
        hmm.rf = Some(take_hmm_annotation(
            query_bytes,
            &mut pos,
            m,
            "RF annotation",
        )?);
    }
    if hmm.flags & P7H_MMASK != 0 {
        hmm.mm = Some(take_hmm_annotation(
            query_bytes,
            &mut pos,
            m,
            "MM annotation",
        )?);
    }
    if hmm.flags & P7H_CONS != 0 {
        hmm.consensus = Some(take_hmm_annotation(
            query_bytes,
            &mut pos,
            m,
            "consensus annotation",
        )?);
    }
    if hmm.flags & P7H_CS != 0 {
        hmm.cs = Some(take_hmm_annotation(
            query_bytes,
            &mut pos,
            m,
            "CS annotation",
        )?);
    }
    if hmm.flags & P7H_CA != 0 {
        hmm.ca = Some(take_hmm_annotation(
            query_bytes,
            &mut pos,
            m,
            "CA annotation",
        )?);
    }
    if hmm.flags & P7H_MAP != 0 {
        let mut map = Vec::with_capacity(m + 1);
        for _ in 0..=m {
            map.push(take_native_i32(query_bytes, &mut pos, "HMM map")?);
        }
        hmm.map = Some(map);
    }
    Ok(ClientQuery::Hmm(hmm))
}

fn take_hmm_annotation(
    bytes: &[u8],
    pos: &mut usize,
    m: usize,
    label: &str,
) -> Result<Vec<u8>, String> {
    let n = m
        .checked_add(2)
        .ok_or_else(|| format!("HMMD_SEARCH_CMD {label} length overflow"))?;
    let end = pos
        .checked_add(n)
        .ok_or_else(|| format!("HMMD_SEARCH_CMD {label} length overflow"))?;
    if end > bytes.len() {
        return Err(format!("short HMMD_SEARCH_CMD {label}"));
    }
    let value = bytes[*pos..end].to_vec();
    *pos = end;
    Ok(value)
}

pub(super) fn take_hmmd_query_string(
    bytes: &[u8],
    start: usize,
    label: &str,
) -> Result<(String, usize), String> {
    if start >= bytes.len() {
        return Err(format!("HMMD_SEARCH_CMD {label} is missing"));
    }
    let end = bytes[start..]
        .iter()
        .position(|&b| b == 0)
        .map(|offset| start + offset)
        .ok_or_else(|| format!("HMMD_SEARCH_CMD {label} is unterminated"))?;
    let value = std::str::from_utf8(&bytes[start..end])
        .map_err(|e| format!("HMMD_SEARCH_CMD {label} is not valid UTF-8: {e}"))?;
    Ok((value.to_string(), end + 1))
}

pub(super) fn read_native_u32(body: &[u8], offset: usize) -> Result<u32, String> {
    if offset + 4 > body.len() {
        return Err("short HMMD_SEARCH_CMD body".to_string());
    }
    Ok(u32::from_ne_bytes(
        body[offset..offset + 4].try_into().unwrap(),
    ))
}

fn read_native_i32(body: &[u8], offset: usize) -> Result<i32, String> {
    if offset + 4 > body.len() {
        return Err("short HMMD_SEARCH_CMD body".to_string());
    }
    Ok(i32::from_ne_bytes(
        body[offset..offset + 4].try_into().unwrap(),
    ))
}

fn read_native_f32(body: &[u8], offset: usize) -> Result<f32, String> {
    if offset + 4 > body.len() {
        return Err("short HMMD_SEARCH_CMD body".to_string());
    }
    Ok(f32::from_ne_bytes(
        body[offset..offset + 4].try_into().unwrap(),
    ))
}

fn read_native_usize(body: &[u8], offset: usize) -> Result<usize, String> {
    let width = std::mem::size_of::<usize>();
    if offset + width > body.len() {
        return Err("short HMMD_SEARCH_CMD body".to_string());
    }
    let mut value = [0u8; std::mem::size_of::<usize>()];
    value.copy_from_slice(&body[offset..offset + width]);
    Ok(usize::from_ne_bytes(value))
}

fn take_native_f32(bytes: &[u8], pos: &mut usize, label: &str) -> Result<f32, String> {
    let value =
        read_native_f32(bytes, *pos).map_err(|_| format!("short HMMD_SEARCH_CMD {label}"))?;
    *pos += 4;
    Ok(value)
}

fn take_native_i32(bytes: &[u8], pos: &mut usize, label: &str) -> Result<i32, String> {
    let value =
        read_native_i32(bytes, *pos).map_err(|_| format!("short HMMD_SEARCH_CMD {label}"))?;
    *pos += 4;
    Ok(value)
}

pub(super) fn write_i32_ne(out: &mut [u8], offset: usize, value: i32) {
    out[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
}

pub(super) fn write_u32_ne(out: &mut [u8], offset: usize, value: u32) {
    out[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
}

pub(super) fn write_f32_ne(out: &mut [u8], offset: usize, value: f32) {
    out[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
}

pub(super) fn write_ptr_value(out: &mut [u8], offset: usize, value: usize) {
    let width = std::mem::size_of::<usize>();
    out[offset..offset + width].copy_from_slice(&value.to_ne_bytes());
}
