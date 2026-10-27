use super::*;

pub(super) fn parse_worker_payload(payload: &[u8]) -> std::io::Result<ParsedWorkerPayload> {
    if payload.len() < HMMD_SEARCH_STATS_SIZE {
        return Err(std::io::Error::other("short worker search payload"));
    }

    let _nmodels = read_payload_u64(payload, 42)?;
    let _nseqs = read_payload_u64(payload, 50)?;
    let stats = WorkerSearchStats {
        n_past_msv: read_payload_u64(payload, 58)?,
        n_past_bias: read_payload_u64(payload, 66)?,
        n_past_vit: read_payload_u64(payload, 74)?,
        n_past_fwd: read_payload_u64(payload, 82)?,
        nreported: read_payload_u64(payload, 98)?,
        nincluded: read_payload_u64(payload, 106)?,
    };
    let nhits_u64 = read_payload_u64(payload, 90)?;
    let nhits = usize::try_from(nhits_u64)
        .map_err(|_| std::io::Error::other("worker search payload hit count too large"))?;
    if nhits == 0 {
        if payload.len() != HMMD_SEARCH_STATS_SIZE {
            return Err(std::io::Error::other(
                "zero-hit worker payload has trailing bytes",
            ));
        }
        let sentinel = read_payload_u64(payload, 114)?;
        if sentinel != u64::MAX {
            return Err(std::io::Error::other(
                "zero-hit worker payload missing offset sentinel",
            ));
        }
        return Ok(ParsedWorkerPayload {
            stats,
            hits: Vec::new(),
        });
    }
    let offsets_start: usize = 114;
    let offset_bytes = nhits
        .checked_mul(8)
        .ok_or_else(|| std::io::Error::other("worker hit offset table overflow"))?;
    let stats_size = offsets_start
        .checked_add(offset_bytes)
        .ok_or_else(|| std::io::Error::other("worker hit offset table overflow"))?;
    if stats_size > payload.len() {
        return Err(std::io::Error::other("short worker hit offset table"));
    }

    let mut hit_offsets = Vec::with_capacity(nhits);
    for i in 0..nhits {
        let offset_pos = offsets_start
            .checked_add(
                i.checked_mul(8)
                    .ok_or_else(|| std::io::Error::other("worker hit offset table overflow"))?,
            )
            .ok_or_else(|| std::io::Error::other("worker hit offset table overflow"))?;
        let offset = usize::try_from(read_payload_u64(payload, offset_pos)?)
            .map_err(|_| std::io::Error::other("worker hit offset too large"))?;
        hit_offsets.push(offset);
    }
    let hits = match parse_worker_hits_with_offsets(payload, stats_size, &hit_offsets).ok() {
        Some(hits) => hits,
        None => parse_worker_hits_sequential(payload, nhits)?,
    };

    Ok(ParsedWorkerPayload { stats, hits })
}

fn parse_worker_hits_with_offsets(
    payload: &[u8],
    hit_base: usize,
    hit_offsets: &[usize],
) -> std::io::Result<Vec<SerializedHit>> {
    if !hit_offsets.windows(2).all(|window| window[0] < window[1]) {
        return Err(std::io::Error::other("invalid worker hit offset order"));
    }

    let mut hits = Vec::with_capacity(hit_offsets.len());
    for (idx, offset) in hit_offsets.iter().copied().enumerate() {
        let start = hit_base
            .checked_add(offset)
            .ok_or_else(|| std::io::Error::other("worker hit offset overflow"))?;
        let end = if let Some(next_offset) = hit_offsets.get(idx + 1).copied() {
            hit_base
                .checked_add(next_offset)
                .ok_or_else(|| std::io::Error::other("worker hit offset overflow"))?
        } else {
            payload.len()
        };
        hits.push(parse_serialized_hit_slice(payload, start, end)?);
    }
    Ok(hits)
}

fn parse_worker_hits_sequential(
    payload: &[u8],
    nhits: usize,
) -> std::io::Result<Vec<SerializedHit>> {
    let mut hits = Vec::with_capacity(nhits);
    let mut start = HMMD_SEARCH_STATS_SIZE;
    for _ in 0..nhits {
        let end = serialized_hit_payload_end(payload, start)?;
        hits.push(parse_serialized_hit_slice(payload, start, end)?);
        start = end;
    }
    if start != payload.len() {
        return Err(std::io::Error::other("invalid worker hit trailing payload"));
    }
    Ok(hits)
}

fn parse_serialized_hit_slice(
    payload: &[u8],
    start: usize,
    end: usize,
) -> std::io::Result<SerializedHit> {
    if start >= end || end > payload.len() {
        return Err(std::io::Error::other("invalid worker hit size"));
    }
    validate_serialized_hit_payload(&payload[start..end])?;
    let sortkey = f64::from_bits(u64::from_be_bytes(
        payload[start + 8..start + 16].try_into().unwrap(),
    ));
    let name = serialized_hit_name(&payload[start..end])?;
    let (first_domain_iali, first_domain_jali) =
        serialized_hit_first_domain_alipos(&payload[start..end])?;
    Ok(SerializedHit {
        sortkey,
        name,
        first_domain_iali,
        first_domain_jali,
        bytes: payload[start..end].to_vec(),
    })
}

pub(super) fn serialized_hit_name(bytes: &[u8]) -> std::io::Result<String> {
    let hit_size = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let name_start = P7_HIT_BASE_SIZE;
    let Some(name_len) = bytes[name_start..hit_size].iter().position(|&b| b == 0) else {
        return Err(std::io::Error::other("worker hit name is unterminated"));
    };
    std::str::from_utf8(&bytes[name_start..name_start + name_len])
        .map(|name| name.to_string())
        .map_err(|e| std::io::Error::other(format!("worker hit name is not valid UTF-8: {e}")))
}

pub(super) fn serialized_hit_first_domain_alipos(bytes: &[u8]) -> std::io::Result<(i64, i64)> {
    let ndom = i32::from_be_bytes(bytes[72..76].try_into().unwrap());
    if ndom <= 0 {
        return Ok((0, 1));
    }
    let hit_size = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let domain_start = hit_size;
    if domain_start + P7_DOMAIN_BASE_SIZE > bytes.len() {
        return Err(std::io::Error::other("short worker domain payload"));
    }
    let iali = i64::from_be_bytes(
        bytes[domain_start + 20..domain_start + 28]
            .try_into()
            .unwrap(),
    );
    let jali = i64::from_be_bytes(
        bytes[domain_start + 28..domain_start + 36]
            .try_into()
            .unwrap(),
    );
    Ok((iali, jali))
}

pub(super) fn validate_serialized_hit_payload(bytes: &[u8]) -> std::io::Result<()> {
    let end = serialized_hit_payload_end(bytes, 0)?;
    if end != bytes.len() {
        return Err(std::io::Error::other("invalid worker hit trailing payload"));
    }
    Ok(())
}

pub(super) fn serialized_hit_payload_end(bytes: &[u8], start: usize) -> std::io::Result<usize> {
    if start > bytes.len() {
        return Err(std::io::Error::other("short worker hit payload"));
    }
    let bytes = &bytes[start..];
    if bytes.len() < P7_HIT_BASE_SIZE {
        return Err(std::io::Error::other("short worker hit payload"));
    }
    let hit_size = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
    if hit_size < P7_HIT_BASE_SIZE || hit_size > bytes.len() {
        return Err(std::io::Error::other("invalid worker hit size"));
    }

    let ndom = i32::from_be_bytes(bytes[72..76].try_into().unwrap());
    if ndom < 0 {
        return Err(std::io::Error::other("invalid worker domain count"));
    }

    let presence = bytes[108];
    let mut pos = P7_HIT_BASE_SIZE;
    take_serialized_c_string(bytes, &mut pos, hit_size, "worker hit name")?;
    if presence & P7_HIT_ACC_PRESENT != 0 {
        take_serialized_c_string(bytes, &mut pos, hit_size, "worker hit accession")?;
    }
    if presence & P7_HIT_DESC_PRESENT != 0 {
        take_serialized_c_string(bytes, &mut pos, hit_size, "worker hit description")?;
    }
    if pos != hit_size {
        return Err(std::io::Error::other("invalid worker hit string layout"));
    }

    for _ in 0..ndom {
        pos = validate_serialized_domain_payload(bytes, pos)?;
    }
    Ok(start + pos)
}

fn validate_serialized_domain_payload(bytes: &[u8], start: usize) -> std::io::Result<usize> {
    if start + P7_DOMAIN_BASE_SIZE > bytes.len() {
        return Err(std::io::Error::other("short worker domain payload"));
    }
    let domain_size = u32::from_be_bytes(bytes[start..start + 4].try_into().unwrap()) as usize;
    if domain_size != P7_DOMAIN_BASE_SIZE {
        return Err(std::io::Error::other("invalid worker domain size"));
    }
    validate_serialized_alidisplay_payload(bytes, start + domain_size)
}

fn validate_serialized_alidisplay_payload(bytes: &[u8], start: usize) -> std::io::Result<usize> {
    if start + P7_ALIDISPLAY_BASE_SIZE > bytes.len() {
        return Err(std::io::Error::other("short worker alignment payload"));
    }
    let ad_size = u32::from_be_bytes(bytes[start..start + 4].try_into().unwrap()) as usize;
    if ad_size < P7_ALIDISPLAY_BASE_SIZE || start + ad_size > bytes.len() {
        return Err(std::io::Error::other(
            "invalid worker alignment payload size",
        ));
    }
    let n = i32::from_be_bytes(bytes[start + 4..start + 8].try_into().unwrap());
    if n < 0 {
        return Err(std::io::Error::other("invalid worker alignment length"));
    }
    let n = n as usize;
    let end = start + ad_size;
    let presence = bytes[start + 44];
    let mut pos = start + P7_ALIDISPLAY_BASE_SIZE;
    if presence & P7_ALIDISPLAY_RFLINE_PRESENT != 0 {
        take_fixed_serialized_c_string(bytes, &mut pos, end, n, "worker alignment RF line")?;
    }
    if presence & P7_ALIDISPLAY_MMLINE_PRESENT != 0 {
        take_fixed_serialized_c_string(bytes, &mut pos, end, n, "worker alignment MM line")?;
    }
    if presence & P7_ALIDISPLAY_CSLINE_PRESENT != 0 {
        take_fixed_serialized_c_string(bytes, &mut pos, end, n, "worker alignment CS line")?;
    }
    take_fixed_serialized_c_string(bytes, &mut pos, end, n, "worker alignment model line")?;
    take_fixed_serialized_c_string(bytes, &mut pos, end, n, "worker alignment match line")?;
    if presence & P7_ALIDISPLAY_ASEQ_PRESENT != 0 {
        take_fixed_serialized_c_string(bytes, &mut pos, end, n, "worker alignment sequence line")?;
    }
    if presence & P7_ALIDISPLAY_NTSEQ_PRESENT != 0 {
        take_fixed_serialized_c_string(
            bytes,
            &mut pos,
            end,
            n.saturating_mul(3),
            "worker alignment nucleotide line",
        )?;
    }
    if presence & P7_ALIDISPLAY_PPLINE_PRESENT != 0 {
        take_fixed_serialized_c_string(bytes, &mut pos, end, n, "worker alignment posterior line")?;
    }
    for label in [
        "worker alignment HMM name",
        "worker alignment HMM accession",
        "worker alignment HMM description",
        "worker alignment sequence name",
        "worker alignment sequence accession",
        "worker alignment sequence description",
    ] {
        take_serialized_c_string(bytes, &mut pos, end, label)?;
    }
    if pos != end {
        return Err(std::io::Error::other(
            "invalid worker alignment trailing payload",
        ));
    }
    Ok(end)
}

fn take_fixed_serialized_c_string(
    bytes: &[u8],
    pos: &mut usize,
    end: usize,
    len: usize,
    label: &str,
) -> std::io::Result<()> {
    let next = pos
        .checked_add(len)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| std::io::Error::other(format!("{label} length overflow")))?;
    if next > end || bytes[next - 1] != 0 {
        return Err(std::io::Error::other(format!("{label} is malformed")));
    }
    *pos = next;
    Ok(())
}

fn take_serialized_c_string(
    bytes: &[u8],
    pos: &mut usize,
    end: usize,
    label: &str,
) -> std::io::Result<()> {
    if *pos >= end {
        return Err(std::io::Error::other(format!("{label} is missing")));
    }
    let Some(offset) = bytes[*pos..end].iter().position(|&b| b == 0) else {
        return Err(std::io::Error::other(format!("{label} is unterminated")));
    };
    *pos += offset + 1;
    Ok(())
}

fn read_payload_u64(payload: &[u8], offset: usize) -> std::io::Result<u64> {
    if offset + 8 > payload.len() {
        return Err(std::io::Error::other("short worker stats payload"));
    }
    Ok(u64::from_be_bytes(
        payload[offset..offset + 8].try_into().unwrap(),
    ))
}
