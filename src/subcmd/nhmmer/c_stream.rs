use super::*;

#[derive(Debug)]
pub(super) struct NhmmerFmRecord {
    pub(super) term_loc: usize,
    pub(super) seq_offset: usize,
    pub(super) seq_count: usize,
    pub(super) ambig_offset: usize,
    pub(super) ambig_count: usize,
    pub(super) overlap_bases: usize,
    pub(super) text_bases_len: usize,
    pub(super) occ_b_offset: usize,
    pub(super) occ_b_count: usize,
    pub(super) occ_sb_offset: usize,
    pub(super) occ_sb_count: usize,
    pub(super) freq_sa: usize,
    pub(super) freq_cnt_b: usize,
    pub(super) bwt_len: usize,
    pub(super) packed_bwt_offset: usize,
    pub(super) packed_bwt_bytes: usize,
    pub(super) packed_bwt: Vec<u8>,
    pub(super) packed_text_offset: usize,
    pub(super) packed_text_bytes: usize,
    pub(super) packed_text: Vec<u8>,
    pub(super) raw_sa_offset: usize,
    pub(super) raw_sa_count: usize,
    pub(super) raw_sa_samples: Vec<i32>,
    pub(super) occ_b_codes: Vec<u16>,
    pub(super) occ_sb_codes: Vec<u32>,
    pub(super) text: Vec<u8>,
}

#[allow(clippy::type_complexity)]
pub(super) fn read_makehmmerdb_c_metadata_payload(
    cursor: &mut Cursor<&[u8]>,
) -> Result<
    (
        bool,
        usize,
        usize,
        usize,
        Vec<NhmmerFmSequenceMeta>,
        Vec<(usize, usize)>,
    ),
    String,
> {
    let fwd_only = read_u8(cursor)? != 0;
    let alph_type = read_u8(cursor)?;
    let alph_size = read_u8(cursor)?;
    let char_bits = read_u8(cursor)?;
    if (alph_type, alph_size, char_bits) != (0, 4, 2) {
        return Err("makehmmerdb FM-index is not a DNA 2-bit database".to_string());
    }
    let freq_sa = read_u32(cursor)? as usize;
    let _freq_cnt_sb = read_u32(cursor)? as usize;
    let freq_cnt_b = read_u32(cursor)? as usize;
    if freq_sa == 0 {
        return Err("makehmmerdb FM-index suffix-array sampling frequency is zero".to_string());
    }
    if freq_cnt_b == 0 {
        return Err("makehmmerdb FM-index occurrence sampling frequency is zero".to_string());
    }
    let block_count = read_u16(cursor)? as usize;
    let seq_count = read_u32(cursor)? as usize;
    let ambig_count = read_u32(cursor)? as usize;
    let _char_count = u64_to_usize(read_u64(cursor)?, "makehmmerdb metadata character count")?;

    let sequence_bytes = checked_count_bytes(
        seq_count,
        MAKEHMMERDB_C_MIN_SEQUENCE_META_BYTES,
        "makehmmerdb sequence metadata",
    )?;
    let ambiguity_bytes = checked_count_bytes(
        ambig_count,
        MAKEHMMERDB_C_AMBIGUITY_BYTES,
        "makehmmerdb ambiguity metadata",
    )?;
    let min_payload_bytes = sequence_bytes
        .checked_add(ambiguity_bytes)
        .ok_or_else(|| "makehmmerdb metadata payload byte span overflows usize".to_string())?;
    if makehmmerdb_remaining(cursor) < min_payload_bytes {
        return Err("makehmmerdb metadata payload is truncated".to_string());
    }

    let mut sequences = Vec::with_capacity(seq_count);
    for _ in 0..seq_count {
        let _target_id = read_u32(cursor)?;
        let target_start = u64_to_usize(read_u64(cursor)?, "makehmmerdb sequence target_start")?;
        let fm_start = read_u32(cursor)? as usize;
        let length = read_u32(cursor)? as usize;
        let name_len = read_u16(cursor)? as usize;
        let acc_len = read_u16(cursor)? as usize;
        let source_len = read_u16(cursor)? as usize;
        let desc_len = read_u16(cursor)? as usize;
        let name = read_nul_string(cursor, name_len)?;
        let acc = read_nul_string(cursor, acc_len)?;
        let _source = read_nul_string(cursor, source_len)?;
        let desc = read_nul_string(cursor, desc_len)?;
        sequences.push(NhmmerFmSequenceMeta {
            name,
            acc,
            desc,
            target_start,
            fm_start,
            length,
        });
    }
    ensure_makehmmerdb_remaining(
        cursor,
        ambig_count,
        MAKEHMMERDB_C_AMBIGUITY_BYTES,
        "makehmmerdb ambiguity metadata",
    )?;
    let mut ambiguities = Vec::with_capacity(ambig_count);
    for _ in 0..ambig_count {
        let lower = read_i32(cursor)?;
        let upper = read_i32(cursor)?;
        if lower < 0 || upper < lower {
            return Err("makehmmerdb ambiguity metadata has invalid coordinates".to_string());
        }
        ambiguities.push((lower as usize, upper as usize));
    }
    Ok((
        fwd_only,
        freq_sa,
        freq_cnt_b,
        block_count,
        sequences,
        ambiguities,
    ))
}

pub(super) fn read_makehmmerdb_c_fm_record(
    cursor: &mut Cursor<&[u8]>,
    freq_sa: usize,
    freq_cnt_b: usize,
    has_text_and_sa: bool,
    payload_base_offset: usize,
    read_raw_sa_samples: bool,
) -> Result<NhmmerFmRecord, String> {
    if freq_sa == 0 {
        return Err("makehmmerdb FM-index suffix-array sampling frequency is zero".to_string());
    }
    if freq_cnt_b == 0 {
        return Err("makehmmerdb FM-index occurrence sampling frequency is zero".to_string());
    }
    let n = u64_to_usize(read_u64(cursor)?, "makehmmerdb FM-index record length")?;
    let term_loc = read_u32(cursor)? as usize;
    let seq_offset = read_u32(cursor)? as usize;
    let ambig_offset = read_u32(cursor)? as usize;
    let overlap_bases = read_u32(cursor)? as usize;
    let seq_count = read_u32(cursor)? as usize;
    let ambig_count = read_u32(cursor)? as usize;

    let mut packed_text = Vec::new();
    let mut raw_sa_samples = Vec::new();
    let packed_len = n.div_ceil(4);
    let mut packed_text_offset = 0usize;
    let mut packed_text_bytes = 0usize;
    let text_bytes = if has_text_and_sa { packed_len } else { 0 };
    let bwt_bytes = packed_len;
    let sa_count = if has_text_and_sa {
        1usize
            .checked_add(n / freq_sa)
            .ok_or_else(|| "makehmmerdb FM-index suffix-array span overflows usize".to_string())?
    } else {
        0
    };
    let sa_bytes = checked_count_bytes(sa_count, 4, "makehmmerdb FM-index suffix array")?;
    let occ_b_count = 1usize
        .checked_add(n.div_ceil(freq_cnt_b))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| "makehmmerdb FM-index occurrence table span overflows usize".to_string())?;
    let occ_b_bytes = checked_count_bytes(occ_b_count, 2, "makehmmerdb FM-index occurrence table")?;
    let occ_sb_count = 1usize
        .checked_add(n.div_ceil(65_536))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| {
            "makehmmerdb FM-index superblock occurrence table span overflows usize".to_string()
        })?;
    let occ_sb_bytes = checked_count_bytes(
        occ_sb_count,
        4,
        "makehmmerdb FM-index superblock occurrence table",
    )?;
    let record_payload_bytes = text_bytes
        .checked_add(bwt_bytes)
        .and_then(|value| value.checked_add(sa_bytes))
        .and_then(|value| value.checked_add(occ_b_bytes))
        .and_then(|value| value.checked_add(occ_sb_bytes))
        .ok_or_else(|| "makehmmerdb FM-index record byte span overflows usize".to_string())?;
    if makehmmerdb_remaining(cursor) < record_payload_bytes {
        return Err("makehmmerdb FM-index record payload is truncated".to_string());
    }

    if has_text_and_sa {
        packed_text_offset = payload_base_offset
            .checked_add(cursor.position() as usize)
            .ok_or_else(|| "makehmmerdb FM-index text offset overflows usize".to_string())?;
        packed_text_bytes = text_bytes;
        if read_raw_sa_samples {
            packed_text = read_packed_dna_quads(cursor, n)?;
        } else {
            advance_makehmmerdb_cursor(cursor, text_bytes)?;
        }
        if n == 0 {
            return Err("makehmmerdb FM-index text record is missing terminal symbol".to_string());
        }
    }
    let packed_bwt_offset = payload_base_offset
        .checked_add(cursor.position() as usize)
        .ok_or_else(|| "makehmmerdb FM-index BWT offset overflows usize".to_string())?;
    let packed_bwt_bytes = bwt_bytes;
    let packed_bwt = if read_raw_sa_samples {
        read_packed_dna_quads(cursor, n)?
    } else {
        advance_makehmmerdb_cursor(cursor, bwt_bytes)?;
        Vec::new()
    };
    if term_loc >= n {
        return Err("makehmmerdb FM-index terminal location is outside BWT".to_string());
    }
    let raw_sa_offset = payload_base_offset
        .checked_add(cursor.position() as usize)
        .ok_or_else(|| "makehmmerdb FM-index suffix-array offset overflows usize".to_string())?;
    if has_text_and_sa {
        if read_raw_sa_samples {
            raw_sa_samples.reserve(sa_count);
            for _ in 0..sa_count {
                raw_sa_samples.push(read_i32(cursor)?);
            }
        } else {
            advance_makehmmerdb_cursor(cursor, sa_bytes)?;
        }
    }
    let occ_b_offset = payload_base_offset
        .checked_add(cursor.position() as usize)
        .ok_or_else(|| {
            "makehmmerdb FM-index occurrence table offset overflows usize".to_string()
        })?;
    let mut occ_b_codes = Vec::with_capacity(occ_b_count);
    for _ in 0..occ_b_count {
        occ_b_codes.push(read_u16(cursor)?);
    }
    let occ_sb_offset = payload_base_offset
        .checked_add(cursor.position() as usize)
        .ok_or_else(|| {
            "makehmmerdb FM-index superblock occurrence table offset overflows usize".to_string()
        })?;
    let mut occ_sb_codes = Vec::with_capacity(occ_sb_count);
    for _ in 0..occ_sb_count {
        occ_sb_codes.push(read_u32(cursor)?);
    }

    if has_text_and_sa {
        Ok(NhmmerFmRecord {
            term_loc,
            seq_offset,
            seq_count,
            ambig_offset,
            ambig_count,
            overlap_bases,
            text_bases_len: n,
            occ_b_offset,
            occ_b_count,
            occ_sb_offset,
            occ_sb_count,
            freq_sa,
            freq_cnt_b,
            bwt_len: n,
            packed_bwt_offset,
            packed_bwt_bytes,
            packed_bwt,
            packed_text_offset,
            packed_text_bytes,
            packed_text,
            raw_sa_offset,
            raw_sa_count: sa_count,
            raw_sa_samples,
            occ_b_codes,
            occ_sb_codes,
            text: Vec::new(),
        })
    } else {
        Ok(NhmmerFmRecord {
            term_loc,
            seq_offset,
            seq_count,
            ambig_offset,
            ambig_count,
            overlap_bases,
            text_bases_len: 0,
            occ_b_offset,
            occ_b_count,
            occ_sb_offset,
            occ_sb_count,
            freq_sa,
            freq_cnt_b,
            bwt_len: n,
            packed_bwt_offset,
            packed_bwt_bytes,
            packed_bwt,
            packed_text_offset,
            packed_text_bytes,
            packed_text,
            raw_sa_offset,
            raw_sa_count: sa_count,
            raw_sa_samples,
            occ_b_codes,
            occ_sb_codes,
            text: Vec::new(),
        })
    }
}

pub(super) fn unpack_dna_quads_range(
    packed: &[u8],
    start: usize,
    len: usize,
) -> Result<Vec<u8>, String> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| "packed FM DNA text range overflows usize".to_string())?;
    if packed.len() < end.div_ceil(4) {
        return Err("packed FM DNA text range is truncated".to_string());
    }
    let mut out = Vec::with_capacity(len);
    for pos in start..end {
        out.push(
            match packed_dna_quad_dsq_at(packed, pos)
                .ok_or_else(|| "packed FM DNA text range is truncated".to_string())?
            {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                3 => b'T',
                _ => unreachable!(),
            },
        );
    }
    Ok(out)
}

pub(super) fn unpack_mapped_dna_quads_range(
    packed: &MappedPackedBytes,
    start: usize,
    len: usize,
) -> Result<Vec<u8>, String> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| "packed FM DNA text range overflows usize".to_string())?;
    if packed.len < end.div_ceil(4) {
        return Err("packed FM DNA text range is truncated".to_string());
    }
    let mut out = Vec::with_capacity(len);
    for pos in start..end {
        out.push(
            match packed_dna_quad_dsq_at_mapped(packed, pos)
                .ok_or_else(|| "packed FM DNA text range is truncated".to_string())?
            {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                3 => b'T',
                _ => unreachable!(),
            },
        );
    }
    Ok(out)
}

fn read_packed_dna_quads<R: Read>(cursor: &mut R, len: usize) -> Result<Vec<u8>, String> {
    let mut packed = vec![0u8; len.div_ceil(4)];
    cursor
        .read_exact(&mut packed)
        .map_err(|e| format!("truncated packed FM DNA text: {e}"))?;
    Ok(packed)
}

fn advance_makehmmerdb_cursor(cursor: &mut Cursor<&[u8]>, byte_count: usize) -> Result<(), String> {
    let next = (cursor.position() as usize)
        .checked_add(byte_count)
        .filter(|&pos| pos <= cursor.get_ref().len())
        .ok_or_else(|| "makehmmerdb FM-index record payload is truncated".to_string())?;
    cursor.set_position(next as u64);
    Ok(())
}

pub(super) fn read_nul_string<R: Read>(cursor: &mut R, len: usize) -> Result<String, String> {
    let mut bytes = vec![0u8; len + 1];
    cursor
        .read_exact(&mut bytes)
        .map_err(|e| format!("truncated makehmmerdb string: {e}"))?;
    if bytes[len] != 0 {
        return Err("makehmmerdb string is missing NUL terminator".to_string());
    }
    if bytes[..len].contains(&0) {
        return Err("makehmmerdb string contains embedded NUL byte".to_string());
    }
    String::from_utf8(bytes[..len].to_vec())
        .map_err(|e| format!("makehmmerdb string is not UTF-8: {e}"))
}

pub(super) fn makehmmerdb_remaining(cursor: &Cursor<&[u8]>) -> usize {
    cursor
        .get_ref()
        .len()
        .saturating_sub(cursor.position() as usize)
}

pub(super) fn u64_to_usize(value: u64, what: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{what} exceeds usize"))
}

pub(super) fn checked_count_bytes(
    count: usize,
    element_size: usize,
    what: &str,
) -> Result<usize, String> {
    count
        .checked_mul(element_size)
        .ok_or_else(|| format!("{what} byte span overflows usize"))
}

pub(super) fn ensure_makehmmerdb_remaining(
    cursor: &Cursor<&[u8]>,
    count: usize,
    element_size: usize,
    what: &str,
) -> Result<(), String> {
    let bytes = checked_count_bytes(count, element_size, what)?;
    if makehmmerdb_remaining(cursor) < bytes {
        return Err(format!("{what} is truncated"));
    }
    Ok(())
}

fn read_u8<R: Read>(cursor: &mut R) -> Result<u8, String> {
    let mut buf = [0u8; 1];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| format!("truncated makehmmerdb byte: {e}"))?;
    Ok(buf[0])
}

fn read_u16<R: Read>(cursor: &mut R) -> Result<u16, String> {
    let mut buf = [0u8; 2];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| format!("truncated makehmmerdb u16: {e}"))?;
    Ok(u16::from_le_bytes(buf))
}

pub(super) fn read_u32<R: Read>(cursor: &mut R) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| format!("truncated makehmmerdb u32: {e}"))?;
    Ok(u32::from_le_bytes(buf))
}

pub(super) fn read_i32<R: Read>(cursor: &mut R) -> Result<i32, String> {
    let mut buf = [0u8; 4];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| format!("truncated makehmmerdb i32: {e}"))?;
    Ok(i32::from_le_bytes(buf))
}

pub(super) fn read_u64<R: Read>(cursor: &mut R) -> Result<u64, String> {
    let mut buf = [0u8; 8];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| format!("truncated makehmmerdb u64: {e}"))?;
    Ok(u64::from_le_bytes(buf))
}

pub(super) fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
