use super::*;

/// Validate that hydrated pressed optimized profiles line up with `.h3m`
/// HMM records in count and order before scan code indexes them in parallel.
pub fn validate_pressed_oprofiles_match_hmms(
    hmms: &[Hmm],
    oprofiles: &[OProfile],
) -> HmmerResult<()> {
    if oprofiles.len() != hmms.len() {
        return Err(HmmerError::Format(format!(
            ".h3f/.h3p contain {} records but .h3m contains {}",
            oprofiles.len(),
            hmms.len()
        )));
    }

    for (idx, (hmm, om)) in hmms.iter().zip(oprofiles.iter()).enumerate() {
        let expected_abc = alphabet_dimensions(alphabet_type_code(hmm.abc_type)?, ".h3m")?;
        if hmm.name != om.name
            || hmm.m != om.m
            || hmm.abc_k != om.abc_k
            || expected_abc.1 != om.abc_kp
        {
            return Err(HmmerError::Format(format!(
                "Pressed optimized profile {} does not match .h3m record {}: .h3m has {} M={} K={} Kp={}, .h3f/.h3p has {} M={} K={} Kp={}",
                idx + 1,
                idx + 1,
                hmm.name,
                hmm.m,
                hmm.abc_k,
                expected_abc.1,
                om.name,
                om.m,
                om.abc_k,
                om.abc_kp
            )));
        }
    }

    Ok(())
}

pub(super) fn validate_h3i_against_h3m(
    path: &Path,
    hmm_path: &Path,
    h3m_path: &Path,
) -> HmmerResult<()> {
    let mut file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let h3i_len = file.metadata().map_err(HmmerError::Io)?.len();
    let magic = read_u32_be(&mut file).map_err(|e| {
        if matches!(e, HmmerError::Io(ref io) if io.kind() == ErrorKind::UnexpectedEof) {
            HmmerError::Format(format!(
                "Pressed SSI sidecar {} is truncated",
                path.display()
            ))
        } else {
            e
        }
    })?;
    if magic != SSI_V30_MAGIC {
        return Err(HmmerError::Format(format!(
            "Bad .h3i SSI magic in {}: {:#x}",
            path.display(),
            magic
        )));
    }
    let _flags = read_u32_be(&mut file)?;
    let offsz = read_u32_be(&mut file)?;
    let nfiles = read_u16_be(&mut file)?;
    let nprimary = read_u64_be(&mut file)?;
    let nsecondary = read_u64_be(&mut file)?;
    let flen = read_u32_be(&mut file)? as usize;
    let plen = read_u32_be(&mut file)? as usize;
    let slen = read_u32_be(&mut file)? as usize;
    let frecsize = read_u32_be(&mut file)? as usize;
    let precsize = read_u32_be(&mut file)? as usize;
    let srecsize = read_u32_be(&mut file)? as usize;
    let foffset = read_u64_be(&mut file)?;
    let poffset = read_u64_be(&mut file)?;
    let soffset = read_u64_be(&mut file)?;

    if offsz != 8 || nfiles != 1 {
        return Err(HmmerError::Format(format!(
            "Pressed SSI sidecar {} has unsupported header: offsz={} nfiles={}",
            path.display(),
            offsz,
            nfiles
        )));
    }
    validate_h3i_fixed_lengths(path, flen, plen, slen)?;
    let expected_frecsize =
        checked_add_usize(flen, 4 * std::mem::size_of::<u32>(), ".h3i file record")?;
    let expected_precsize = checked_add_usize(
        checked_add_usize(plen, std::mem::size_of::<u16>(), ".h3i primary record")?,
        2 * 8 + 8,
        ".h3i primary record",
    )?;
    let expected_srecsize = checked_add_usize(slen, plen, ".h3i secondary record")?;
    if frecsize != expected_frecsize
        || precsize != expected_precsize
        || srecsize != expected_srecsize
    {
        return Err(HmmerError::Format(format!(
            "Pressed SSI sidecar {} has inconsistent record sizes",
            path.display()
        )));
    }
    validate_h3i_extent(
        path,
        "file table",
        foffset,
        frecsize as u64,
        nfiles as u64,
        h3i_len,
    )?;
    validate_h3i_extent(
        path,
        "primary table",
        poffset,
        precsize as u64,
        nprimary,
        h3i_len,
    )?;
    validate_h3i_extent(
        path,
        "secondary table",
        soffset,
        srecsize as u64,
        nsecondary,
        h3i_len,
    )?;

    file.seek(SeekFrom::Start(foffset))
        .map_err(HmmerError::Io)?;
    let indexed_file = read_fixed_string(&mut file, flen)?;
    if Path::new(&indexed_file).file_name() != hmm_path.file_name() {
        return Err(HmmerError::Format(format!(
            "Pressed SSI sidecar {} was built for {}, not {}",
            path.display(),
            indexed_file,
            hmm_path.display()
        )));
    }
    skip_exact(&mut file, 4 * std::mem::size_of::<u32>())?;

    let h3m_len = std::fs::metadata(h3m_path).map_err(HmmerError::Io)?.len();
    let mut h3m = std::fs::File::open(h3m_path).map_err(HmmerError::Io)?;
    let mut primary_names = HashSet::new();
    let mut primary_acc_by_name = HashMap::new();
    let mut last_primary_key: Option<String> = None;
    file.seek(SeekFrom::Start(poffset))
        .map_err(HmmerError::Io)?;
    for _ in 0..nprimary {
        let key = read_fixed_string(&mut file, plen)?;
        if key.is_empty() {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} contains empty primary key",
                path.display()
            )));
        }
        if !primary_names.insert(key.clone()) {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} contains duplicate primary key {}",
                path.display(),
                key
            )));
        }
        if last_primary_key
            .as_ref()
            .is_some_and(|previous| previous >= &key)
        {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} primary keys are not strictly sorted",
                path.display()
            )));
        }
        last_primary_key = Some(key.clone());
        let file_idx = read_u16_be(&mut file)?;
        let offset = read_u64_be(&mut file)?;
        let data_offset = read_u64_be(&mut file)?;
        let record_len = read_i64_be(&mut file)?;
        if data_offset != 0 || record_len != 0 {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} primary key {} has unsupported data_offset={} record_len={}",
                path.display(),
                key,
                data_offset,
                record_len
            )));
        }
        if file_idx != 0 || offset >= h3m_len {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} has invalid offset for key {}",
                path.display(),
                key
            )));
        }
        h3m.seek(SeekFrom::Start(offset)).map_err(HmmerError::Io)?;
        let Some(hmm) = hmmfile_binary::read_binary_hmm(&mut h3m)? else {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} offset for key {} does not point to an HMM",
                path.display(),
                key
            )));
        };
        if hmm.name != key {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} key {} points to HMM {}",
                path.display(),
                key,
                hmm.name
            )));
        }
        primary_acc_by_name.insert(key, hmm.acc.unwrap_or_default());
    }

    let mut h3m_scan = BufReader::new(std::fs::File::open(h3m_path).map_err(HmmerError::Io)?);
    let mut h3m_count = 0u64;
    while hmmfile_binary::read_binary_hmm(&mut h3m_scan)?.is_some() {
        h3m_count += 1;
    }
    if h3m_count != nprimary {
        return Err(HmmerError::Format(format!(
            "Pressed SSI sidecar {} indexes {} primary keys but .h3m contains {} records",
            path.display(),
            nprimary,
            h3m_count
        )));
    }

    file.seek(SeekFrom::Start(soffset))
        .map_err(HmmerError::Io)?;
    let mut secondary_keys = HashSet::new();
    let mut last_secondary_key: Option<String> = None;
    for _ in 0..nsecondary {
        let acc = read_fixed_string(&mut file, slen)?;
        let primary = read_fixed_string(&mut file, plen)?;
        if acc.is_empty() {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} contains empty secondary key",
                path.display()
            )));
        }
        if !secondary_keys.insert(acc.clone()) {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} contains duplicate secondary key {}",
                path.display(),
                acc
            )));
        }
        if last_secondary_key
            .as_ref()
            .is_some_and(|previous| previous >= &acc)
        {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} secondary keys are not strictly sorted",
                path.display()
            )));
        }
        last_secondary_key = Some(acc.clone());
        let Some(expected_acc) = primary_acc_by_name.get(&primary) else {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} secondary key {} references missing primary {}",
                path.display(),
                acc,
                primary
            )));
        };
        if !primary_names.contains(&primary) || expected_acc != &acc {
            return Err(HmmerError::Format(format!(
                "Pressed SSI sidecar {} secondary key {} does not match primary {} accession",
                path.display(),
                acc,
                primary
            )));
        }
    }
    Ok(())
}

fn validate_h3i_fixed_lengths(
    path: &Path,
    flen: usize,
    plen: usize,
    slen: usize,
) -> HmmerResult<()> {
    if flen == 0 || plen == 0 {
        return Err(HmmerError::Format(format!(
            "Pressed SSI sidecar {} has zero-length required fixed field",
            path.display()
        )));
    }
    if flen > MAX_PRESSED_STRING_LEN
        || plen > MAX_PRESSED_STRING_LEN
        || slen > MAX_PRESSED_STRING_LEN
    {
        return Err(HmmerError::Format(format!(
            "Pressed SSI sidecar {} has excessive fixed field length",
            path.display()
        )));
    }
    Ok(())
}

fn validate_h3i_extent(
    path: &Path,
    label: &str,
    offset: u64,
    record_size: u64,
    count: u64,
    file_len: u64,
) -> HmmerResult<()> {
    if offset < SSI_HEADER_LEN {
        return Err(HmmerError::Format(format!(
            "Pressed SSI sidecar {} {label} starts inside the header",
            path.display()
        )));
    }
    let byte_len = record_size.checked_mul(count).ok_or_else(|| {
        HmmerError::Format(format!(
            "Pressed SSI sidecar {} {label} extent overflows",
            path.display()
        ))
    })?;
    let end = offset.checked_add(byte_len).ok_or_else(|| {
        HmmerError::Format(format!(
            "Pressed SSI sidecar {} {label} extent overflows",
            path.display()
        ))
    })?;
    if end > file_len {
        return Err(HmmerError::Format(format!(
            "Pressed SSI sidecar {} {label} extent exceeds file length",
            path.display()
        )));
    }
    Ok(())
}

fn read_u16_be<R: Read>(r: &mut R) -> HmmerResult<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u16::from_be_bytes(buf))
}

fn read_u32_be<R: Read>(r: &mut R) -> HmmerResult<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_u64_be<R: Read>(r: &mut R) -> HmmerResult<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u64::from_be_bytes(buf))
}

fn read_i64_be<R: Read>(r: &mut R) -> HmmerResult<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(i64::from_be_bytes(buf))
}

fn read_fixed_string<R: Read>(r: &mut R, len: usize) -> HmmerResult<String> {
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    let end = buf.iter().position(|&b| b == 0).ok_or_else(|| {
        HmmerError::Format("Pressed SSI fixed-width string is missing NUL terminator".to_string())
    })?;
    if buf[end + 1..].iter().any(|&b| b != 0) {
        return Err(HmmerError::Format(
            "Pressed SSI fixed-width string has nonzero padding after NUL terminator".to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&buf[..end]).to_string())
}

fn skip_exact<R: Read>(r: &mut R, len: usize) -> HmmerResult<()> {
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(HmmerError::Io)
}
