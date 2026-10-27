use super::*;

/// Load the Easel SSI v3 sidecar for an HMM file, if it exists.
pub fn read_hmm_ssi(hmm_path: &Path) -> HmmerResult<Option<OnDiskIndex>> {
    let ssi_path = path_with_appended_suffix(hmm_path, ".ssi");
    if !ssi_path.exists() {
        return Ok(None);
    }
    let on_disk = read_ssi_path(&ssi_path)?;
    if on_disk.indexed_path.file_name() != hmm_path.file_name() {
        return Err(HmmerError::Format(format!(
            "SSI index {} was built for {} not {}",
            ssi_path.display(),
            on_disk.indexed_path.display(),
            hmm_path.display()
        )));
    }
    Ok(Some(on_disk))
}

pub fn read_ssi_path(ssi_path: &Path) -> HmmerResult<OnDiskIndex> {
    let mut file = std::fs::File::open(ssi_path).map_err(HmmerError::Io)?;
    let file_len = file.metadata().map_err(HmmerError::Io)?.len();
    let magic = read_u32(&mut file)?;
    if magic != SSI_V30_MAGIC {
        return Err(HmmerError::Format(format!(
            "Bad SSI magic in {}: {magic:#x}",
            ssi_path.display()
        )));
    }
    let _flags = read_u32(&mut file)?;
    let offsz = read_u32(&mut file)?;
    let nfiles = read_u16(&mut file)?;
    let nprimary = read_u64(&mut file)?;
    let nsecondary = read_u64(&mut file)?;
    let flen = read_u32(&mut file)? as usize;
    let plen = read_u32(&mut file)? as usize;
    let slen = read_u32(&mut file)? as usize;
    let frecsize = read_u32(&mut file)? as usize;
    let precsize = read_u32(&mut file)? as usize;
    let srecsize = read_u32(&mut file)? as usize;
    let foffset = read_offset(&mut file, offsz)?;
    let poffset = read_offset(&mut file, offsz)?;
    let soffset = read_offset(&mut file, offsz)?;

    if (offsz != SSI_OFFSZ && offsz != SSI_OFFSZ_32) || nfiles != 1 {
        return Err(HmmerError::Format(format!(
            "Unsupported SSI header in {}: offsz={offsz} nfiles={nfiles}",
            ssi_path.display()
        )));
    }
    if flen > MAX_SSI_FIELD_LEN || plen > MAX_SSI_FIELD_LEN || slen > MAX_SSI_FIELD_LEN {
        return Err(HmmerError::Format(format!(
            "SSI index {} has excessive fixed field length",
            ssi_path.display()
        )));
    }
    if flen == 0 || plen == 0 || (nsecondary > 0 && slen == 0) {
        return Err(HmmerError::Format(format!(
            "SSI index {} has zero-length fixed key field",
            ssi_path.display()
        )));
    }
    let expected_frecsize = flen + 4 * std::mem::size_of::<u32>();
    let expected_precsize = plen + std::mem::size_of::<u16>() + 2 * offsz as usize + 8;
    let expected_srecsize = slen + plen;
    if frecsize != expected_frecsize
        || precsize != expected_precsize
        || srecsize != expected_srecsize
    {
        return Err(HmmerError::Format(format!(
            "SSI index {} has inconsistent record sizes",
            ssi_path.display()
        )));
    }
    validate_ssi_extent(
        ssi_path,
        "file table",
        foffset,
        frecsize as u64,
        nfiles as u64,
        file_len,
    )?;
    let header_len = file.stream_position().map_err(HmmerError::Io)?;
    if foffset < header_len || poffset < header_len || (nsecondary > 0 && soffset < header_len) {
        return Err(HmmerError::Format(format!(
            "SSI index {} table offset points into header",
            ssi_path.display()
        )));
    }
    validate_ssi_extent(
        ssi_path,
        "primary table",
        poffset,
        precsize as u64,
        nprimary,
        file_len,
    )?;
    validate_ssi_extent(
        ssi_path,
        "secondary table",
        soffset,
        srecsize as u64,
        nsecondary,
        file_len,
    )?;

    file.seek(SeekFrom::Start(foffset))
        .map_err(HmmerError::Io)?;
    let indexed_path = read_fixed_path(&mut file, flen)?;
    skip_exact(&mut file, 4 * std::mem::size_of::<u32>())?;

    let mut primary_offsets = HashMap::new();
    file.seek(SeekFrom::Start(poffset))
        .map_err(HmmerError::Io)?;
    for _ in 0..nprimary {
        let key = read_fixed_string(&mut file, plen)?;
        if key.is_empty() {
            return Err(HmmerError::Format(format!(
                "SSI index {} contains empty primary key",
                ssi_path.display()
            )));
        }
        let file_idx = read_u16(&mut file)?;
        let offset = read_offset(&mut file, offsz)?;
        let data_offset = read_offset(&mut file, offsz)?;
        let record_len = read_i64(&mut file)?;
        if file_idx != 0 || data_offset != 0 || record_len != 0 {
            return Err(HmmerError::Format(format!(
                "SSI index {} contains unsupported primary record for {key}",
                ssi_path.display()
            )));
        }
        if primary_offsets.insert(key.clone(), offset).is_some() {
            return Err(HmmerError::Format(format!(
                "SSI index {} contains duplicate primary key {key}",
                ssi_path.display()
            )));
        }
    }

    let mut secondary_to_primary = HashMap::new();
    file.seek(SeekFrom::Start(soffset))
        .map_err(HmmerError::Io)?;
    for _ in 0..nsecondary {
        let key = read_fixed_string(&mut file, slen)?;
        let primary = read_fixed_string(&mut file, plen)?;
        if key.is_empty() || primary.is_empty() {
            return Err(HmmerError::Format(format!(
                "SSI index {} contains empty secondary key",
                ssi_path.display()
            )));
        }
        if !primary_offsets.contains_key(&primary) {
            return Err(HmmerError::Format(format!(
                "SSI index {} secondary key {key} references missing primary {primary}",
                ssi_path.display()
            )));
        }
        if secondary_to_primary.insert(key.clone(), primary).is_some() {
            return Err(HmmerError::Format(format!(
                "SSI index {} contains duplicate secondary key {key}",
                ssi_path.display()
            )));
        }
    }

    Ok(OnDiskIndex {
        indexed_path,
        primary_offsets,
        secondary_to_primary,
    })
}

fn validate_ssi_extent(
    ssi_path: &Path,
    label: &str,
    offset: u64,
    record_size: u64,
    count: u64,
    file_len: u64,
) -> HmmerResult<()> {
    let byte_len = record_size.checked_mul(count).ok_or_else(|| {
        HmmerError::Format(format!(
            "SSI index {} {label} extent overflows",
            ssi_path.display()
        ))
    })?;
    let end = offset.checked_add(byte_len).ok_or_else(|| {
        HmmerError::Format(format!(
            "SSI index {} {label} extent overflows",
            ssi_path.display()
        ))
    })?;
    if end > file_len {
        return Err(HmmerError::Format(format!(
            "SSI index {} {label} extent exceeds file length",
            ssi_path.display()
        )));
    }
    Ok(())
}
fn read_fixed_string<R: Read>(r: &mut R, len: usize) -> HmmerResult<String> {
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    let end = validate_fixed_string_padding(&buf)?;
    Ok(String::from_utf8_lossy(&buf[..end]).to_string())
}

fn read_fixed_path<R: Read>(r: &mut R, len: usize) -> HmmerResult<PathBuf> {
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    let end = validate_fixed_string_padding(&buf)?;
    Ok(path_from_bytes(&buf[..end]))
}

pub(super) fn validate_fixed_string_padding(buf: &[u8]) -> HmmerResult<usize> {
    let end = buf.iter().position(|&b| b == 0).ok_or_else(|| {
        HmmerError::Format("SSI fixed-width string is missing NUL terminator".to_string())
    })?;
    if buf[end + 1..].iter().any(|&b| b != 0) {
        return Err(HmmerError::Format(
            "SSI fixed-width string has nonzero padding after NUL terminator".to_string(),
        ));
    }
    Ok(end)
}

fn skip_exact<R: Read>(r: &mut R, len: usize) -> HmmerResult<()> {
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(HmmerError::Io)
}

fn read_u16<R: Read>(r: &mut R) -> HmmerResult<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u16::from_be_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> HmmerResult<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> HmmerResult<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u64::from_be_bytes(buf))
}

fn read_offset<R: Read>(r: &mut R, offsz: u32) -> HmmerResult<u64> {
    match offsz {
        SSI_OFFSZ_32 => read_u32(r).map(u64::from),
        SSI_OFFSZ => read_u64(r),
        _ => Err(HmmerError::Format(format!(
            "Unsupported SSI offset size {offsz}"
        ))),
    }
}

fn read_i64<R: Read>(r: &mut R) -> HmmerResult<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(i64::from_be_bytes(buf))
}
