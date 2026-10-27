use super::*;

/// Write a C/Easel-compatible SSI v3 index for an ASCII HMM file.
///
/// HMM records have no data offset or length in C HMMER's `hmmfetch --index`,
/// so primary keys store only record offsets and accessions are secondary keys.
pub fn write_hmm_ssi(hmm_path: &Path) -> HmmerResult<(PathBuf, usize, usize)> {
    let index = Index::build_from_hmm_file(hmm_path)?;
    let ssi_path = path_with_appended_suffix(hmm_path, ".ssi");
    if ssi_path.exists() {
        return Err(HmmerError::Format(format!(
            "SSI index {} already exists; delete or rename it",
            ssi_path.display()
        )));
    }

    // C `hmmfetch --index` passes the full filename to esl_newssi_AddFile (which
    // sizes flen from strlen(path)+1) but stores only the basename via
    // esl_FileTail. Mirror that: size from the full path, store the basename.
    let stored_path = path_file_name(hmm_path);
    write_hmm_ssi_records_with_stored_path(
        hmm_path,
        &stored_path,
        &ssi_path,
        index.primary_keys.iter().map(|p| {
            let acc = index
                .secondary_keys
                .iter()
                .find(|s| s.primary_key == p.key)
                .map(|s| s.key.clone());
            (p.key.clone(), acc, p.offset)
        }),
        false,
    )
}
/// Write an Easel SSI v3 index from already-known HMM record offsets.
///
/// This is used by `hmmpress`, whose index points at generated `.h3m` binary
/// records rather than the original ASCII input file.
pub fn write_hmm_ssi_records<I>(
    indexed_path: &Path,
    ssi_path: &Path,
    records: I,
    overwrite: bool,
) -> HmmerResult<(PathBuf, usize, usize)>
where
    I: IntoIterator<Item = (String, Option<String>, u64)>,
{
    write_hmm_ssi_records_with_stored_path(indexed_path, indexed_path, ssi_path, records, overwrite)
}

/// Write an SSI v3 index while allowing the fixed file-table width and stored
/// file-table name to differ. Easel sizes the field from the path passed to
/// `esl_newssi_AddFile()`, but stores only its basename.
pub fn write_hmm_ssi_records_with_stored_path<I>(
    indexed_path: &Path,
    stored_path: &Path,
    ssi_path: &Path,
    records: I,
    overwrite: bool,
) -> HmmerResult<(PathBuf, usize, usize)>
where
    I: IntoIterator<Item = (String, Option<String>, u64)>,
{
    if ssi_path.exists() && !overwrite {
        return Err(HmmerError::Format(format!(
            "SSI index {} already exists; delete or rename it",
            ssi_path.display()
        )));
    }

    let mut primary = Vec::new();
    let mut secondary = Vec::new();
    for (name, acc, offset) in records {
        primary.push(PrimaryKey {
            key: name.clone(),
            offset,
        });
        if let Some(acc) = acc {
            if !acc.is_empty() {
                secondary.push(SecondaryKey {
                    key: acc,
                    primary_key: name,
                });
            }
        }
    }

    primary.sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));
    secondary.sort_by(|a, b| a.key.as_bytes().cmp(b.key.as_bytes()));

    let flen = fixed_path_len(indexed_path);
    let plen = primary.iter().map(|p| fixed_len(&p.key)).max().unwrap_or(0);
    let slen = secondary
        .iter()
        .map(|s| fixed_len(&s.key))
        .max()
        .unwrap_or(0);

    let frecsize = checked_add_usize(flen, 4 * std::mem::size_of::<u32>(), "SSI file record")?;
    let precsize = checked_add_usize(
        checked_add_usize(plen, std::mem::size_of::<u16>(), "SSI primary record")?,
        2 * SSI_OFFSZ as usize + 8,
        "SSI primary record",
    )?;
    let srecsize = checked_add_usize(slen, plen, "SSI secondary record")?;
    let foffset = checked_add_usize(
        checked_add_usize(
            checked_add_usize(
                9 * std::mem::size_of::<u32>(),
                2 * std::mem::size_of::<u64>(),
                "SSI header",
            )?,
            std::mem::size_of::<u16>(),
            "SSI header",
        )?,
        3 * SSI_OFFSZ as usize,
        "SSI header",
    )?;
    let poffset = checked_add_usize(foffset, frecsize, "SSI primary offset")?;
    let soffset = checked_add_usize(
        poffset,
        checked_mul_usize(precsize, primary.len(), "SSI secondary offset")?,
        "SSI secondary offset",
    )?;

    let mut out = std::fs::File::create(ssi_path).map_err(HmmerError::Io)?;
    write_u32(&mut out, SSI_V30_MAGIC)?;
    write_u32(&mut out, 0)?;
    write_u32(&mut out, SSI_OFFSZ)?;
    write_u16(&mut out, 1)?;
    write_u64(
        &mut out,
        checked_u64_usize(primary.len(), "SSI primary count")?,
    )?;
    write_u64(
        &mut out,
        checked_u64_usize(secondary.len(), "SSI secondary count")?,
    )?;
    write_u32(&mut out, checked_u32_usize(flen, "SSI file name length")?)?;
    write_u32(&mut out, checked_u32_usize(plen, "SSI primary key length")?)?;
    write_u32(
        &mut out,
        checked_u32_usize(slen, "SSI secondary key length")?,
    )?;
    write_u32(
        &mut out,
        checked_u32_usize(frecsize, "SSI file record size")?,
    )?;
    write_u32(
        &mut out,
        checked_u32_usize(precsize, "SSI primary record size")?,
    )?;
    write_u32(
        &mut out,
        checked_u32_usize(srecsize, "SSI secondary record size")?,
    )?;
    write_u64(
        &mut out,
        checked_u64_usize(foffset, "SSI file table offset")?,
    )?;
    write_u64(
        &mut out,
        checked_u64_usize(poffset, "SSI primary table offset")?,
    )?;
    write_u64(
        &mut out,
        checked_u64_usize(soffset, "SSI secondary table offset")?,
    )?;

    write_fixed_path(&mut out, stored_path, flen)?;
    write_u32(&mut out, 0)?; // HMM file format code
    write_u32(&mut out, 0)?; // file flags
    write_u32(&mut out, 0)?; // bpl
    write_u32(&mut out, 0)?; // rpl

    let mut previous: Option<&str> = None;
    for p in &primary {
        if previous == Some(p.key.as_str()) {
            return Err(HmmerError::Format(format!(
                "Duplicate HMM name '{}' cannot be indexed",
                p.key
            )));
        }
        previous = Some(&p.key);
        write_fixed_str(&mut out, &p.key, plen)?;
        write_u16(&mut out, 0)?;
        write_u64(&mut out, p.offset)?;
        write_u64(&mut out, 0)?;
        write_i64(&mut out, 0)?;
    }

    previous = None;
    for s in &secondary {
        if previous == Some(s.key.as_str()) {
            return Err(HmmerError::Format(format!(
                "Duplicate HMM accession '{}' cannot be indexed",
                s.key
            )));
        }
        previous = Some(&s.key);
        write_fixed_str(&mut out, &s.key, slen)?;
        write_fixed_str(&mut out, &s.primary_key, plen)?;
    }

    Ok((ssi_path.to_path_buf(), primary.len(), secondary.len()))
}

pub(super) fn fixed_len(s: &str) -> usize {
    s.len() + 1
}

fn fixed_path_len(path: &Path) -> usize {
    path_bytes(path).len() + 1
}

pub(super) fn write_fixed_str<W: Write>(w: &mut W, s: &str, len: usize) -> HmmerResult<()> {
    let mut buf = vec![0u8; len];
    let bytes = s.as_bytes();
    if bytes.contains(&0) {
        return Err(HmmerError::Format(
            "SSI fixed string contains embedded NUL byte".to_string(),
        ));
    }
    if bytes.len() >= len {
        return Err(HmmerError::Format(format!(
            "SSI string '{}' exceeds fixed field length {}",
            s, len
        )));
    }
    buf[..bytes.len()].copy_from_slice(bytes);
    w.write_all(&buf).map_err(HmmerError::Io)
}

fn write_fixed_path<W: Write>(w: &mut W, path: &Path, len: usize) -> HmmerResult<()> {
    let mut buf = vec![0u8; len];
    let bytes = path_bytes(path);
    if bytes.contains(&0) {
        return Err(HmmerError::Format(
            "SSI fixed path contains embedded NUL byte".to_string(),
        ));
    }
    if bytes.len() >= len {
        return Err(HmmerError::Format(format!(
            "SSI path '{}' exceeds fixed field length {}",
            path.display(),
            len
        )));
    }
    buf[..bytes.len()].copy_from_slice(&bytes);
    w.write_all(&buf).map_err(HmmerError::Io)
}
pub(super) fn write_u16<W: Write>(w: &mut W, v: u16) -> HmmerResult<()> {
    w.write_all(&v.to_be_bytes()).map_err(HmmerError::Io)
}

pub(super) fn write_u32<W: Write>(w: &mut W, v: u32) -> HmmerResult<()> {
    w.write_all(&v.to_be_bytes()).map_err(HmmerError::Io)
}

pub(super) fn write_u64<W: Write>(w: &mut W, v: u64) -> HmmerResult<()> {
    w.write_all(&v.to_be_bytes()).map_err(HmmerError::Io)
}

pub(super) fn write_i64<W: Write>(w: &mut W, v: i64) -> HmmerResult<()> {
    w.write_all(&v.to_be_bytes()).map_err(HmmerError::Io)
}
