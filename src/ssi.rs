//! Simple sequence/HMM index for fast random access by name.
//!
//! The in-memory index is used for direct Rust lookup. `write_hmm_ssi()` emits
//! Easel SSI v3 files compatible with C HMMER's `hmmfetch --index`.

use std::collections::HashMap;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::errors::{HmmerError, HmmerResult};
use crate::hmmfile;
use crate::hmmfile_binary;

const SSI_V30_MAGIC: u32 = 0xd3d3c9b3;
const SSI_OFFSZ: u32 = 8;
const SSI_OFFSZ_32: u32 = 4;
const MAX_SSI_FIELD_LEN: usize = 1 << 20;

#[derive(Debug, Clone)]
struct PrimaryKey {
    key: String,
    offset: u64,
}

#[derive(Debug, Clone)]
struct SecondaryKey {
    key: String,
    primary_key: String,
}

/// Return `<path><suffix>` without formatting the path through UTF-8.
pub fn path_with_appended_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut path = path.as_os_str().to_os_string();
    path.push(suffix);
    PathBuf::from(path)
}

/// An index mapping names to file offsets.
#[derive(Debug)]
pub struct Index {
    /// Map from name to byte offset in the file
    pub name_to_offset: HashMap<String, u64>,
    /// Map from accession to byte offset
    pub acc_to_offset: HashMap<String, u64>,
    primary_keys: Vec<PrimaryKey>,
    secondary_keys: Vec<SecondaryKey>,
}

/// An Easel SSI v3 index loaded from disk.
#[derive(Debug)]
pub struct OnDiskIndex {
    pub indexed_path: PathBuf,
    primary_offsets: HashMap<String, u64>,
    secondary_to_primary: HashMap<String, String>,
}

impl OnDiskIndex {
    /// Look up `key` first as a primary HMM name, then as a secondary accession.
    pub fn lookup(&self, key: &str) -> Option<u64> {
        self.primary_offsets.get(key).copied().or_else(|| {
            self.secondary_to_primary
                .get(key)
                .and_then(|primary| self.primary_offsets.get(primary))
                .copied()
        })
    }
}

impl Index {
    /// Build an in-memory name/accession index from an ASCII or binary HMM file.
    /// Records each HMM's byte offset for SSI-style random access. Simpler
    /// in-memory alternative to the on-disk SSI files built by Easel's
    /// `esl_newssi_*` functions.
    pub fn build_from_hmm_file(path: &Path) -> HmmerResult<Self> {
        let records = scan_hmm_records(path)?;
        let mut name_to_offset = HashMap::new();
        let mut acc_to_offset = HashMap::new();
        let mut primary_keys = Vec::with_capacity(records.len());
        let mut secondary_keys = Vec::new();

        for record in records {
            if name_to_offset
                .insert(record.name.clone(), record.offset)
                .is_some()
            {
                return Err(HmmerError::Format(format!(
                    "Duplicate HMM name '{}' cannot be indexed",
                    record.name
                )));
            }
            primary_keys.push(PrimaryKey {
                key: record.name.clone(),
                offset: record.offset,
            });
            if let Some(acc) = record.acc {
                if acc_to_offset.insert(acc.clone(), record.offset).is_some() {
                    return Err(HmmerError::Format(format!(
                        "Duplicate HMM accession '{}' cannot be indexed",
                        acc
                    )));
                }
                secondary_keys.push(SecondaryKey {
                    key: acc,
                    primary_key: record.name,
                });
            }
        }

        Ok(Index {
            name_to_offset,
            acc_to_offset,
            primary_keys,
            secondary_keys,
        })
    }

    /// Look up `key` first as a name, then as an accession, and return the
    /// file offset of the containing HMM record (or `None` if not found).
    /// Analog of `esl_ssi_FindName()` from Easel's `esl_ssi.c`.
    pub fn lookup(&self, key: &str) -> Option<u64> {
        self.name_to_offset
            .get(key)
            .or_else(|| self.acc_to_offset.get(key))
            .copied()
    }

    /// Number of indexed primary names (excludes accession-only aliases).
    pub fn len(&self) -> usize {
        self.name_to_offset.len()
    }

    pub fn is_empty(&self) -> bool {
        self.name_to_offset.is_empty()
    }
}

struct HmmRecord {
    name: String,
    acc: Option<String>,
    offset: u64,
}

fn scan_hmm_records(path: &Path) -> HmmerResult<Vec<HmmRecord>> {
    if hmmfile_binary::looks_like_binary_hmm_file(path)? {
        return scan_binary_hmm_records(path);
    }
    scan_ascii_hmm_records(path)
}

fn scan_ascii_hmm_records(path: &Path) -> HmmerResult<Vec<HmmRecord>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    let mut records = Vec::new();
    let mut record_offset: Option<u64> = None;
    let mut record_text = String::new();

    loop {
        let offset = reader.stream_position().map_err(HmmerError::Io)?;
        let Some(line) =
            hmmfile::read_capped_text_line(&mut reader, hmmfile::MAX_ASCII_HMM_LINE_LEN)?
        else {
            break;
        };

        let trimmed = line.trim();
        if trimmed.starts_with("HMMER3/") {
            if record_offset.is_some() {
                return Err(HmmerError::Format(
                    "Missing // terminator before next HMM record".to_string(),
                ));
            }
            record_offset = Some(offset);
            record_text.clear();
        }
        if record_offset.is_some() {
            record_text.push_str(&line);
        }
        if trimmed == "//" {
            finish_ascii_record(&mut records, record_offset.take(), &record_text)?;
            record_text.clear();
        }
    }

    if record_offset.is_some() {
        return Err(HmmerError::Format(
            "Missing // terminator at end of HMM file".to_string(),
        ));
    }
    Ok(records)
}

fn scan_binary_hmm_records(path: &Path) -> HmmerResult<Vec<HmmRecord>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    let mut records = Vec::new();

    loop {
        let offset = reader.stream_position().map_err(HmmerError::Io)?;
        let Some(hmm) = hmmfile_binary::read_binary_hmm(&mut reader)? else {
            break;
        };
        records.push(HmmRecord {
            name: hmm.name,
            acc: hmm.acc,
            offset,
        });
    }

    Ok(records)
}

fn finish_ascii_record(
    records: &mut Vec<HmmRecord>,
    record_offset: Option<u64>,
    record_text: &str,
) -> HmmerResult<()> {
    let Some(offset) = record_offset else {
        return Ok(());
    };
    let mut hmms =
        hmmfile::read_hmms(BufReader::new(Cursor::new(record_text.as_bytes().to_vec())))?;
    if hmms.len() != 1 {
        return Err(HmmerError::Format(format!(
            "Expected one HMM record at offset {offset}, found {}",
            hmms.len()
        )));
    }
    let hmm = hmms.remove(0);
    let name = hmm.name;
    if name.is_empty() {
        return Err(HmmerError::Format(
            "Every HMM must have a name to be indexed".to_string(),
        ));
    }
    let acc = hmm.acc;
    records.push(HmmRecord { name, acc, offset });
    Ok(())
}

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

fn fixed_len(s: &str) -> usize {
    s.len() + 1
}

fn fixed_path_len(path: &Path) -> usize {
    path_bytes(path).len() + 1
}

fn write_fixed_str<W: Write>(w: &mut W, s: &str, len: usize) -> HmmerResult<()> {
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

fn checked_add_usize(a: usize, b: usize, field: &str) -> HmmerResult<usize> {
    a.checked_add(b)
        .ok_or_else(|| HmmerError::Format(format!("{field} size overflows usize")))
}

fn checked_mul_usize(a: usize, b: usize, field: &str) -> HmmerResult<usize> {
    a.checked_mul(b)
        .ok_or_else(|| HmmerError::Format(format!("{field} size overflows usize")))
}

fn checked_u32_usize(value: usize, field: &str) -> HmmerResult<u32> {
    u32::try_from(value)
        .map_err(|_| HmmerError::Format(format!("{field} exceeds u32 range: {value}")))
}

fn checked_u64_usize(value: usize, field: &str) -> HmmerResult<u64> {
    u64::try_from(value)
        .map_err(|_| HmmerError::Format(format!("{field} exceeds u64 range: {value}")))
}

#[cfg(unix)]
fn path_bytes(path: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;

    path.as_os_str().as_bytes().to_vec()
}

pub fn path_file_name(path: &Path) -> PathBuf {
    path.file_name()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(""))
}

#[cfg(not(unix))]
fn path_bytes(path: &Path) -> Vec<u8> {
    path.to_string_lossy().as_bytes().to_vec()
}

#[cfg(unix)]
fn path_from_bytes(bytes: &[u8]) -> PathBuf {
    use std::os::unix::ffi::OsStringExt;

    PathBuf::from(std::ffi::OsString::from_vec(bytes.to_vec()))
}

#[cfg(not(unix))]
fn path_from_bytes(bytes: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(bytes).into_owned())
}

fn write_u16<W: Write>(w: &mut W, v: u16) -> HmmerResult<()> {
    w.write_all(&v.to_be_bytes()).map_err(HmmerError::Io)
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> HmmerResult<()> {
    w.write_all(&v.to_be_bytes()).map_err(HmmerError::Io)
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> HmmerResult<()> {
    w.write_all(&v.to_be_bytes()).map_err(HmmerError::Io)
}

fn write_i64<W: Write>(w: &mut W, v: i64) -> HmmerResult<()> {
    w.write_all(&v.to_be_bytes()).map_err(HmmerError::Io)
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

fn validate_fixed_string_padding(buf: &[u8]) -> HmmerResult<usize> {
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

#[cfg(test)]
#[path = "../tests/unit/ssi_test.rs"]
mod tests;
