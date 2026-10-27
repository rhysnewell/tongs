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

mod scan;
use scan::*;
mod read;
pub use read::*;
mod write;
pub use write::*;

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

#[cfg(test)]
#[path = "../../tests/unit/ssi_test.rs"]
mod tests;
