//! Read/write C HMMER pressed database format (.h3f, .h3p, .h3i).
//! Enables reading databases created by C hmmpress.
#![allow(clippy::needless_range_loop)]

use std::collections::{HashMap, HashSet};
use std::io::{BufReader, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::alphabet::AlphabetType;
use crate::errors::{HmmerError, HmmerResult};
use crate::hmm::Hmm;
use crate::hmmfile_binary;
use crate::simd::oprofile::AlignedF32x4;
use crate::simd::oprofile::{nqb, nqf, nqw, OProfile};

mod write;
pub use write::*;
mod read;
pub use read::*;
mod h3i;
pub use h3i::*;

// Magic numbers from C HMMER impl_sse/io.c, v3f format.
const V3F_FMAGIC: u32 = 0xb3e6e6f3; // .h3f sentinel
#[allow(dead_code)]
const V3F_PMAGIC: u32 = 0xb3e6f0f3; // .h3p sentinel (used for future .h3p reader)

const SSI_V30_MAGIC: u32 = 0xd3d3c9b3;
const P7_MAXABET: usize = 20;
const P7_NOFFSETS: usize = 3;
const P7O_EXTRA_SB: usize = 17;
const MAX_PRESSED_MODEL_LENGTH: i32 = 1_000_000;
const MAX_PRESSED_STRING_LEN: usize = 1 << 20;
const SSI_HEADER_LEN: u64 = 9 * 4 + 2 * 8 + 2 + 3 * 8;

/// Read a native-endian `u32` from `r`.
fn read_u32_ne<R: Read>(r: &mut R) -> HmmerResult<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(u32::from_ne_bytes(buf))
}

fn read_record_magic_ne<R: Read>(r: &mut R) -> HmmerResult<Option<u32>> {
    let mut buf = [0u8; 4];
    match r.read(&mut buf[..1]) {
        Ok(0) => return Ok(None),
        Ok(1) => {}
        Ok(_) => unreachable!(),
        Err(e) => return Err(HmmerError::Io(e)),
    }
    r.read_exact(&mut buf[1..]).map_err(HmmerError::Io)?;
    Ok(Some(u32::from_ne_bytes(buf)))
}

/// Read a native-endian `i32` from `r`.
fn read_i32_ne<R: Read>(r: &mut R) -> HmmerResult<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(i32::from_ne_bytes(buf))
}

/// Read a native-endian `f32` from `r`.
fn read_f32_ne<R: Read>(r: &mut R) -> HmmerResult<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(f32::from_ne_bytes(buf))
}

/// Read one byte from `r`.
fn read_u8<R: Read>(r: &mut R) -> HmmerResult<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(buf[0])
}

/// Read exactly `n` bytes from `r` into a fresh `Vec<u8>`.
fn read_bytes<R: Read>(r: &mut R, n: usize) -> HmmerResult<Vec<u8>> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf).map_err(HmmerError::Io)?;
    Ok(buf)
}

fn read_nonnegative_len<R: Read>(r: &mut R, field: &str) -> HmmerResult<usize> {
    let len = read_i32_ne(r)?;
    if len < 0 {
        return Err(HmmerError::Format(format!(
            "Negative {field} in pressed profile record: {len}"
        )));
    }
    let len = len as usize;
    if len > MAX_PRESSED_STRING_LEN {
        return Err(HmmerError::Format(format!(
            "{field} in pressed profile record is too large: {len}"
        )));
    }
    Ok(len)
}

fn read_len_prefixed_cstr<R: Read>(r: &mut R, len: usize, field: &str) -> HmmerResult<String> {
    if len == 0 {
        return Ok(String::new());
    }
    let Some(stored_len) = len.checked_add(1) else {
        return Err(HmmerError::Format(format!(
            "{field} length overflows usize"
        )));
    };
    let bytes = read_bytes(r, stored_len)?;
    if bytes.last() != Some(&0) {
        return Err(HmmerError::Format(format!(
            "{field} is missing trailing NUL terminator"
        )));
    }
    if bytes[..len].contains(&0) {
        return Err(HmmerError::Format(format!(
            "{field} contains embedded NUL byte"
        )));
    }
    Ok(String::from_utf8_lossy(&bytes[..len]).to_string())
}

fn require_nonempty_required_string(value: String, field: &str) -> HmmerResult<String> {
    if value.is_empty() {
        return Err(HmmerError::Format(format!("{field} is empty")));
    }
    Ok(value)
}

fn validate_pressed_record_header(m: i32, abc_type: i32, sidecar: &str) -> HmmerResult<usize> {
    if !(1..=MAX_PRESSED_MODEL_LENGTH).contains(&m) {
        return Err(HmmerError::Format(format!(
            "Invalid {sidecar} model length: {m}"
        )));
    }
    if !matches!(abc_type, 1..=3) {
        return Err(HmmerError::Format(format!(
            "Invalid {sidecar} alphabet code: {abc_type}"
        )));
    }
    Ok(m as usize)
}

fn checked_mul_usize(a: usize, b: usize, field: &str) -> HmmerResult<usize> {
    a.checked_mul(b)
        .ok_or_else(|| HmmerError::Format(format!("{field} size overflows usize")))
}

fn checked_add_usize(a: usize, b: usize, field: &str) -> HmmerResult<usize> {
    a.checked_add(b)
        .ok_or_else(|| HmmerError::Format(format!("{field} size overflows usize")))
}

fn pressed_vector_bytes(factors: &[usize], field: &str) -> HmmerResult<usize> {
    factors
        .iter()
        .try_fold(1usize, |acc, &factor| checked_mul_usize(acc, factor, field))
}

fn alphabet_type_code(abc_type: AlphabetType) -> HmmerResult<i32> {
    match abc_type {
        AlphabetType::Rna => Ok(1),
        AlphabetType::Dna => Ok(2),
        AlphabetType::Amino => Ok(3),
        AlphabetType::Unknown => Err(HmmerError::Format(
            "Cannot press HMM with unknown alphabet".to_string(),
        )),
    }
}

/// Return `Ok(true)` when every pressed-database sidecar exists, `Ok(false)`
/// when none exist, and an error for partial sidecar sets.
pub fn pressed_db_sidecars_complete(hmm_path: &Path) -> HmmerResult<bool> {
    let paths = pressed_sidecar_paths(hmm_path);
    let existing = paths.iter().filter(|p| p.exists()).count();
    if existing == 0 {
        return Ok(false);
    }
    if existing != paths.len() {
        return Err(HmmerError::Format(format!(
            "Incomplete pressed database for {}: expected .h3f/.h3p/.h3i/.h3m sidecars",
            hmm_path.display()
        )));
    }
    Ok(true)
}

pub fn pressed_h3m_path(hmm_path: &Path) -> PathBuf {
    pressed_sidecar_paths(hmm_path)[3].clone()
}

/// Return `Ok(true)` only when a complete, minimally readable pressed database
/// exists. Return an error for partial or malformed sidecar sets, because C
/// HMMER treats those as pressed-database failures rather than silently falling
/// back to the source HMM.
pub fn pressed_db_available(hmm_path: &Path) -> HmmerResult<bool> {
    if !pressed_db_sidecars_complete(hmm_path)? {
        return Ok(false);
    }
    let paths = pressed_sidecar_paths(hmm_path);

    let mut h3f = std::fs::File::open(&paths[0]).map_err(HmmerError::Io)?;
    let Some(msv) = read_h3f_record(&mut h3f)? else {
        return Err(HmmerError::Format(format!(
            "Pressed MSV sidecar {} contains no records",
            paths[0].display()
        )));
    };
    let mut h3p = std::fs::File::open(&paths[1]).map_err(HmmerError::Io)?;
    let Some(profile) = read_h3p_record(&mut h3p)? else {
        return Err(HmmerError::Format(format!(
            "Pressed profile sidecar {} contains no records",
            paths[1].display()
        )));
    };
    if (msv.name.as_str(), msv.m, msv.abc_type)
        != (profile.name.as_str(), profile.m, profile.abc_type)
    {
        return Err(HmmerError::Format(format!(
            "Pressed sidecar first-record mismatch: .h3f has {} M={} alphabet={}, .h3p has {} M={} alphabet={}",
            msv.name, msv.m, msv.abc_type, profile.name, profile.m, profile.abc_type
        )));
    }

    validate_h3i_against_h3m(&paths[2], hmm_path, &paths[3])?;
    let h3m = std::fs::File::open(&paths[3]).map_err(HmmerError::Io)?;
    let mut h3m = BufReader::new(h3m);
    let Some(hmm) = hmmfile_binary::read_binary_hmm(&mut h3m)? else {
        return Err(HmmerError::Format(format!(
            "Pressed binary model sidecar {} contains no records",
            paths[3].display()
        )));
    };
    let h3m_abc = alphabet_type_code(hmm.abc_type)?;
    if (msv.name.as_str(), msv.m, msv.abc_type) != (hmm.name.as_str(), hmm.m, h3m_abc) {
        return Err(HmmerError::Format(format!(
            "Pressed sidecar/model mismatch: sidecars have {} M={} alphabet={}, .h3m has {} M={} alphabet={}",
            msv.name, msv.m, msv.abc_type, hmm.name, hmm.m, h3m_abc
        )));
    }

    let hmms = hmmfile_binary::read_binary_hmm_file(&paths[3])?;
    let oprofiles = read_pressed_oprofiles(hmm_path)?;
    validate_pressed_oprofiles_match_hmms(&hmms, &oprofiles)?;

    Ok(true)
}

/// Read all optimized profiles from a complete C-style pressed database.
///
/// Profiles are returned in the same order as their corresponding `.h3m`
/// records. Scans still need the `.h3m` HMMs for generic profile
/// configuration, thresholds, alignment, and domain definition.
pub fn read_pressed_oprofiles(hmm_path: &Path) -> HmmerResult<Vec<OProfile>> {
    let paths = pressed_sidecar_paths(hmm_path);
    let mut h3f = std::fs::File::open(&paths[0]).map_err(HmmerError::Io)?;
    let mut h3p = std::fs::File::open(&paths[1]).map_err(HmmerError::Io)?;
    let h3m = std::fs::File::open(&paths[3]).map_err(HmmerError::Io)?;
    let mut h3m = BufReader::new(h3m);

    let mut profiles = Vec::new();
    let mut max_h3p_position = 0_u64;
    loop {
        let current_foffset = h3f.stream_position().map_err(HmmerError::Io)? as i64;
        let current_moffset = h3m.stream_position().map_err(HmmerError::Io)? as i64;
        match read_h3f_record(&mut h3f)? {
            Some(msv) => {
                if msv.offsets[1] != current_foffset {
                    return Err(HmmerError::Format(format!(
                        "Pressed .h3f FOFFSET mismatch for {}: record starts at {}, offset says {}",
                        msv.name, current_foffset, msv.offsets[1]
                    )));
                }
                if msv.offsets[0] != current_moffset {
                    return Err(HmmerError::Format(format!(
                        "Pressed .h3f MOFFSET mismatch for {}: .h3m record starts at {}, offset says {}",
                        msv.name, current_moffset, msv.offsets[0]
                    )));
                }
                let Some(hmm) = hmmfile_binary::read_binary_hmm(&mut h3m)? else {
                    return Err(HmmerError::Format(format!(
                        "Pressed .h3f record {} has no matching .h3m record",
                        msv.name
                    )));
                };
                if msv.name != hmm.name {
                    return Err(HmmerError::Format(format!(
                        "Pressed .h3f record {} points to .h3m record {}",
                        msv.name, hmm.name
                    )));
                }
                let h3m_abc = alphabet_type_code(hmm.abc_type)?;
                if msv.abc_type != h3m_abc {
                    return Err(HmmerError::Format(format!(
                        "Pressed .h3f record {} alphabet {} does not match .h3m alphabet {}",
                        msv.name, msv.abc_type, h3m_abc
                    )));
                }
                let p_offset = msv.offsets[2];
                if p_offset < 0 {
                    return Err(HmmerError::Format(format!(
                        "Pressed database has negative .h3p offset for {}: {}",
                        msv.name, p_offset
                    )));
                }
                h3p.seek(SeekFrom::Start(p_offset as u64))
                    .map_err(HmmerError::Io)?;
                let Some(profile) = read_h3p_record(&mut h3p)? else {
                    return Err(HmmerError::Format(format!(
                        "Pressed database has .h3f record {} without matching .h3p record",
                        msv.name
                    )));
                };
                max_h3p_position =
                    max_h3p_position.max(h3p.stream_position().map_err(HmmerError::Io)?);
                profiles.push(oprofile_from_pressed_records(&msv, &profile)?);
            }
            None => break,
        }
    }
    let h3p_len = h3p.metadata().map_err(HmmerError::Io)?.len();
    if max_h3p_position != h3p_len {
        return Err(HmmerError::Format(format!(
            "Pressed .h3p sidecar has {} trailing bytes after matched records",
            h3p_len.saturating_sub(max_h3p_position)
        )));
    }

    Ok(profiles)
}

fn pressed_sidecar_paths(hmm_path: &Path) -> [PathBuf; 4] {
    let sidecar_path = |suffix: &str| {
        let mut path = hmm_path.as_os_str().to_os_string();
        path.push(suffix);
        PathBuf::from(path)
    };
    [
        sidecar_path(".h3f"),
        sidecar_path(".h3p"),
        sidecar_path(".h3i"),
        sidecar_path(".h3m"),
    ]
}

#[cfg(test)]
#[path = "../../tests/unit/pressed_test.rs"]
mod tests;
