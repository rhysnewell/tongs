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

fn write_i32_ne<W: Write>(w: &mut W, v: i32) -> HmmerResult<()> {
    w.write_all(&v.to_ne_bytes()).map_err(HmmerError::Io)
}

fn write_u32_ne<W: Write>(w: &mut W, v: u32) -> HmmerResult<()> {
    w.write_all(&v.to_ne_bytes()).map_err(HmmerError::Io)
}

fn write_f32_ne<W: Write>(w: &mut W, v: f32) -> HmmerResult<()> {
    w.write_all(&v.to_ne_bytes()).map_err(HmmerError::Io)
}

fn checked_i32_len(len: usize, field: &str) -> HmmerResult<i32> {
    i32::try_from(len)
        .map_err(|_| HmmerError::Format(format!("{field} length exceeds i32 range: {len}")))
}

fn checked_i32_usize(value: usize, field: &str) -> HmmerResult<i32> {
    i32::try_from(value)
        .map_err(|_| HmmerError::Format(format!("{field} exceeds i32 range: {value}")))
}

fn write_len_prefixed_cstr<W: Write>(w: &mut W, s: &str) -> HmmerResult<()> {
    if s.as_bytes().contains(&0) {
        return Err(HmmerError::Format(
            "Pressed profile string contains embedded NUL byte".to_string(),
        ));
    }
    write_i32_ne(w, checked_i32_len(s.len(), "Pressed profile string")?)?;
    w.write_all(s.as_bytes()).map_err(HmmerError::Io)?;
    w.write_all(&[0]).map_err(HmmerError::Io)
}

fn write_optional_len_prefixed_cstr<W: Write>(w: &mut W, s: Option<&str>) -> HmmerResult<()> {
    match s {
        Some(s) if !s.is_empty() => write_len_prefixed_cstr(w, s),
        _ => write_i32_ne(w, 0),
    }
}

fn write_annotation<W: Write>(w: &mut W, annotation: Option<&[u8]>, m: usize) -> HmmerResult<()> {
    let len = m + 2;
    match annotation {
        // C dumps `om->rf/mm/cs/consensus` of width M+2 (impl_sse/io.c:146-149).
        // Those arrays are `strcpy`'d from the gm strings (p7_oprofile.c:1112),
        // so index M+1 is the copied `'\0'` terminator. Our in-memory arrays
        // keep a `' '` (space) sentinel at index M+1; force the trailing byte to
        // a NUL to match C byte-for-byte (same class as the hmmfile_binary fix).
        Some(bytes) => {
            let mut buf = vec![0u8; len];
            let n = bytes.len().min(len);
            buf[..n].copy_from_slice(&bytes[..n]);
            buf[len - 1] = 0;
            w.write_all(&buf).map_err(HmmerError::Io)
        }
        None => {
            let zeros = vec![0u8; len];
            w.write_all(&zeros).map_err(HmmerError::Io)
        }
    }
}

/// Write one C HMMER3/f `.h3f` MSV-filter record.
pub fn write_h3f_record<W: Write>(
    w: &mut W,
    hmm: &Hmm,
    om: &OProfile,
    offsets: [i64; P7_NOFFSETS],
) -> HmmerResult<()> {
    let abc_type = alphabet_type_code(hmm.abc_type)?;
    let q16 = nqb(om.m);
    let q16x = q16 + P7O_EXTRA_SB;

    write_u32_ne(w, V3F_FMAGIC)?;
    write_i32_ne(w, checked_i32_usize(om.m, ".h3f model length")?)?;
    write_i32_ne(w, abc_type)?;
    write_len_prefixed_cstr(w, &om.name)?;
    // C `impl_sse/io.c:102` writes `om->max_length` here (the per-model upper
    // bound on emitted length, from the HMM `MAXL` line), NOT the configured L.
    write_i32_ne(w, om.max_length)?;
    w.write_all(&[om.tbm_b, om.tec_b, om.tjb_b])
        .map_err(HmmerError::Io)?;
    write_f32_ne(w, om.scale_b)?;
    w.write_all(&[om.base_b, om.bias_b])
        .map_err(HmmerError::Io)?;

    for x in 0..om.abc_kp {
        for q in 0..q16x {
            w.write_all(&om.sbv[x][q]).map_err(HmmerError::Io)?;
        }
    }
    for x in 0..om.abc_kp {
        for q in 0..q16 {
            w.write_all(&om.rbv[x][q]).map_err(HmmerError::Io)?;
        }
    }
    for value in om.evparam {
        write_f32_ne(w, value)?;
    }
    for offset in offsets {
        w.write_all(&offset.to_ne_bytes()).map_err(HmmerError::Io)?;
    }
    // C writes p7_MAXABET(=20) compo floats (impl_sse/io.c:118). For nucleic
    // (K=4) models the unused tail entries are 0.0 in C's array, whereas our
    // `om.compo` keeps COMPO_UNSET (-1.0) there. Emit the real composition for
    // [0..K) and 0.0 for the tail so the .h3f matches C byte-for-byte. Amino
    // (K=20) uses every entry, so this is a no-op for amino models.
    for (i, value) in om.compo.iter().enumerate() {
        let out = if i < om.abc_k { *value } else { 0.0 };
        write_f32_ne(w, out)?;
    }
    write_u32_ne(w, V3F_FMAGIC)
}

/// Write one C HMMER3/f `.h3p` optimized-profile remainder record.
pub fn write_h3p_record<W: Write>(w: &mut W, hmm: &Hmm, om: &OProfile) -> HmmerResult<()> {
    let abc_type = alphabet_type_code(hmm.abc_type)?;
    let q8 = nqw(om.m);
    let q4 = nqf(om.m);

    write_u32_ne(w, V3F_PMAGIC)?;
    write_i32_ne(w, checked_i32_usize(om.m, ".h3p model length")?)?;
    write_i32_ne(w, abc_type)?;
    write_len_prefixed_cstr(w, &om.name)?;
    write_optional_len_prefixed_cstr(w, hmm.acc.as_deref())?;
    write_optional_len_prefixed_cstr(w, hmm.desc.as_deref())?;

    write_annotation(w, hmm.rf.as_deref(), om.m)?;
    write_annotation(w, hmm.mm.as_deref(), om.m)?;
    write_annotation(w, hmm.cs.as_deref(), om.m)?;
    write_annotation(w, hmm.consensus.as_deref(), om.m)?;

    for q in 0..8 * q8 {
        for value in om.twv[q] {
            w.write_all(&value.to_ne_bytes()).map_err(HmmerError::Io)?;
        }
    }
    for x in 0..om.abc_kp {
        for q in 0..q8 {
            for value in om.rwv[x][q] {
                w.write_all(&value.to_ne_bytes()).map_err(HmmerError::Io)?;
            }
        }
    }
    for state in 0..4 {
        for trans in 0..2 {
            w.write_all(&om.xw[state][trans].to_ne_bytes())
                .map_err(HmmerError::Io)?;
        }
    }
    write_f32_ne(w, om.scale_w)?;
    w.write_all(&om.base_w.to_ne_bytes())
        .map_err(HmmerError::Io)?;
    w.write_all(&om.ddbound_w.to_ne_bytes())
        .map_err(HmmerError::Io)?;
    write_f32_ne(w, om.ncj_roundoff)?;

    for q in 0..8 * q4 {
        for value in *om.tfv[q] {
            write_f32_ne(w, value)?;
        }
    }
    for x in 0..om.abc_kp {
        for q in 0..q4 {
            for value in *om.rfv[x][q] {
                write_f32_ne(w, value)?;
            }
        }
    }
    for state in 0..4 {
        for trans in 0..2 {
            write_f32_ne(w, om.xf[state][trans])?;
        }
    }
    for value in om.cutoff {
        write_f32_ne(w, value)?;
    }
    write_f32_ne(w, om.nj)?;
    write_i32_ne(w, om.mode)?;
    write_i32_ne(w, om.l)?;
    write_u32_ne(w, V3F_PMAGIC)
}

/// MSV filter data from .h3f file for one profile.
pub struct MsvFilterData {
    pub name: String,
    pub m: usize,
    pub abc_type: i32,
    pub max_length: i32,
    pub tbm_b: u8,
    pub tec_b: u8,
    pub tjb_b: u8,
    pub scale_b: f32,
    pub base_b: u8,
    pub bias_b: u8,
    /// Raw SSV/MSV byte scores (flattened __m128i arrays)
    pub msv_scores: Vec<u8>,
    pub evparam: [f32; 6],
    pub offsets: [i64; P7_NOFFSETS],
    pub compo: [f32; 20],
}

fn alphabet_dimensions(abc_type: i32, sidecar: &str) -> HmmerResult<(usize, usize)> {
    match abc_type {
        1 | 2 => Ok((4, 18)),
        3 => Ok((20, 29)),
        _ => Err(HmmerError::Format(format!(
            "Invalid {sidecar} alphabet code: {abc_type}"
        ))),
    }
}

/// Read one MSV filter (`.h3f`) record produced by `hmmpress`.
/// Reads the V3F magic, the SSV/MSV byte score table, and footer magic; mirrors
/// `p7_oprofile_ReadMSV` in `hmmer/src/p7_hmmfile.c`. Returns `Ok(None)` at EOF.
pub fn read_h3f_record<R: Read>(r: &mut R) -> HmmerResult<Option<MsvFilterData>> {
    let Some(magic) = read_record_magic_ne(r)? else {
        return Ok(None);
    };
    if magic != V3F_FMAGIC {
        return Err(HmmerError::Format(format!("Bad .h3f magic: {:#x}", magic)));
    }

    let m = read_i32_ne(r)?;
    let abc_type = read_i32_ne(r)?;
    let m = validate_pressed_record_header(m, abc_type, ".h3f")?;
    let name_len = read_nonnegative_len(r, ".h3f name length")?;
    let name = require_nonempty_required_string(
        read_len_prefixed_cstr(r, name_len, ".h3f name")?,
        ".h3f name",
    )?;
    let max_length = read_i32_ne(r)?;

    let tbm_b = read_u8(r)?;
    let tec_b = read_u8(r)?;
    let tjb_b = read_u8(r)?;
    let scale_b = read_f32_ne(r)?;
    let base_b = read_u8(r)?;
    let bias_b = read_u8(r)?;

    // Read MSV score vectors (raw bytes): sbv has Q16 + p7O_EXTRA_SB
    // vectors per alphabet symbol; rbv has Q16.
    let (_k, k) = alphabet_dimensions(abc_type, ".h3f")?;
    let q = ((m.max(1) - 1) / 16 + 1).max(2);
    let q_total = checked_add_usize(
        checked_add_usize(q, P7O_EXTRA_SB, ".h3f MSV vector block")?,
        q,
        ".h3f MSV vector block",
    )?;
    let msv_bytes = pressed_vector_bytes(&[k, q_total, 16], ".h3f MSV vector block")?; // sbv + rbv
    let msv_scores = read_bytes(r, msv_bytes)?;

    let mut evparam = [0.0f32; 6];
    for i in 0..6 {
        evparam[i] = read_f32_ne(r)?;
    }

    let mut offsets = [0i64; P7_NOFFSETS];
    for offset in &mut offsets {
        let mut buf = [0u8; std::mem::size_of::<i64>()];
        r.read_exact(&mut buf).map_err(HmmerError::Io)?;
        *offset = i64::from_ne_bytes(buf);
    }

    let mut compo = [0.0f32; P7_MAXABET];
    for i in 0..P7_MAXABET {
        compo[i] = read_f32_ne(r)?;
    }

    // Footer magic
    let footer = read_u32_ne(r)?;
    if footer != V3F_FMAGIC {
        return Err(HmmerError::Format("Bad .h3f footer".to_string()));
    }

    Ok(Some(MsvFilterData {
        name,
        m,
        abc_type,
        max_length,
        tbm_b,
        tec_b,
        tjb_b,
        scale_b,
        base_b,
        bias_b,
        msv_scores,
        evparam,
        offsets,
        compo,
    }))
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

fn validate_h3i_against_h3m(path: &Path, hmm_path: &Path, h3m_path: &Path) -> HmmerResult<()> {
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

/// Viterbi/Forward profile data from .h3p file for one profile.
pub struct ProfileData {
    pub name: String,
    pub acc: String,
    pub desc: String,
    pub m: usize,
    pub abc_type: i32,
    /// Raw Viterbi transition/emission data
    pub viterbi_data: Vec<u8>,
    /// Raw Forward transition/emission data
    pub forward_data: Vec<u8>,
    pub xw: [[i16; 2]; 4],
    pub xf: [[f32; 2]; 4],
    pub scale_w: f32,
    pub base_w: i16,
    pub ddbound_w: i16,
    pub ncj_roundoff: f32,
    pub cutoff: [f32; 6],
    pub nj: f32,
    pub mode: i32,
    pub l: i32,
}

/// Read one Viterbi/Forward profile (`.h3p`) record produced by `hmmpress`.
/// Reads name/acc/desc, annotation strings, the striped Viterbi (16-bit) and
/// Forward (float) parameter blocks, and trailing cutoffs/metadata. Mirrors
/// `p7_oprofile_ReadRest` in `hmmer/src/p7_hmmfile.c`. Returns `Ok(None)` at EOF.
pub fn read_h3p_record<R: Read>(r: &mut R) -> HmmerResult<Option<ProfileData>> {
    let Some(magic) = read_record_magic_ne(r)? else {
        return Ok(None);
    };
    if magic != V3F_PMAGIC {
        return Err(HmmerError::Format(format!("Bad .h3p magic: {:#x}", magic)));
    }

    let m = read_i32_ne(r)?;
    let abc_type = read_i32_ne(r)?;
    let m = validate_pressed_record_header(m, abc_type, ".h3p")?;

    // Name
    let name_len = read_nonnegative_len(r, ".h3p name length")?;
    let name = require_nonempty_required_string(
        read_len_prefixed_cstr(r, name_len, ".h3p name")?,
        ".h3p name",
    )?;

    // Accession (optional, length-prefixed)
    let acc_len = read_nonnegative_len(r, ".h3p accession length")?;
    let acc = read_len_prefixed_cstr(r, acc_len, ".h3p accession")?;

    // Description
    let desc_len = read_nonnegative_len(r, ".h3p description length")?;
    let desc = read_len_prefixed_cstr(r, desc_len, ".h3p description")?;

    // RF, MM, CS, consensus strings
    for _ in 0..4 {
        let annotation_len = checked_add_usize(m, 2, ".h3p annotation")?;
        read_bytes(r, annotation_len)?; // each is M+2 bytes
    }

    // Viterbi data (twv + rwv)
    let (_k, kp) = alphabet_dimensions(abc_type, ".h3p")?;
    let qw = ((m.max(1) - 1) / 8 + 1).max(2);
    let twv_size = pressed_vector_bytes(&[8, qw, 16], ".h3p Viterbi block")?;
    let rwv_size = pressed_vector_bytes(&[kp, qw, 16], ".h3p Viterbi block")?;
    let vit_size = checked_add_usize(twv_size, rwv_size, ".h3p Viterbi block")?;
    let viterbi_data = read_bytes(r, vit_size)?;

    // Viterbi special states
    let mut xw = [[0i16; 2]; 4];
    for s in 0..4 {
        for t in 0..2 {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf).map_err(HmmerError::Io)?;
            xw[s][t] = i16::from_ne_bytes(buf);
        }
    }
    let scale_w = read_f32_ne(r)?;
    let mut buf2 = [0u8; 2];
    r.read_exact(&mut buf2).map_err(HmmerError::Io)?;
    let base_w = i16::from_ne_bytes(buf2);
    r.read_exact(&mut buf2).map_err(HmmerError::Io)?; // ddbound_w
    let ddbound_w = i16::from_ne_bytes(buf2);
    let ncj_roundoff = read_f32_ne(r)?;

    // Forward data (tfv + rfv)
    let qf = ((m.max(1) - 1) / 4 + 1).max(2);
    let tfv_size = pressed_vector_bytes(&[8, qf, 16], ".h3p Forward block")?;
    let rfv_size = pressed_vector_bytes(&[kp, qf, 16], ".h3p Forward block")?;
    let fwd_size = checked_add_usize(tfv_size, rfv_size, ".h3p Forward block")?;
    let forward_data = read_bytes(r, fwd_size)?;

    // Forward special states
    let mut xf = [[0.0f32; 2]; 4];
    for s in 0..4 {
        for t in 0..2 {
            xf[s][t] = read_f32_ne(r)?;
        }
    }

    // Cutoffs
    let mut cutoff = [0.0f32; 6];
    for i in 0..6 {
        cutoff[i] = read_f32_ne(r)?;
    }

    let nj = read_f32_ne(r)?;
    let mode = read_i32_ne(r)?;
    let l = read_i32_ne(r)?;

    // Footer
    let footer = read_u32_ne(r)?;
    if footer != V3F_PMAGIC {
        return Err(HmmerError::Format("Bad .h3p footer".to_string()));
    }

    Ok(Some(ProfileData {
        name,
        acc,
        desc,
        m,
        abc_type,
        viterbi_data,
        forward_data,
        xw,
        xf,
        scale_w,
        base_w,
        ddbound_w,
        ncj_roundoff,
        cutoff,
        nj,
        mode,
        l,
    }))
}

fn read_i16_vec8(bytes: &[u8], offset: &mut usize, field: &str) -> HmmerResult<[i16; 8]> {
    if bytes.len().saturating_sub(*offset) < 16 {
        return Err(HmmerError::Format(format!(
            "Truncated pressed {field} vector"
        )));
    }
    let mut out = [0i16; 8];
    for value in &mut out {
        let mut buf = [0u8; 2];
        buf.copy_from_slice(&bytes[*offset..*offset + 2]);
        *value = i16::from_ne_bytes(buf);
        *offset += 2;
    }
    Ok(out)
}

fn read_f32_vec4(bytes: &[u8], offset: &mut usize, field: &str) -> HmmerResult<[f32; 4]> {
    if bytes.len().saturating_sub(*offset) < 16 {
        return Err(HmmerError::Format(format!(
            "Truncated pressed {field} vector"
        )));
    }
    let mut out = [0.0f32; 4];
    for value in &mut out {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&bytes[*offset..*offset + 4]);
        *value = f32::from_ne_bytes(buf);
        *offset += 4;
    }
    Ok(out)
}

/// Reconstruct an optimized profile from matching `.h3f` and `.h3p` records.
///
/// This intentionally does not replace scan's current `.h3m` path yet: scan
/// still needs the generic profile/HMM for thresholding, alignment, and domain
/// work. The constructor gives the acceleration payload a focused parity point.
pub fn oprofile_from_pressed_records(
    msv: &MsvFilterData,
    profile: &ProfileData,
) -> HmmerResult<OProfile> {
    if (msv.name.as_str(), msv.m, msv.abc_type)
        != (profile.name.as_str(), profile.m, profile.abc_type)
    {
        return Err(HmmerError::Format(format!(
            "Pressed profile record mismatch: .h3f has {} M={} alphabet={}, .h3p has {} M={} alphabet={}",
            msv.name, msv.m, msv.abc_type, profile.name, profile.m, profile.abc_type
        )));
    }

    let (abc_k, abc_kp) = alphabet_dimensions(msv.abc_type, "pressed profile")?;
    let q16 = nqb(msv.m);
    let q16x = q16 + P7O_EXTRA_SB;
    let q8 = nqw(msv.m);
    let q4 = nqf(msv.m);

    let q16_total = checked_add_usize(q16x, q16, "pressed .h3f vector block")?;
    let expected_msv = pressed_vector_bytes(&[abc_kp, q16_total, 16], "pressed .h3f vector block")?;
    if msv.msv_scores.len() != expected_msv {
        return Err(HmmerError::Format(format!(
            "Pressed .h3f vector block has {} bytes, expected {}",
            msv.msv_scores.len(),
            expected_msv
        )));
    }
    let expected_vit = checked_add_usize(
        pressed_vector_bytes(&[8, q8, 16], "pressed .h3p Viterbi block")?,
        pressed_vector_bytes(&[abc_kp, q8, 16], "pressed .h3p Viterbi block")?,
        "pressed .h3p Viterbi block",
    )?;
    if profile.viterbi_data.len() != expected_vit {
        return Err(HmmerError::Format(format!(
            "Pressed .h3p Viterbi block has {} bytes, expected {}",
            profile.viterbi_data.len(),
            expected_vit
        )));
    }
    let expected_fwd = checked_add_usize(
        pressed_vector_bytes(&[8, q4, 16], "pressed .h3p Forward block")?,
        pressed_vector_bytes(&[abc_kp, q4, 16], "pressed .h3p Forward block")?,
        "pressed .h3p Forward block",
    )?;
    if profile.forward_data.len() != expected_fwd {
        return Err(HmmerError::Format(format!(
            "Pressed .h3p Forward block has {} bytes, expected {}",
            profile.forward_data.len(),
            expected_fwd
        )));
    }

    let mut offset = 0;
    let mut sbv = vec![vec![[0u8; 16]; q16x]; abc_kp];
    for rows in &mut sbv {
        for vector in rows {
            vector.copy_from_slice(&msv.msv_scores[offset..offset + 16]);
            offset += 16;
        }
    }
    let mut rbv = vec![vec![[0u8; 16]; q16]; abc_kp];
    for rows in &mut rbv {
        for vector in rows {
            vector.copy_from_slice(&msv.msv_scores[offset..offset + 16]);
            offset += 16;
        }
    }

    let mut offset = 0;
    let mut twv = vec![[0i16; 8]; 8 * q8];
    for vector in &mut twv {
        *vector = read_i16_vec8(&profile.viterbi_data, &mut offset, ".h3p twv")?;
    }
    let mut rwv = vec![vec![[0i16; 8]; q8]; abc_kp];
    for rows in &mut rwv {
        for vector in rows {
            *vector = read_i16_vec8(&profile.viterbi_data, &mut offset, ".h3p rwv")?;
        }
    }

    let mut offset = 0;
    let mut tfv = vec![AlignedF32x4([0.0f32; 4]); 8 * q4];
    for vector in &mut tfv {
        vector.0 = read_f32_vec4(&profile.forward_data, &mut offset, ".h3p tfv")?;
    }
    let mut rfv = vec![vec![AlignedF32x4([0.0f32; 4]); q4]; abc_kp];
    for rows in &mut rfv {
        for vector in rows {
            vector.0 = read_f32_vec4(&profile.forward_data, &mut offset, ".h3p rfv")?;
        }
    }

    let ssv = crate::simd::ssv::SsvProfile::new(&sbv, msv.m);
    let om = OProfile {
        rbv: std::sync::Arc::new(rbv),
        sbv: std::sync::Arc::new(sbv),
        ssv,
        vit: None,
        tbm_b: msv.tbm_b,
        tec_b: msv.tec_b,
        tjb_b: msv.tjb_b,
        scale_b: msv.scale_b,
        base_b: msv.base_b,
        bias_b: msv.bias_b,
        rwv: std::sync::Arc::new(rwv),
        twv: std::sync::Arc::new(twv),
        xw: profile.xw,
        scale_w: profile.scale_w,
        base_w: profile.base_w,
        ddbound_w: profile.ddbound_w,
        ncj_roundoff: profile.ncj_roundoff,
        rfv: std::sync::Arc::new(rfv),
        tfv: std::sync::Arc::new(tfv),
        xf: profile.xf,
        m: msv.m,
        l: profile.l,
        max_length: msv.max_length,
        nj: profile.nj,
        mode: profile.mode,
        evparam: msv.evparam,
        cutoff: profile.cutoff,
        compo: msv.compo,
        name: msv.name.clone(),
        abc_k,
        abc_kp,
    };
    let om = OProfile {
        vit: crate::simd::vit_wide::VitProfile::new(&om),
        ..om
    };
    Ok(om)
}

#[cfg(test)]
#[path = "../tests/unit/pressed_test.rs"]
mod tests;
