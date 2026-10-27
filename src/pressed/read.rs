use super::*;

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

pub(super) fn alphabet_dimensions(abc_type: i32, sidecar: &str) -> HmmerResult<(usize, usize)> {
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
