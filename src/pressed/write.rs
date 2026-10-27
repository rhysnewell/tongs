use super::*;

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

pub(super) fn write_len_prefixed_cstr<W: Write>(w: &mut W, s: &str) -> HmmerResult<()> {
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
