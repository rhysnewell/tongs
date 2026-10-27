use super::*;

/// Write an H3 HMM in HMMER2 ASCII save-file format (`hmmconvert -2`).
///
/// Mirrors `p7_h2io_WriteASCII` in `hmmer/src/h2_io.c`. HMMER2 stored the null
/// model and the search configuration (local vs glocal) in the HMM file; H3
/// only stores the core HMM, so the output is emitted for HMMER2's default
/// "ls mode" (glocal) with default null-model and special-state (NECJ)
/// transitions. Statistical calibration and the alignment checksum are omitted
/// (H2 and H3 differ too much for those).
pub fn write_hmm_h2_ascii<W: std::io::Write>(w: &mut W, hmm: &Hmm) -> HmmerResult<()> {
    let abc = Alphabet::new(hmm.abc_type);
    let bg = crate::bg::Bg::new(&abc);
    let k = hmm.abc_k;

    // Default H2 null-model / special-state transitions: amino vs nucleic only.
    // (h2_io.c:96-97 — computed as double then narrowed to float.)
    let (pmove, ploop): (f32, f32) = if hmm.abc_type == AlphabetType::Amino {
        ((1.0_f64 / 351.0) as f32, (350.0_f64 / 351.0) as f32)
    } else {
        ((1.0_f64 / 1001.0) as f32, (1000.0_f64 / 1001.0) as f32)
    };

    // magic header (HMMER_VERSION == "3.4")
    writeln!(w, "HMMER2.0  [converted from 3.4]").map_err(HmmerError::Io)?;
    writeln!(w, "NAME  {}", hmm.name).map_err(HmmerError::Io)?;
    if let Some(ref acc) = hmm.acc {
        writeln!(w, "ACC   {}", acc).map_err(HmmerError::Io)?;
    }
    if let Some(ref desc) = hmm.desc {
        writeln!(w, "DESC  {}", desc).map_err(HmmerError::Io)?;
    }
    writeln!(w, "LENG  {}", hmm.m).map_err(HmmerError::Io)?;
    match hmm.abc_type {
        AlphabetType::Amino => writeln!(w, "ALPH  Amino").map_err(HmmerError::Io)?,
        AlphabetType::Dna | AlphabetType::Rna => {
            writeln!(w, "ALPH  Nucleic").map_err(HmmerError::Io)?
        }
        AlphabetType::Unknown => {
            return Err(HmmerError::InvalidArg(
                "Only protein, DNA, RNA HMMs can be saved in H2 format".to_string(),
            ));
        }
    }
    writeln!(
        w,
        "RF    {}",
        if hmm.flags & P7H_RF != 0 { "yes" } else { "no" }
    )
    .map_err(HmmerError::Io)?;
    writeln!(
        w,
        "CS    {}",
        if hmm.flags & P7H_CS != 0 { "yes" } else { "no" }
    )
    .map_err(HmmerError::Io)?;
    writeln!(
        w,
        "MAP   {}",
        if hmm.flags & P7H_MAP != 0 {
            "yes"
        } else {
            "no"
        }
    )
    .map_err(HmmerError::Io)?;

    if let Some(ref comlog) = hmm.comlog {
        h2_multiline(w, "COM   ", comlog)?;
    }
    if hmm.nseq != -1 {
        writeln!(w, "NSEQ  {}", hmm.nseq).map_err(HmmerError::Io)?;
    }
    if let Some(ref ctime) = hmm.ctime {
        writeln!(w, "DATE  {}", ctime).map_err(HmmerError::Io)?;
    }
    // Checksum is intentionally not written (H2 and H3 use different algorithms).

    if hmm.flags & P7H_GA != 0 {
        writeln!(
            w,
            "GA    {} {}",
            fmt_fixed1(hmm.cutoff[P7_GA1] as f64),
            fmt_fixed1(hmm.cutoff[P7_GA2] as f64)
        )
        .map_err(HmmerError::Io)?;
    }
    if hmm.flags & P7H_TC != 0 {
        writeln!(
            w,
            "TC    {} {}",
            fmt_fixed1(hmm.cutoff[P7_TC1] as f64),
            fmt_fixed1(hmm.cutoff[P7_TC2] as f64)
        )
        .map_err(HmmerError::Io)?;
    }
    if hmm.flags & P7H_NC != 0 {
        writeln!(
            w,
            "NC    {} {}",
            fmt_fixed1(hmm.cutoff[P7_NC1] as f64),
            fmt_fixed1(hmm.cutoff[P7_NC2] as f64)
        )
        .map_err(HmmerError::Io)?;
    }

    // XT: special-state transitions (NB NN EC EJ CT CC JB JJ), default ls config.
    write!(w, "XT     ").map_err(HmmerError::Io)?;
    h2_printprob(w, 6, pmove, 1.0)?; // NB
    h2_printprob(w, 6, ploop, 1.0)?; // NN
    h2_printprob(w, 6, 0.5, 1.0)?; // EC
    h2_printprob(w, 6, 0.5, 1.0)?; // EJ
    h2_printprob(w, 6, pmove, 1.0)?; // CT
    h2_printprob(w, 6, ploop, 1.0)?; // CC
    h2_printprob(w, 6, pmove, 1.0)?; // JB
    h2_printprob(w, 6, ploop, 1.0)?; // JJ
    writeln!(w).map_err(HmmerError::Io)?;

    // NULT: default H2 null-model transitions (NOT H3's).
    write!(w, "NULT   ").map_err(HmmerError::Io)?;
    h2_printprob(w, 6, ploop, 1.0)?; // 1-p1
    h2_printprob(w, 6, pmove, 1.0)?; // p1
    writeln!(w).map_err(HmmerError::Io)?;

    // NULE: H3 null-model emissions (these really are H3's).
    write!(w, "NULE   ").map_err(HmmerError::Io)?;
    let nule_null = (1.0_f64 / k as f64) as f32;
    for x in 0..k {
        h2_printprob(w, 6, bg.f[x], nule_null)?;
    }
    writeln!(w).map_err(HmmerError::Io)?;

    // Main model section.
    write!(w, "HMM      ").map_err(HmmerError::Io)?;
    for x in 0..k {
        write!(w, "  {}    ", abc.sym[x] as char).map_err(HmmerError::Io)?;
    }
    writeln!(w).map_err(HmmerError::Io)?;
    writeln!(
        w,
        "       {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "m->m", "m->i", "m->d", "i->m", "i->i", "d->m", "d->d", "b->m", "m->e"
    )
    .map_err(HmmerError::Io)?;

    // BEGIN node (k=0) transition line: m->m = 1 - t[0][MD], m->i = '*', m->d = t[0][MD].
    write!(w, "      ").map_err(HmmerError::Io)?;
    h2_printprob(w, 6, 1.0 - hmm.t[0][MD], 1.0)?;
    write!(w, " {:>6}", "*").map_err(HmmerError::Io)?;
    h2_printprob(w, 6, hmm.t[0][MD], 1.0)?;
    writeln!(w).map_err(HmmerError::Io)?;

    for node in 1..=hmm.m {
        // Line 1: node index, match emissions, optional map.
        write!(w, " {:>5} ", node).map_err(HmmerError::Io)?;
        for x in 0..k {
            h2_printprob(w, 6, hmm.mat[node][x], bg.f[x])?;
        }
        if hmm.flags & P7H_MAP != 0 {
            let mapval = hmm.map.as_ref().map(|m| m[node]).unwrap_or(0);
            write!(w, " {:>5}", mapval).map_err(HmmerError::Io)?;
        }
        writeln!(w).map_err(HmmerError::Io)?;

        // Line 2: RF char + insert emissions (0.0 at the final node).
        let rf_char = if hmm.flags & P7H_RF != 0 {
            hmm.rf.as_ref().map(|r| r[node]).unwrap_or(b'-') as char
        } else {
            '-'
        };
        write!(w, " {:>5} ", rf_char).map_err(HmmerError::Io)?;
        for x in 0..k {
            let p = if node < hmm.m { hmm.ins[node][x] } else { 0.0 };
            h2_printprob(w, 6, p, bg.f[x])?;
        }
        writeln!(w).map_err(HmmerError::Io)?;

        // Line 3: CS char + transitions (b->m only at node 1; m->e only at the last node).
        let cs_char = if hmm.flags & P7H_CS != 0 {
            hmm.cs.as_ref().map(|c| c[node]).unwrap_or(b'-') as char
        } else {
            '-'
        };
        write!(w, " {:>5} ", cs_char).map_err(HmmerError::Io)?;
        for ts in 0..7 {
            let p = if node < hmm.m { hmm.t[node][ts] } else { 0.0 };
            h2_printprob(w, 6, p, 1.0)?;
        }
        let bm = if node == 1 { hmm.t[0][MM] } else { 0.0 };
        h2_printprob(w, 6, bm, 1.0)?;
        let me = if node < hmm.m { 0.0 } else { 1.0 };
        h2_printprob(w, 6, me, 1.0)?;
        writeln!(w).map_err(HmmerError::Io)?;
    }
    writeln!(w, "//").map_err(HmmerError::Io)?;
    Ok(())
}

/// Print one probability as an H2-format integer log-odds score, with a leading
/// space, right-justified in `fieldwidth`. Mirrors `printprob()` in
/// `hmmer/src/h2_io.c`:
///   - `p == 0.0`                  → `" %*s"` with `"*"`
///   - `null == 1.0 && p == 1.0`   → `" %*d"` with `0`
///   - else                        → `" %*d"` with `floor(0.5 + 1442.695 * log(p/null))`
fn h2_printprob<W: std::io::Write>(
    w: &mut W,
    fieldwidth: usize,
    p: f32,
    null: f32,
) -> HmmerResult<()> {
    if p == 0.0 {
        write!(w, " {:>width$}", "*", width = fieldwidth).map_err(HmmerError::Io)
    } else if null == 1.0 && p == 1.0 {
        write!(w, " {:>width$}", 0, width = fieldwidth).map_err(HmmerError::Io)
    } else {
        // C: (int) floor(0.5 + 1442.695 * log(p/null)). p/null is computed in
        // f32 (both are float), promoted to f64 for log() (libc, via c_log_f64).
        let score = (0.5 + 1442.695 * c_log_f64((p / null) as f64)).floor() as i32;
        write!(w, " {:>width$}", score, width = fieldwidth).map_err(HmmerError::Io)
    }
}

/// Print a multi-line record (e.g. the command log) with a fixed prefix, one
/// line per `\n`-separated segment. Mirrors `h2_multiline()` in `h2_io.c`
/// (note: H2 records bare commands, with no `[n]` numbering — the Rust comlog is
/// already stored without the numbering).
fn h2_multiline<W: std::io::Write>(w: &mut W, pfx: &str, s: &str) -> HmmerResult<()> {
    let bytes = s.as_bytes();
    let mut start = 0usize;
    loop {
        match bytes[start..].iter().position(|&c| c == b'\n') {
            Some(rel) => {
                let end = start + rel;
                write!(w, "{} ", pfx).map_err(HmmerError::Io)?;
                w.write_all(&bytes[start..end]).map_err(HmmerError::Io)?;
                writeln!(w).map_err(HmmerError::Io)?;
                start = end + 1;
                if start >= bytes.len() {
                    break; // C loop ends when *sptr == '\0' after the final '\n'.
                }
            }
            None => {
                writeln!(w, "{} {}", pfx, &s[start..]).map_err(HmmerError::Io)?;
                break;
            }
        }
    }
    Ok(())
}

/// Format a probability `p` as `-ln(p)` (or `*` if zero) using single-precision
/// `logf`, matching the C HMMER ASCII writer's field width and digits.
///
/// Mirrors `printprob(fp, 8, p)` from `hmmer/src/p7_hmmfile.c:2091`:
///   - `p == 0.0` → `" %*s"` with `fieldwidth=8` and `"*"` → `"       *"` (7 spaces + `*`)
///   - `p == 1.0` → `" %*.5f"` with `fieldwidth=8` and `0.0` → `" 0.00000"` (8 chars)
///   - else       → `" %*.5f"` with `fieldwidth=8` and `-logf(p)` → 8-char float string
///
/// Returns an 8-wide string; the caller prepends one space via `" {}"` to
/// reproduce C's `fprintf(fp, " %*.5f", 8, ...)` = 9 bytes per probability value.
pub(super) fn fmt_prob(p: f32) -> String {
    // C's `printprob` uses `%8s` / `%8.5f` (8-wide field).
    if p <= 0.0 {
        // C: fprintf(fp, " %*s", 8, "*") -> "       *" (7 spaces + '*', 8 chars)
        "       *".to_string()
    } else if p == 1.0 {
        // C: fprintf(fp, " %*.5f", 8, 0.0) -> " 0.00000" (8 chars)
        fmt_hmm_prob(0.0)
    } else {
        // HMMER's C writer uses logf(), not double-precision log().
        // fmt_hmm_prob is now %8.5f, directly matching C's fieldwidth=8.
        fmt_hmm_prob(-c_logf_to_f32(p) as f64)
    }
}
