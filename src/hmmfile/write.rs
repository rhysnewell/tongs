use super::*;

pub fn write_hmm<W: std::io::Write>(w: &mut W, hmm: &Hmm) -> HmmerResult<()> {
    write_hmm_with_format(w, hmm, HmmAsciiFormat::Hmmer3f)
}

pub fn write_hmm_with_format<W: std::io::Write>(
    w: &mut W,
    hmm: &Hmm,
    format: HmmAsciiFormat,
) -> HmmerResult<()> {
    let k = hmm.abc_k;

    if format == HmmAsciiFormat::Hmmer3f {
        writeln!(w, "HMMER3/f [3.4 | Aug 2023]").map_err(HmmerError::Io)?;
    } else {
        writeln!(
            w,
            "HMMER3/{} [3.4 | Aug 2023; reverse compatibility mode]",
            format.code()
        )
        .map_err(HmmerError::Io)?;
    }
    writeln!(w, "NAME  {}", hmm.name).map_err(HmmerError::Io)?;
    if let Some(ref acc) = hmm.acc {
        writeln!(w, "ACC   {}", acc).map_err(HmmerError::Io)?;
    }
    if let Some(ref desc) = hmm.desc {
        writeln!(w, "DESC  {}", desc).map_err(HmmerError::Io)?;
    }
    writeln!(w, "LENG  {}", hmm.m).map_err(HmmerError::Io)?;
    if format.is_at_least(HmmAsciiFormat::Hmmer3c) && hmm.max_length > 0 {
        writeln!(w, "MAXL  {}", hmm.max_length).map_err(HmmerError::Io)?;
    }
    let alph = match hmm.abc_type {
        AlphabetType::Amino => "amino",
        AlphabetType::Dna => "DNA",
        AlphabetType::Rna => "RNA",
        AlphabetType::Unknown => "unknown",
    };
    writeln!(w, "ALPH  {}", alph).map_err(HmmerError::Io)?;
    writeln!(
        w,
        "RF    {}",
        if hmm.flags & P7H_RF != 0 { "yes" } else { "no" }
    )
    .map_err(HmmerError::Io)?;
    if format.is_at_least(HmmAsciiFormat::Hmmer3f) {
        writeln!(
            w,
            "MM    {}",
            if hmm.flags & P7H_MMASK != 0 {
                "yes"
            } else {
                "no"
            }
        )
        .map_err(HmmerError::Io)?;
    }
    if format.is_at_least(HmmAsciiFormat::Hmmer3e) {
        writeln!(
            w,
            "CONS  {}",
            if hmm.flags & P7H_CONS != 0 {
                "yes"
            } else {
                "no"
            }
        )
        .map_err(HmmerError::Io)?;
    }
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
    if let Some(ref ctime) = hmm.ctime {
        writeln!(w, "DATE  {}", ctime).map_err(HmmerError::Io)?;
    }
    if let Some(ref comlog) = hmm.comlog {
        for (idx, command) in comlog.lines().enumerate() {
            writeln!(w, "COM   [{}] {}", idx + 1, command).map_err(HmmerError::Io)?;
        }
    }
    if hmm.nseq >= 0 {
        writeln!(w, "NSEQ  {}", hmm.nseq).map_err(HmmerError::Io)?;
    }
    if hmm.eff_nseq >= 0.0 {
        writeln!(w, "EFFN  {}", fmt_fixed6(hmm.eff_nseq as f64)).map_err(HmmerError::Io)?;
    }
    if hmm.flags & P7H_CHKSUM != 0 {
        writeln!(w, "CKSUM {}", hmm.checksum).map_err(HmmerError::Io)?;
    }
    // C (p7_hmmfile.c:546-554): nucleic-acid models emit a single cutoff value;
    // only amino emits both the per-sequence and per-domain cutoffs.
    let nucleic = matches!(hmm.abc_type, AlphabetType::Dna | AlphabetType::Rna);
    if hmm.flags & P7H_GA != 0 {
        if nucleic {
            writeln!(w, "GA    {}", fmt_fixed2(hmm.cutoff[P7_GA1] as f64))
                .map_err(HmmerError::Io)?;
        } else {
            writeln!(
                w,
                "GA    {} {}",
                fmt_fixed2(hmm.cutoff[P7_GA1] as f64),
                fmt_fixed2(hmm.cutoff[P7_GA2] as f64)
            )
            .map_err(HmmerError::Io)?;
        }
    }
    if hmm.flags & P7H_TC != 0 {
        if nucleic {
            writeln!(w, "TC    {}", fmt_fixed2(hmm.cutoff[P7_TC1] as f64))
                .map_err(HmmerError::Io)?;
        } else {
            writeln!(
                w,
                "TC    {} {}",
                fmt_fixed2(hmm.cutoff[P7_TC1] as f64),
                fmt_fixed2(hmm.cutoff[P7_TC2] as f64)
            )
            .map_err(HmmerError::Io)?;
        }
    }
    if hmm.flags & P7H_NC != 0 {
        if nucleic {
            writeln!(w, "NC    {}", fmt_fixed2(hmm.cutoff[P7_NC1] as f64))
                .map_err(HmmerError::Io)?;
        } else {
            writeln!(
                w,
                "NC    {} {}",
                fmt_fixed2(hmm.cutoff[P7_NC1] as f64),
                fmt_fixed2(hmm.cutoff[P7_NC2] as f64)
            )
            .map_err(HmmerError::Io)?;
        }
    }
    if hmm.flags & P7H_STATS != 0 {
        if format == HmmAsciiFormat::Hmmer3a {
            writeln!(
                w,
                "STATS LOCAL     VLAMBDA {}",
                fmt_fixed6(hmm.evparam[P7_MLAMBDA] as f64)
            )
            .map_err(HmmerError::Io)?;
            writeln!(
                w,
                "STATS LOCAL         VMU {}",
                fmt_fixed6(hmm.evparam[P7_MMU] as f64)
            )
            .map_err(HmmerError::Io)?;
            writeln!(
                w,
                "STATS LOCAL        FTAU {}",
                fmt_fixed6(hmm.evparam[P7_FTAU] as f64)
            )
            .map_err(HmmerError::Io)?;
        } else {
            // C: fprintf(fp, "STATS LOCAL MSV      %8.4f %8.5f\n", ...)
            // (label + spaces, then a width-8 mu, one space, width-8 lambda).
            // `fmt_fixed4`/`fmt_fixed5` reproduce C's `%.4f`/`%.5f`; right-
            // justifying in width 8 reproduces the `%8.4f`/`%8.5f` field width.
            writeln!(
                w,
                "STATS LOCAL MSV      {:>8} {:>8}",
                fmt_fixed4(hmm.evparam[P7_MMU] as f64),
                fmt_fixed5(hmm.evparam[P7_MLAMBDA] as f64)
            )
            .map_err(HmmerError::Io)?;
            writeln!(
                w,
                "STATS LOCAL VITERBI  {:>8} {:>8}",
                fmt_fixed4(hmm.evparam[P7_VMU] as f64),
                fmt_fixed5(hmm.evparam[P7_VLAMBDA] as f64)
            )
            .map_err(HmmerError::Io)?;
            writeln!(
                w,
                "STATS LOCAL FORWARD  {:>8} {:>8}",
                fmt_fixed4(hmm.evparam[P7_FTAU] as f64),
                fmt_fixed5(hmm.evparam[P7_FLAMBDA] as f64)
            )
            .map_err(HmmerError::Io)?;
        }
    }

    // Alphabet header
    let abc = crate::alphabet::Alphabet::new(hmm.abc_type);
    write!(w, "HMM     ").map_err(HmmerError::Io)?;
    for i in 0..k {
        write!(w, "     {}   ", abc.sym[i] as char).map_err(HmmerError::Io)?;
    }
    writeln!(w).map_err(HmmerError::Io)?;

    // Transition label line
    writeln!(
        w,
        "            m->m     m->i     m->d     i->m     i->i     d->m     d->d"
    )
    .map_err(HmmerError::Io)?;

    // COMPO line
    if hmm.flags & P7H_COMPO != 0 {
        write!(w, "  COMPO ").map_err(HmmerError::Io)?;
        for i in 0..k {
            write!(w, " {}", fmt_prob(hmm.compo[i])).map_err(HmmerError::Io)?;
        }
        writeln!(w).map_err(HmmerError::Io)?;
    }

    // Node 0 insert emissions
    write!(w, "        ").map_err(HmmerError::Io)?;
    for i in 0..k {
        write!(w, " {}", fmt_prob(hmm.ins[0][i])).map_err(HmmerError::Io)?;
    }
    writeln!(w).map_err(HmmerError::Io)?;

    // Node 0 transitions
    write!(w, "        ").map_err(HmmerError::Io)?;
    for i in 0..NTRANSITIONS {
        write!(w, " {}", fmt_prob(hmm.t[0][i])).map_err(HmmerError::Io)?;
    }
    writeln!(w).map_err(HmmerError::Io)?;

    // Nodes 1..M
    for node in 1..=hmm.m {
        // Match emission line: C writes " %6d " (leading + trailing space).
        write!(w, " {:>6} ", node).map_err(HmmerError::Io)?;
        for i in 0..k {
            write!(w, " {}", fmt_prob(hmm.mat[node][i])).map_err(HmmerError::Io)?;
        }
        if let Some(ref map) = hmm.map {
            write!(w, " {:>6}", map[node]).map_err(HmmerError::Io)?;
        } else {
            write!(w, " {:>6}", "-").map_err(HmmerError::Io)?;
        }
        let cons_ch = hmm
            .consensus
            .as_ref()
            .map(|cons| cons[node] as char)
            .unwrap_or('-');
        let rf_ch = hmm.rf.as_ref().map(|rf| rf[node] as char).unwrap_or('-');
        let mm_ch = hmm.mm.as_ref().map(|mm| mm[node] as char).unwrap_or('-');
        let cs_ch = hmm.cs.as_ref().map(|cs| cs[node] as char).unwrap_or('-');
        if format.is_at_least(HmmAsciiFormat::Hmmer3e) {
            write!(w, " {}", cons_ch).map_err(HmmerError::Io)?;
        }
        write!(w, " {}", rf_ch).map_err(HmmerError::Io)?;
        if format.is_at_least(HmmAsciiFormat::Hmmer3f) {
            write!(w, " {}", mm_ch).map_err(HmmerError::Io)?;
        }
        write!(w, " {}", cs_ch).map_err(HmmerError::Io)?;
        writeln!(w).map_err(HmmerError::Io)?;

        // Insert emission line
        write!(w, "        ").map_err(HmmerError::Io)?;
        for i in 0..k {
            write!(w, " {}", fmt_prob(hmm.ins[node][i])).map_err(HmmerError::Io)?;
        }
        writeln!(w).map_err(HmmerError::Io)?;

        // Transition line
        write!(w, "        ").map_err(HmmerError::Io)?;
        for i in 0..NTRANSITIONS {
            write!(w, " {}", fmt_prob(hmm.t[node][i])).map_err(HmmerError::Io)?;
        }
        writeln!(w).map_err(HmmerError::Io)?;
    }

    writeln!(w, "//").map_err(HmmerError::Io)?;
    Ok(())
}
