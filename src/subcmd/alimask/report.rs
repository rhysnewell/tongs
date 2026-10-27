use super::*;

pub(super) fn write_summary_report(
    args: &Args,
    seed_was_requested: bool,
    wid_was_requested: bool,
) -> std::io::Result<()> {
    let mut file;
    let stdout;
    let out: &mut dyn Write = if let Some(ref path) = args.summary_out {
        file = std::fs::File::create(path)?;
        &mut file
    } else {
        stdout = std::io::stdout();
        let mut lock = stdout.lock();
        write_summary_report_to(&mut lock, args, seed_was_requested, wid_was_requested)?;
        lock.flush()?;
        return Ok(());
    };

    write_summary_report_to(out, args, seed_was_requested, wid_was_requested)
}

fn write_summary_report_to<W: Write + ?Sized>(
    out: &mut W,
    args: &Args,
    seed_was_requested: bool,
    wid_was_requested: bool,
) -> std::io::Result<()> {
    writeln!(
        out,
        "# alimask :: append modelmask line to a multiple sequence alignment"
    )?;
    writeln!(out, "# HMMER 3.4 (Aug 2023); http://hmmer.org/")?;
    writeln!(out, "# Copyright (C) 2023 Howard Hughes Medical Institute.")?;
    writeln!(
        out,
        "# Freely distributed under the BSD open source license."
    )?;
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )?;
    writeln!(
        out,
        "# input alignment file:             {}",
        args.msafile.display()
    )?;
    if let Some(outfile) = args.outfile.as_ref() {
        if args.alirange.is_some() || args.modelrange.is_some() {
            writeln!(
                out,
                "# output alignment file:            {}",
                outfile.display()
            )?;
        }
    }
    if let Some(ref range) = args.alirange {
        writeln!(out, "# alignment range:                  {range}")?;
    }
    if let Some(ref range) = args.modelrange {
        writeln!(out, "# model range:                      {range}")?;
    }
    if args.appendmask {
        writeln!(out, "# add to existing mask:             [on]")?;
    }
    if let Some(ref range) = args.model2ali {
        writeln!(out, "# ali ranges for model range:      {range}")?;
    }
    if let Some(ref range) = args.ali2model {
        writeln!(out, "# model ranges for ali range:      {range}")?;
    }
    if let Some(ref path) = args.summary_out {
        writeln!(
            out,
            "# output directed to file:          {}",
            path.display()
        )?;
    }
    if args.amino {
        writeln!(out, "# input alignment is asserted as:   protein")?;
    }
    if args.dna {
        writeln!(out, "# input alignment is asserted as:   DNA")?;
    }
    if args.rna {
        writeln!(out, "# input alignment is asserted as:   RNA")?;
    }
    if args.hand {
        writeln!(
            out,
            "# model architecture construction:  hand-specified by RF annotation"
        )?;
    }
    if args.symfrac != 0.5 {
        writeln!(
            out,
            "# sym fraction for model structure: {}",
            fmt_fixed3(args.symfrac)
        )?;
    }
    if args.fragthresh != 0.5 {
        writeln!(
            out,
            "# seq called frag if L <= x*alen:   {}",
            fmt_fixed3(args.fragthresh)
        )?;
    }
    // C (alimask.c:171-174): gated on IsUsed("--seed"), which is FALSE when the
    // supplied value equals the default (42). For seed 0 C prints BOTH lines:
    // the first `if (seed==0 && fprintf(...)<0)` actually evaluates the fprintf
    // (printing "one-time arbitrary"), the condition is false, so the `else if`
    // fprintf("set to: 0") also fires.
    if seed_was_requested {
        if args.seed == 0 {
            writeln!(
                out,
                "# random number seed:               one-time arbitrary"
            )?;
            writeln!(out, "# random number seed set to:        0")?;
        } else {
            writeln!(out, "# random number seed set to:        {}", args.seed)?;
        }
    }
    // C (alimask.c) gates the scheme line on esl_opt_IsUsed of a NON-default
    // weighting: --wpb is the default, so it never prints a scheme line (even
    // when given explicitly); only --wgsc/--wblosum/--wnone do.
    if args.wgsc {
        writeln!(out, "# relative weighting scheme:        G/S/C")?;
    }
    if args.wblosum {
        writeln!(out, "# relative weighting scheme:        BLOSUM filter")?;
    }
    if args.wnone {
        writeln!(out, "# relative weighting scheme:        none")?;
    }
    // C (alimask.c:170): the BLOSUM identity-cutoff line is gated on
    // IsUsed("--wid"), i.e. printed only when --wid is explicitly supplied,
    // NOT on --wblosum. (--wgiven has no header line in C at all, F6.)
    if wid_was_requested {
        writeln!(out, "# frac id cutoff for BLOSUM wgts:   {:.6}", args.wid)?;
    }
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )?;
    writeln!(out)?;
    Ok(())
}
