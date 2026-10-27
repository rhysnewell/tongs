use super::*;

pub(super) fn output_header(
    out: &mut dyn Write,
    args: &Args,
    requested: &Requested,
    cmdline: &str,
) {
    writeln!(
        out,
        "# jackhmmer :: iteratively search a protein sequence against a protein database"
    )
    .unwrap();
    writeln!(out, "# HMMER 3.4 (Aug 2023); http://hmmer.org/").unwrap();
    writeln!(out, "# Copyright (C) 2023 Howard Hughes Medical Institute.").unwrap();
    writeln!(
        out,
        "# Freely distributed under the BSD open source license."
    )
    .unwrap();
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(
        out,
        "# query sequence file:             {}",
        args.seqfile.display()
    )
    .unwrap();
    if let Some(format) = args.qformat.as_deref() {
        writeln!(out, "# query <seqfile> format asserted: {format}").unwrap();
    }
    writeln!(
        out,
        "# target sequence database:        {}",
        args.seqdb.display()
    )
    .unwrap();
    if let Some(format) = args.tformat.as_deref() {
        writeln!(out, "# target <seqdb> format asserted:  {format}").unwrap();
    }
    writeln!(
        out,
        "# maximum iterations set to:       {}",
        args.max_iterations
    )
    .unwrap();
    if let Some(output) = &args.output {
        writeln!(
            out,
            "# output directed to file:         {}",
            output.display()
        )
        .unwrap();
    }
    if let Some(tblout) = &args.tblout {
        writeln!(
            out,
            "# per-seq hits tabular output:     {}",
            tblout.display()
        )
        .unwrap();
    }
    if let Some(domtblout) = &args.domtblout {
        writeln!(
            out,
            "# per-dom hits tabular output:     {}",
            domtblout.display()
        )
        .unwrap();
    }
    if let Some(ali_outfile) = &args.ali_outfile {
        writeln!(
            out,
            "# MSA of hits saved to file:       {}",
            ali_outfile.display()
        )
        .unwrap();
    }
    if args.show_acc {
        writeln!(out, "# prefer accessions over names:    yes").unwrap();
    }
    if args.noali {
        writeln!(out, "# show alignments in output:       no").unwrap();
    }
    if requested.matrix {
        writeln!(out, "# subst score matrix (built-in):   {}", args.matrix).unwrap();
    }
    if requested.mxfile {
        writeln!(
            out,
            "# subst score matrix (file):       {}",
            args.mxfile.as_ref().unwrap().display()
        )
        .unwrap();
    }
    if requested.popen {
        writeln!(out, "# gap open probability:            {:.6}", args.popen).unwrap();
    }
    if requested.pextend {
        writeln!(
            out,
            "# gap extend probability:          {:.6}",
            args.pextend
        )
        .unwrap();
    }
    write_jackhmmer_option_header(out, args, cmdline);
    if args.notextw {
        writeln!(out, "# max ASCII text line length:      unlimited").unwrap();
    } else if args.textw != 120 {
        writeln!(out, "# max ASCII text line length:      {}", args.textw).unwrap();
    }
    if args.max {
        writeln!(
            out,
            "# Max sensitivity mode:            on [all heuristic filters off]"
        )
        .unwrap();
    }
    if args.fragthresh != 0.5 {
        writeln!(
            out,
            "# define fragments if <= x*alen:   {:.3}",
            args.fragthresh
        )
        .unwrap();
    }
    if args.wpb {
        writeln!(out, "# relative weighting scheme:       Henikoff PB").unwrap();
    }
    if args.wgsc {
        writeln!(out, "# relative weighting scheme:       G/S/C").unwrap();
    }
    if args.wblosum {
        writeln!(out, "# relative weighting scheme:       BLOSUM filter").unwrap();
        writeln!(out, "# frac id cutoff for BLOSUM wgts:  {:.6}", args.wid).unwrap();
    }
    if args.wnone {
        writeln!(out, "# relative weighting scheme:       none").unwrap();
    }
    if args.eent {
        writeln!(out, "# effective seq number scheme:     entropy weighting").unwrap();
    }
    if args.eentexp {
        writeln!(
            out,
            "# effective seq number scheme:     entropy weighting using exponent-based scaling"
        )
        .unwrap();
    }
    if args.eclust {
        writeln!(
            out,
            "# effective seq number scheme:     single linkage clusters"
        )
        .unwrap();
        writeln!(out, "# frac id cutoff for --eclust:     {:.6}", args.eid).unwrap();
    }
    if args.enone {
        writeln!(out, "# effective seq number scheme:     none").unwrap();
    }
    if let Some(eset) = args.eset {
        writeln!(out, "# effective seq number:            set to {:.6}", eset).unwrap();
    }
    if let Some(ere) = args.ere {
        writeln!(out, "# minimum rel entropy target:      {:.6} bits", ere).unwrap();
    }
    if args.esigma != 45.0 {
        writeln!(
            out,
            "# entropy target sigma parameter:  {:.6} bits",
            args.esigma
        )
        .unwrap();
    }
    if args.pnone {
        writeln!(out, "# prior scheme:                    none").unwrap();
    }
    if args.plaplace {
        writeln!(out, "# prior scheme:                    Laplace +1").unwrap();
    }
    if requested.cpu {
        writeln!(out, "# number of worker threads:        {}", args.cpu).unwrap();
    }
    if requested.em_l {
        writeln!(out, "# seq length, MSV Gumbel mu fit:   {}", args.em_l).unwrap();
    }
    if requested.em_n {
        writeln!(out, "# seq number, MSV Gumbel mu fit:   {}", args.em_n).unwrap();
    }
    if requested.ev_l {
        writeln!(out, "# seq length, Vit Gumbel mu fit:   {}", args.ev_l).unwrap();
    }
    if requested.ev_n {
        writeln!(out, "# seq number, Vit Gumbel mu fit:   {}", args.ev_n).unwrap();
    }
    if requested.ef_l {
        writeln!(out, "# seq length, Fwd exp tau fit:     {}", args.ef_l).unwrap();
    }
    if requested.ef_n {
        writeln!(out, "# seq number, Fwd exp tau fit:     {}", args.ef_n).unwrap();
    }
    if requested.eft {
        writeln!(out, "# tail mass for Fwd exp tau fit:   {:.6}", args.eft).unwrap();
    }
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(out).unwrap();
}

pub(super) fn normalize_jackhmmer_table_cmdline(cmdline: &str) -> String {
    let mut tokens = cmdline.split_whitespace();
    match (tokens.next(), tokens.next()) {
        (Some(_wrapper), Some("jackhmmer")) => {
            let rest = tokens.collect::<Vec<_>>();
            if rest.is_empty() {
                "jackhmmer".to_string()
            } else {
                format!("jackhmmer {}", rest.join(" "))
            }
        }
        _ => cmdline.to_string(),
    }
}

fn write_jackhmmer_option_header(out: &mut dyn Write, args: &Args, cmdline: &str) {
    if command_line_has_option(cmdline, "-E") {
        writeln!(
            out,
            "# sequence reporting threshold:    E-value <= {}",
            tongs::output::fmt_g(args.e_value)
        )
        .unwrap();
    }
    if let Some(score) = args.score_threshold {
        writeln!(
            out,
            // C jackhmmer prints "<=" here, despite score thresholds being minimums.
            "# sequence reporting threshold:    score <= {}",
            tongs::output::fmt_g(score)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--domE") {
        writeln!(
            out,
            "# domain reporting threshold:      E-value <= {}",
            tongs::output::fmt_g(args.dom_e)
        )
        .unwrap();
    }
    if let Some(score) = args.dom_t {
        writeln!(
            out,
            // C jackhmmer prints "<=" here, despite score thresholds being minimums.
            "# domain reporting threshold:      score <= {}",
            tongs::output::fmt_g(score)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--incE") {
        writeln!(
            out,
            "# sequence inclusion threshold:    E-value <= {}",
            tongs::output::fmt_g(args.inc_e)
        )
        .unwrap();
    }
    if let Some(score) = args.inc_t {
        writeln!(
            out,
            "# sequence inclusion threshold:    score >= {}",
            tongs::output::fmt_g(score)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--incdomE") {
        writeln!(
            out,
            "# domain inclusion threshold:      E-value <= {}",
            tongs::output::fmt_g(args.incdom_e)
        )
        .unwrap();
    }
    if let Some(score) = args.incdom_t {
        writeln!(
            out,
            "# domain inclusion threshold:      score >= {}",
            tongs::output::fmt_g(score)
        )
        .unwrap();
    }
    if args.max {
        writeln!(
            out,
            "# Max sensitivity mode:            on [all heuristic filters off]"
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--F1") {
        writeln!(
            out,
            "# MSV filter P threshold:       <= {}",
            tongs::output::fmt_g(args.f1)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--F2") {
        writeln!(
            out,
            "# Vit filter P threshold:       <= {}",
            tongs::output::fmt_g(args.f2)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--F3") {
        writeln!(
            out,
            "# Fwd filter P threshold:       <= {}",
            tongs::output::fmt_g(args.f3)
        )
        .unwrap();
    }
    if args.nobias {
        writeln!(out, "# biased composition HMM filter:   off").unwrap();
    }
    if args.nonull2 {
        writeln!(out, "# null2 bias corrections:          off").unwrap();
    }
    if let Some(z) = args.z_value {
        writeln!(
            out,
            "# sequence search space set to:    {}",
            tongs::output::fmt_fixed0(z)
        )
        .unwrap();
    }
    if let Some(domz) = args.domz_value {
        writeln!(
            out,
            "# domain search space set to:      {}",
            tongs::output::fmt_fixed0(domz)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--seed") {
        if args.seed == 0 {
            writeln!(out, "# random number seed:              one-time arbitrary").unwrap();
        } else {
            writeln!(out, "# random number seed set to:       {}", args.seed).unwrap();
        }
    }
}

fn command_line_has_option(cmdline: &str, option: &str) -> bool {
    let compact_short = option
        .strip_prefix('-')
        .filter(|rest| !rest.starts_with('-') && rest.chars().count() == 1)
        .map(|_| option);
    cmdline.split_whitespace().any(|token| {
        token == option
            || token.starts_with(&format!("{option}="))
            || compact_short
                .is_some_and(|short| token.starts_with(short) && token.len() > short.len())
    })
}
