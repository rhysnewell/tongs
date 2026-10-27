use std::io::Write;

use super::Args;

pub(super) fn write_header(
    out: &mut dyn Write,
    args: &Args,
    cmdline: &str,
    cpu_was_requested: bool,
) {
    writeln!(
        out,
        "# hmmsearch :: search profile(s) against a sequence database"
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
        "# query HMM file:                  {}",
        args.hmmfile.display()
    )
    .unwrap();
    writeln!(
        out,
        "# target sequence database:        {}",
        args.seqdb.display()
    )
    .unwrap();
    if let Some(format) = args.tformat.as_deref() {
        writeln!(out, "# targ <seqfile> format asserted:  {format}").unwrap();
    }
    if let Some(ref path) = args.outfile {
        writeln!(out, "# output directed to file:         {}", path.display()).unwrap();
    }
    if let Some(ref path) = args.tblout {
        writeln!(out, "# per-seq hits tabular output:     {}", path.display()).unwrap();
    }
    if let Some(ref path) = args.domtblout {
        writeln!(out, "# per-dom hits tabular output:     {}", path.display()).unwrap();
    }
    if let Some(ref path) = args.pfamtblout {
        writeln!(out, "# pfam-style tabular hit output:   {}", path.display()).unwrap();
    }
    if let Some(ref path) = args.ali_outfile {
        writeln!(out, "# MSA of all hits saved to file:   {}", path.display()).unwrap();
    }
    if args.noali {
        writeln!(out, "# show alignments in output:       no").unwrap();
    }
    if args.show_acc {
        writeln!(out, "# prefer accessions over names:    yes").unwrap();
    }
    write_hmmsearch_option_header(out, args, cmdline);
    if cpu_was_requested {
        writeln!(out, "# number of worker threads:        {}", args.cpu).unwrap();
    }
    if args.notextw {
        writeln!(out, "# max ASCII text line length:      unlimited").unwrap();
    } else if args.textw != 120 {
        writeln!(out, "# max ASCII text line length:      {}", args.textw).unwrap();
    }
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(out).unwrap();
    out.flush().unwrap_or_else(|e| {
        eprintln!("Error writing output: {}", e);
        std::process::exit(1);
    });
}

fn write_hmmsearch_option_header(out: &mut dyn Write, args: &Args, cmdline: &str) {
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
            "# sequence reporting threshold:    score >= {}",
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
            "# domain reporting threshold:      score >= {}",
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
            tongs::output::fmt_g(args.inc_dome)
        )
        .unwrap();
    }
    if let Some(score) = args.inc_dom_t {
        writeln!(
            out,
            "# domain inclusion threshold:      score >= {}",
            tongs::output::fmt_g(score)
        )
        .unwrap();
    }
    if args.cut_ga {
        writeln!(out, "# model-specific thresholding:     GA cutoffs").unwrap();
    }
    if args.cut_nc {
        writeln!(out, "# model-specific thresholding:     NC cutoffs").unwrap();
    }
    if args.cut_tc {
        writeln!(out, "# model-specific thresholding:     TC cutoffs").unwrap();
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
    if let Some(stkey) = &args.restrictdb_stkey {
        writeln!(out, "# Restrict db to start at seq key: {}", stkey).unwrap();
    }
    if let Some(n) = args.restrictdb_n {
        writeln!(out, "# Restrict db to # target seqs:    {}", n).unwrap();
    }
    if let Some(ssifile) = &args.ssifile {
        writeln!(
            out,
            "# Override ssi file to:            {}",
            ssifile.display()
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
