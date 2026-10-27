use super::*;

pub(super) fn output_header(
    out: &mut dyn Write,
    args: &Args,
    requested: &Requested,
    cmdline: &str,
) {
    writeln!(
        out,
        "# nhmmer :: search a DNA model, alignment, or sequence against a DNA database"
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
        "# query file:                      {}",
        args.hmmfile.display()
    )
    .unwrap();
    writeln!(
        out,
        "# target sequence database:        {}",
        args.seqdb.display()
    )
    .unwrap();
    if args
        .tformat
        .as_deref()
        .is_some_and(|format| format.eq_ignore_ascii_case("fasta"))
    {
        writeln!(out, "# target format asserted:          fasta").unwrap();
    }
    if args
        .tformat
        .as_deref()
        .is_some_and(is_fmindex_target_format)
    {
        writeln!(out, "# target format asserted:          fmindex").unwrap();
    }
    if let Some(path) = &args.tblout {
        writeln!(out, "# hits tabular output:             {}", path.display()).unwrap();
    }
    if let Some(path) = &args.dfamtblout {
        writeln!(out, "# hits output in Dfam format:      {}", path.display()).unwrap();
    }
    if let Some(path) = &args.ali_outfile {
        writeln!(out, "# MSA of all hits saved to file:   {}", path.display()).unwrap();
    }
    if let Some(path) = &args.aliscoresout {
        writeln!(out, "# alignment scores output:         {}", path.display()).unwrap();
    }
    if let Some(path) = &args.hmmout {
        writeln!(out, "# hmm output:                      {}", path.display()).unwrap();
    }
    if args.show_acc {
        writeln!(out, "# prefer accessions over names:    yes").unwrap();
    }
    if args.noali {
        writeln!(out, "# show alignments in output:       no").unwrap();
    }
    write_nhmmer_option_header(out, args, cmdline);
    if args.notextw {
        writeln!(out, "# max ASCII text line length:      unlimited").unwrap();
    } else if args.textw != 120 {
        writeln!(out, "# max ASCII text line length:      {}", args.textw).unwrap();
    }
    if args.singlemx {
        writeln!(out, "# Use score matrix for 1-seq MSAs:  on").unwrap();
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
    // C prints the alphabet assertion only when the flag is given (nhmmer.c:341-342).
    if args.dna {
        writeln!(out, "# input query is asserted as:      DNA").unwrap();
    }
    if args.rna {
        writeln!(out, "# input query is asserted as:      RNA").unwrap();
    }
    if let Some(format) = &args.qformat {
        writeln!(out, "# query format asserted:           {}", format).unwrap();
    }
    if let Some(w_beta) = args.w_beta {
        writeln!(out, "# window length beta value:        {}", w_beta).unwrap();
    }
    if let Some(w_length) = args.w_length {
        writeln!(out, "# window length :                  {}", w_length).unwrap();
    }
    if let Some(block_length) = args.block_length {
        writeln!(out, "# block length :                   {}", block_length).unwrap();
    }
    if args.qsingle_seqs {
        writeln!(out, "# query contains individual seqs:  on").unwrap();
    }
    if args.b1 != 110 {
        writeln!(out, "# biased comp SSV window len:      {}", args.b1).unwrap();
    }
    if args.b2 != 240 {
        writeln!(out, "# biased comp Viterbi window len:  {}", args.b2).unwrap();
    }
    if args.b3 != 1000 {
        writeln!(out, "# biased comp Forward window len:  {}", args.b3).unwrap();
    }
    if let Some(path) = &args.bgfile {
        writeln!(out, "# file with custom bg probs:       {}", path.display()).unwrap();
    }
    if requested.seed_max_depth {
        writeln!(
            out,
            "# FM Seed length:                  {}",
            args.seed_max_depth
        )
        .unwrap();
    }
    if requested.seed_sc_thresh {
        writeln!(
            out,
            "# FM score threshold (bits):       {}",
            args.seed_sc_thresh
        )
        .unwrap();
    }
    if requested.seed_sc_density {
        writeln!(
            out,
            "# FM score density (bits/pos):     {}",
            args.seed_sc_density
        )
        .unwrap();
    }
    if requested.seed_drop_max_len {
        writeln!(
            out,
            "# FM max neg-growth length:        {}",
            args.seed_drop_max_len
        )
        .unwrap();
    }
    if requested.seed_drop_lim {
        writeln!(
            out,
            "# FM max run drop:                 {}",
            args.seed_drop_lim
        )
        .unwrap();
    }
    if requested.seed_req_pos {
        writeln!(
            out,
            "# FM req positive run length:      {}",
            args.seed_req_pos
        )
        .unwrap();
    }
    if requested.seed_consens_match {
        writeln!(
            out,
            "# FM consec consensus match req:   {}",
            args.seed_consens_match
        )
        .unwrap();
    }
    if requested.seed_ssv_length {
        writeln!(
            out,
            "# FM len used for Vit window:      {}",
            args.seed_ssv_length
        )
        .unwrap();
    }
    writeln!(out, "# number of worker threads:        {}", args.cpu).unwrap();
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
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(out).unwrap();
}

fn write_nhmmer_option_header(out: &mut dyn Write, args: &Args, cmdline: &str) {
    if command_line_has_option(cmdline, "-E") {
        writeln!(
            out,
            "# sequence reporting threshold:    E-value <= {}",
            fmt_g(args.e_value)
        )
        .unwrap();
    }
    if let Some(score) = args.score_threshold {
        writeln!(
            out,
            "# sequence reporting threshold:    score >= {}",
            fmt_g(score)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--incE") {
        writeln!(
            out,
            "# sequence inclusion threshold:    E-value <= {}",
            fmt_g(args.inc_e)
        )
        .unwrap();
    }
    if let Some(score) = args.inc_t {
        writeln!(
            out,
            "# sequence inclusion threshold:    score >= {}",
            fmt_g(score)
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
        writeln!(out, "# SSV filter P threshold:       <= {}", fmt_g(args.f1)).unwrap();
    }
    if command_line_has_option(cmdline, "--F2") {
        writeln!(out, "# Vit filter P threshold:       <= {}", fmt_g(args.f2)).unwrap();
    }
    if command_line_has_option(cmdline, "--F3") {
        writeln!(out, "# Fwd filter P threshold:       <= {}", fmt_g(args.f3)).unwrap();
    }
    if args.nobias {
        writeln!(out, "# biased composition HMM filter:   off").unwrap();
    }
    if args.nonull2 {
        writeln!(out, "# null2 bias corrections:          off").unwrap();
    }
    if args.watson {
        writeln!(out, "# search only top strand:          on").unwrap();
    }
    if args.crick {
        writeln!(out, "# search only bottom strand:       on").unwrap();
    }
    if let Some(z) = args.z_value {
        writeln!(out, "# database size is set to:         {:.1} Mb", z).unwrap();
    }
    if command_line_has_option(cmdline, "--seed") {
        if args.seed == 0 {
            writeln!(out, "# random number seed:              one-time arbitrary").unwrap();
        }
        writeln!(out, "# random number seed set to:       {}", args.seed).unwrap();
    }
}
