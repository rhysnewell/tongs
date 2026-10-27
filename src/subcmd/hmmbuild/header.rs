use super::*;

pub(super) fn output_header(
    summary: &mut dyn Write,
    args: &Args,
    requested: &Requested,
    singlemx_matrix_label: &str,
) {
    writeln!(
        summary,
        "# hmmbuild :: profile HMM construction from multiple sequence alignments"
    )
    .unwrap();
    writeln!(summary, "# HMMER 3.4 (Aug 2023); http://hmmer.org/").unwrap();
    writeln!(
        summary,
        "# Copyright (C) 2023 Howard Hughes Medical Institute."
    )
    .unwrap();
    writeln!(
        summary,
        "# Freely distributed under the BSD open source license."
    )
    .unwrap();
    writeln!(
        summary,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(
        summary,
        "# input alignment file:             {}",
        args.msafile.display()
    )
    .unwrap();
    writeln!(
        summary,
        "# output HMM file:                  {}",
        args.hmmfile.display()
    )
    .unwrap();
    if let Some(ref name) = args.name {
        writeln!(summary, "# name (the single) HMM:            {name}").unwrap();
    }
    if let Some(ref path) = args.summary_out {
        writeln!(
            summary,
            "# output directed to file:          {}",
            path.display()
        )
        .unwrap();
    }
    if let Some(ref path) = args.msa_out {
        writeln!(
            summary,
            "# processed alignment resaved to:   {}",
            path.display()
        )
        .unwrap();
    }
    if args.amino {
        writeln!(summary, "# input alignment is asserted as:   protein").unwrap();
    }
    if args.dna {
        writeln!(summary, "# input alignment is asserted as:   DNA").unwrap();
    }
    if args.rna {
        writeln!(summary, "# input alignment is asserted as:   RNA").unwrap();
    }
    if args.hand {
        writeln!(
            summary,
            "# model architecture construction:  hand-specified by RF annotation"
        )
        .unwrap();
    }
    if args.symfrac != 0.5 {
        writeln!(
            summary,
            "# sym fraction for model structure: {}",
            fmt_fixed3(args.symfrac as f64)
        )
        .unwrap();
    }
    if args.fragthresh != 0.5 {
        writeln!(
            summary,
            "# seq called frag if L <= x*alen:  {}",
            fmt_fixed3(args.fragthresh as f64)
        )
        .unwrap();
    }
    if args.singlemx {
        writeln!(
            summary,
            "# single sequence builder:          substitution matrix"
        )
        .unwrap();
        writeln!(summary, "# use score matrix for 1-seq MSAs:  on").unwrap();
        writeln!(
            summary,
            "# substitution score matrix:        {}",
            singlemx_matrix_label
        )
        .unwrap();
    }
    if requested.popen {
        writeln!(
            summary,
            "# gap open probability:             {:.6}",
            args.popen
        )
        .unwrap();
    }
    if requested.pextend {
        writeln!(
            summary,
            "# gap extend probability:           {:.6}",
            args.pextend
        )
        .unwrap();
    }
    if args.wnone {
        writeln!(summary, "# relative weighting scheme:        none").unwrap();
    }
    if args.wgsc {
        writeln!(summary, "# relative weighting scheme:        G/S/C").unwrap();
    }
    if args.wblosum {
        writeln!(summary, "# relative weighting scheme:        BLOSUM filter").unwrap();
    }
    if args.wgiven {
        writeln!(summary, "# relative weighting scheme:        given").unwrap();
    }
    if args.wblosum {
        writeln!(
            summary,
            "# frac id cutoff for BLOSUM wgts:   {:.6}",
            args.wid
        )
        .unwrap();
    }
    if args.enone {
        writeln!(summary, "# effective seq number scheme:      none").unwrap();
    }
    if args.eentexp {
        writeln!(
            summary,
            "# effective seq number scheme:      entropy weighting using exponent-based scaling"
        )
        .unwrap();
    }
    if args.eclust {
        writeln!(
            summary,
            "# effective seq number scheme:      single linkage clusters"
        )
        .unwrap();
    }
    if let Some(eset) = args.eset {
        writeln!(
            summary,
            "# effective seq number:             set to {:.6}",
            eset
        )
        .unwrap();
    }
    if args.eclust {
        writeln!(
            summary,
            "# frac id cutoff for --eclust:      {:.6}",
            args.eid
        )
        .unwrap();
    }
    if args.pnone {
        writeln!(summary, "# prior scheme:                     none").unwrap();
    }
    if args.plaplace {
        writeln!(summary, "# prior scheme:                     Laplace").unwrap();
    }
    if let Some(ere) = args.ere {
        writeln!(
            summary,
            "# minimum rel entropy target:       {:.6} bits",
            ere
        )
        .unwrap();
    }
    if args.esigma != 45.0 {
        writeln!(
            summary,
            "# entropy target sigma parameter:   {:.6} bits",
            args.esigma
        )
        .unwrap();
    }
    if requested.seed {
        if args.seed == 0 {
            writeln!(
                summary,
                "# random number seed:               one-time arbitrary"
            )
            .unwrap();
        } else {
            writeln!(summary, "# random number seed set to:        {}", args.seed).unwrap();
        }
    }
    if requested.cpu {
        writeln!(summary, "# number of worker threads:         {}", args.cpu).unwrap();
    }
    if requested.em_l {
        writeln!(summary, "# seq length for MSV Gumbel mu fit: {}", args.em_l).unwrap();
    }
    if requested.em_n {
        writeln!(summary, "# seq number for MSV Gumbel mu fit: {}", args.em_n).unwrap();
    }
    if requested.ev_l {
        writeln!(summary, "# seq length for Vit Gumbel mu fit: {}", args.ev_l).unwrap();
    }
    if requested.ev_n {
        writeln!(summary, "# seq number for Vit Gumbel mu fit: {}", args.ev_n).unwrap();
    }
    if requested.ef_l {
        writeln!(summary, "# seq length for Fwd exp tau fit:   {}", args.ef_l).unwrap();
    }
    if requested.ef_n {
        writeln!(summary, "# seq number for Fwd exp tau fit:   {}", args.ef_n).unwrap();
    }
    if requested.eft {
        writeln!(
            summary,
            "# tail mass for Fwd exp tau fit:    {:.6}",
            args.eft
        )
        .unwrap();
    }
    if let Some(max_insert_len) = args.max_insert_len {
        writeln!(
            summary,
            "# max insert length:                {}",
            max_insert_len
        )
        .unwrap();
    }
    if let Some(w_beta) = args.w_beta {
        writeln!(
            summary,
            "# window length tail mass:          {} bits",
            w_beta
        )
        .unwrap();
    }
    if let Some(w_length) = args.w_length {
        writeln!(summary, "# window length :                   {}", w_length).unwrap();
    }
    writeln!(
        summary,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(summary).unwrap();
}
