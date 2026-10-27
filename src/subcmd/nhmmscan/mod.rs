use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use tongs::alphabet::Alphabet;
use tongs::bg::Bg;
use tongs::builder::{self, DEFAULT_WINDOW_BETA};
use tongs::hmmfile_binary;
use tongs::logsum;
use tongs::output::{fmt_bias, fmt_fixed0, fmt_g, fmt_score};
use tongs::pipeline::{BitCutoff, Pipeline};
use tongs::pressed;
use tongs::profile::{self, Profile, P7_LOCAL};
use tongs::sequence::Sequence;
use tongs::subcmd::cli::{
    command_line_has_option, normalize_table_cmdline, parse_positive_f64, parse_textw,
    parse_window_beta, parse_window_length,
};
use tongs::subcmd::inputs::{open_seq_file_as, validate_sequence_format};
use tongs::subcmd::outputs::{create_file_or_exit, display_name};
use tongs::tophits::TopHits;
use tongs::util::cmath::c_exp_f64;

mod options;
use options::*;
mod report;
use report::*;
mod setup;
use setup::*;

pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let cmdline = args.join(" ");
    let table_cmdline = normalize_table_cmdline(&cmdline, &["nhmmscan"]);
    let args = tongs::util::apply_hmmer_ncpu_env_default(args);
    let args = Args::parse_from(&args);
    validate_sequence_format("nhmmscan --qformat", args.qformat.as_deref());
    if matches!(args.w_length, Some(0..=3)) {
        eprintln!("Invalid window length value");
        std::process::exit(1);
    }

    logsum::p7_flogsuminit();

    if args.cpu > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.cpu)
            .start_handler(|_| tongs::util::simd_env::init())
            .build_global()
            .ok();
    }

    // C nhmmscan requires a pressed HMM database and refuses plain ASCII HMMs.
    let pressed_available = pressed::pressed_db_available(&args.hmmdb).unwrap_or_else(|e| {
        eprintln!("Error opening pressed HMM database: {}", e);
        std::process::exit(1);
    });
    if !pressed_available {
        eprintln!(
            "Error opening pressed HMM database: {} is not pressed; use hmmpress first",
            args.hmmdb.display()
        );
        std::process::exit(1);
    }
    let h3m_path = pressed::pressed_h3m_path(&args.hmmdb);
    let hmms = hmmfile_binary::read_binary_hmm_file(&h3m_path).unwrap_or_else(|e| {
        eprintln!("Error reading HMM database: {}", e);
        std::process::exit(1);
    });
    let pressed_oprofiles = pressed::read_pressed_oprofiles(&args.hmmdb).unwrap_or_else(|e| {
        eprintln!("Error reading pressed optimized profiles: {}", e);
        std::process::exit(1);
    });
    if let Err(e) = pressed::validate_pressed_oprofiles_match_hmms(&hmms, &pressed_oprofiles) {
        eprintln!("Error reading pressed optimized profiles: {}", e);
        std::process::exit(1);
    };
    // nhmmscan.c:360: C reads the first optimized profile up front and refuses
    // the database if it carries no window length.
    if hmms.first().is_some_and(|hmm| hmm.max_length == -1) {
        eprintln!(
            "\nError: No MAXL field in model(s); is this an old model format?\nnhmmer/hmmscan require HMMER 3.1 models or later."
        );
        std::process::exit(1);
    }
    let bit_cutoff = selected_bit_cutoff(&args);
    if let Some(cutoff) = bit_cutoff {
        for hmm in &hmms {
            let mut pli = Pipeline::new();
            pli.use_bit_cutoffs = cutoff;
            if let Err(e) = pli.new_model_thresholds(&hmm.cutoff) {
                eprintln!("Error: {} for model {}", e, hmm.name);
                std::process::exit(1);
            }
        }
    }

    let mut tblout_file = args
        .tblout
        .as_ref()
        .map(|p| create_file_or_exit(p, "tblout file"));
    let mut dfamtblout_file = args
        .dfamtblout
        .as_ref()
        .map(|p| create_file_or_exit(p, "dfamtblout file"));

    let mut output_file = args
        .output
        .as_ref()
        .map(|p| create_file_or_exit(p, "output file"));
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();
    let out: &mut dyn Write = match output_file {
        Some(ref mut file) => file,
        None => &mut stdout_lock,
    };

    writeln!(
        out,
        "# nhmmscan :: search DNA sequence(s) against a DNA profile database"
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
        writeln!(out, "# input seqfile format asserted:   {format}").unwrap();
    }
    if let Some(w_beta) = args.w_beta {
        writeln!(out, "# window length beta value:        {}", w_beta).unwrap();
    }
    if let Some(w_length) = args.w_length {
        writeln!(out, "# window length :                  {}", w_length).unwrap();
    }
    if args.b1 != 110 {
        writeln!(out, "# biased comp MSV window len:      {}", args.b1).unwrap();
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
    writeln!(
        out,
        "# target HMM database:             {}",
        args.hmmdb.display()
    )
    .unwrap();
    if let Some(ref path) = args.output {
        writeln!(out, "# output directed to file:         {}", path.display()).unwrap();
    }
    if let Some(ref path) = args.tblout {
        writeln!(out, "# per-seq hits tabular output:     {}", path.display()).unwrap();
    }
    if let Some(ref path) = args.dfamtblout {
        writeln!(out, "# hits output in Dfam format:      {}", path.display()).unwrap();
    }
    if args.acc {
        writeln!(out, "# prefer accessions over names:    yes").unwrap();
    }
    if args.noali {
        writeln!(out, "# show alignments in output:       no").unwrap();
    }
    if command_line_has_option(&cmdline, "--cpu") || std::env::var_os("HMMER_NCPU").is_some() {
        if args.cpu == 0 {
            writeln!(out, "# multithread parallelization:     off").unwrap();
        } else {
            writeln!(
                out,
                "# multithread parallelization:     {} workers",
                args.cpu
            )
            .unwrap();
        }
    }
    if args.notextw {
        writeln!(out, "# max ASCII text line length:      unlimited").unwrap();
    }
    if let Some(textw) = args.textw {
        writeln!(out, "# max ASCII text line length:      {}", textw).unwrap();
    }
    if let Some(z) = args.z_value {
        writeln!(out, "# sequence search space set to:    {}", fmt_fixed0(z)).unwrap();
    }
    write_nhmmscan_option_header(out, &args, &cmdline);
    if command_line_has_option(&cmdline, "--seed") {
        if args.seed == 0 {
            writeln!(out, "# random number seed:              one-time arbitrary").unwrap();
        } else {
            writeln!(out, "# random number seed set to:       {}", args.seed).unwrap();
        }
    }
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(out).unwrap();

    // For each query sequence, search all HMMs
    let abc = nucleotide_alphabet_from_hmms(&hmms).unwrap_or_else(|e| {
        eprintln!("Error reading HMM database: {}", e);
        std::process::exit(1);
    });
    let mut bg = Bg::new(&abc);
    if let Some(path) = &args.bgfile {
        if let Err(e) = bg.read_file(&abc, path) {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }

    let mut sqf =
        open_seq_file_as(&args.seqfile, &abc, args.qformat.as_deref()).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        });

    let mut sq = Sequence::new();
    let mut query_idx = 0usize;
    let search_space_ln = tongs::util::cmath::c_log_f64(hmms.len() as f64);
    while sqf.read(&mut sq).unwrap_or_else(|e| {
        eprintln!("Error reading sequence file: {}", e);
        std::process::exit(1);
    }) {
        query_idx += 1;
        writeln!(out, "Query:       {}  [L={}]", sq.name, sq.n).unwrap();

        let scan_model = |idx: usize, hmm: &tongs::hmm::Hmm| {
            tongs::util::simd_env::init();
            let mut local_bg = bg.clone();
            local_bg.set_filter(hmm.m, &hmm.compo);
            local_bg.set_length(sq.n);

            let mut gm = Profile::new(hmm.m, &abc);
            profile::profile_config(hmm, &local_bg, &mut gm, sq.n as i32, P7_LOCAL);
            let om = pressed_oprofiles[idx].clone();
            let mut threshold_pli = Pipeline::new();
            threshold_pli.new_model(&gm);
            configure_thresholds(&mut threshold_pli, &args);
            if let Some(cutoff) = bit_cutoff {
                threshold_pli.use_bit_cutoffs = cutoff;
                if threshold_pli.new_model_thresholds(&hmm.cutoff).is_err() {
                    return Ok(ScanModelResult {
                        hits: Vec::new(),
                        stats: PipelineStats::default(),
                    });
                }
            }
            let threshold_config =
                crate::subcmd::nhmmer::NhmmerThresholdConfig::from_pipeline(&threshold_pli);
            let bias_windows = crate::subcmd::nhmmer::NhmmerBiasWindowLengths {
                b1: args.b1,
                b2: args.b2,
                b3: args.b3,
            };

            let max_length = nhmmscan_max_length(hmm, args.w_length, args.w_beta);
            let mut search_hmm = hmm.clone();
            search_hmm.max_length = max_length;
            let effective_f1 = if args.max { 0.3 } else { args.f1 };
            let effective_f2 = if args.max { 1.0 } else { args.f2 };
            let effective_f3 = if args.max { 1.0 } else { args.f3 };
            let effective_nobias = args.max || args.nobias;
            let do_watson = !args.crick;
            let do_crick = !args.watson;
            let mut hits = Vec::new();
            let msv_counter = std::sync::atomic::AtomicU64::new(0);
            let bias_counter = std::sync::atomic::AtomicU64::new(0);
            let vit_counter = std::sync::atomic::AtomicU64::new(0);
            let fwd_counter = std::sync::atomic::AtomicU64::new(0);

            if do_crick && abc.complement.is_some() {
                let mut rc_dsq = sq.dsq.clone();
                abc.revcomp(&mut rc_dsq, sq.n);
                let rc_sq = Sequence {
                    name: sq.name.clone(),
                    acc: sq.acc.clone(),
                    desc: sq.desc.clone(),
                    dsq: rc_dsq,
                    n: sq.n,
                    l: sq.l,
                    taxid: -1,
                };
                let mut rc_hits = crate::subcmd::nhmmer::search_sequence(
                    &rc_sq,
                    &search_hmm,
                    &gm,
                    &om,
                    &local_bg,
                    max_length,
                    effective_f1,
                    effective_f2,
                    effective_f3,
                    args.max,
                    effective_nobias,
                    args.nonull2,
                    args.seed,
                    bias_windows,
                    threshold_config,
                    true,
                    &msv_counter,
                    &bias_counter,
                    &vit_counter,
                    &fwd_counter,
                    None,
                    None,
                )?;
                convert_crick_hits_to_forward(&mut rc_hits, sq.n);
                hits.extend(rc_hits);
            }

            if do_watson {
                hits.extend(crate::subcmd::nhmmer::search_sequence(
                    &sq,
                    &search_hmm,
                    &gm,
                    &om,
                    &local_bg,
                    max_length,
                    effective_f1,
                    effective_f2,
                    effective_f3,
                    args.max,
                    effective_nobias,
                    args.nonull2,
                    args.seed,
                    bias_windows,
                    threshold_config,
                    false,
                    &msv_counter,
                    &bias_counter,
                    &vit_counter,
                    &fwd_counter,
                    None,
                    None,
                )?);
            }

            let strands_searched = (do_watson as usize) + (do_crick as usize);
            let seq_len = sq.n * strands_searched.max(1);
            let nhmmscan_ln =
                nhmmscan_long_target_ln_adjustment(seq_len, max_length, search_space_ln);
            for hit in &mut hits {
                hit.lnp += nhmmscan_ln;
                hit.sortkey = hit.lnp;
                if let Some(dom0) = hit.dcl.first_mut() {
                    dom0.lnp = hit.lnp;
                }
                hit.name = hmm.name.clone();
                hit.acc = hmm.acc.clone().unwrap_or_default();
                hit.desc = hmm.desc.clone().unwrap_or_default();
                hit.n = hmm.m;
            }
            if let Some(cutoff) = bit_cutoff {
                let mut local_th = TopHits::new();
                local_th.hits = hits;
                let mut tmp_pli = Pipeline::new();
                tmp_pli.long_target = true;
                configure_thresholds(&mut tmp_pli, &args);
                tmp_pli.use_bit_cutoffs = cutoff;
                if tmp_pli.new_model_thresholds(&hmm.cutoff).is_ok() {
                    local_th.threshold(&tmp_pli, 1.0, 1.0);
                }
                hits = local_th.hits;
            }
            Ok(ScanModelResult {
                hits,
                stats: PipelineStats {
                    n_past_msv: msv_counter.load(std::sync::atomic::Ordering::Relaxed),
                    n_past_bias: bias_counter.load(std::sync::atomic::Ordering::Relaxed),
                    n_past_vit: vit_counter.load(std::sync::atomic::Ordering::Relaxed),
                    n_past_fwd: fwd_counter.load(std::sync::atomic::Ordering::Relaxed),
                },
            })
        };
        let scan_results: Vec<Result<ScanModelResult, String>> = if args.cpu == 0 {
            hmms.iter()
                .enumerate()
                .map(|(idx, hmm)| scan_model(idx, hmm))
                .collect()
        } else {
            use rayon::prelude::*;
            hmms.par_iter()
                .enumerate()
                .map(|(idx, hmm)| scan_model(idx, hmm))
                .collect()
        };
        let mut stats = PipelineStats::default();
        let all_hits: Vec<tongs::tophits::Hit> = scan_results
            .into_iter()
            .flat_map(|result| {
                let result = result.unwrap_or_else(|err| {
                    eprintln!("Error: {err}");
                    std::process::exit(1);
                });
                stats.n_past_msv += result.stats.n_past_msv;
                stats.n_past_bias += result.stats.n_past_bias;
                stats.n_past_vit += result.stats.n_past_vit;
                stats.n_past_fwd += result.stats.n_past_fwd;
                result.hits
            })
            .collect();

        let mut th = TopHits::new();
        th.hits = all_hits;
        let evalue_scale = 1.0;
        th.sort_by_modelname_and_alipos();
        th.remove_duplicates();
        th.sort_by_sortkey();
        {
            let mut tmp_pli = Pipeline::new();
            tmp_pli.long_target = true;
            configure_thresholds(&mut tmp_pli, &args);
            if let Some(cutoff) = bit_cutoff {
                tmp_pli.use_bit_cutoffs = cutoff;
            }
            th.threshold(&tmp_pli, evalue_scale, evalue_scale);
        }

        if let Some(ref mut f) = tblout_file {
            crate::subcmd::nhmmer::write_nhmmer_tblout(
                f,
                "nhmmscan",
                "SCAN",
                &sq.name,
                Some(&sq.acc),
                &th,
                &args.seqfile,
                &args.hmmdb,
                &cmdline,
                query_idx == 1,
                false,
                evalue_scale,
                " modlen",
            );
        }
        if let Some(ref mut f) = dfamtblout_file {
            crate::subcmd::nhmmer::write_nhmmer_dfamtblout(
                f,
                &sq.name,
                None,
                "modlen",
                &th,
                evalue_scale,
            );
        }

        let textw = if args.notextw {
            0
        } else {
            args.textw.unwrap_or(120)
        };
        write_nhmmscan_stdout_hits(
            out,
            &th,
            evalue_scale,
            &sq.name,
            args.acc,
            args.noali,
            textw,
        );
        let stats_residue_multiplier = if args.watson { 1 } else { 2 };
        let pos_output = reported_aligned_residues(&th);
        write_nhmmscan_pipeline_stats(
            out,
            (sq.n * stats_residue_multiplier) as u64,
            hmms.len() as u64,
            hmms.iter().map(|hmm| hmm.m as u64).sum(),
            th.nreported as u64,
            pos_output,
            &stats,
            if args.max { 0.3 } else { args.f1 },
            if args.max { 1.0 } else { args.f2 },
            if args.max { 1.0 } else { args.f3 },
        );
        writeln!(out, "//").unwrap();

        sq.reuse();
    }

    if query_idx == 0 {
        eprintln!("Error: no sequences found in {}", args.seqfile.display());
        std::process::exit(1);
    }

    if let Some(ref mut f) = tblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "nhmmscan",
            "SCAN",
            &args.seqfile,
            &args.hmmdb,
            &table_cmdline,
        );
    }
    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
#[path = "../../../tests/unit/subcmd/nhmmscan_test.rs"]
mod tests;
