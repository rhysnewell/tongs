//! hmmscan — search sequence(s) against an HMM database.
//! Reverses hmmsearch: query=sequences, targets=HMMs.

#![allow(clippy::too_many_arguments)]

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use tongs::alphabet::Alphabet;
use tongs::bg::Bg;
use tongs::hmmfile_binary;
use tongs::logsum;
use tongs::output::{fmt_g, fmt_width5_1};
use tongs::pipeline::{BitCutoff, Pipeline};
use tongs::pressed;
use tongs::profile::{self, Profile, P7_LOCAL};
use tongs::sequence::{self, Sequence, SequenceFormat};
use tongs::simd::oprofile::OProfile;
use tongs::tophits::TopHits;
use tongs::util::cmath::c_exp_f64;

mod options;
use options::*;
mod header;
use header::*;
mod report;
use report::*;
mod scan;
use scan::*;

/// Entry point for `hmmscan`: scan each query sequence against every HMM in a
/// database and report a ranked, E-value-thresholded hit table.
///
/// Per-query, profiles are built once-per-HMM in parallel via rayon, the
/// pipeline (MSV -> Viterbi -> Forward + domain decoding) is run, the best
/// hit per HMM is collected, and the top-hits list is sorted, thresholded, and
/// printed in the canonical HMMER per-sequence tabular layout. Corresponds to
/// `serial_master()` in hmmer/src/hmmscan.c. MPI is omitted; pressed HMM
/// databases are required and read through the `.h3*` sidecar path.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let cmdline = args.join(" ");
    let table_cmdline = normalize_hmmscan_table_cmdline(&cmdline);
    let args = tongs::util::apply_hmmer_ncpu_env_default(args);
    let args = Args::parse_from(&args);

    validate_sequence_format("hmmscan --qformat", args.qformat.as_deref());

    logsum::p7_flogsuminit();

    let use_parallel = args.cpu > 1;
    if use_parallel {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.cpu)
            .start_handler(|_| tongs::util::simd_env::init())
            .build_global()
            .ok();
    }

    // C hmmscan requires a pressed HMM database and refuses plain ASCII HMMs.
    let pressed_available =
        pressed::pressed_db_sidecars_complete(&args.hmmdb).unwrap_or_else(|e| {
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
    let mut pressed_oprofiles = pressed::read_pressed_oprofiles(&args.hmmdb).unwrap_or_else(|e| {
        eprintln!("Error reading pressed optimized profiles: {}", e);
        std::process::exit(1);
    });
    if let Err(e) = pressed::validate_pressed_oprofiles_match_hmms(&hmms, &pressed_oprofiles) {
        eprintln!("Error reading pressed optimized profiles: {}", e);
        std::process::exit(1);
    };
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

    let mut tblout_file = args.tblout.as_ref().map(|p| {
        crate::subcmd::hmmsearch::create_output_file_or_exit(
            p,
            "Failed to open tabular per-seq output file {path} for writing",
        )
    });
    let mut domtblout_file = args.domtblout.as_ref().map(|p| {
        crate::subcmd::hmmsearch::create_output_file_or_exit(
            p,
            "Failed to open tabular per-dom output file {path} for writing",
        )
    });
    let mut pfamtblout_file = args.pfamtblout.as_ref().map(|p| {
        crate::subcmd::hmmsearch::create_output_file_or_exit(
            p,
            "Failed to open pfam-style tabular output file {path} for writing",
        )
    });

    let mut output_file = args.output.as_ref().map(|p| {
        crate::subcmd::hmmsearch::create_output_file_or_exit(
            p,
            "Failed to open output file {path} for writing",
        )
    });
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();
    let out: &mut dyn Write = match output_file {
        Some(ref mut file) => file,
        None => &mut stdout_lock,
    };

    writeln!(
        out,
        "# hmmscan :: search sequence(s) against a profile database"
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
    if let Some(ref path) = args.domtblout {
        writeln!(out, "# per-dom hits tabular output:     {}", path.display()).unwrap();
    }
    if let Some(ref path) = args.pfamtblout {
        writeln!(out, "# pfam-style tabular hit output:   {}", path.display()).unwrap();
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
    write_hmmscan_option_header(out, &args, &cmdline);
    writeln!(
        out,
        "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
    )
    .unwrap();
    writeln!(out).unwrap();

    // For each query sequence, search all HMMs
    let abc = alphabet_from_hmms(&hmms).unwrap_or_else(|e| {
        eprintln!("Error reading HMM database: {}", e);
        std::process::exit(1);
    });
    let bg = Bg::new(&abc);
    let mut profiles: Vec<Profile> = hmms
        .iter()
        .map(|hmm| {
            let mut gm = Profile::new(hmm.m, &abc);
            profile::profile_config(hmm, &bg, &mut gm, 100, P7_LOCAL);
            gm
        })
        .collect();

    let mut sqf =
        open_query_seq_file(&args.seqfile, &abc, args.qformat.as_deref()).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        });

    let mut sq = Sequence::new();
    let mut query_idx = 0usize;
    while sqf.read(&mut sq).unwrap_or_else(|e| {
        eprintln!("Error reading sequence file: {}", e);
        std::process::exit(1);
    }) {
        query_idx += 1;
        writeln!(out, "Query:       {}  [L={}]", sq.name, sq.n).unwrap();
        if !sq.acc.is_empty() {
            writeln!(out, "Accession:   {}", sq.acc).unwrap();
        }
        if !sq.desc.is_empty() {
            writeln!(out, "Description: {}", sq.desc).unwrap();
        }

        let scored: Vec<(Option<tongs::tophits::Hit>, PipelineStats)> = if !use_parallel {
            hmms.iter()
                .zip(pressed_oprofiles.iter_mut())
                .zip(profiles.iter_mut())
                .enumerate()
                .map(|(idx, ((hmm, oprofile), profile))| {
                    score_hmmscan_model(hmm, profile, oprofile, &bg, &sq, &args, bit_cutoff, idx)
                })
                .collect()
        } else {
            use rayon::prelude::*;
            hmms.par_iter()
                .enumerate()
                .map(|(idx, hmm)| {
                    let mut profile = profiles[idx].clone();
                    let mut oprofile = pressed_oprofiles[idx].clone();
                    score_hmmscan_model(
                        hmm,
                        &mut profile,
                        &mut oprofile,
                        &bg,
                        &sq,
                        &args,
                        bit_cutoff,
                        idx,
                    )
                })
                .collect()
        };

        let mut th = TopHits::new();
        let mut stats = PipelineStats::default();
        th.hits = scored
            .into_iter()
            .filter_map(|(hit, s)| {
                stats.n_past_msv += s.n_past_msv;
                stats.n_past_bias += s.n_past_bias;
                stats.n_past_vit += s.n_past_vit;
                stats.n_past_fwd += s.n_past_fwd;
                hit
            })
            .collect();
        let z = args.z_value.unwrap_or(hmms.len() as f64);
        th.sort_by_sortkey();
        let mut output_pli = Pipeline::new();
        configure_thresholds(&mut output_pli, &args);
        configure_acceleration(&mut output_pli, &args);
        if let Some(cutoff) = bit_cutoff {
            output_pli.use_bit_cutoffs = cutoff;
        }
        {
            th.threshold(&output_pli, z, z);
        }
        let domz = args.domz_value.unwrap_or(th.nreported as f64);
        if domz != z {
            th.threshold(&output_pli, z, domz);
        }

        if let Some(ref mut f) = tblout_file {
            crate::subcmd::hmmsearch::write_tblout(
                f,
                &sq.name,
                Some(&sq.acc),
                &th,
                z,
                query_idx == 1,
            );
        }
        if let Some(ref mut f) = domtblout_file {
            crate::subcmd::hmmsearch::write_domtblout(
                f,
                &sq.name,
                Some(&sq.acc),
                sq.n,
                &th,
                z,
                domz,
                query_idx == 1,
            );
        }
        if let Some(ref mut f) = pfamtblout_file {
            crate::subcmd::hmmsearch::write_pfamtblout_with_pipeline(
                f,
                &sq.name,
                Some(&sq.acc),
                sq.n,
                &th,
                &output_pli,
                z,
                domz,
            );
        }

        let textw = if args.notextw {
            0
        } else {
            args.textw.unwrap_or(120)
        };
        write_hmmscan_score_table(out, &th, z, args.acc, textw);
        write_hmmscan_domain_annotation(
            out, &th, z, domz, &sq.name, &sq.acc, sq.n, &hmms, args.acc, args.noali, textw,
        );
        write_scan_pipeline_stats(
            out,
            hmms.len() as u64,
            hmms.iter().map(|hmm| hmm.m as u64).sum(),
            sq.n as u64,
            z,
            domz,
            args.z_value.is_some(),
            args.domz_value.is_some(),
            &stats,
            if args.max { 1.0 } else { args.f1 },
            if args.max { 1.0 } else { args.f2 },
            if args.max { 1.0 } else { args.f3 },
        );
        writeln!(out, "//").unwrap();

        sq.reuse();
    }

    if let Some(ref mut f) = tblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "hmmscan",
            "SCAN",
            &args.seqfile,
            &args.hmmdb,
            &table_cmdline,
        );
    }
    if let Some(ref mut f) = domtblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "hmmscan",
            "SCAN",
            &args.seqfile,
            &args.hmmdb,
            &table_cmdline,
        );
    }
    if let Some(ref mut f) = pfamtblout_file {
        // C hmmscan deliberately passes p7_SEARCH_SEQS -> "SEARCH" for the
        // pfam-style (TabularXfam) tail (hmmscan.c:519,971), unlike the
        // tblout/domtblout tails above which use p7_SCAN_MODELS -> "SCAN".
        // Verified byte-for-byte against the bundled C binary. Do NOT "fix"
        // this to SCAN — that would break C parity.
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "hmmscan",
            "SEARCH",
            &args.seqfile,
            &args.hmmdb,
            &table_cmdline,
        );
    }

    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
#[path = "../../../tests/unit/subcmd/hmmscan_test.rs"]
mod tests;
