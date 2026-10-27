use std::io::Write;
use std::path::{Path, PathBuf};

use clap::Parser;

use tongs::alphabet::Alphabet;
use tongs::bg::Bg;
use tongs::builder::Builder;
use tongs::calibrate::CalibrationConfig;
use tongs::logsum;
use tongs::pipeline::Pipeline;
use tongs::profile::{self, Profile, P7_LOCAL};
use tongs::sequence::{self, Sequence};
use tongs::simd::oprofile::OProfile;
use tongs::subcmd::cli::{
    command_line_has_option, normalize_table_cmdline, parse_open_unit_f64, parse_pextend,
    parse_popen, parse_positive_f64, parse_positive_usize, parse_textw,
};
use tongs::subcmd::inputs::{open_seq_file_as, validate_sequence_format};
use tongs::subcmd::outputs::{create_output_file_or_exit, write_pipeline_stats, PipelineStats};
use tongs::tophits::{Hit, TopHits};

mod options;
use options::*;
mod header;
use header::*;
mod input;
use input::*;

/// Number of target sequences read into memory per batch in the multi-threaded
/// search loop. Mirrors `hmmsearch::TARGET_BATCH_SIZE` so phmmer's peak resident
/// set is bounded by the batch rather than the entire target database.
const TARGET_BATCH_SIZE: usize = 4096;

pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let cmdline = args.join(" ");
    let requested = Requested::from_args(&args);
    let args = tongs::util::apply_hmmer_ncpu_env_default(args);
    let args = Args::parse_from(&args);
    validate_sequence_format("phmmer --qformat", args.qformat.as_deref());
    validate_sequence_format("phmmer --tformat", args.tformat.as_deref());

    logsum::p7_flogsuminit();

    if args.cpu > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.cpu)
            .start_handler(|_| tongs::util::simd_env::init())
            .build_global()
            .ok();
    }

    let abc = Alphabet::amino();
    let bg = Bg::new(&abc);

    // p7_builder_Create() + p7_builder_{Set,Load}ScoreSystem(), phmmer.c:446-447:
    // --mxfile takes precedence, else the --mx built-in (default BLOSUM62).
    let mut builder = Builder::new(abc.abc_type).with_seed(args.seed);
    let score_system = if let Some(mxfile) = args.mxfile.as_ref() {
        builder.set_score_system(Some(mxfile), args.popen, args.pextend, &bg, &abc)
    } else {
        builder.load_score_system(&args.matrix, args.popen, args.pextend, &bg, &abc)
    };
    if let Err(e) = score_system {
        eprintln!("\nError: Failed to set single query seq score system:\n{e}\n");
        std::process::exit(1);
    }
    if args.seqfile.as_path() == Path::new("-") && args.seqdb.as_path() == Path::new("-") {
        eprintln!("Error: Either <seqfile> or <seqdb> may be '-' but not both");
        std::process::exit(1);
    }
    if args.seqdb.as_path() == Path::new("-") && count_query_sequences(&args.seqfile, &abc) > 1 {
        eprintln!(
            "Error: target sequence file - isn't rewindable; can't search it with multiple queries"
        );
        std::process::exit(1);
    }

    let mut tblout_file = args.tblout.as_ref().map(|p| {
        create_output_file_or_exit(
            p,
            "Failed to open tabular per-seq output file {path} for writing",
        )
    });
    let mut domtblout_file = args.domtblout.as_ref().map(|p| {
        create_output_file_or_exit(
            p,
            "Failed to open tabular per-dom output file {path} for writing",
        )
    });
    let mut pfamtblout_file = args.pfamtblout.as_ref().map(|p| {
        create_output_file_or_exit(
            p,
            "Failed to open pfam-style tabular output file {path} for writing",
        )
    });
    let mut ali_file = args.ali_outfile.as_ref().map(|p| {
        create_output_file_or_exit(p, "Failed to open alignment output file {path} for writing")
    });
    let mut output_file = args
        .output
        .as_ref()
        .map(|p| create_output_file_or_exit(p, "Failed to open output file {path} for writing"));
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();
    let mut out: &mut dyn Write = match output_file {
        Some(ref mut file) => file,
        None => &mut stdout_lock,
    };

    let textw = if args.notextw { 0 } else { args.textw };
    output_header(out, &args, &requested, &cmdline);
    out.flush().unwrap_or_else(|e| {
        eprintln!("Error writing output: {}", e);
        std::process::exit(1);
    });

    let calibration_config = CalibrationConfig {
        em_l: args.em_l,
        em_n: args.em_n,
        ev_l: args.ev_l,
        ev_n: args.ev_n,
        ef_l: args.ef_l,
        ef_n: args.ef_n,
        eft: args.eft,
    };
    // bld->EmL .. bld->Eft (p7_builder.c:113-119).
    builder.calibration = calibration_config;

    // Read query sequences
    let mut query_sqf = open_seq_file_as(&args.seqfile, &abc, args.qformat.as_deref())
        .unwrap_or_else(|e| {
            eprintln!("Error opening query file: {}", e);
            std::process::exit(1);
        });

    let mut query_sq = Sequence::new();
    let mut n_queries = 0usize;
    while query_sqf.read(&mut query_sq).unwrap_or_else(|e| {
        eprintln!("Error reading query file: {}", e);
        std::process::exit(1);
    }) {
        n_queries += 1;
        // Build HMM from query sequence
        // p7_SingleBuilder(), phmmer.c:557.
        let hmm = builder
            .single_builder(&query_sq.name, &query_sq.dsq, query_sq.n, &abc, &bg)
            .unwrap_or_else(|e| {
                eprintln!("\nError: build failed: {e}");
                std::process::exit(1);
            });

        let mut local_bg = bg.clone();
        local_bg.set_filter(hmm.m, &hmm.compo);

        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(&hmm, &local_bg, &mut gm, 400, P7_LOCAL);
        let om = OProfile::convert(&gm);

        writeln!(out, "Query:       {}  [L={}]", query_sq.name, query_sq.n).unwrap();
        if !query_sq.acc.is_empty() {
            writeln!(out, "Accession:   {}", query_sq.acc).unwrap();
        }
        if !query_sq.desc.is_empty() {
            writeln!(out, "Description: {}", query_sq.desc).unwrap();
        }
        out.flush().unwrap_or_else(|e| {
            eprintln!("Error writing output: {}", e);
            std::process::exit(1);
        });

        let mut th = TopHits::new();
        let mut stats = PipelineStats::default();
        let mut total_residues: u64 = 0;
        let mut n_targets: u64 = 0;
        let wants_alignment_file = args.ali_outfile.is_some();
        let do_alignment = true;
        let do_alignment_display = !args.noali || wants_alignment_file;
        let effective_f1 = if args.max { 1.0 } else { args.f1 };
        let effective_f2 = if args.max { 1.0 } else { args.f2 };
        let effective_f3 = if args.max { 1.0 } else { args.f3 };
        let effective_biasfilter = !(args.max || args.nobias);
        {
            let mut sqf = open_seq_file_as(&args.seqdb, &abc, args.tformat.as_deref())
                .unwrap_or_else(|e| {
                    eprintln!("Error opening target database: {}", e);
                    std::process::exit(1);
                });
            let mut restrict_started = args.restrictdb_stkey.is_none();
            if let Some(stkey) = args.restrictdb_stkey.as_deref() {
                sqf = crate::subcmd::hmmsearch::open_restricted_target_seq_file(
                    &args.seqdb,
                    &abc,
                    args.tformat.as_deref(),
                    stkey,
                    args.ssifile.as_deref(),
                )
                .unwrap_or_else(|e| {
                    eprintln!("Error opening restricted target database: {}", e);
                    std::process::exit(1);
                });
                restrict_started = true;
            }
            let mut restrict_seen = 0usize;
            let mut restrict_done = false;

            if args.cpu <= 1 {
                let mut lb = local_bg.clone();
                let mut lgm = gm.clone();
                let mut lom = om.clone();
                let mut lpli = Pipeline::new();
                lpli.f1 = effective_f1;
                lpli.f2 = effective_f2;
                lpli.f3 = effective_f3;
                lpli.do_max = args.max;
                lpli.do_biasfilter = effective_biasfilter;
                lpli.do_null2 = !args.nonull2;
                lpli.do_alignment = do_alignment;
                lpli.do_alignment_display = do_alignment_display;
                lpli.seed = args.seed;
                lpli.new_model(&lgm);
                configure_thresholds(&mut lpli, &args);
                configure_search_spaces(&mut lpli, &args);

                let mut sq = Sequence::new();
                while sqf.read(&mut sq).unwrap_or_else(|e| {
                    eprintln!("Error reading target database: {}", e);
                    std::process::exit(1);
                }) {
                    if !restrict_started {
                        if args
                            .restrictdb_stkey
                            .as_deref()
                            .is_some_and(|key| sq.name == key)
                        {
                            restrict_started = true;
                        } else {
                            sq.reuse();
                            continue;
                        }
                    }
                    if args
                        .restrictdb_n
                        .is_some_and(|limit| restrict_seen >= limit)
                    {
                        break;
                    }
                    restrict_seen += 1;
                    total_residues += sq.n as u64;
                    n_targets += 1;

                    // Reset per-target pipeline counters; the scratch matrices
                    // inside `lpli` are kept and grown as needed.
                    lpli.n_targets = n_targets.saturating_sub(1);
                    lpli.n_past_msv = 0;
                    lpli.n_past_bias = 0;
                    lpli.n_past_vit = 0;
                    lpli.n_past_fwd = 0;
                    lb.set_length(sq.n);

                    let mut lth = TopHits::new();
                    if lpli.run(&mut lgm, &mut lom, &lb, &hmm, &sq, &mut lth) {
                        if let Some(hit) = lth.hits.into_iter().next() {
                            th.hits.push(hit);
                        }
                    }
                    stats.n_past_msv += lpli.n_past_msv;
                    stats.n_past_bias += lpli.n_past_bias;
                    stats.n_past_vit += lpli.n_past_vit;
                    stats.n_past_fwd += lpli.n_past_fwd;
                    sq.reuse();
                }
            } else {
                use rayon::prelude::*;
                use std::sync::Arc;
                let shared_gm = Arc::new(gm.clone());
                let shared_om = Arc::new(om.clone());
                let shared_bg = Arc::new(local_bg.clone());

                let f1 = effective_f1;
                let f2 = effective_f2;
                let f3 = effective_f3;
                let do_max = args.max;
                let do_biasfilter = effective_biasfilter;
                let do_null2 = !args.nonull2;
                let seed = args.seed;
                let cli_args = &args;

                let mut sq = Sequence::new();
                let mut batch: Vec<(u64, Sequence)> = Vec::with_capacity(TARGET_BATCH_SIZE);

                loop {
                    while batch.len() < TARGET_BATCH_SIZE {
                        match sqf.read(&mut sq) {
                            Ok(true) => {
                                if !restrict_started {
                                    if args
                                        .restrictdb_stkey
                                        .as_deref()
                                        .is_some_and(|key| sq.name == key)
                                    {
                                        restrict_started = true;
                                    } else {
                                        sq.reuse();
                                        continue;
                                    }
                                }
                                if args
                                    .restrictdb_n
                                    .is_some_and(|limit| restrict_seen >= limit)
                                {
                                    restrict_done = true;
                                    break;
                                }
                                restrict_seen += 1;
                                total_residues += sq.n as u64;
                                n_targets += 1;
                                batch.push((n_targets, sq.clone()));
                                sq.reuse();
                            }
                            Ok(false) => break,
                            Err(e) => {
                                eprintln!("Error reading target database: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }

                    if batch.is_empty() {
                        break;
                    }

                    let results: Vec<(Option<Hit>, u64, u64, u64, u64)> = batch
                        .par_iter()
                        .map_init(
                            || {
                                tongs::util::simd_env::init();
                                let local_bg = (*shared_bg).clone();
                                let local_gm = (*shared_gm).clone();
                                let local_om = (*shared_om).clone();
                                let mut local_pli = Pipeline::new();
                                local_pli.f1 = f1;
                                local_pli.f2 = f2;
                                local_pli.f3 = f3;
                                local_pli.do_max = do_max;
                                local_pli.do_biasfilter = do_biasfilter;
                                local_pli.do_null2 = do_null2;
                                local_pli.do_alignment = do_alignment;
                                local_pli.do_alignment_display = do_alignment_display;
                                local_pli.seed = seed;
                                local_pli.new_model(&local_gm);
                                configure_thresholds(&mut local_pli, cli_args);
                                configure_search_spaces(&mut local_pli, cli_args);
                                (local_bg, local_gm, local_om, local_pli)
                            },
                            |(local_bg, local_gm, local_om, local_pli), (target_idx, sq)| {
                                // Reset per-target pipeline counters; the
                                // scratch matrices inside `local_pli` are
                                // kept and grown as needed.
                                local_pli.n_targets = target_idx.saturating_sub(1);
                                local_pli.n_past_msv = 0;
                                local_pli.n_past_bias = 0;
                                local_pli.n_past_vit = 0;
                                local_pli.n_past_fwd = 0;
                                local_bg.set_length(sq.n);

                                let mut local_th = TopHits::new();
                                let hit = if local_pli.run(
                                    local_gm,
                                    local_om,
                                    local_bg,
                                    &hmm,
                                    sq,
                                    &mut local_th,
                                ) {
                                    local_th.hits.into_iter().next()
                                } else {
                                    None
                                };
                                (
                                    hit,
                                    local_pli.n_past_msv,
                                    local_pli.n_past_bias,
                                    local_pli.n_past_vit,
                                    local_pli.n_past_fwd,
                                )
                            },
                        )
                        .collect();

                    for (hit, msv, bias, vit, fwd) in results {
                        stats.n_past_msv += msv;
                        stats.n_past_bias += bias;
                        stats.n_past_vit += vit;
                        stats.n_past_fwd += fwd;
                        if let Some(h) = hit {
                            th.hits.push(h);
                        }
                    }

                    batch.clear();
                    if restrict_done {
                        break;
                    }
                }
            }
        }
        if n_targets == 0 {
            eprintln!("Error: no sequences found in {}", args.seqdb.display());
            std::process::exit(1);
        }

        let z = args.z_value.unwrap_or(n_targets as f64);
        th.sort_by_sortkey();
        {
            let tmp_pli = threshold_pipeline_for_output(
                &args,
                effective_f1,
                effective_f2,
                effective_f3,
                effective_biasfilter,
            );
            th.threshold(&tmp_pli, z, z);
            let domz = args.domz_value.unwrap_or(th.nreported as f64);
            if domz != z {
                th.threshold(&tmp_pli, z, domz);
            }
        }
        let domz = args.domz_value.unwrap_or(th.nreported as f64);
        let output_pli = threshold_pipeline_for_output(
            &args,
            effective_f1,
            effective_f2,
            effective_f3,
            effective_biasfilter,
        );

        crate::subcmd::hmmsearch::write_standard_stdout_tables(
            &mut out,
            &th,
            &hmm.name,
            hmm.acc.as_deref().unwrap_or(""),
            hmm.m,
            hmm.cs.as_deref(),
            z,
            domz,
            textw,
            args.show_acc,
            args.noali,
            false,
        );
        write_pipeline_stats(
            &mut out,
            hmm.m,
            n_targets,
            total_residues,
            th.nreported as u64,
            z,
            domz,
            args.z_value.is_some(),
            args.domz_value.is_some(),
            &stats,
            effective_f1,
            effective_f2,
            effective_f3,
        );
        writeln!(out, "//").unwrap();

        if let Some(ref mut f) = tblout_file {
            crate::subcmd::hmmsearch::write_tblout(
                f,
                &query_sq.name,
                Some(&query_sq.acc),
                &th,
                z,
                n_queries == 1,
            );
        }
        if let Some(ref mut f) = domtblout_file {
            crate::subcmd::hmmsearch::write_domtblout(
                f,
                &query_sq.name,
                Some(&query_sq.acc),
                query_sq.n,
                &th,
                z,
                domz,
                n_queries == 1,
            );
        }
        if let Some(ref mut f) = pfamtblout_file {
            crate::subcmd::hmmsearch::write_pfamtblout_with_pipeline(
                f,
                &query_sq.name,
                Some(&query_sq.acc),
                query_sq.n,
                &th,
                &output_pli,
                z,
                domz,
            );
        }
        if let Some(ref mut f) = ali_file {
            // C phmmer.c:620-638: build the included-domain MSA (named after the
            // model/query, with the query's acc/desc when present), write it,
            // then echo the confirmation line carrying the MSA's nseq.
            let nseq = crate::subcmd::hmmsearch::write_ali_output(
                f,
                &abc,
                hmm.m,
                &hmm.name,
                Some(query_sq.acc.as_str()),
                Some(query_sq.desc.as_str()),
                "phmmer (HMMER 3.4)",
                &th,
                textw,
            );
            match nseq {
                Some(nseq) => writeln!(
                    out,
                    "# Alignment of {} hits satisfying inclusion thresholds saved to: {}",
                    nseq,
                    args.ali_outfile.as_ref().unwrap().display()
                )
                .unwrap(),
                None => writeln!(
                    out,
                    "# No hits satisfy inclusion thresholds; no alignment saved"
                )
                .unwrap(),
            }
        }

        query_sq.reuse();
    }
    if n_queries == 0 {
        eprintln!("Error: no sequences found in {}", args.seqfile.display());
        std::process::exit(1);
    }

    let table_cmdline = normalize_table_cmdline(&cmdline, &["phmmer"]);
    if let Some(ref mut f) = tblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "phmmer",
            "SEARCH",
            &args.seqfile,
            &args.seqdb,
            &table_cmdline,
        );
    }
    if let Some(ref mut f) = domtblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "phmmer",
            "SEARCH",
            &args.seqfile,
            &args.seqdb,
            &table_cmdline,
        );
    }
    if let Some(ref mut f) = pfamtblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "phmmer",
            "SEARCH",
            &args.seqfile,
            &args.seqdb,
            &table_cmdline,
        );
    }

    if let Some(ref mut f) = tblout_file {
        f.flush().unwrap();
    }
    if let Some(ref mut f) = domtblout_file {
        f.flush().unwrap();
    }
    if let Some(ref mut f) = pfamtblout_file {
        f.flush().unwrap();
    }

    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
#[path = "../../../tests/unit/subcmd/phmmer_test.rs"]
mod tests;
