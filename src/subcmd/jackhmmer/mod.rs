//! jackhmmer — iteratively search a protein sequence against a protein database.
//! Builds single-seq HMM, searches, collects hits into MSA, rebuilds, repeats.

use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{ArgAction, Parser};

use tongs::alphabet::Alphabet;
use tongs::bg::Bg;
use tongs::builder;
use tongs::builder::Builder;
use tongs::calibrate::CalibrationConfig;
use tongs::logsum;
use tongs::msa::Msa;
use tongs::pipeline::Pipeline;
use tongs::prior::PriorStrategy;
use tongs::profile::{self, Profile, P7_LOCAL};
use tongs::sequence::{self, Sequence, SequenceFormat};
use tongs::simd::oprofile::OProfile;
use tongs::tophits::{Hit, TopHits};
use tongs::trace::{State, Trace};

mod options;
use options::*;
mod header;
use header::*;
mod input;
use input::*;
mod report;
use report::*;
mod checkpoint;
use checkpoint::*;
mod stockholm;
pub(crate) use stockholm::*;

const TARGET_BATCH_SIZE: usize = 256;

/// Entry point for `jackhmmer`: iteratively search a protein query against a
/// protein database, rebuilding the HMM each round from included hits until
/// convergence or `-N` iterations.
///
/// Round 1 builds a single-sequence HMM (phmmer-style); subsequent rounds
/// reuse the previous round's included-hit MSA augmented with the original
/// query to rebuild the model. Each round runs the standard pipeline against
/// the target DB in bounded parallel batches, applies E/incE thresholds,
/// reports the canonical per-sequence tabular block, and optionally writes
/// `--chkhmm`/`--chkali`/`--tblout`/`--domtblout` checkpoints. Corresponds to
/// `serial_master()` in hmmer/src/jackhmmer.c.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let cmdline = args.join(" ");
    let requested = Requested::from_args(&args);
    let args = tongs::util::apply_hmmer_ncpu_env_default(args);
    let args = Args::parse_from(&args);
    if args.cut_ga {
        println!("Failed to parse command line: jackhmmer does not accept a --cut-ga option");
        std::process::exit(1);
    }
    if args.cut_nc {
        println!("Failed to parse command line: jackhmmer does not accept a --cut-nc option");
        std::process::exit(1);
    }
    if args.cut_tc {
        println!("Failed to parse command line: jackhmmer does not accept a --cut-tc option");
        std::process::exit(1);
    }
    if args.fast {
        println!("Failed to parse command line: jackhmmer does not accept a --fast option");
        std::process::exit(1);
    }
    if args.hand {
        println!("Failed to parse command line: jackhmmer does not accept a --hand option");
        std::process::exit(1);
    }
    if args.symfrac.is_some() {
        println!("Failed to parse command line: jackhmmer does not accept a --symfrac option");
        std::process::exit(1);
    }
    if args.wgiven {
        println!("Failed to parse command line: jackhmmer does not accept a --wgiven option");
        std::process::exit(1);
    }
    if requested.wid && !args.wblosum {
        eprintln!("Error: --wid only works in combination with --wblosum");
        std::process::exit(1);
    }
    if requested.eid && !args.eclust {
        eprintln!("Error: --eid only works in combination with --eclust");
        std::process::exit(1);
    }
    validate_sequence_format("jackhmmer --qformat", args.qformat.as_deref());
    validate_sequence_format("jackhmmer --tformat", args.tformat.as_deref());
    let rebuild_weighting = if args.wnone {
        builder::RelativeWeighting::None
    } else if args.wgsc {
        builder::RelativeWeighting::Gsc
    } else if args.wblosum {
        builder::RelativeWeighting::Blosum {
            identity_cutoff: args.wid,
        }
    } else {
        builder::RelativeWeighting::PositionBased
    };
    let rebuild_effn = if args.enone {
        builder::EffectiveSeqNumber::None
    } else if args.eclust {
        builder::EffectiveSeqNumber::Cluster {
            identity_cutoff: args.eid,
        }
    } else if let Some(eset) = args.eset {
        builder::EffectiveSeqNumber::Set(eset)
    } else if args.eentexp {
        builder::EffectiveSeqNumber::EntropyExp {
            target_re: args.ere,
            target_sigma: Some(args.esigma),
        }
    } else {
        builder::EffectiveSeqNumber::Entropy {
            target_re: args.ere,
            target_sigma: Some(args.esigma),
        }
    };
    let rebuild_prior = if args.pnone {
        PriorStrategy::None
    } else if args.plaplace {
        PriorStrategy::Laplace
    } else {
        PriorStrategy::Default
    };
    if args.seqdb.as_path() == Path::new("-") {
        eprintln!("Error: target sequence database may not be '-' for jackhmmer");
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

    let abc = Alphabet::amino();
    let bg = Bg::new(&abc);

    // p7_builder_Create() + p7_builder_{Set,Load}ScoreSystem(),
    // jackhmmer.c:938-939.
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

    let mut output_file = args.output.as_ref().map(|p| {
        crate::subcmd::hmmsearch::create_output_file_or_exit(
            p,
            "Failed to open output file {path} for writing",
        )
    });
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();
    let mut out: &mut dyn Write = match output_file {
        Some(ref mut file) => file,
        None => &mut stdout_lock,
    };
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
    let mut ali_file = args.ali_outfile.as_ref().map(|p| {
        crate::subcmd::hmmsearch::create_output_file_or_exit(
            p,
            "Failed to open alignment output file {path} for writing",
        )
    });

    let textw = if args.notextw { 0 } else { args.textw };
    output_header(out, &args, &requested, &cmdline);

    let table_cmdline = normalize_jackhmmer_table_cmdline(&cmdline);
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

    // Read query sequences. C jackhmmer runs the full iterative search once
    // for each FASTA record and keeps the outer report/footer shared.
    let mut query_sqf = open_sequence_file(&args.seqfile, &abc, args.qformat.as_deref())
        .unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        });
    let mut query_sq = Sequence::new();
    let mut saw_query = false;
    let mut wrote_tblout_header = false;
    let mut wrote_domtblout_header = false;
    loop {
        query_sq.reuse();
        if !query_sqf.read(&mut query_sq).unwrap_or_else(|e| {
            eprintln!("Error reading query file: {}", e);
            std::process::exit(1);
        }) {
            if !saw_query {
                eprintln!("Error: no query sequence found");
                std::process::exit(1);
            }
            break;
        }
        saw_query = true;

        writeln!(out, "Query:       {}  [L={}]", query_sq.name, query_sq.n).unwrap();
        if !query_sq.acc.is_empty() {
            writeln!(out, "Accession:   {}", query_sq.acc).unwrap();
        }
        if !query_sq.desc.is_empty() {
            writeln!(out, "Description: {}", query_sq.desc).unwrap();
        }
        writeln!(out).unwrap();

        let query_tr = exact_match_query_trace(query_sq.n);

        let mut prev_included_names: Vec<String> = Vec::new();
        let mut prev_msa_nseq = 1usize;
        let mut prev_hmm: Option<tongs::hmm::Hmm> = None;
        let mut final_hits: Option<TopHits> = None;
        let mut final_z: Option<f64> = None;
        let mut final_domz: Option<f64> = None;
        let mut final_model_len: Option<usize> = None;
        // MSA of included hits from the previous round. Built once at the
        // bottom of each iteration (matching C jackhmmer.c:683) and reused at
        // the top of the next iteration to rebuild the HMM, by `--chkali` at
        // the bottom of the current iteration, and (after the loop) by `-A`.
        // C builds the same `msa` once per round (jackhmmer.c:683-691, 723).
        let mut final_msa: Option<Msa> = None;

        for iteration in 1..=args.max_iterations {
            // Build HMM for this iteration
            let hmm = if iteration == 1 {
                // First iteration: single-sequence HMM (phmmer-style)
                // p7_SingleBuilder(), jackhmmer.c:1020.
                builder
                    .single_builder(&query_sq.name, &query_sq.dsq, query_sq.n, &abc, &bg)
                    .unwrap_or_else(|e| {
                        eprintln!("\nError: build failed: {e}");
                        std::process::exit(1);
                    })
            } else {
                // C jackhmmer.c:683 builds the round's MSA once at end of
                // iteration; the same `msa` object is then handed to
                // `p7_Builder` at the top of the next round (line 605) and
                // freed at line 620. Mirror that: the MSA built at end of
                // the prior iteration is stashed in `final_msa`; take it
                // here so the rebuild consumes it and the storage is freed
                // before the new round's pipeline allocates `all_hits`.
                let msa = final_msa.take();
                // Match C jackhmmer.c:595's "destroy info->th at top of round":
                // the prior round's tophits (with all alidisplay payloads) is
                // no longer needed once its MSA has been materialized. Drop it
                // before the new round's all_hits accumulator and pipeline
                // scratch start growing.
                final_hits = None;

                let Some(msa) = msa else {
                    writeln!(out, "@@ No hits to build MSA from. Stopping.").unwrap();
                    break;
                };
                let prev_included_count = prev_included_names.len();
                let _ = prev_hmm
                    .as_ref()
                    .expect("jackhmmer rebuild requested without previous-round HMM");
                let mut hmm = builder::build_hmm_from_msa_with_prior(
                    &msa,
                    &abc,
                    &bg,
                    0.5,
                    args.fragthresh,
                    true,
                    rebuild_weighting,
                    rebuild_effn,
                    rebuild_prior,
                    // jackhmmer.c uses one P7_BUILDER (created once with the
                    // user's --EmL/--EmN/--EvL/--EvN/--EfL/--EfN/--Eft) for
                    // every round, so round 2+ rebuilds must calibrate with
                    // the same config as round 1, not Easel defaults.
                    calibration_config,
                    args.seed,
                )
                // jackhmmer.c:683 does not check p7_Builder's status, but
                // p7_Calibrate's failure is fatal in every driver that does;
                // report it the same way rather than search with bad STATS.
                .unwrap_or_else(|e| {
                    eprintln!("\nError: build failed: {e}");
                    std::process::exit(1);
                });
                // Stash counts needed for the header banner, then drop the
                // MSA — C does the equivalent at jackhmmer.c:620 right after
                // the builder consumes it.
                let msa_nseq_for_banner = msa.nseq;
                drop(msa);
                if !query_sq.desc.is_empty() {
                    hmm.desc = Some(query_sq.desc.clone());
                    hmm.flags |= tongs::hmm::P7H_DESC;
                }
                writeln!(out, "@@").unwrap();
                writeln!(out, "@@ Round:                  {}", iteration).unwrap();
                writeln!(
                out,
                "@@ Included in MSA:        {} subsequences (query + {} subseqs from {} targets)",
                msa_nseq_for_banner,
                msa_nseq_for_banner.saturating_sub(1),
                prev_included_count
            )
                .unwrap();
                writeln!(out, "@@ Model size:             {} positions", hmm.m).unwrap();
                writeln!(out, "@@").unwrap();
                writeln!(out).unwrap();
                hmm
            };

            if let Some(prefix) = &args.chkhmm {
                write_hmm_checkpoint(prefix, iteration, &hmm);
            }

            // Configure profile and search
            let mut local_bg = bg.clone();
            local_bg.set_filter(hmm.m, &hmm.compo);
            let mut gm = Profile::new(hmm.m, &abc);
            profile::profile_config(&hmm, &local_bg, &mut gm, 400, P7_LOCAL);
            let om = OProfile::convert(&gm);

            // Search the target DB in bounded batches so RSS scales with
            // the batch size instead of the full database size.
            use rayon::prelude::*;
            let mut all_hits = Vec::new();
            let mut z = 0usize;
            let mut total_residues = 0u64;
            let mut stats = PipelineStats::default();
            {
                let mut sqf = open_sequence_file(&args.seqdb, &abc, args.tformat.as_deref())
                    .unwrap_or_else(|e| {
                        eprintln!("Error: {}", e);
                        std::process::exit(1);
                    });
                let mut sq = Sequence::new();
                let mut batch: Vec<(u64, Sequence)> = Vec::with_capacity(TARGET_BATCH_SIZE);

                loop {
                    while batch.len() < TARGET_BATCH_SIZE {
                        match sqf.read(&mut sq) {
                            Ok(true) => {
                                z += 1;
                                total_residues += sq.n as u64;
                                batch.push((z as u64, sq.clone()));
                                sq.reuse();
                            }
                            Ok(false) => break,
                            Err(e) => {
                                eprintln!("Error: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }

                    if batch.is_empty() {
                        break;
                    }

                    let batch_hits: Vec<(Option<Hit>, u64, u64, u64, u64)> = batch
                        .par_iter()
                        .map_init(
                            || {
                                tongs::util::simd_env::init();
                                let lb = local_bg.clone();
                                let lgm = gm.clone();
                                let lom = om.clone();
                                let mut lpli = Pipeline::new();
                                configure_pipeline(&mut lpli, &args);
                                lpli.do_biasfilter = !args.nobias;
                                lpli.do_null2 = !args.nonull2;
                                lpli.seed = args.seed;
                                lpli.new_model(&lgm);
                                (lb, lgm, lom, lpli)
                            },
                            |(lb, lgm, lom, lpli), (target_idx, sq)| {
                                // Mirror C's `p7_pipeline_Reuse`: zero the per-target
                                // counters before each target so the returned counts
                                // reflect only this sequence (the outer loop sums them
                                // into `stats`). DP scratch / profile / oprofile / bg
                                // are reused across all of this worker's targets.
                                lpli.n_targets = target_idx.saturating_sub(1);
                                lpli.n_past_msv = 0;
                                lpli.n_past_bias = 0;
                                lpli.n_past_vit = 0;
                                lpli.n_past_fwd = 0;

                                lb.set_length(sq.n);

                                let mut lth = TopHits::new();
                                let hit = if lpli.run(lgm, lom, lb, &hmm, sq, &mut lth) {
                                    lth.hits.into_iter().next()
                                } else {
                                    None
                                };
                                (
                                    hit,
                                    lpli.n_past_msv,
                                    lpli.n_past_bias,
                                    lpli.n_past_vit,
                                    lpli.n_past_fwd,
                                )
                            },
                        )
                        .collect();
                    for (hit, msv, bias, vit, fwd) in batch_hits {
                        stats.n_past_msv += msv;
                        stats.n_past_bias += bias;
                        stats.n_past_vit += vit;
                        stats.n_past_fwd += fwd;
                        if let Some(hit) = hit {
                            all_hits.push(hit);
                        }
                    }
                    batch.clear();
                }
            }

            let mut th = TopHits::new();
            th.hits = all_hits;
            let n_targets = z as u64;
            let z = args.z_value.unwrap_or(n_targets as f64);
            final_z = Some(z);
            th.sort_by_sortkey();
            {
                let mut tmp_pli = Pipeline::new();
                configure_pipeline(&mut tmp_pli, &args);
                tmp_pli.do_biasfilter = !args.nobias;
                tmp_pli.do_null2 = !args.nonull2;
                tmp_pli.seed = args.seed;
                th.threshold(&tmp_pli, z, z);
                let domz = args.domz_value.unwrap_or(th.nreported as f64);
                if domz != z {
                    th.threshold(&tmp_pli, z, domz);
                }
                final_domz = Some(domz);
            }
            for hit in &mut th.hits {
                if hit.flags & tongs::tophits::P7_IS_INCLUDED != 0
                    && !prev_included_names.iter().any(|prev| prev == &hit.name)
                {
                    hit.flags |= tongs::tophits::P7_IS_NEW;
                } else {
                    hit.flags &= !tongs::tophits::P7_IS_NEW;
                }
            }
            crate::subcmd::hmmsearch::write_standard_stdout_tables(
                &mut out,
                &th,
                &hmm.name,
                hmm.acc.as_deref().unwrap_or(""),
                hmm.m,
                hmm.cs.as_deref(),
                z,
                final_domz.unwrap_or(z),
                textw,
                args.show_acc,
                args.noali,
                true,
            );
            write_pipeline_stats(
                &mut out,
                hmm.m,
                n_targets,
                total_residues,
                th.nreported as u64,
                z,
                final_domz.unwrap_or(z),
                args.z_value.is_some(),
                args.domz_value.is_some(),
                &stats,
                if args.max { 1.0 } else { args.f1 },
                if args.max { 1.0 } else { args.f2 },
                if args.max { 1.0 } else { args.f3 },
            );
            writeln!(out).unwrap();
            let new_included_names: Vec<String> = th
                .hits
                .iter()
                .filter(|hit| hit.flags & tongs::tophits::P7_IS_INCLUDED != 0)
                .map(|hit| hit.name.clone())
                .collect();
            let msa_nseq = th
                .hits
                .iter()
                .filter(|hit| hit.flags & tongs::tophits::P7_IS_INCLUDED != 0)
                .map(|hit| hit.dcl.iter().filter(|dom| dom.is_included).count())
                .sum::<usize>()
                + 1;
            final_hits = Some(th);
            final_model_len = Some(hmm.m);
            prev_hmm = Some(hmm.clone());

            // Build the round MSA ONCE (matching C jackhmmer.c:683's single
            // `p7_tophits_Alignment` call per round). The same object is then
            // reused by `--chkali` below, by the next iteration's HMM build
            // (taken via `final_msa.take()` at the top of the loop), and by
            // `-A` after the loop ends. The previous round's MSA was either
            // consumed by this round's rebuild or dropped at the top of this
            // iteration, so the stash holds at most one MSA at a time.
            let msa = tongs::tophits::included_alignment(
                final_hits.as_ref().unwrap(),
                &abc,
                hmm.m,
                Some((&query_sq, &query_tr)),
                true, // p7_ALL_CONSENSUS_COLS (jackhmmer.c:683)
                &format!("{}-i{}", query_sq.name, iteration),
            )
            .map(|mut msa| {
                if !query_sq.desc.is_empty() {
                    msa.desc = Some(query_sq.desc.clone());
                }
                msa.author = Some("jackhmmer (HMMER 3.4)".to_string());
                msa
            });

            if let (Some(prefix), Some(ref msa)) = (&args.chkali, &msa) {
                write_msa_checkpoint(prefix, iteration, msa);
            }

            final_msa = msa;

            // Check convergence
            let n_new = new_included_names
                .iter()
                .filter(|name| !prev_included_names.iter().any(|prev| prev == *name))
                .count();
            writeln!(out, "@@ New targets included:   {}", n_new).unwrap();
            writeln!(
                out,
                "@@ New alignment includes: {} subseqs (was {}), including original query",
                msa_nseq, prev_msa_nseq
            )
            .unwrap();

            // Convergence test, faithful to jackhmmer.c serial_master() line 702:
            //   `if (nnew_targets == 0 && msa->nseq <= prv_msa_nseq)`.
            // C compares the new MSA's subsequence count against the previous
            // round's; it has no `iteration > 1` guard (round 1 with no
            // included hits converges immediately, msa->nseq==1<=prv==1).
            if n_new == 0 && msa_nseq <= prev_msa_nseq {
                writeln!(out, "@@").unwrap();
                writeln!(out, "@@ CONVERGED (in {} rounds). ", iteration).unwrap();
                writeln!(out, "@@").unwrap();
                writeln!(out).unwrap();
                break;
            }

            if iteration < args.max_iterations {
                writeln!(out, "@@ Continuing to next round.").unwrap();
                writeln!(out).unwrap();
            }

            prev_included_names = new_included_names;
            prev_msa_nseq = msa_nseq;
        }

        if let Some(ref mut f) = tblout_file {
            if let (Some(ref th), Some(z)) = (&final_hits, final_z) {
                let show_header = !wrote_tblout_header;
                crate::subcmd::hmmsearch::write_tblout(
                    f,
                    &query_sq.name,
                    Some(&query_sq.acc),
                    th,
                    z,
                    show_header,
                );
                wrote_tblout_header = true;
            }
        }
        if let Some(ref mut f) = domtblout_file {
            if let (Some(ref th), Some(z), Some(domz), Some(qlen)) =
                (&final_hits, final_z, final_domz, final_model_len)
            {
                let show_header = !wrote_domtblout_header;
                crate::subcmd::hmmsearch::write_domtblout(
                    f,
                    &query_sq.name,
                    Some(&query_sq.acc),
                    qlen,
                    th,
                    z,
                    domz,
                    show_header,
                );
                wrote_domtblout_header = true;
            }
        }
        if let Some(ref mut f) = ali_file {
            // Reuse the MSA already built at the end of the final iteration
            // (matching C jackhmmer.c:723's reuse of the single per-round
            // `msa`). When no rounds ran or no hits were found, `final_msa`
            // is None and no `-A` block is emitted.
            if let Some(ref msa) = final_msa {
                // jackhmmer.c:723-724: Stockholm if textw > 0, else Pfam.
                write_stockholm_msa_for_textw(f, msa, textw);
                writeln!(
                    out,
                    "# Alignment of {} hits satisfying inclusion thresholds saved to: {}",
                    msa.nseq,
                    args.ali_outfile.as_ref().unwrap().display()
                )
                .unwrap();
            }
            f.flush().unwrap();
        }

        writeln!(out, "//").unwrap();
    }

    if let Some(ref mut f) = tblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "jackhmmer",
            "SEARCH",
            &args.seqfile,
            &args.seqdb,
            &table_cmdline,
        );
        f.flush().unwrap();
    }
    if let Some(ref mut f) = domtblout_file {
        crate::subcmd::hmmsearch::write_table_footer(
            f,
            "jackhmmer",
            "SEARCH",
            &args.seqfile,
            &args.seqdb,
            &table_cmdline,
        );
        f.flush().unwrap();
    }

    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
#[path = "../../../tests/unit/subcmd/jackhmmer_test.rs"]
mod tests;
