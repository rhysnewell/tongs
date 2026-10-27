//! hmmbuild — build profile HMM(s) from multiple sequence alignment(s).

use std::io::{BufReader, Read, Write};
use std::path::PathBuf;

use clap::{ArgAction, Parser};

use tongs::alphabet::{Alphabet, AlphabetType};
use tongs::bg::Bg;
use tongs::builder;
use tongs::builder::Builder;
use tongs::calibrate::CalibrationConfig;
use tongs::hmmfile;
use tongs::msa;
use tongs::output::{fmt_fixed3, fmt_width6_3, fmt_width8_2};
use tongs::prior::PriorStrategy;
use tongs::seqmodel;
use tongs::util::cmath::{c_log_f64, ESL_CONST_LOG2R};

mod options;
use options::*;
mod header;
use header::*;
mod input;
use input::*;
mod build;
use build::*;

struct BuildAlignment {
    msa: msa::Msa,
    cutoffs: msa::StockholmCutoffs,
    force_hand_arch: bool,
    original_stockholm: Option<msa::FullStockholm>,
}

impl std::ops::Deref for BuildAlignment {
    type Target = msa::Msa;

    fn deref(&self) -> &Self::Target {
        &self.msa
    }
}

/// Entry point for `hmmbuild`: build profile HMM(s) from MSA(s) and write them
/// to a single output file.
///
/// Streams a Stockholm input, calls the builder pipeline per alignment, applies
/// optional name/alphabet overrides, and prints HMMER 3.4's per-MSA summary line
/// (`idx`, name, nodes, nseq, eff_nseq). Corresponds to `main()` /
/// `output_result()` in hmmer/src/hmmbuild.c (single-threaded path only).
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let requested = Requested::from_args(&args);
    let args = Args::parse_from(&args);
    let _stall_compat = args.stall;

    // Alphabet (--amino/--dna/--rna) and construction (--fast/--hand) mutual
    // exclusion, the --symfrac/--fragthresh 0<=x<=1 range, and "--symfrac
    // requires --fast" (clap: conflicts_with --hand) are all enforced at parse
    // time by clap, matching C's ALPHOPTS/CONOPTS toggles, range columns, and
    // --symfrac reqs --fast in hmmbuild.c.
    if requested.wid && !args.wblosum {
        eprintln!("Error: --wid only works in combination with --wblosum");
        std::process::exit(1);
    }
    if requested.eid && !args.eclust {
        eprintln!("Error: --eid only works in combination with --eclust");
        std::process::exit(1);
    }
    if matches!(args.w_length, Some(1..=3)) {
        eprintln!("Invalid window length value");
        std::process::exit(1);
    }
    if args.hmmfile == std::path::Path::new("-") {
        eprintln!("Error: hmmbuild cannot write <hmmfile_out> to stdout; use a file path");
        std::process::exit(1);
    }
    if !args.singlemx && (requested.mx || args.mxfile.is_some()) {
        eprintln!("Error: hmmbuild --mx and --mxfile currently require --singlemx");
        std::process::exit(1);
    }
    if args.singlemx {
        if let Some(path) = args.mxfile.as_ref() {
            std::fs::File::open(path).unwrap_or_else(|_| {
                // p7_builder_SetScoreSystem (p7_builder.c:307). C's hmmbuild
                // opens the alignment file before reaching this check, so it
                // reports a missing alignment first; this port validates the
                // score system up front. See TODO.md.
                eprintln!(
                    "\nError: Failed to set single query seq score system:\nFailed to find or open matrix file {}\n",
                    path.display()
                );
                std::process::exit(1);
            });
        } else if requested.mx && !seqmodel::is_known_builtin_score_matrix_name(&args.matrix) {
            // p7_builder_LoadScoreSystem (p7_builder.c:213).
            eprintln!(
                "\nError: Failed to set single query seq score system:\nno matrix named {} is available as a built-in\n",
                args.matrix
            );
            std::process::exit(1);
        }
    }
    if args.msafile == std::path::Path::new("-") && args.informat.is_none() {
        println!("Must specify --informat to read <alifile> from stdin ('-')");
        std::process::exit(1);
    }
    if let Some(ref informat) = args.informat {
        if !is_supported_msa_format(informat) {
            eprintln!("{informat} is not a recognized input alignment file format");
            std::process::exit(1);
        }
    }

    let alignments = read_build_alignments_maybe_stdin(&args.msafile, args.informat.as_deref())
        .unwrap_or_else(|e| {
            eprintln!("Error reading MSA file: {}", e);
            std::process::exit(1);
        });
    if args.name.is_some() && alignments.len() > 1 {
        eprintln!("Error: You can't use -n with an alignment database");
        std::process::exit(1);
    }
    let Some(first_alignment) = alignments.first() else {
        eprintln!("Error: no alignments found in {}", args.msafile.display());
        std::process::exit(1);
    };
    let abc = if args.dna {
        Alphabet::dna()
    } else if args.rna {
        Alphabet::rna()
    } else if args.amino {
        Alphabet::amino()
    } else {
        Alphabet::new(
            guess_msa_alphabet(&first_alignment.msa).unwrap_or_else(|e| {
                eprintln!("{e}; please specify --amino, --dna, or --rna");
                std::process::exit(1);
            }),
        )
    };
    for alignment in &alignments {
        if let Err(e) = alignment.validate_digitizable(&abc) {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
    let bg = Bg::new(&abc);
    let nucleotide_singlemx =
        args.singlemx && matches!(abc.abc_type, AlphabetType::Dna | AlphabetType::Rna);
    let singlemx_popen = if nucleotide_singlemx && !requested.popen {
        0.03125
    } else {
        args.popen
    };
    let singlemx_pextend = if nucleotide_singlemx && !requested.pextend {
        0.75
    } else {
        args.pextend
    };
    // The label hmmbuild echoes for --singlemx. C prints the --mxfile path
    // (hmmbuild.c, "# subst score matrix (file)") when one is given, otherwise
    // the --mx value, with DNA1 substituted for nucleotide input
    // (hmmbuild.c:592).
    let singlemx_matrix_label: String = match args.mxfile.as_ref() {
        Some(path) => path.display().to_string(),
        None if nucleotide_singlemx && !requested.mx => "DNA1".to_string(),
        None => args.matrix.clone(),
    };
    let score_matrix = if args.singlemx {
        if let Some(path) = args.mxfile.as_ref() {
            seqmodel::ScoreMatrix::from_file_for_alphabet(path, &abc).unwrap_or_else(|e| {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            })
        } else {
            let matrix_name = if nucleotide_singlemx && !requested.mx {
                "DNA1"
            } else {
                args.matrix.as_str()
            };
            seqmodel::ScoreMatrix::builtin_for_alphabet(matrix_name, &abc).unwrap_or_else(|e| {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            })
        }
    } else {
        seqmodel::ScoreMatrix::blosum62()
    };

    let mut summary_file = args.summary_out.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating summary output file: {}", e);
            std::process::exit(1);
        })
    });
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();
    let summary: &mut dyn Write = match summary_file {
        Some(ref mut file) => file,
        None => &mut stdout_lock,
    };

    // Build output file
    let mut out_file = std::fs::File::create(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error creating HMM file: {}", e);
        std::process::exit(1);
    });

    output_header(summary, &args, &requested, &singlemx_matrix_label);
    if abc.abc_type == AlphabetType::Amino {
        writeln!(
            summary,
            "# idx name                  nseq  alen  mlen eff_nseq re/pos description"
        )
        .unwrap();
        writeln!(
            summary,
            "#---- -------------------- ----- ----- ----- -------- ------ -----------"
        )
        .unwrap();
    } else {
        writeln!(
            summary,
            "# idx name                  nseq  alen  mlen     W eff_nseq re/pos description"
        )
        .unwrap();
        writeln!(
            summary,
            "#---- -------------------- ----- ----- ----- ----- -------- ------ -----------"
        )
        .unwrap();
    }

    let mut msa_out = args.msa_out.as_ref().map(|p| {
        std::fs::File::create(p).unwrap_or_else(|e| {
            eprintln!("Error creating processed alignment output file: {}", e);
            std::process::exit(1);
        })
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

    for (idx, alignment) in alignments.iter().enumerate() {
        let hand_arch = args.hand || alignment.force_hand_arch;
        if hand_arch && alignment.rf.is_none() {
            eprintln!("Model file does not contain an RF line, required for --hand.");
            std::process::exit(1);
        }
        if args.singlemx && alignment.nseq != 1 {
            eprintln!("Error: hmmbuild --singlemx requires alignments with exactly one sequence");
            std::process::exit(1);
        }
        let weighting_strategy = if args.wnone {
            builder::RelativeWeighting::None
        } else if args.wgsc {
            builder::RelativeWeighting::Gsc
        } else if args.wblosum {
            builder::RelativeWeighting::Blosum {
                identity_cutoff: args.wid,
            }
        } else if args.wgiven {
            builder::RelativeWeighting::Given
        } else {
            builder::RelativeWeighting::PositionBased
        };
        if let Some(ref mut out) = msa_out {
            let model_mask = builder::model_mask_from_msa(
                alignment,
                &abc,
                args.symfrac,
                args.fragthresh,
                hand_arch,
                weighting_strategy,
            );
            write_processed_msa(
                out,
                alignment,
                &model_mask,
                &abc,
                args.fragthresh,
                hand_arch,
                weighting_strategy,
            );
        }
        let mut hmm = if args.singlemx {
            build_single_sequence_hmm(
                alignment,
                &abc,
                &bg,
                &score_matrix,
                singlemx_popen,
                singlemx_pextend,
                args.seed,
                calibration_config,
                args.w_length,
                args.w_beta,
            )
        } else {
            builder::build_hmm_from_msa_with_prior_and_max_insert(
                alignment,
                &abc,
                &bg,
                args.symfrac,
                args.fragthresh,
                hand_arch,
                weighting_strategy,
                if args.enone {
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
                },
                if args.pnone {
                    PriorStrategy::None
                } else if args.plaplace {
                    PriorStrategy::Laplace
                } else {
                    PriorStrategy::Default
                },
                calibration_config,
                args.seed,
                args.max_insert_len,
            )
            // hmmbuild.c:1045: p7_Fail("build failed: %s", bld->errbuf).
            .unwrap_or_else(|e| {
                eprintln!("\nError: build failed: {e}");
                std::process::exit(1);
            })
        };
        if args.singlemx {
            // Nothing to do: p7_SingleBuilder() already calibrated with this
            // builder's seed/config and set max_length from w_len/w_beta.
        } else if requested.popen || requested.pextend {
            apply_window_length_options(&mut hmm, abc.abc_type, args.w_length, args.w_beta);
            apply_fixed_gap_params(
                &mut hmm,
                requested.popen.then_some(args.popen),
                requested.pextend.then_some(args.pextend),
            );
            if let Err(e) = tongs::calibrate::calibrate_with_config(
                &mut hmm,
                &abc,
                &bg,
                args.seed,
                calibration_config,
            ) {
                eprintln!("\nError: build failed: {e}");
                std::process::exit(1);
            }
        } else {
            apply_window_length_options(&mut hmm, abc.abc_type, args.w_length, args.w_beta);
        }
        builder::copy_stockholm_cutoffs_to_hmm(alignment.cutoffs, &mut hmm);

        if let Some(ref name) = args.name {
            hmm.name = name.clone();
        }

        let rel_entropy = mean_match_relative_entropy(&hmm, &bg);
        let description = hmm.desc.as_deref().unwrap_or("");
        // hmmbuild.c:1291-1311 prints msa->name here, not hmm->name. The two
        // differ under --singlemx, where p7_SingleBuilder names the model after
        // the sequence (seqmodel.c:88) while the table still shows the
        // alignment's name. -n renames the MSA before the build
        // (hmmbuild.c:1361-1363), so it shows here too.
        let table_name: &str = args.name.as_deref().unwrap_or(&alignment.msa.name);
        if abc.abc_type == AlphabetType::Amino {
            writeln!(
                summary,
                "{:<5} {:<20} {:>5} {:>5} {:>5} {} {} {}",
                idx + 1,
                table_name,
                alignment.nseq,
                alignment.alen,
                hmm.m,
                fmt_width8_2(hmm.eff_nseq as f64),
                fmt_width6_3(rel_entropy as f64),
                description,
            )
            .unwrap();
        } else {
            writeln!(
                summary,
                "{:<5} {:<20} {:>5} {:>5} {:>5} {:>5} {} {} {}",
                idx + 1,
                table_name,
                alignment.nseq,
                alignment.alen,
                hmm.m,
                hmm.max_length,
                fmt_width8_2(hmm.eff_nseq as f64),
                fmt_width6_3(rel_entropy as f64),
                description,
            )
            .unwrap();
        }

        hmmfile::write_hmm(&mut out_file, &hmm).unwrap_or_else(|e| {
            eprintln!("Error writing HMM: {}", e);
            std::process::exit(1);
        });
    }
    writeln!(summary).unwrap();

    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
#[path = "../../../tests/unit/subcmd/hmmbuild_test.rs"]
mod tests;
