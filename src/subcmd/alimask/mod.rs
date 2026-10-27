//! alimask — add/modify mask annotation on a Stockholm alignment.

use std::io::{BufReader, Write};
use std::path::PathBuf;

use clap::{ArgAction, Parser};

use tongs::alphabet::{Alphabet, AlphabetType};
use tongs::builder::{self, RelativeWeighting};
use tongs::msa;
use tongs::output::fmt_fixed3;

mod report;
use report::*;
mod ranges;
use ranges::*;
mod input;
use input::*;

#[derive(Parser)]
#[command(
    name = "alimask",
    about = "Add mask annotation to a Stockholm alignment"
)]
struct Args {
    /// Input alignment file (Stockholm)
    msafile: PathBuf,

    /// Output alignment file
    outfile: Option<PathBuf>,

    /// Direct summary output to file, not stdout
    #[arg(short = 'o')]
    summary_out: Option<PathBuf>,

    /// Assert input alignment format
    #[arg(long = "informat")]
    informat: Option<String>,

    /// Output alignment format
    #[arg(long = "outformat", default_value = "Stockholm")]
    outformat: String,

    /// Use protein alphabet
    #[arg(long, action = ArgAction::SetTrue)]
    amino: bool,

    /// Use DNA alphabet
    #[arg(long, action = ArgAction::SetTrue)]
    dna: bool,

    /// Use RNA alphabet
    #[arg(long, action = ArgAction::SetTrue)]
    rna: bool,

    /// Assign columns with >= symfrac residues as consensus
    #[arg(long = "fast", conflicts_with = "hand")]
    fast: bool,

    /// Use RF annotation as consensus columns
    #[arg(long = "hand", conflicts_with = "fast")]
    hand: bool,

    /// Consensus residue fraction for --fast construction
    #[arg(long = "symfrac", default_value = "0.5", value_parser = parse_fraction)]
    symfrac: f64,

    /// Sequence is called a fragment if L <= x*alignment_length
    #[arg(long = "fragthresh", default_value = "0.5", value_parser = parse_fraction)]
    fragthresh: f64,

    /// Random number seed
    #[arg(long = "seed", default_value = "42")]
    seed: u32,

    /// Henikoff position-based weights for model-coordinate construction
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wgsc", "wblosum", "wnone", "wgiven"])]
    wpb: bool,

    /// Gerstein/Sonnhammer/Chothia tree weights for model-coordinate construction
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wblosum", "wnone", "wgiven"])]
    wgsc: bool,

    /// Henikoff simple filter weights for model-coordinate construction
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wnone", "wgiven"])]
    wblosum: bool,

    /// No relative sequence weighting for model-coordinate construction
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wblosum", "wgiven"])]
    wnone: bool,

    /// Use weights as given in the MSA file for model-coordinate construction
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wblosum", "wnone"])]
    wgiven: bool,

    /// For --wblosum: set identity cutoff
    #[arg(long = "wid", default_value = "0.62", value_parser = parse_fraction)]
    wid: f64,

    /// Mask alignment-coordinate range(s), for example 3..8,12..15
    #[arg(long = "alirange")]
    alirange: Option<String>,

    /// Mask model-coordinate range(s)
    #[arg(long = "modelrange")]
    modelrange: Option<String>,

    /// Map model-coordinate range(s) to alignment coordinates
    #[arg(long = "model2ali")]
    model2ali: Option<String>,

    /// Map alignment-coordinate range(s) to model coordinates
    #[arg(long = "ali2model")]
    ali2model: Option<String>,

    /// Add to existing mask annotation instead of replacing it
    #[arg(long = "appendmask", action = ArgAction::SetTrue)]
    appendmask: bool,

    /// Model mask to apply (string of 'm' and '.' characters)
    #[arg(long = "modelmask")]
    modelmask: Option<String>,
}

fn parse_fraction(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid fraction: {e}"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("value must be between 0 and 1".to_string())
    }
}

/// Entry point for the `alimask` subcommand: append a modelmask line to a Stockholm MSA.
///
/// Parses the command line, reads the input alignment, and writes a Stockholm output
/// optionally augmented with a `#=GC MM` model mask line. Corresponds to `main()` in
/// hmmer/src/alimask.c, with the Rust port focused on the simple mask-annotation flow.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let wid_on_cmdline = args
        .iter()
        .any(|arg| arg == "--wid" || arg.starts_with("--wid="));
    let seed_on_cmdline = args
        .iter()
        .any(|arg| arg == "--seed" || arg.starts_with("--seed="));
    let args = Args::parse_from(&args);
    // C uses esl_opt_IsUsed("--seed"/"--wid"), which is FALSE when the supplied
    // value equals the option default (42 / 0.62 respectively). So `--seed 42`
    // and `--wid 0.62` print no header line, matching the no-flag case.
    let seed_was_requested = seed_on_cmdline && args.seed != 42;
    let wid_was_requested = wid_on_cmdline && args.wid != 0.62;
    if [args.amino, args.dna, args.rna]
        .into_iter()
        .filter(|v| *v)
        .count()
        > 1
    {
        eprintln!("Error: options --amino, --dna, and --rna are mutually exclusive");
        std::process::exit(1);
    }
    let mode_count = [
        args.alirange.is_some(),
        args.modelrange.is_some(),
        args.model2ali.is_some(),
        args.ali2model.is_some(),
    ]
    .into_iter()
    .filter(|enabled| *enabled)
    .count();
    if mode_count == 0 && args.modelmask.is_none() {
        eprintln!(
            "Must specify one of --modelrange, --alirange, --model2ali, --ali2model, or --modelmask"
        );
        std::process::exit(1);
    }
    if mode_count > 1 {
        eprintln!(
            "Options --modelrange, --alirange, --model2ali, and --ali2model are mutually exclusive"
        );
        std::process::exit(1);
    }
    if wid_was_requested && !args.wblosum {
        eprintln!("Error: --wid only works in combination with --wblosum");
        std::process::exit(1);
    }
    let mapping_mode = args.model2ali.is_some() || args.ali2model.is_some();
    let outfile = args.outfile.as_ref();
    if !mapping_mode && outfile.is_none() {
        eprintln!("alimask --alirange requires <postmsafile>");
        std::process::exit(1);
    }
    if args.msafile == std::path::Path::new("-") && args.informat.is_none() {
        println!("Must specify --informat to read <alifile> from stdin ('-')");
        std::process::exit(1);
    }
    if let Some(ref informat) = args.informat {
        if !is_stockholm_format(informat) {
            eprintln!("{informat} is not a recognized input sequence file format");
            std::process::exit(1);
        }
    }
    if !is_stockholm_format(&args.outformat) {
        eprintln!(
            "{} is not a recognized output MSA file format",
            args.outformat
        );
        std::process::exit(1);
    }

    let (msas, full_msas) = read_stockholm_maybe_stdin(&args.msafile).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });
    let needs_model_map = args.modelrange.is_some() || mapping_mode;
    let inferred_abc = if needs_model_map && !args.hand {
        Some(if args.dna {
            Alphabet::dna()
        } else if args.rna {
            Alphabet::rna()
        } else if args.amino {
            Alphabet::amino()
        } else {
            let Some(first_msa) = msas.first() else {
                eprintln!("Error: no alignments found in {}", args.msafile.display());
                std::process::exit(1);
            };
            Alphabet::new(guess_msa_alphabet(&first_msa.msa).unwrap_or_else(|e| {
                eprintln!("{e}; please specify --amino, --dna, or --rna");
                std::process::exit(1);
            }))
        })
    } else {
        None
    };
    let weighting_strategy = if args.wnone {
        RelativeWeighting::None
    } else if args.wgsc {
        RelativeWeighting::Gsc
    } else if args.wblosum {
        RelativeWeighting::Blosum {
            identity_cutoff: args.wid,
        }
    } else if args.wgiven {
        RelativeWeighting::Given
    } else {
        RelativeWeighting::PositionBased
    };
    let coordinate_ranges = args
        .alirange
        .as_deref()
        .or(args.modelrange.as_deref())
        .or(args.model2ali.as_deref())
        .or(args.ali2model.as_deref())
        .map(parse_ranges)
        .transpose();
    let coordinate_ranges = coordinate_ranges.unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    if mapping_mode {
        write_summary_report(&args, seed_was_requested, wid_was_requested).unwrap_or_else(|e| {
            eprintln!("Error writing summary output: {}", e);
            std::process::exit(1);
        });
        for stockholm in &msas {
            let alignment = &stockholm.msa;
            let model_to_ali = make_model_to_alignment_map(
                alignment,
                inferred_abc.as_ref(),
                args.hand,
                args.symfrac,
                args.fragthresh,
                weighting_strategy,
            )
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            });
            let ranges = coordinate_ranges.as_deref().unwrap_or(&[]);
            if args.model2ali.is_some() {
                write_model_to_alignment_map(ranges, &model_to_ali).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                });
            } else {
                write_alignment_to_model_map(ranges, &model_to_ali, alignment.alen).unwrap_or_else(
                    |e| {
                        eprintln!("{e}");
                        std::process::exit(1);
                    },
                );
            }
        }
        return std::process::ExitCode::SUCCESS;
    }

    let outfile = outfile.expect("checked above");
    let mut out = std::fs::File::create(outfile).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });
    write_summary_report(&args, seed_was_requested, wid_was_requested).unwrap_or_else(|e| {
        eprintln!("Error writing summary output: {}", e);
        std::process::exit(1);
    });

    for (stockholm, full) in msas.iter().zip(full_msas.iter()) {
        let alignment = &stockholm.msa;
        if let Some(ref mask) = args.modelmask {
            if mask.len() != alignment.alen {
                eprintln!(
                    "Model mask length {} does not match alignment length {}",
                    mask.len(),
                    alignment.alen
                );
                std::process::exit(1);
            }
            if !mask.bytes().all(|b| b == b'm' || b == b'.') {
                eprintln!("Model mask may only contain 'm' and '.' characters");
                std::process::exit(1);
            }
        }

        let ali_ranges = if args.modelrange.is_some() {
            let model_to_ali = make_model_to_alignment_map(
                alignment,
                inferred_abc.as_ref(),
                args.hand,
                args.symfrac,
                args.fragthresh,
                weighting_strategy,
            )
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1);
            });
            coordinate_ranges
                .as_ref()
                .map(|ranges| model_ranges_to_alignment_ranges(ranges, &model_to_ali))
                .transpose()
                .unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1);
                })
        } else {
            coordinate_ranges.clone()
        };

        let mut range_mask = ali_ranges.as_ref().map(|ranges| {
            // C (alimask.c:368-374): keep_mm = IsUsed("--appendmask"). If the
            // input MSA has no #=GC MM line, msa->mm is freshly allocated and
            // keep_mm forced FALSE. When !keep_mm the mask is reset to all '.'.
            // So with --appendmask AND an existing MM line, start from the
            // existing mask and OR the new range columns on top; otherwise
            // start from a fresh all-'.' mask.
            let mut mask = match (args.appendmask, alignment.mm.as_ref()) {
                (true, Some(existing)) if existing.len() == alignment.alen => existing.clone(),
                _ => vec![b'.'; alignment.alen],
            };
            for &(start, end) in ranges {
                if start == 0 {
                    eprintln!("Mask ranges can not start before position 1; start 0 is invalid");
                    std::process::exit(1);
                }
                if end > alignment.alen {
                    eprintln!(
                        "Maximum mask range {} exceeds alignment length {}",
                        end, alignment.alen
                    );
                    std::process::exit(1);
                }
                for pos in start..=end {
                    mask[pos - 1] = b'm';
                }
            }
            mask
        });

        if args.appendmask {
            if let (Some(range_mask), Some(modelmask)) = (&mut range_mask, &args.modelmask) {
                for (dst, src) in range_mask.iter_mut().zip(modelmask.bytes()) {
                    if src == b'm' {
                        *dst = b'm';
                    }
                }
            }
        } else if range_mask.is_none() {
            range_mask = args.modelmask.as_ref().map(|mask| mask.as_bytes().to_vec());
        }

        // C re-serializes the whole MSA via esl_msafile_Write(...
        // eslMSAFILE_STOCKHOLM) -> stockholm_write(fp, msa, cpl=200). C opens
        // the input through esl_msafile_Open(&abc, ...) with an explicit or
        // autodetected alphabet (digital mode), so stockholm_write textizes the
        // sequences: gap chars normalize to '-' and residues become canonical.
        // For model-coordinate modes, C first calls esl_msa_MarkFragments_old(),
        // which rewrites leading/trailing non-residues in short fragments to
        // missing data '~'. We apply the same mutation to the full-fidelity
        // text model before writing.
        let write_abc = inferred_abc.clone().unwrap_or_else(|| {
            if args.dna {
                Alphabet::dna()
            } else if args.rna {
                Alphabet::rna()
            } else if args.amino {
                Alphabet::amino()
            } else {
                Alphabet::new(guess_msa_alphabet(alignment).unwrap_or_else(|e| {
                    eprintln!("{e}; please specify --amino, --dna, or --rna");
                    std::process::exit(1);
                }))
            }
        });
        let mut full = full.clone();
        if needs_model_map {
            mark_fragments_old_full(&mut full, &write_abc, args.fragthresh);
        }
        if let Some(mask) = range_mask.as_deref() {
            full.mm = Some(mask.to_vec());
        }
        // C (alimask.c:380-396): in --modelrange mode the MSA is relative-weighted
        // before the map is built. --wpb/--wgsc/--wblosum raise eslMSA_HASWGTS, so
        // the output MSA carries #=GS ... WT lines; --wnone/--wgiven do not.
        if args.modelrange.is_some() {
            if let Some(weights) =
                compute_output_weights(alignment, &write_abc, args.hand, weighting_strategy)
            {
                full.has_wgts = true;
                full.wgt = weights;
            }
        }
        let cpl = if is_pfam_format(&args.outformat) {
            full.alen
        } else {
            200
        };
        msa::write_stockholm_full(&mut out, &full, cpl, Some(&write_abc)).unwrap();
    }

    std::process::ExitCode::SUCCESS
}
