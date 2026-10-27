#![allow(
    clippy::approx_constant,
    clippy::manual_clamp,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::too_many_arguments
)]

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use tongs::alphabet::Alphabet;
use tongs::bg::Bg;
use tongs::calibrate::p7_lambda;
use tongs::dp::generic_fwdback::{g_forward, g_hybrid};
use tongs::dp::generic_msv::g_msv;
use tongs::dp::generic_viterbi::g_viterbi;
use tongs::dp::gmx::Gmx;
use tongs::hmm::Hmm;
use tongs::hmmfile;
use tongs::output::{fmt_fixed4, fmt_fixed6, fmt_g};
use tongs::profile::{
    self, Profile, P7P_C, P7P_J, P7P_LOOP, P7P_MOVE, P7P_N, P7_GLOCAL, P7_LOCAL, P7_UNIGLOCAL,
    P7_UNILOCAL,
};
use tongs::simd::oprofile::OProfile;
use tongs::stats::{exponential, gumbel};
use tongs::subcmd::cli::{command_line_has_option, parse_open_unit_f64, parse_positive_usize};
use tongs::trace::{g_trace, State};
use tongs::util::cmath::{c_log_f64, c_logf_to_f32, c_sqrt_f64, ESL_CONST_LOG2};
use tongs::util::random::Mt19937;

mod calibrate;
use calibrate::*;
mod report;
use report::*;
mod fit;
use fit::*;

#[derive(Parser)]
#[command(name = "hmmsim", about = "Score random sequences against an HMM")]
struct Args {
    hmmfile: PathBuf,

    #[arg(short = 'N', default_value = "1000")]
    n: usize,

    #[arg(short = 'L', default_value = "100")]
    l: usize,

    // C reads --seed as an unranged int into a uint32_t, so a negative seed wraps (hmmsim.c:79).
    // allow_hyphen_values lets `--seed -5` parse as it does in C.
    #[arg(long = "seed", default_value = "0", allow_hyphen_values = true)]
    seed: i32,

    #[arg(short = 'o')]
    outfile: Option<PathBuf>,

    #[arg(short = 'a')]
    alignment_stats: bool,

    #[arg(short = 'v')]
    verbose: bool,

    #[arg(
        long = "stall",
        help = "arrest after start: for debugging MPI under gdb"
    )]
    stall: bool,

    #[arg(long = "pfile")]
    pfile: Option<PathBuf>,

    #[arg(long = "efile")]
    efile: Option<PathBuf>,

    #[arg(long = "ffile")]
    ffile: Option<PathBuf>,

    #[arg(long = "xfile")]
    xfile: Option<PathBuf>,

    #[arg(long = "afile")]
    afile: Option<PathBuf>,

    #[arg(long = "pthresh", default_value = "0.02")]
    pthresh: f64,

    // C gives --tmin, --tmax and --tpoints no range, so zero and negative values are accepted.
    // allow_hyphen_values lets `--tmin -1` parse as it does in C.
    #[arg(long = "tmin", default_value = "0.02", allow_hyphen_values = true)]
    tmin: f64,

    #[arg(long = "tmax", default_value = "0.02", allow_hyphen_values = true)]
    tmax: f64,

    #[arg(long = "tpoints", default_value = "1", allow_hyphen_values = true)]
    tpoints: usize,

    #[arg(long = "tlinear")]
    tlinear: bool,

    #[arg(long = "vit", alias = "viterbi")]
    viterbi: bool,

    #[arg(long = "fwd", alias = "forward")]
    forward: bool,

    #[arg(long = "hyb")]
    hybrid: bool,

    #[arg(long = "msv")]
    msv: bool,

    /// Use optimized scoring kernels when available (accepted as compatibility no-op)
    #[arg(long = "fast")]
    fast: bool,

    #[arg(long = "bgflat")]
    bgflat: bool,

    #[arg(long = "bgcomp")]
    bgcomp: bool,

    #[arg(long = "x-no-lengthmodel")]
    no_lengthmodel: bool,

    #[arg(long = "nu")]
    nu: Option<f32>,

    #[arg(long = "EmL", default_value = "200", value_parser = parse_positive_usize)]
    em_l: usize,

    #[arg(long = "EmN", default_value = "200", value_parser = parse_positive_usize)]
    em_n: usize,

    #[arg(long = "EvL", default_value = "200", value_parser = parse_positive_usize)]
    ev_l: usize,

    #[arg(long = "EvN", default_value = "200", value_parser = parse_positive_usize)]
    ev_n: usize,

    #[arg(long = "EfL", default_value = "100", value_parser = parse_positive_usize)]
    ef_l: usize,

    #[arg(long = "EfN", default_value = "200", value_parser = parse_positive_usize)]
    ef_n: usize,

    #[arg(long = "Eft", default_value = "0.04", value_parser = parse_open_unit_f64)]
    eft: f64,

    #[arg(long = "fs")]
    fs: bool,

    #[arg(long = "sw")]
    sw: bool,

    #[arg(long = "ls")]
    ls: bool,

    #[arg(long = "s")]
    s: bool,
}

pub fn run(args: Vec<String>) -> ExitCode {
    let pthresh_given = command_line_has_option(&args.join(" "), "--pthresh");
    let args = Args::parse_from(&args);
    let _fast_compat = args.fast;
    let _verbose_compat = args.verbose;
    let _stall_compat = args.stall;

    if args.n == 0 {
        eprintln!("Invalid number of samples: -N must be > 0");
        return ExitCode::FAILURE;
    }
    if args.l == 0 {
        eprintln!("Invalid sequence length: -L must be > 0");
        return ExitCode::FAILURE;
    }
    let score_modes = [args.viterbi, args.forward, args.hybrid, args.msv]
        .iter()
        .filter(|&&enabled| enabled)
        .count();
    if score_modes > 1 {
        eprintln!("hmmsim scoring options --vit, --fwd, --hyb, and --msv are mutually exclusive");
        return ExitCode::FAILURE;
    }
    let alignment_modes = [args.fs, args.sw, args.ls, args.s]
        .iter()
        .filter(|&&enabled| enabled)
        .count();
    if alignment_modes > 1 {
        eprintln!("hmmsim alignment options --fs, --sw, --ls, and --s are mutually exclusive");
        return ExitCode::FAILURE;
    }
    if args.alignment_stats && score_modes > 0 && !args.viterbi {
        eprintln!("hmmsim -a requires Viterbi scoring");
        return ExitCode::FAILURE;
    }
    if args.afile.is_some() && !args.alignment_stats {
        eprintln!("hmmsim --afile requires -a");
        return ExitCode::FAILURE;
    }
    if pthresh_given && args.ffile.is_none() {
        eprintln!("hmmsim --pthresh requires --ffile");
        return ExitCode::FAILURE;
    }
    if args.nu.is_some() && !args.msv {
        eprintln!("hmmsim --nu is only supported with --msv");
        return ExitCode::FAILURE;
    }
    if args.nu.is_some() && args.fast {
        eprintln!("hmmsim --nu cannot be used with --fast");
        return ExitCode::FAILURE;
    }
    let nu = args.nu.unwrap_or(2.0);
    if nu <= 1.0 || !nu.is_finite() {
        eprintln!("hmmsim --nu must be finite and > 1.0");
        return ExitCode::FAILURE;
    }
    if !(0.0..=1.0).contains(&args.pthresh) || !args.pthresh.is_finite() {
        eprintln!("hmmsim --pthresh must be finite and between 0.0 and 1.0");
        return ExitCode::FAILURE;
    }

    tongs::logsum::p7_flogsuminit();

    let hmms = hmmfile::read_hmm_file_auto(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });
    if hmms.is_empty() {
        eprintln!("Error: no HMMs found in {}", args.hmmfile.display());
        return ExitCode::FAILURE;
    }

    let hmm = &hmms[0];
    let abc = Alphabet::new(hmm.abc_type);
    let mut bg = if args.bgflat {
        Bg::new_uniform(&abc)
    } else {
        Bg::new(&abc)
    };
    if args.bgcomp {
        bg.f = profile::hmm_average_match_composition(hmm);
        bg.fhmm_e0 = bg.f.clone();
        bg.fhmm_e1 = bg.f.clone();
    }
    // Match C esl_randomness_Create(uint32_t): a negative int seed is
    // reinterpreted as its wrapping u32; only an exact 0 picks an arbitrary
    // clock seed (handled inside Mt19937::new via resolve_seed).
    let mut rng = Mt19937::new(args.seed as u32);
    let score_mode = score_mode_from_args(&args);

    let mut gm = Profile::new(hmm.m, &abc);
    let mode = if args.sw {
        P7_UNILOCAL
    } else if args.ls {
        P7_GLOCAL
    } else if args.s {
        P7_UNIGLOCAL
    } else {
        P7_LOCAL
    };
    profile::profile_config(hmm, &bg, &mut gm, args.l as i32, mode);
    let model_params = calibrated_model_params(score_mode, hmm, &abc, &bg, &args, &mut rng);
    if args.no_lengthmodel {
        elide_length_model(&mut gm, &mut bg);
    }
    bg.set_length(args.l);

    let mut out: Box<dyn Write> = match args.outfile {
        Some(ref path) => Box::new(std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("Error creating output {}: {}", path.display(), e);
            std::process::exit(1);
        })),
        None => Box::new(std::io::stdout()),
    };

    let mut scores = Vec::with_capacity(args.n);
    let mut alilens = Vec::with_capacity(args.n);

    for _ in 0..args.n {
        let mut dsq = vec![tongs::alphabet::DSQ_SENTINEL];
        for _ in 0..args.l {
            dsq.push(rng.sample_residue(&bg.f));
        }
        dsq.push(tongs::alphabet::DSQ_SENTINEL);

        let null_sc = bg.null_one(args.l);
        let mut gx = Gmx::new(gm.m, args.l);

        let bits = if args.forward {
            let sc = g_forward(&dsq, args.l, &gm, &mut gx);
            ((sc - null_sc) as f64 / ESL_CONST_LOG2) as f32
        } else if args.hybrid {
            let sc = g_hybrid(&dsq, args.l, &gm, &mut gx);
            ((sc - null_sc) as f64 / ESL_CONST_LOG2) as f32
        } else if args.msv {
            let sc = g_msv(&dsq, args.l, &gm, &mut gx, nu);
            ((sc - null_sc) as f64 / ESL_CONST_LOG2) as f32
        } else {
            let sc = g_viterbi(&dsq, args.l, &gm, &mut gx);
            ((sc - null_sc) as f64 / ESL_CONST_LOG2) as f32
        };

        let alilen = if score_mode == ScoreMode::Viterbi {
            let tr = g_trace(&dsq, args.l, &gm, &gx);
            tr.st
                .iter()
                .filter(|&&state| matches!(state, State::M | State::D | State::I))
                .count()
        } else {
            args.l
        };
        scores.push(bits as f64);
        alilens.push(alilen);
        if args.verbose {
            writeln!(out, "{:.3}", bits as f64).unwrap();
        }
    }

    let histogram = ScoreHistogram::from_scores(&scores, -50.0, 50.0, 0.2);
    match score_mode {
        ScoreMode::Forward => {
            write_forward_tail_summary(&args, &mut out, &hmm.name, model_params, &histogram)
                .unwrap();
        }
        ScoreMode::Viterbi | ScoreMode::Hybrid | ScoreMode::Msv => {
            let alilens_opt = if args.alignment_stats {
                Some(alilens.as_slice())
            } else {
                None
            };
            write_gumbel_summary(&mut out, &hmm.name, model_params, &histogram, alilens_opt)
                .unwrap();
        }
    }

    if let Err(e) = write_optional_outputs(
        &args,
        &hmm.name,
        model_params,
        score_mode,
        &scores,
        &alilens,
    ) {
        eprintln!("Error writing hmmsim output: {}", e);
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScoreMode {
    Viterbi,
    Forward,
    Hybrid,
    Msv,
}

fn score_mode_from_args(args: &Args) -> ScoreMode {
    if args.forward {
        ScoreMode::Forward
    } else if args.hybrid {
        ScoreMode::Hybrid
    } else if args.msv {
        ScoreMode::Msv
    } else {
        ScoreMode::Viterbi
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/subcmd/hmmsim_test.rs"]
mod tests;
