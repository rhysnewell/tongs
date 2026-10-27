#![allow(clippy::if_same_then_else, clippy::needless_range_loop)]

use std::io::Write;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use clap::{ArgAction, FromArgMatches, Parser};

use tongs::alphabet::{Alphabet, Dsq, DSQ_SENTINEL};
use tongs::bg::Bg;
use tongs::hmm::{Hmm, DD, DM, II, IM, MD, MM};
use tongs::profile::{
    profile_config, Profile, P7P_BM, P7P_C, P7P_E, P7P_J, P7P_LOOP, P7P_MOVE, P7P_N, P7_GLOCAL,
    P7_LOCAL, P7_UNIGLOCAL, P7_UNILOCAL,
};
use tongs::sequence::Sequence;
use tongs::subcmd::inputs::read_hmms;
use tongs::subcmd::outputs::create_file_or_exit;
use tongs::trace::{State, Trace};
use tongs::util::cmath::c_exp_to_f32;
use tongs::util::random::MersenneTwister;

mod emit;
use emit::*;
mod output;
use output::*;

/// Easel's interleaved Stockholm writer wraps residues at this column count
/// (`stockholm_write(fp, msa, 200)` for `eslMSAFILE_STOCKHOLM`,
/// `esl_msafile_stockholm.c`).
const STOCKHOLM_CPL: usize = 200;

#[derive(Parser)]
#[command(
    name = "hmmemit",
    about = "Sample or emit sequences from a profile HMM"
)]
struct Args {
    /// Direct output to file, not stdout
    #[arg(short = 'o')]
    outfile: Option<PathBuf>,

    /// HMM file
    hmmfile: PathBuf,

    /// Emit alignment
    #[arg(short = 'a', action = ArgAction::SetTrue,
          conflicts_with_all = ["consensus", "fancy_consensus", "profile"])]
    alignment: bool,

    /// Emit consensus sequence
    // C (hmmemit.c:32): -c is short-only (no long form). Do NOT add `long`.
    #[arg(short = 'c', action = ArgAction::SetTrue,
          conflicts_with_all = ["alignment", "fancy_consensus", "profile"])]
    consensus: bool,

    /// Emit fancy consensus sequence
    #[arg(short = 'C', action = ArgAction::SetTrue,
          conflicts_with_all = ["alignment", "consensus", "profile"])]
    fancy_consensus: bool,

    /// Sample sequences from profile, not core model
    #[arg(short = 'p', action = ArgAction::SetTrue,
          conflicts_with_all = ["alignment", "consensus", "fancy_consensus"])]
    profile: bool,

    /// Expected sequence length for profile emission
    /// (requires -p; enforced in run() so the default value alone does not trip it)
    #[arg(short = 'L', default_value = "400")]
    length: usize,

    /// Configure profile in multihit local mode
    #[arg(long = "local", action = ArgAction::SetTrue, requires = "profile",
          conflicts_with_all = ["unilocal", "glocal", "uniglocal"])]
    local: bool,

    /// Configure profile in unihit local mode
    #[arg(long = "unilocal", action = ArgAction::SetTrue, requires = "profile",
          conflicts_with_all = ["local", "glocal", "uniglocal"])]
    unilocal: bool,

    /// Configure profile in multihit glocal mode
    #[arg(long = "glocal", action = ArgAction::SetTrue, requires = "profile",
          conflicts_with_all = ["local", "unilocal", "uniglocal"])]
    glocal: bool,

    /// Configure profile in unihit glocal mode
    #[arg(long = "uniglocal", action = ArgAction::SetTrue, requires = "profile",
          conflicts_with_all = ["local", "unilocal", "glocal"])]
    uniglocal: bool,

    /// Fancy consensus: use any-residue unless best residue probability is at least this
    /// (range 0<=x<=1; requires -C, enforced in run())
    #[arg(long = "minl", default_value = "0.0", value_parser = parse_unit_interval)]
    minl: f32,

    /// Fancy consensus: uppercase best residue when probability is at least this
    /// (range 0<=x<=1; requires -C, enforced in run())
    #[arg(long = "minu", default_value = "0.0", value_parser = parse_unit_interval)]
    minu: f32,

    /// Number of sequences to emit
    #[arg(short = 'N', conflicts_with_all = ["consensus", "fancy_consensus"])]
    n: Option<NonZeroUsize>,

    /// Random number seed
    #[arg(long = "seed", default_value = "0")]
    seed: u64,
}

/// clap value parser for `--minl`/`--minu`: a real number in the closed range
/// `0 <= x <= 1`, matching C's `eslARG_REAL` range `"0<=x<=1"`.
fn parse_unit_interval(s: &str) -> Result<f32, String> {
    let v: f32 = s
        .parse()
        .map_err(|_| format!("takes real-valued arg in range 0<=x<=1; got {s}"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("takes real-valued arg in range 0<=x<=1; got {s}"))
    }
}

pub fn run(args: Vec<String>) -> std::process::ExitCode {
    use clap::parser::ValueSource;
    use clap::CommandFactory;

    // C's reqs on -L, --minl and --minu fire on presence, not value, and all three have defaults.
    // ArgMatches tells a value given on the command line from a clap default.
    let matches = match Args::command().try_get_matches_from(&args) {
        Ok(m) => m,
        Err(e) => {
            e.print().ok();
            std::process::exit(2);
        }
    };
    let given = |id: &str| matches.value_source(id) == Some(ValueSource::CommandLine);
    let args = match Args::from_arg_matches(&matches) {
        Ok(a) => a,
        Err(e) => {
            e.print().ok();
            std::process::exit(2);
        }
    };

    // C: `-L` reqs `-p`; `--minl`/`--minu` reqs `-C`. Enforced on presence.
    if given("length") && !args.profile {
        eprintln!("Error: option -L requires (or has no effect without) option -p");
        std::process::exit(1);
    }
    if (given("minl") || given("minu")) && !args.fancy_consensus {
        eprintln!("Error: options --minl and --minu require (or have no effect without) option -C");
        std::process::exit(1);
    }

    let hmms = read_hmms(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });
    if hmms.is_empty() {
        eprintln!(
            "Empty HMM file {}? No HMM data found.",
            args.hmmfile.display()
        );
        std::process::exit(1);
    }

    let mut outfile = args
        .outfile
        .as_ref()
        .map(|p| create_file_or_exit(p, "output file"));
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();
    let out: &mut dyn Write = match outfile {
        Some(ref mut file) => file,
        None => &mut stdout_lock,
    };

    let n = args.n.map(NonZeroUsize::get).unwrap_or(1);
    let mut rng = MersenneTwister::new(args.seed as u32);

    for h in &hmms {
        let abc = Alphabet::new(h.abc_type);
        let bg = Bg::new(&abc);

        if args.consensus {
            let seq = simple_consensus(h, &abc);
            write_fasta(out, &format!("{}-consensus", h.name), &seq);
        } else if args.fancy_consensus {
            let seq = fancy_consensus(h, &abc, args.minl, args.minu);
            write_fasta(out, &format!("{}-consensus", h.name), &seq);
        } else if args.alignment {
            let mut samples = Vec::with_capacity(n);
            for i in 0..n {
                let (seq, trace) = core_emit(h, &abc, &mut rng);
                samples.push((format!("{}-sample{}", h.name, i + 1), seq, trace));
            }
            write_stockholm_alignment(out, h, &abc, &samples);
        } else {
            let mut gm = if args.profile {
                let mut profile = Profile::new(h.m, &abc);
                profile_config(
                    h,
                    &bg,
                    &mut profile,
                    args.length as i32,
                    profile_mode(&args),
                );
                Some(profile)
            } else {
                None
            };
            for i in 0..n {
                let (seq, _) = if let Some(profile) = gm.as_mut() {
                    profile_emit(h, profile, &bg, &mut rng)
                } else {
                    core_emit(h, &abc, &mut rng)
                };
                write_fasta(
                    out,
                    &format!("{}-sample{}", h.name, i + 1),
                    &sequence_text(&abc, &seq),
                );
            }
        }
    }
    std::process::ExitCode::SUCCESS
}

fn profile_mode(args: &Args) -> i32 {
    if args.unilocal {
        P7_UNILOCAL
    } else if args.glocal {
        P7_GLOCAL
    } else if args.uniglocal {
        P7_UNIGLOCAL
    } else {
        P7_LOCAL
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/subcmd/hmmemit_test.rs"]
mod tests;
