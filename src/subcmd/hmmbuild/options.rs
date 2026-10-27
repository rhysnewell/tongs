use super::*;

#[derive(Parser)]
#[command(
    name = "hmmbuild",
    about = "Build profile HMM(s) from multiple sequence alignment(s)"
)]
pub(super) struct Args {
    /// Output HMM file
    pub(super) hmmfile: PathBuf,
    /// Input alignment file (Stockholm format)
    pub(super) msafile: PathBuf,

    /// Name the HMM
    #[arg(short = 'n')]
    pub(super) name: Option<String>,

    /// Direct summary output to file, not stdout
    #[arg(short = 'o')]
    pub(super) summary_out: Option<PathBuf>,

    /// Resave processed input alignment to file
    #[arg(short = 'O')]
    pub(super) msa_out: Option<PathBuf>,

    /// Assert input alignment file format
    #[arg(long = "informat")]
    pub(super) informat: Option<String>,

    /// Use DNA alphabet
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["rna", "amino"])]
    pub(super) dna: bool,

    /// Use RNA alphabet
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["dna", "amino"])]
    pub(super) rna: bool,

    /// Use protein alphabet
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["dna", "rna"])]
    pub(super) amino: bool,

    /// Assign consensus columns from RF annotation
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "fast")]
    pub(super) hand: bool,

    /// Assign consensus columns by residue fraction
    #[arg(long, action = ArgAction::SetTrue)]
    pub(super) fast: bool,

    /// Henikoff position-based weights (default; accepted for C compatibility)
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wgsc", "wblosum", "wnone", "wgiven"])]
    pub(super) wpb: bool,

    /// Gerstein/Sonnhammer/Chothia tree weights
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wblosum", "wnone", "wgiven"])]
    pub(super) wgsc: bool,

    /// Henikoff simple filter weights
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wnone", "wgiven"])]
    pub(super) wblosum: bool,

    /// No relative sequence weighting; set all weights to 1
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wblosum", "wgiven"])]
    pub(super) wnone: bool,

    /// Use weights as given in MSA file
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wblosum", "wnone"])]
    pub(super) wgiven: bool,

    /// For --wblosum: set identity cutoff
    #[arg(long = "wid", default_value = "0.62", value_parser = parse_unit_f64)]
    pub(super) wid: f64,

    /// Entropy effective sequence weighting (default; accepted for C compatibility)
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["eentexp", "eclust", "enone", "eset"])]
    pub(super) eent: bool,

    /// Entropy effective sequence weighting with exponent-based scaling
    #[arg(long = "eentexp", action = ArgAction::SetTrue, conflicts_with_all = ["eent", "eclust", "enone", "eset"])]
    pub(super) eentexp: bool,

    /// Set effective sequence number to single-linkage cluster count
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["eent", "eentexp", "enone", "eset"])]
    pub(super) eclust: bool,

    /// For --eclust: set identity cutoff
    #[arg(long = "eid", default_value = "0.62", value_parser = parse_unit_f64)]
    pub(super) eid: f64,

    /// No effective sequence weighting; use the raw sequence count
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["eent", "eentexp", "eclust", "eset"])]
    pub(super) enone: bool,

    /// Set effective sequence number for all models
    #[arg(long = "eset", conflicts_with_all = ["eent", "eentexp", "eclust", "enone"])]
    pub(super) eset: Option<f32>,

    /// No prior; use observed counts only
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "plaplace")]
    pub(super) pnone: bool,

    /// Use a Laplace +1 prior
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "pnone")]
    pub(super) plaplace: bool,

    /// Minimum relative entropy target for entropy effective sequence weighting
    #[arg(long = "ere", value_parser = parse_positive_f64)]
    pub(super) ere: Option<f64>,

    /// Entropy target sigma parameter
    #[arg(long = "esigma", default_value = "45.0", value_parser = parse_positive_f64)]
    pub(super) esigma: f64,

    /// Random number seed for E-value calibration
    #[arg(long = "seed", default_value = "42")]
    pub(super) seed: u32,

    /// Number of parallel CPU workers (accepted for C compatibility)
    #[arg(long = "cpu", default_value = "2")]
    pub(super) cpu: usize,

    #[arg(
        long = "stall",
        action = ArgAction::SetTrue,
        help = "arrest after start: for attaching debugger to process"
    )]
    pub(super) stall: bool,

    /// Sym fraction threshold for match/insert (default 0.5)
    #[arg(long = "symfrac", default_value = "0.5", value_parser = parse_unit_f32, conflicts_with = "hand")]
    pub(super) symfrac: f32,

    /// Sequence is called a fragment if L <= x*alignment_length
    #[arg(long = "fragthresh", default_value = "0.5", value_parser = parse_unit_f32)]
    pub(super) fragthresh: f32,

    /// Use single-sequence builder path for one-sequence alignments
    #[arg(long = "singlemx", action = ArgAction::SetTrue)]
    pub(super) singlemx: bool,

    /// Substitution score matrix choice for --singlemx
    #[arg(long = "mx", default_value = "BLOSUM62", conflicts_with = "mxfile")]
    pub(super) matrix: String,

    /// Read substitution score matrix from file
    #[arg(long = "mxfile", conflicts_with = "matrix")]
    pub(super) mxfile: Option<PathBuf>,

    /// Gap open probability for --singlemx
    #[arg(long = "popen", default_value = "0.02", value_parser = parse_popen)]
    pub(super) popen: f64,

    /// Gap extend probability for --singlemx
    #[arg(long = "pextend", default_value = "0.4", value_parser = parse_pextend)]
    pub(super) pextend: f64,

    /// Length of sequences for MSV Gumbel mu fit
    #[arg(long = "EmL", default_value = "200", value_parser = parse_positive_usize)]
    pub(super) em_l: usize,

    /// Number of sequences for MSV Gumbel mu fit
    #[arg(long = "EmN", default_value = "200", value_parser = parse_positive_usize)]
    pub(super) em_n: usize,

    /// Length of sequences for Viterbi Gumbel mu fit
    #[arg(long = "EvL", default_value = "200", value_parser = parse_positive_usize)]
    pub(super) ev_l: usize,

    /// Number of sequences for Viterbi Gumbel mu fit
    #[arg(long = "EvN", default_value = "200", value_parser = parse_positive_usize)]
    pub(super) ev_n: usize,

    /// Length of sequences for Forward exp tail tau fit
    #[arg(long = "EfL", default_value = "100", value_parser = parse_positive_usize)]
    pub(super) ef_l: usize,

    /// Number of sequences for Forward exp tail tau fit
    #[arg(long = "EfN", default_value = "200", value_parser = parse_positive_usize)]
    pub(super) ef_n: usize,

    /// Tail mass for Forward exponential tail tau fit
    #[arg(long = "Eft", default_value = "0.04", value_parser = parse_open_unit_f64)]
    pub(super) eft: f64,

    /// Tail mass for deriving nucleotide model window length
    #[arg(long = "w_beta", value_parser = parse_window_beta)]
    pub(super) w_beta: Option<f64>,

    /// Nucleotide model window length
    #[arg(long = "w_length", value_parser = parse_window_length)]
    pub(super) w_length: Option<usize>,

    /// Pretend all inserts are length <= n
    #[arg(long = "maxinsertlen", value_parser = parse_maxinsertlen)]
    pub(super) max_insert_len: Option<usize>,
}
fn parse_positive_f64(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid positive number: {e}"))?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err("value must be > 0".to_string())
    }
}

fn parse_positive_usize(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid positive integer: {e}"))?;
    if value > 0 {
        Ok(value)
    } else {
        Err("value must be > 0".to_string())
    }
}

fn parse_open_unit_f64(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid probability: {e}"))?;
    if value > 0.0 && value < 1.0 {
        Ok(value)
    } else {
        Err("value must be > 0 and < 1".to_string())
    }
}

fn parse_window_beta(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid window-length beta value: {e}"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("Invalid window-length beta value".to_string())
    }
}

fn parse_window_length(s: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|e| format!("invalid window length: {e}"))
}

fn parse_maxinsertlen(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid max insert length: {e}"))?;
    if value >= 5 {
        Ok(value)
    } else {
        Err("value must be >= 5".to_string())
    }
}

fn parse_unit_f64(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid probability: {e}"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("value must be >= 0 and <= 1".to_string())
    }
}

/// Parse an f32 in C's `0<=x<=1` range (used for --symfrac and --fragthresh).
fn parse_unit_f32(s: &str) -> Result<f32, String> {
    let value = s
        .parse::<f32>()
        .map_err(|e| format!("invalid real value: {e}"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("value must be >= 0 and <= 1".to_string())
    }
}

fn parse_popen(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid gap open probability: {e}"))?;
    if (0.0..0.5).contains(&value) {
        Ok(value)
    } else {
        Err("--popen must be >= 0 and < 0.5".to_string())
    }
}

fn parse_pextend(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid gap extend probability: {e}"))?;
    if (0.0..1.0).contains(&value) {
        Ok(value)
    } else {
        Err("--pextend must be >= 0 and < 1".to_string())
    }
}

pub(super) struct Requested {
    pub(super) seed: bool,
    pub(super) cpu: bool,
    pub(super) wid: bool,
    pub(super) eid: bool,
    pub(super) em_l: bool,
    pub(super) em_n: bool,
    pub(super) ev_l: bool,
    pub(super) ev_n: bool,
    pub(super) ef_l: bool,
    pub(super) ef_n: bool,
    pub(super) eft: bool,
    pub(super) popen: bool,
    pub(super) pextend: bool,
    pub(super) mx: bool,
}

impl Requested {
    pub(super) fn from_args(args: &[String]) -> Self {
        let given = |flag: &str| {
            args.iter().any(|arg| {
                arg == flag
                    || arg
                        .strip_prefix(flag)
                        .is_some_and(|rest| rest.starts_with('='))
            })
        };
        Requested {
            seed: given("--seed"),
            cpu: given("--cpu"),
            wid: given("--wid"),
            eid: given("--eid"),
            em_l: given("--EmL"),
            em_n: given("--EmN"),
            ev_l: given("--EvL"),
            ev_n: given("--EvN"),
            ef_l: given("--EfL"),
            ef_n: given("--EfN"),
            eft: given("--Eft"),
            popen: given("--popen"),
            pextend: given("--pextend"),
            mx: given("--mx"),
        }
    }
}
