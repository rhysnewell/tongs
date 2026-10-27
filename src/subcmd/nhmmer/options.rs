use super::*;

#[derive(Debug, Clone, Copy)]
pub(super) struct NhmmerFmSeedConfig {
    pub(super) max_depth: usize,
    pub(super) score_threshold_bits: f32,
    pub(super) score_density_bits: f32,
    pub(super) drop_max_len: usize,
    pub(super) drop_lim_bits: f32,
    pub(super) consec_pos_req: usize,
    pub(super) consensus_match_req: usize,
    pub(super) ssv_length: usize,
}

impl Default for NhmmerFmSeedConfig {
    fn default() -> Self {
        Self {
            max_depth: NHMMER_FM_SEED_MAX_DEPTH,
            score_threshold_bits: NHMMER_FM_SEED_SCORE_THRESHOLD_BITS,
            score_density_bits: NHMMER_FM_SEED_SCORE_DENSITY_BITS,
            drop_max_len: NHMMER_FM_SEED_DROP_MAX_LEN,
            drop_lim_bits: NHMMER_FM_SEED_DROP_LIMIT_BITS,
            consec_pos_req: NHMMER_FM_SEED_CONSEC_POS_REQ,
            consensus_match_req: NHMMER_FM_CONSENSUS_MATCH_REQ,
            ssv_length: NHMMER_FM_SEED_SSV_LENGTH,
        }
    }
}

#[derive(Parser)]
#[command(
    name = "nhmmer",
    about = "Search DNA/RNA HMM(s) against a nucleotide sequence database"
)]
pub(super) struct Args {
    /// HMM file, alignment file, or query sequence
    pub(super) hmmfile: PathBuf,
    /// Target sequence database (FASTA or makehmmerdb FM-index)
    pub(super) seqdb: PathBuf,

    // --- Output options ---
    /// Direct output to file, not stdout
    #[arg(short = 'o')]
    pub(super) outfile: Option<PathBuf>,

    /// Save multiple alignment of all hits to file
    #[arg(short = 'A')]
    pub(super) ali_outfile: Option<PathBuf>,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    pub(super) tblout: Option<PathBuf>,

    /// Save hits to Dfam-style tabular file
    #[arg(long = "dfamtblout")]
    pub(super) dfamtblout: Option<PathBuf>,

    /// Save scores for each position in each alignment to file
    #[arg(long = "aliscoresout")]
    pub(super) aliscoresout: Option<PathBuf>,

    /// Write query HMMs to file
    #[arg(long = "hmmout")]
    pub(super) hmmout: Option<PathBuf>,

    /// Don't output alignments
    #[arg(long = "noali")]
    pub(super) noali: bool,

    /// Prefer accessions over names in output
    #[arg(long = "acc")]
    pub(super) show_acc: bool,

    /// Unlimit ASCII text output line width
    #[arg(long = "notextw", conflicts_with = "textw")]
    pub(super) notextw: bool,

    /// Set max width of ASCII text output lines
    #[arg(long = "textw", default_value = "120", value_parser = parse_textw)]
    pub(super) textw: usize,

    // --- Reporting thresholds ---
    /// Report sequences <= this E-value threshold
    #[arg(
        short = 'E',
        default_value = "10.0",
        value_parser = parse_positive_f64,
        conflicts_with = "score_threshold"
    )]
    pub(super) e_value: f64,

    /// Report sequences >= this score threshold
    // `allow_hyphen_values` so negative bit-score thresholds parse in the
    // space-separated form C/Easel uses (e.g. `-T -20`); clap otherwise treats
    // `-20` as flags. The `-T=-20` form also still works.
    #[arg(short = 'T', conflicts_with = "e_value", allow_hyphen_values = true)]
    pub(super) score_threshold: Option<f64>,

    /// Include sequences <= this E-value threshold
    #[arg(
        long = "incE",
        default_value = "0.01",
        value_parser = parse_positive_f64,
        conflicts_with = "inc_t"
    )]
    pub(super) inc_e: f64,

    /// Include sequences >= this score threshold
    #[arg(long = "incT", conflicts_with = "inc_e", allow_hyphen_values = true)]
    pub(super) inc_t: Option<f64>,

    // --- Model-specific cutoffs ---
    /// Use model's GA gathering cutoffs to set all thresholding
    #[arg(
        long = "cut_ga",
        conflicts_with_all = ["cut_tc", "cut_nc", "e_value", "score_threshold", "inc_e", "inc_t"]
    )]
    pub(super) cut_ga: bool,

    /// Use model's NC noise cutoffs to set all thresholding
    #[arg(
        long = "cut_nc",
        conflicts_with_all = ["cut_ga", "cut_tc", "e_value", "score_threshold", "inc_e", "inc_t"]
    )]
    pub(super) cut_nc: bool,

    /// Use model's TC trusted cutoffs to set all thresholding
    #[arg(
        long = "cut_tc",
        conflicts_with_all = ["cut_ga", "cut_nc", "e_value", "score_threshold", "inc_e", "inc_t"]
    )]
    pub(super) cut_tc: bool,

    // --- Acceleration heuristics ---
    /// Turn all heuristic filters off (less speed, more power)
    #[arg(long = "max", conflicts_with_all = ["f1", "f2", "f3", "nobias"])]
    pub(super) max: bool,

    /// Stage 1 (SSV) threshold
    #[arg(long = "F1", default_value = "0.02", allow_hyphen_values = true)]
    pub(super) f1: f64,

    /// Stage 2 (Vit) threshold
    #[arg(long = "F2", default_value = "3e-3", allow_hyphen_values = true)]
    pub(super) f2: f64,

    /// Stage 3 (Fwd) threshold
    #[arg(long = "F3", default_value = "3e-5", allow_hyphen_values = true)]
    pub(super) f3: f64,

    /// Turn off composition bias filter
    #[arg(long = "nobias")]
    pub(super) nobias: bool,

    /// Gap open probability for single-sequence query models
    #[arg(long = "popen", default_value = "0.03125", value_parser = parse_gap_open)]
    pub(super) popen: f64,

    /// Gap extend probability for single-sequence query models
    #[arg(long = "pextend", default_value = "0.75", value_parser = parse_gap_extend)]
    pub(super) pextend: f64,

    /// Use substitution score matrix for single-sequence MSA-format inputs
    #[arg(long = "singlemx")]
    pub(super) singlemx: bool,

    /// Substitution score matrix choice for --singlemx-compatible inputs
    #[arg(
        long = "mx",
        default_value = "DNA1",
        conflicts_with = "mxfile",
        hide = true
    )]
    pub(super) matrix: String,

    /// Read substitution score matrix from file
    #[arg(long = "mxfile", conflicts_with = "matrix")]
    pub(super) mxfile: Option<PathBuf>,

    // --- Alphabet selection ---
    /// Use DNA alphabet
    #[arg(long)]
    pub(super) dna: bool,

    /// Use RNA alphabet
    #[arg(long)]
    pub(super) rna: bool,

    // --- Expert options ---
    /// Turn off biased composition score corrections
    #[arg(long = "nonull2")]
    pub(super) nonull2: bool,

    /// Set database size (Megabases) for E-value calculation
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    pub(super) z_value: Option<f64>,

    /// Retained for C command-line compatibility; not used by nhmmer output
    #[arg(long = "domZ", value_parser = parse_positive_f64, hide = true)]
    pub(super) domz_value: Option<f64>,

    /// Retained for C command-line compatibility; not used by nhmmer output
    #[arg(long = "domE", default_value = "10.0", value_parser = parse_positive_f64, conflicts_with = "dom_t", hide = true)]
    pub(super) dom_e: f64,

    /// Retained for C command-line compatibility; not used by nhmmer output
    #[arg(
        long = "domT",
        conflicts_with = "dom_e",
        hide = true,
        allow_hyphen_values = true
    )]
    pub(super) dom_t: Option<f64>,

    /// Retained for C command-line compatibility; not used by nhmmer output
    #[arg(long = "incdomE", default_value = "0.01", value_parser = parse_positive_f64, conflicts_with = "inc_dom_t", hide = true)]
    pub(super) inc_dome: f64,

    /// Retained for C command-line compatibility; not used by nhmmer output
    #[arg(
        long = "incdomT",
        conflicts_with = "inc_dome",
        hide = true,
        allow_hyphen_values = true
    )]
    pub(super) inc_dom_t: Option<f64>,

    /// Set RNG seed (0: one-time arbitrary seed)
    #[arg(long = "seed", default_value = "42")]
    pub(super) seed: u32,

    /// Only search the top strand
    #[arg(long = "watson", conflicts_with = "crick")]
    pub(super) watson: bool,

    /// Only search the bottom strand
    #[arg(long = "crick", conflicts_with = "watson")]
    pub(super) crick: bool,

    /// Window length (max expected hit length)
    #[arg(long = "w_length", value_parser = parse_window_length)]
    pub(super) w_length: Option<i32>,

    /// Length of blocks read from target database in threaded C nhmmer
    #[arg(long = "block_length", value_parser = parse_block_length)]
    pub(super) block_length: Option<usize>,

    /// Tail mass for deriving window length
    #[arg(long = "w_beta", value_parser = parse_window_beta)]
    pub(super) w_beta: Option<f64>,

    /// Assert target sequence file format (FASTA or FM-index)
    #[arg(long = "tformat")]
    pub(super) tformat: Option<String>,

    /// Assert query file format (HMM auto-detected; Stockholm/AFA MSA supported)
    #[arg(long = "qformat")]
    pub(super) qformat: Option<String>,

    /// Force query to be read as individual sequences, even if MSA-looking
    #[arg(long = "qsingle_seqs")]
    pub(super) qsingle_seqs: bool,

    /// Window length for biased-composition modifier at SSV stage
    #[arg(long = "B1", default_value = "110", conflicts_with_all = ["max", "nobias"], hide = true)]
    pub(super) b1: usize,

    /// Window length for biased-composition modifier at Viterbi stage
    #[arg(long = "B2", default_value = "240", conflicts_with_all = ["max", "nobias"], hide = true)]
    pub(super) b2: usize,

    /// Window length for biased-composition modifier at Forward stage
    #[arg(long = "B3", default_value = "1000", conflicts_with_all = ["max", "nobias"], hide = true)]
    pub(super) b3: usize,

    /// Override default background probabilities; accepted for compatibility
    #[arg(long = "bgfile", hide = true)]
    pub(super) bgfile: Option<PathBuf>,

    /// FM seed length at which bit threshold must be met
    #[arg(long = "seed_max_depth", default_value = "15")]
    pub(super) seed_max_depth: usize,

    /// Required FM seed score in bits
    #[arg(long = "seed_sc_thresh", default_value = "14")]
    pub(super) seed_sc_thresh: f32,

    /// Required FM seed score density in bits per position
    #[arg(long = "seed_sc_density", default_value = "0.75")]
    pub(super) seed_sc_density: f32,

    /// Maximum run length with score under max minus drop limit
    #[arg(long = "seed_drop_max_len", default_value = "4")]
    pub(super) seed_drop_max_len: usize,

    /// Maximum drop in a low-growth FM seed run
    #[arg(long = "seed_drop_lim", default_value = "0.3")]
    pub(super) seed_drop_lim: f32,

    /// Minimum consecutive positive scores in FM seed
    #[arg(long = "seed_req_pos", default_value = "5")]
    pub(super) seed_req_pos: usize,

    /// Consecutive consensus matches that override FM score threshold
    #[arg(long = "seed_consens_match", default_value = "11")]
    pub(super) seed_consens_match: usize,

    /// Window length around FM seed for full SSV diagonal
    #[arg(long = "seed_ssv_length", default_value = "100")]
    pub(super) seed_ssv_length: usize,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    pub(super) cpu: usize,

    /// Start restricted target search at this sequence key
    #[arg(long = "restrictdb_stkey", hide = true)]
    pub(super) restrictdb_stkey: Option<String>,

    /// Search only this many target sequences from the restricted start
    #[arg(long = "restrictdb_n", value_parser = parse_positive_usize, hide = true)]
    pub(super) restrictdb_n: Option<usize>,

    /// SSI index file for C-compatible restricted database options
    #[arg(long = "ssifile", hide = true)]
    pub(super) ssifile: Option<PathBuf>,
}

impl NhmmerFmSeedConfig {
    pub(super) fn from_args(args: &Args) -> Self {
        Self {
            max_depth: args.seed_max_depth,
            score_threshold_bits: args.seed_sc_thresh,
            score_density_bits: args.seed_sc_density,
            drop_max_len: args.seed_drop_max_len,
            drop_lim_bits: args.seed_drop_lim,
            consec_pos_req: args.seed_req_pos,
            consensus_match_req: args.seed_consens_match,
            ssv_length: args.seed_ssv_length,
        }
    }
}

fn parse_textw(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid text width: {e}"))?;
    if value < 120 {
        Err("--textw must be >= 120".to_string())
    } else {
        Ok(value)
    }
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

fn parse_window_length(s: &str) -> Result<i32, String> {
    let value = s
        .parse::<i32>()
        .map_err(|e| format!("invalid window length: {e}"))?;
    if value > 0 {
        Ok(value)
    } else {
        Err("--w_length must be > 0 and fit in a 32-bit signed integer".to_string())
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

fn parse_block_length(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid block length: {e}"))?;
    if value >= 50_000 {
        Ok(value)
    } else {
        Err("value must be >= 50000".to_string())
    }
}

fn parse_gap_open(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid gap open probability: {e}"))?;
    if (0.0..0.5).contains(&value) {
        Ok(value)
    } else {
        Err("--popen must be >= 0 and < 0.5".to_string())
    }
}

fn parse_gap_extend(s: &str) -> Result<f64, String> {
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
    pub(super) matrix: bool,
    pub(super) mxfile: bool,
    pub(super) seed_max_depth: bool,
    pub(super) seed_sc_thresh: bool,
    pub(super) seed_sc_density: bool,
    pub(super) seed_drop_max_len: bool,
    pub(super) seed_drop_lim: bool,
    pub(super) seed_req_pos: bool,
    pub(super) seed_consens_match: bool,
    pub(super) seed_ssv_length: bool,
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
            matrix: given("--mx"),
            mxfile: given("--mxfile"),
            seed_max_depth: given("--seed_max_depth"),
            seed_sc_thresh: given("--seed_sc_thresh"),
            seed_sc_density: given("--seed_sc_density"),
            seed_drop_max_len: given("--seed_drop_max_len"),
            seed_drop_lim: given("--seed_drop_lim"),
            seed_req_pos: given("--seed_req_pos"),
            seed_consens_match: given("--seed_consens_match"),
            seed_ssv_length: given("--seed_ssv_length"),
        }
    }
}
