use super::*;

#[derive(Parser)]
#[command(
    name = "nhmmscan",
    about = "Search nucleotide sequence(s) against a DNA HMM database"
)]
pub(super) struct Args {
    /// Direct output to file <f>, not stdout
    #[arg(short = 'o')]
    pub(super) output: Option<PathBuf>,

    /// HMM database file
    pub(super) hmmdb: PathBuf,
    /// Query sequence file (FASTA)
    pub(super) seqfile: PathBuf,

    /// Assert query sequence file format
    #[arg(long = "qformat")]
    pub(super) qformat: Option<String>,

    /// Report models <= this E-value threshold
    #[arg(
        short = 'E',
        default_value = "10.0",
        value_parser = parse_positive_f64,
        conflicts_with = "score_threshold"
    )]
    pub(super) e_value: f64,

    /// Report models >= this score threshold
    #[arg(short = 'T', conflicts_with = "e_value", allow_hyphen_values = true)]
    pub(super) score_threshold: Option<f64>,

    /// Include models <= this E-value threshold
    #[arg(
        long = "incE",
        default_value = "0.01",
        value_parser = parse_positive_f64,
        conflicts_with = "inc_t"
    )]
    pub(super) inc_e: f64,

    /// Include models >= this score threshold
    #[arg(long = "incT", conflicts_with = "inc_e", allow_hyphen_values = true)]
    pub(super) inc_t: Option<f64>,

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

    /// Turn off biased composition score corrections
    #[arg(long = "nonull2")]
    pub(super) nonull2: bool,

    /// Set number of comparisons done for E-value calculation
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    pub(super) z_value: Option<f64>,

    /// Random number seed
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

    /// Tail mass for deriving window length
    #[arg(long = "w_beta", value_parser = parse_window_beta)]
    pub(super) w_beta: Option<f64>,

    /// Window length for biased-composition modifier at MSV stage
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

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "0")]
    pub(super) cpu: usize,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    pub(super) tblout: Option<PathBuf>,

    /// Save Dfam-style table of hits and domains
    #[arg(long = "dfamtblout")]
    pub(super) dfamtblout: Option<PathBuf>,

    /// Prefer accessions over names in output
    #[arg(long = "acc")]
    pub(super) acc: bool,

    /// Do not output alignments
    #[arg(long = "noali")]
    pub(super) noali: bool,

    /// Unlimit ASCII text output line width
    #[arg(long = "notextw", conflicts_with = "textw")]
    pub(super) notextw: bool,

    /// Set max width of ASCII text output lines
    #[arg(long = "textw", value_parser = parse_textw)]
    pub(super) textw: Option<usize>,
}

pub(super) fn selected_bit_cutoff(args: &Args) -> Option<BitCutoff> {
    if args.cut_ga {
        Some(BitCutoff::GA)
    } else if args.cut_tc {
        Some(BitCutoff::TC)
    } else if args.cut_nc {
        Some(BitCutoff::NC)
    } else {
        None
    }
}

pub(super) fn configure_thresholds(pli: &mut Pipeline, args: &Args) {
    pli.e_value_threshold = args.e_value;
    pli.inc_e = args.inc_e;
    if let Some(t) = args.score_threshold {
        pli.t = Some(t);
        pli.by_e = false;
    }
    if let Some(t) = args.inc_t {
        pli.inc_t = Some(t);
        pli.inc_by_e = false;
    }
}
