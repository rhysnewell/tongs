use super::*;

#[derive(Parser)]
#[command(
    name = "hmmscan",
    about = "Search sequence(s) against a profile HMM database"
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

    /// Report profiles <= this E-value threshold
    #[arg(
        short = 'E',
        default_value = "10.0",
        value_parser = parse_positive_f64,
        conflicts_with = "score_threshold"
    )]
    pub(super) e_value: f64,

    /// Report profiles >= this score threshold
    #[arg(short = 'T', conflicts_with = "e_value", allow_hyphen_values = true)]
    pub(super) score_threshold: Option<f64>,

    /// Report domains <= this E-value threshold
    #[arg(
        long = "domE",
        default_value = "10.0",
        value_parser = parse_positive_f64,
        conflicts_with = "dom_t"
    )]
    pub(super) dom_e: f64,

    /// Report domains >= this score threshold
    #[arg(long = "domT", conflicts_with = "dom_e", allow_hyphen_values = true)]
    pub(super) dom_t: Option<f64>,

    /// Include profiles <= this E-value threshold
    #[arg(
        long = "incE",
        default_value = "0.01",
        value_parser = parse_positive_f64,
        conflicts_with = "inc_t"
    )]
    pub(super) inc_e: f64,

    /// Include profiles >= this score threshold
    #[arg(long = "incT", conflicts_with = "inc_e", allow_hyphen_values = true)]
    pub(super) inc_t: Option<f64>,

    /// Include domains <= this E-value threshold
    #[arg(
        long = "incdomE",
        default_value = "0.01",
        value_parser = parse_positive_f64,
        conflicts_with = "inc_dom_t"
    )]
    pub(super) inc_dome: f64,

    /// Include domains >= this score threshold
    #[arg(
        long = "incdomT",
        conflicts_with = "inc_dome",
        allow_hyphen_values = true
    )]
    pub(super) inc_dom_t: Option<f64>,

    /// Use model's GA gathering cutoffs to set all thresholding
    #[arg(
        long = "cut_ga",
        conflicts_with_all = [
            "cut_tc",
            "cut_nc",
            "e_value",
            "score_threshold",
            "dom_e",
            "dom_t",
            "inc_e",
            "inc_t",
            "inc_dome",
            "inc_dom_t"
        ]
    )]
    pub(super) cut_ga: bool,

    /// Use model's NC noise cutoffs to set all thresholding
    #[arg(
        long = "cut_nc",
        conflicts_with_all = [
            "cut_ga",
            "cut_tc",
            "e_value",
            "score_threshold",
            "dom_e",
            "dom_t",
            "inc_e",
            "inc_t",
            "inc_dome",
            "inc_dom_t"
        ]
    )]
    pub(super) cut_nc: bool,

    /// Use model's TC trusted cutoffs to set all thresholding
    #[arg(
        long = "cut_tc",
        conflicts_with_all = [
            "cut_ga",
            "cut_nc",
            "e_value",
            "score_threshold",
            "dom_e",
            "dom_t",
            "inc_e",
            "inc_t",
            "inc_dome",
            "inc_dom_t"
        ]
    )]
    pub(super) cut_tc: bool,

    /// Turn all heuristic filters off
    #[arg(long = "max", conflicts_with_all = ["f1", "f2", "f3", "nobias"])]
    pub(super) max: bool,

    /// MSV filter threshold
    #[arg(long = "F1", default_value = "0.02", allow_hyphen_values = true)]
    pub(super) f1: f64,

    /// Viterbi filter threshold
    #[arg(long = "F2", default_value = "0.001", allow_hyphen_values = true)]
    pub(super) f2: f64,

    /// Forward filter threshold
    #[arg(long = "F3", default_value = "1e-5", allow_hyphen_values = true)]
    pub(super) f3: f64,

    /// Turn off composition bias filter
    #[arg(long = "nobias")]
    pub(super) nobias: bool,

    /// Turn off biased composition score corrections
    #[arg(long = "nonull2")]
    pub(super) nonull2: bool,

    /// Random number seed
    #[arg(long = "seed", default_value = "42")]
    pub(super) seed: u32,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "0")]
    pub(super) cpu: usize,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    pub(super) tblout: Option<PathBuf>,

    /// Save per-domain hits to tabular file
    #[arg(long = "domtblout")]
    pub(super) domtblout: Option<PathBuf>,

    /// Save Pfam-style table of hits and domains
    #[arg(long = "pfamtblout")]
    pub(super) pfamtblout: Option<PathBuf>,

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

    /// Set number of comparisons for E-value calculation
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    pub(super) z_value: Option<f64>,

    /// Set number of significant seqs for domain E-value calculation
    #[arg(long = "domZ", value_parser = parse_positive_f64)]
    pub(super) domz_value: Option<f64>,
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
    pli.dom_e_value_threshold = args.dom_e;
    pli.inc_e = args.inc_e;
    pli.inc_dome = args.inc_dome;

    if let Some(t) = args.score_threshold {
        pli.t = Some(t);
        pli.by_e = false;
    }
    if let Some(t) = args.dom_t {
        pli.dom_t = Some(t);
        pli.dom_by_e = false;
    }
    if let Some(t) = args.inc_t {
        pli.inc_t = Some(t);
        pli.inc_by_e = false;
    }
    if let Some(t) = args.inc_dom_t {
        pli.inc_dom_t = Some(t);
        pli.incdom_by_e = false;
    }
}

pub(super) fn configure_acceleration(pli: &mut Pipeline, args: &Args) {
    pli.do_max = args.max;
    pli.f1 = if args.max { 1.0 } else { args.f1 };
    pli.f2 = if args.max { 1.0 } else { args.f2 };
    pli.f3 = if args.max { 1.0 } else { args.f3 };
    pli.do_biasfilter = !(args.max || args.nobias);
    pli.do_null2 = !args.nonull2;
    pli.seed = args.seed;
}
