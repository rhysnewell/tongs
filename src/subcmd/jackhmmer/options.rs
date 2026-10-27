use super::*;

#[derive(Parser)]
#[command(
    name = "jackhmmer",
    about = "Iteratively search a protein sequence against a protein database"
)]
pub(super) struct Args {
    /// Direct output to file <f>, not stdout
    #[arg(short = 'o')]
    pub(super) output: Option<PathBuf>,

    /// Query sequence file (FASTA)
    pub(super) seqfile: PathBuf,
    /// Target sequence database (FASTA)
    pub(super) seqdb: PathBuf,

    /// Assert query sequence file format
    #[arg(long = "qformat")]
    pub(super) qformat: Option<String>,

    /// Assert target sequence database format
    #[arg(long = "tformat")]
    pub(super) tformat: Option<String>,

    /// Maximum number of iterations
    #[arg(short = 'N', default_value = "5", value_parser = parse_nonzero_usize)]
    pub(super) max_iterations: usize,

    /// Report sequences <= this E-value threshold
    #[arg(
        short = 'E',
        default_value = "10.0",
        value_parser = parse_positive_f64,
        conflicts_with = "score_threshold"
    )]
    pub(super) e_value: f64,

    /// Report sequences >= this score threshold
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

    /// Include sequences <= this E-value threshold
    #[arg(
        long = "incE",
        default_value = "0.001",
        value_parser = parse_positive_f64,
        conflicts_with = "inc_t"
    )]
    pub(super) inc_e: f64,

    /// Include sequences >= this score threshold
    #[arg(long = "incT", conflicts_with = "inc_e", allow_hyphen_values = true)]
    pub(super) inc_t: Option<f64>,

    /// Include domains <= this E-value threshold
    #[arg(
        long = "incdomE",
        default_value = "0.001",
        value_parser = parse_positive_f64,
        conflicts_with = "incdom_t"
    )]
    pub(super) incdom_e: f64,

    /// Include domains >= this score threshold
    #[arg(
        long = "incdomT",
        conflicts_with = "incdom_e",
        allow_hyphen_values = true
    )]
    pub(super) incdom_t: Option<f64>,

    /// Hidden C-compatible model gathering cutoff flag
    #[arg(long = "cut_ga", hide = true)]
    pub(super) cut_ga: bool,

    /// Hidden C-compatible model noise cutoff flag
    #[arg(long = "cut_nc", hide = true)]
    pub(super) cut_nc: bool,

    /// Hidden C-compatible model trusted cutoff flag
    #[arg(long = "cut_tc", hide = true)]
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

    /// Gap open probability for the single-sequence query model
    #[arg(long = "popen", default_value = "0.02", value_parser = parse_popen)]
    pub(super) popen: f64,

    /// Gap extend probability for the single-sequence query model
    #[arg(long = "pextend", default_value = "0.4", value_parser = parse_pextend)]
    pub(super) pextend: f64,

    /// Substitution score matrix choice
    #[arg(long = "mx", default_value = "BLOSUM62", conflicts_with = "mxfile")]
    pub(super) matrix: String,

    /// Read substitution score matrix from file
    #[arg(long = "mxfile")]
    pub(super) mxfile: Option<PathBuf>,

    /// Hidden C-compatible fast architecture option, prohibited by jackhmmer
    #[arg(long = "fast", hide = true)]
    pub(super) fast: bool,

    /// Hidden C-compatible hand architecture option, prohibited by jackhmmer
    #[arg(long = "hand", hide = true)]
    pub(super) hand: bool,

    /// Hidden C-compatible fast architecture threshold, prohibited by jackhmmer
    #[arg(long = "symfrac", hide = true)]
    pub(super) symfrac: Option<f64>,

    /// Henikoff position-based weights for post-round rebuilds (default)
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wgsc", "wblosum", "wnone"])]
    pub(super) wpb: bool,

    /// Gerstein/Sonnhammer/Chothia tree weights for post-round rebuilds
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wblosum", "wnone"])]
    pub(super) wgsc: bool,

    /// Henikoff simple filter weights for post-round rebuilds
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wnone"])]
    pub(super) wblosum: bool,

    /// No relative sequence weighting in post-round rebuilds
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["wpb", "wgsc", "wblosum"])]
    pub(super) wnone: bool,

    /// Hidden C-compatible given-weight option, prohibited by jackhmmer
    #[arg(long = "wgiven", hide = true)]
    pub(super) wgiven: bool,

    /// For --wblosum: set identity cutoff
    #[arg(long = "wid", default_value = "0.62", value_parser = parse_unit_f64)]
    pub(super) wid: f64,

    /// Entropy effective sequence weighting for post-round rebuilds (default)
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["eentexp", "eclust", "enone", "eset"])]
    pub(super) eent: bool,

    /// Entropy effective sequence weighting with exponent-based scaling
    #[arg(long = "eentexp", action = ArgAction::SetTrue, conflicts_with_all = ["eent", "eclust", "enone", "eset"])]
    pub(super) eentexp: bool,

    /// Set post-round effective sequence number to single-linkage cluster count
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["eent", "eentexp", "enone", "eset"])]
    pub(super) eclust: bool,

    /// For --eclust: set identity cutoff
    #[arg(long = "eid", default_value = "0.62", value_parser = parse_unit_f64)]
    pub(super) eid: f64,

    /// No post-round effective sequence weighting; use the raw sequence count
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["eent", "eentexp", "eclust", "eset"])]
    pub(super) enone: bool,

    /// Set post-round effective sequence number for all models
    #[arg(long = "eset", conflicts_with_all = ["eent", "eentexp", "eclust", "enone"])]
    pub(super) eset: Option<f32>,

    /// No post-round prior; use observed counts only
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "plaplace")]
    pub(super) pnone: bool,

    /// Use a post-round Laplace +1 prior
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "pnone")]
    pub(super) plaplace: bool,

    /// Minimum relative entropy target for post-round entropy weighting
    #[arg(long = "ere", value_parser = parse_positive_f64)]
    pub(super) ere: Option<f64>,

    /// Entropy target sigma parameter for post-round entropy weighting
    #[arg(long = "esigma", default_value = "45.0", value_parser = parse_positive_f64)]
    pub(super) esigma: f64,

    /// Sequence is called a fragment if L <= x*alignment_length
    #[arg(long = "fragthresh", default_value = "0.5", value_parser = parse_unit_f32)]
    pub(super) fragthresh: f32,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    pub(super) cpu: usize,

    /// Turn off composition bias filter
    #[arg(long = "nobias")]
    pub(super) nobias: bool,

    /// Turn off biased composition score corrections
    #[arg(long = "nonull2")]
    pub(super) nonull2: bool,

    /// Set number of comparisons for E-value calculation
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    pub(super) z_value: Option<f64>,

    /// Set number of significant seqs for domain E-value calculation
    #[arg(long = "domZ", value_parser = parse_positive_f64)]
    pub(super) domz_value: Option<f64>,

    /// Random number seed
    #[arg(long = "seed", default_value = "42")]
    pub(super) seed: u32,

    /// Length of sequences for MSV Gumbel mu fit
    #[arg(long = "EmL", default_value = "200", value_parser = parse_nonzero_usize)]
    pub(super) em_l: usize,

    /// Number of sequences for MSV Gumbel mu fit
    #[arg(long = "EmN", default_value = "200", value_parser = parse_nonzero_usize)]
    pub(super) em_n: usize,

    /// Length of sequences for Viterbi Gumbel mu fit
    #[arg(long = "EvL", default_value = "200", value_parser = parse_nonzero_usize)]
    pub(super) ev_l: usize,

    /// Number of sequences for Viterbi Gumbel mu fit
    #[arg(long = "EvN", default_value = "200", value_parser = parse_nonzero_usize)]
    pub(super) ev_n: usize,

    /// Length of sequences for Forward exp tail tau fit
    #[arg(long = "EfL", default_value = "100", value_parser = parse_nonzero_usize)]
    pub(super) ef_l: usize,

    /// Number of sequences for Forward exp tail tau fit
    #[arg(long = "EfN", default_value = "200", value_parser = parse_nonzero_usize)]
    pub(super) ef_n: usize,

    /// Tail mass for Forward exponential tail tau fit
    #[arg(long = "Eft", default_value = "0.04", value_parser = parse_open_unit_f64)]
    pub(super) eft: f64,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    pub(super) tblout: Option<PathBuf>,

    /// Save per-domain hits to tabular file
    #[arg(long = "domtblout")]
    pub(super) domtblout: Option<PathBuf>,

    /// Save multiple alignment of hits to file
    #[arg(short = 'A')]
    pub(super) ali_outfile: Option<PathBuf>,

    /// Prefer accessions over names in output
    #[arg(long = "acc")]
    pub(super) show_acc: bool,

    /// Omit alignments from the main output
    #[arg(long = "noali")]
    pub(super) noali: bool,

    /// Do not line-wrap the main text output
    #[arg(long = "notextw", conflicts_with = "textw")]
    pub(super) notextw: bool,

    /// Set the target line width for the main text output
    #[arg(long = "textw", default_value = "120", value_parser = parse_textw)]
    pub(super) textw: usize,

    /// Save HMM checkpoints to files <f>-<iteration>.hmm
    #[arg(long = "chkhmm")]
    pub(super) chkhmm: Option<PathBuf>,

    /// Save alignment checkpoints to files <f>-<iteration>.sto
    #[arg(long = "chkali")]
    pub(super) chkali: Option<PathBuf>,
}

fn parse_nonzero_usize(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid positive integer: {e}"))?;
    if value > 0 {
        Ok(value)
    } else {
        Err("value must be > 0".to_string())
    }
}

fn parse_unit_f32(s: &str) -> Result<f32, String> {
    let value = s
        .parse::<f32>()
        .map_err(|e| format!("invalid probability: {e}"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("value must be >= 0 and <= 1".to_string())
    }
}

pub(super) struct Requested {
    pub(super) cpu: bool,
    pub(super) matrix: bool,
    pub(super) mxfile: bool,
    pub(super) popen: bool,
    pub(super) pextend: bool,
    pub(super) wid: bool,
    pub(super) eid: bool,
    pub(super) em_l: bool,
    pub(super) em_n: bool,
    pub(super) ev_l: bool,
    pub(super) ev_n: bool,
    pub(super) ef_l: bool,
    pub(super) ef_n: bool,
    pub(super) eft: bool,
}

impl Requested {
    pub(super) fn from_args(args: &[String]) -> Self {
        let cmdline = args.join(" ");
        let given = |flag: &str| command_line_has_option(&cmdline, flag);
        Requested {
            cpu: given("--cpu") || std::env::var_os("HMMER_NCPU").is_some(),
            matrix: given("--mx"),
            mxfile: given("--mxfile"),
            popen: given("--popen"),
            pextend: given("--pextend"),
            wid: given("--wid"),
            eid: given("--eid"),
            em_l: given("--EmL"),
            em_n: given("--EmN"),
            ev_l: given("--EvL"),
            ev_n: given("--EvN"),
            ef_l: given("--EfL"),
            ef_n: given("--EfN"),
            eft: given("--Eft"),
        }
    }
}
pub(super) fn configure_pipeline(pli: &mut Pipeline, args: &Args) {
    pli.e_value_threshold = args.e_value;
    pli.dom_e_value_threshold = args.dom_e;
    pli.inc_e = args.inc_e;
    pli.inc_dome = args.incdom_e;
    pli.do_max = args.max;
    pli.f1 = if args.max { 1.0 } else { args.f1 };
    pli.f2 = if args.max { 1.0 } else { args.f2 };
    pli.f3 = if args.max { 1.0 } else { args.f3 };
    if let Some(z) = args.z_value {
        pli.z = z;
        pli.z_setby = tongs::pipeline::ZSetBy::Option;
    }
    if let Some(domz) = args.domz_value {
        pli.domz = domz;
        pli.domz_setby = tongs::pipeline::ZSetBy::Option;
    }

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
    if let Some(t) = args.incdom_t {
        pli.inc_dom_t = Some(t);
        pli.incdom_by_e = false;
    }
}
