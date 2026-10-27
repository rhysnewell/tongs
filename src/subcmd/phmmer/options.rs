use super::*;

#[derive(Parser)]
#[command(
    name = "phmmer",
    about = "Search a protein sequence against a protein database"
)]
pub(super) struct Args {
    /// Query sequence file (FASTA)
    pub(super) seqfile: PathBuf,
    /// Target sequence database (FASTA)
    pub(super) seqdb: PathBuf,

    /// Direct output to file, not stdout
    #[arg(short = 'o')]
    pub(super) output: Option<PathBuf>,

    /// Assert query sequence file format
    #[arg(long = "qformat")]
    pub(super) qformat: Option<String>,

    /// Assert target sequence file format
    #[arg(long = "tformat")]
    pub(super) tformat: Option<String>,

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
        default_value = "0.01",
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

    /// Hidden C-compatible model gathering cutoff flag
    #[arg(
        long = "cut_ga",
        hide = true,
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

    /// Hidden C-compatible model noise cutoff flag
    #[arg(
        long = "cut_nc",
        hide = true,
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

    /// Hidden C-compatible model trusted cutoff flag
    #[arg(
        long = "cut_tc",
        hide = true,
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

    /// Gap open probability
    #[arg(long = "popen", default_value = "0.02", value_parser = parse_popen)]
    pub(super) popen: f64,

    /// Gap extend probability
    #[arg(long = "pextend", default_value = "0.4", value_parser = parse_pextend)]
    pub(super) pextend: f64,

    /// Substitution score matrix choice
    #[arg(long = "mx", default_value = "BLOSUM62", conflicts_with = "mxfile")]
    pub(super) matrix: String,

    /// Read substitution score matrix from file
    #[arg(long = "mxfile")]
    pub(super) mxfile: Option<PathBuf>,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    pub(super) cpu: usize,

    /// Turn all heuristic filters off
    #[arg(long = "max", conflicts_with_all = ["f1", "f2", "f3", "nobias"])]
    pub(super) max: bool,

    /// MSV threshold
    #[arg(long = "F1", default_value = "0.02", allow_hyphen_values = true)]
    pub(super) f1: f64,

    /// Viterbi threshold
    #[arg(long = "F2", default_value = "0.001", allow_hyphen_values = true)]
    pub(super) f2: f64,

    /// Forward threshold
    #[arg(long = "F3", default_value = "1e-5", allow_hyphen_values = true)]
    pub(super) f3: f64,

    /// Turn off composition bias filter
    #[arg(long = "nobias")]
    pub(super) nobias: bool,

    /// Turn off biased composition score corrections
    #[arg(long = "nonull2")]
    pub(super) nonull2: bool,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    pub(super) tblout: Option<PathBuf>,

    /// Save per-domain hits to tabular file
    #[arg(long = "domtblout")]
    pub(super) domtblout: Option<PathBuf>,

    /// Save Pfam-style table of hits and domains
    #[arg(long = "pfamtblout")]
    pub(super) pfamtblout: Option<PathBuf>,

    /// Save multiple alignment of hits to file
    #[arg(short = 'A')]
    pub(super) ali_outfile: Option<PathBuf>,

    /// Omit alignments from the main output
    #[arg(long = "noali")]
    pub(super) noali: bool,

    /// Prefer accessions over names in output
    #[arg(long = "acc")]
    pub(super) show_acc: bool,

    /// Do not line-wrap the main text output
    #[arg(long = "notextw", conflicts_with = "textw")]
    pub(super) notextw: bool,

    /// Set the target line width for the main text output
    #[arg(long = "textw", default_value = "120", value_parser = parse_textw)]
    pub(super) textw: usize,

    /// Set number of comparisons for E-value calculation
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    pub(super) z_value: Option<f64>,

    /// Set number of significant seqs for domain E-value calculation
    #[arg(long = "domZ", value_parser = parse_positive_f64)]
    pub(super) domz_value: Option<f64>,

    /// Set RNG seed
    #[arg(long = "seed", default_value = "42")]
    pub(super) seed: u32,

    /// Start restricted target search at this sequence key
    #[arg(long = "restrictdb_stkey", hide = true)]
    pub(super) restrictdb_stkey: Option<String>,

    /// Search only this many target sequences from the restricted start
    #[arg(long = "restrictdb_n", value_parser = parse_positive_usize, hide = true)]
    pub(super) restrictdb_n: Option<usize>,

    /// SSI index file for C-compatible restricted database options
    #[arg(long = "ssifile", hide = true)]
    pub(super) ssifile: Option<PathBuf>,

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

fn parse_textw(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid text width: {e}"))?;
    if value >= 120 {
        Ok(value)
    } else {
        Err("--textw must be >= 120".to_string())
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
    pub(super) cpu: bool,
    pub(super) matrix: bool,
    pub(super) mxfile: bool,
    pub(super) popen: bool,
    pub(super) pextend: bool,
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
        let given = |flag: &str| {
            args.iter().any(|arg| {
                arg == flag
                    || arg
                        .strip_prefix(flag)
                        .is_some_and(|rest| rest.starts_with('='))
            })
        };
        Requested {
            cpu: given("--cpu") || std::env::var_os("HMMER_NCPU").is_some(),
            matrix: given("--mx"),
            mxfile: given("--mxfile"),
            popen: given("--popen"),
            pextend: given("--pextend"),
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
    // Model-specific thresholding (--cut_ga/--cut_tc/--cut_nc) is unused in phmmer:
    // the query model is built from a single sequence and never carries GA/TC/NC
    // cutoffs. C (hmmer/src/phmmer.c) still configures the pipeline for these
    // options (p7_pipeline.c:293-316): it sets by_E=FALSE and T=incT=domT=incdomT=0.0
    // and use_bit_cutoffs=p7H_*, then calls p7_pli_NewModel() WITHOUT checking its
    // return status (phmmer.c:565). The subsequent p7_pli_NewModelThresholds() fails
    // silently with eslEINVAL ("bit thresholds unavailable"), leaving T at 0.0, so the
    // effective behavior is "report/include by score >= 0.0". We replicate that exactly
    // here, and (unlike hmmsearch/hmmscan) phmmer never calls new_model_thresholds().
    let cutoff = if args.cut_ga {
        Some(tongs::pipeline::BitCutoff::GA)
    } else if args.cut_tc {
        Some(tongs::pipeline::BitCutoff::TC)
    } else if args.cut_nc {
        Some(tongs::pipeline::BitCutoff::NC)
    } else {
        None
    };
    if let Some(bc) = cutoff {
        pli.use_bit_cutoffs = bc;
        pli.by_e = false;
        pli.dom_by_e = false;
        pli.inc_by_e = false;
        pli.incdom_by_e = false;
        pli.t = Some(0.0);
        pli.dom_t = Some(0.0);
        pli.inc_t = Some(0.0);
        pli.inc_dom_t = Some(0.0);
    }
}

pub(super) fn configure_search_spaces(pli: &mut Pipeline, args: &Args) {
    if let Some(z) = args.z_value {
        pli.z = z;
        pli.z_setby = tongs::pipeline::ZSetBy::Option;
    }
    if let Some(domz) = args.domz_value {
        pli.domz = domz;
        pli.domz_setby = tongs::pipeline::ZSetBy::Option;
    }
}

pub(super) fn threshold_pipeline_for_output(
    args: &Args,
    effective_f1: f64,
    effective_f2: f64,
    effective_f3: f64,
    effective_biasfilter: bool,
) -> Pipeline {
    let mut pli = Pipeline::new();
    configure_thresholds(&mut pli, args);
    configure_search_spaces(&mut pli, args);
    pli.f1 = effective_f1;
    pli.f2 = effective_f2;
    pli.f3 = effective_f3;
    pli.do_max = args.max;
    pli.do_biasfilter = effective_biasfilter;
    pli.do_null2 = !args.nonull2;
    pli
}
