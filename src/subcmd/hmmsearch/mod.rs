#![allow(clippy::if_same_then_else, clippy::too_many_arguments)]

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use rayon::prelude::*;

use tongs::alphabet::Alphabet;
use tongs::logsum;
use tongs::search::{scan_all, Query, Scan};
use tongs::sequence::SequenceFormat;
use tongs::subcmd::cli::{
    command_line_has_option, normalize_table_cmdline, parse_positive_f64, parse_positive_usize,
    parse_textw,
};
use tongs::subcmd::inputs::read_hmms;
use tongs::subcmd::outputs::create_output_file_or_exit;

mod options;
mod pfam;
mod report;
mod stdout;
mod tables;
mod targets;

pub use pfam::write_pfamtblout;
pub(crate) use pfam::write_pfamtblout_with_pipeline;
pub use stdout::write_standard_stdout_tables;
pub(crate) use tables::write_ali_output;
pub use tables::{best_domain, write_domtblout, write_table_footer, write_tblout};
pub(crate) use targets::open_restricted_target_seq_file;

// Targets are held a block at a time, so a large database never sits in memory whole.
const BLOCK_RESIDUES: usize = 1 << 27;

#[derive(Parser)]
#[command(
    name = "hmmsearch",
    about = "Search profile(s) against a sequence database"
)]
struct Args {
    /// HMM file
    hmmfile: PathBuf,
    /// Sequence database (FASTA format)
    seqdb: PathBuf,

    // --- Output options ---
    /// Direct output to file, not stdout
    #[arg(short = 'o')]
    outfile: Option<PathBuf>,

    /// Save per-sequence hits to tabular file
    #[arg(long = "tblout")]
    tblout: Option<PathBuf>,

    /// Save per-domain hits to tabular file
    #[arg(long = "domtblout")]
    domtblout: Option<PathBuf>,

    /// Don't output alignments
    #[arg(long = "noali")]
    noali: bool,

    // --- Reporting thresholds ---
    /// Report sequences <= this E-value threshold
    #[arg(
        short = 'E',
        default_value = "10.0",
        value_parser = parse_positive_f64,
        conflicts_with = "score_threshold"
    )]
    e_value: f64,

    /// Report sequences >= this score threshold
    #[arg(short = 'T', conflicts_with = "e_value", allow_hyphen_values = true)]
    score_threshold: Option<f64>,

    /// Report domains <= this E-value threshold
    #[arg(
        long = "domE",
        default_value = "10.0",
        value_parser = parse_positive_f64,
        conflicts_with = "dom_t"
    )]
    dom_e: f64,

    /// Report domains >= this score threshold
    #[arg(long = "domT", conflicts_with = "dom_e", allow_hyphen_values = true)]
    dom_t: Option<f64>,

    // --- Inclusion thresholds ---
    /// Include sequences <= this E-value threshold
    #[arg(
        long = "incE",
        default_value = "0.01",
        value_parser = parse_positive_f64,
        conflicts_with = "inc_t"
    )]
    inc_e: f64,

    /// Include sequences >= this score threshold
    #[arg(long = "incT", conflicts_with = "inc_e", allow_hyphen_values = true)]
    inc_t: Option<f64>,

    /// Include domains <= this E-value threshold
    #[arg(
        long = "incdomE",
        default_value = "0.01",
        value_parser = parse_positive_f64,
        conflicts_with = "inc_dom_t"
    )]
    inc_dome: f64,

    /// Include domains >= this score threshold
    #[arg(
        long = "incdomT",
        conflicts_with = "inc_dome",
        allow_hyphen_values = true
    )]
    inc_dom_t: Option<f64>,

    // --- Model-specific cutoffs ---
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
    cut_ga: bool,

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
    cut_nc: bool,

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
    cut_tc: bool,

    // --- Acceleration heuristics ---
    /// Skip all filters (run everything through Forward)
    #[arg(long = "max", conflicts_with_all = ["f1", "f2", "f3", "nobias"])]
    max: bool,

    /// MSV threshold
    #[arg(long = "F1", default_value = "0.02", allow_hyphen_values = true)]
    f1: f64,

    /// Viterbi threshold
    #[arg(long = "F2", default_value = "0.001", allow_hyphen_values = true)]
    f2: f64,

    /// Forward threshold
    #[arg(long = "F3", default_value = "1e-5", allow_hyphen_values = true)]
    f3: f64,

    /// Turn off composition bias filter
    #[arg(long = "nobias")]
    nobias: bool,

    // --- Other expert options ---
    /// Turn off biased composition score corrections
    #[arg(long = "nonull2")]
    nonull2: bool,

    /// Set number of comparisons for E-value calculation
    #[arg(short = 'Z', value_parser = parse_positive_f64)]
    z_value: Option<f64>,

    /// Set number of significant seqs for domain E-value calculation
    #[arg(long = "domZ", value_parser = parse_positive_f64)]
    domz_value: Option<f64>,

    /// Set RNG seed (0: one-time arbitrary seed)
    #[arg(long = "seed", default_value = "42")]
    seed: u32,

    /// Number of CPU threads
    #[arg(long = "cpu", default_value = "2")]
    cpu: usize,

    /// Start restricted target search at this sequence key
    #[arg(long = "restrictdb_stkey", hide = true)]
    restrictdb_stkey: Option<String>,

    /// Search only this many target sequences from the restricted start
    #[arg(long = "restrictdb_n", value_parser = parse_positive_usize, hide = true)]
    restrictdb_n: Option<usize>,

    /// SSI index file for C-compatible restricted database options
    #[arg(long = "ssifile", hide = true)]
    ssifile: Option<PathBuf>,

    // --- Output formatting ---
    /// Prefer accessions over names in output
    #[arg(long = "acc")]
    show_acc: bool,

    /// Unlimit ASCII text output line width
    #[arg(long = "notextw", conflicts_with = "textw")]
    notextw: bool,

    /// Set max width of ASCII text output lines
    #[arg(long = "textw", default_value = "120", value_parser = parse_textw)]
    textw: usize,

    /// Save table of hits in Pfam format
    #[arg(long = "pfamtblout")]
    pfamtblout: Option<PathBuf>,

    /// Save multiple alignment of all hits to file
    #[arg(short = 'A')]
    ali_outfile: Option<PathBuf>,

    /// Assert target sequence file format (currently FASTA only)
    #[arg(long = "tformat")]
    tformat: Option<String>,
}

pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let cmdline = args.join(" ");
    let cpu_was_requested =
        command_line_has_option(&cmdline, "--cpu") || std::env::var_os("HMMER_NCPU").is_some();
    let args = tongs::util::apply_hmmer_ncpu_env_default(args);
    let args = Args::parse_from(&args);
    if let Some(format) = args.tformat.as_deref() {
        if SequenceFormat::from_name(format).is_none() {
            eprintln!(
                "hmmsearch --tformat={format} is not a recognized input sequence file format",
            );
            std::process::exit(1);
        }
    }

    // `--cpu 0` is HMMER's explicit serial mode, so it gets one worker like `--cpu 1`.
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.cpu.max(1))
        .start_handler(|_| tongs::util::simd_env::init())
        .build_global()
        .ok();

    logsum::p7_flogsuminit();

    if args.hmmfile.as_path() == Path::new("-") && args.seqdb.as_path() == Path::new("-") {
        eprintln!("Error: Either <hmmfile> or <seqdb> may be '-' but not both");
        std::process::exit(1);
    }

    // Read HMM(s)
    let hmms = read_hmms(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });
    if hmms.len() > 1 && args.seqdb.as_path() == Path::new("-") {
        eprintln!(
            "Error: target sequence file - isn't rewindable; can't search it with multiple queries"
        );
        std::process::exit(1);
    }

    // Output destination: -o file or stdout
    let outfile_handle;
    let stdout;
    let mut out: Box<dyn std::io::Write> = if let Some(ref path) = args.outfile {
        outfile_handle =
            create_output_file_or_exit(path, "Failed to open output file {path} for writing");
        Box::new(BufWriter::new(outfile_handle))
    } else {
        stdout = std::io::stdout();
        Box::new(BufWriter::new(stdout.lock()))
    };

    options::write_header(&mut out, &args, &cmdline, cpu_was_requested);

    // Open tblout/domtblout files if requested
    let tblout_file = args.tblout.as_ref().map(|p| {
        let file = create_output_file_or_exit(
            p,
            "Failed to open tabular per-seq output file {path} for writing",
        );
        BufWriter::new(file)
    });
    let domtblout_file = args.domtblout.as_ref().map(|p| {
        let file = create_output_file_or_exit(
            p,
            "Failed to open tabular per-dom output file {path} for writing",
        );
        BufWriter::new(file)
    });
    let pfamtblout_file = args.pfamtblout.as_ref().map(|p| {
        let file = create_output_file_or_exit(
            p,
            "Failed to open pfam-style tabular output file {path} for writing",
        );
        BufWriter::new(file)
    });
    let ali_outfile = args.ali_outfile.as_ref().map(|p| {
        let file =
            create_output_file_or_exit(p, "Failed to open alignment file {path} for writing");
        BufWriter::new(file)
    });

    // The reader stays on this thread while the pool configures models and scans the block
    // already read, so reading never holds the workers up.
    let mut reader = targets::Reader::open(&args, &Alphabet::new(hmms[0].abc_type));
    let (queries, mut block) = std::thread::scope(|s| {
        let setup = s.spawn(|| {
            hmms.par_iter()
                .map(|hmm| configure(hmm, &args))
                .collect::<Vec<_>>()
        });
        let block = reader.next_block(BLOCK_RESIDUES);
        (setup.join().unwrap(), block)
    });
    let mut scans = queries.iter().map(|_| Scan::default()).collect::<Vec<_>>();
    loop {
        let (first, targets) = match std::mem::replace(&mut block, Ok(None)) {
            Ok(Some(next)) => next,
            Ok(None) => break,
            Err(e) => {
                // C reads targets inside the first query's search, so its header precedes the error.
                report::write_query_header(&mut out, queries[0].hmm);
                out.flush().unwrap();
                eprintln!("Error reading sequence file: {}", e);
                std::process::exit(1);
            }
        };
        let found = std::thread::scope(|s| {
            let scan = s.spawn(|| scan_all(&queries, &targets, first));
            block = reader.next_block(BLOCK_RESIDUES);
            scan.join().unwrap()
        });
        for (scan, more) in scans.iter_mut().zip(found) {
            scan.absorb(more);
        }
    }
    if reader.count == 0 {
        eprintln!("Error: no sequences found in {}", args.seqdb.display());
        std::process::exit(1);
    }

    let mut sinks = report::Sinks {
        out,
        tblout: tblout_file,
        domtblout: domtblout_file,
        pfamtblout: pfamtblout_file,
        ali: ali_outfile,
    };
    for (query_idx, (query, scan)) in queries.iter().zip(scans).enumerate() {
        report::write(
            &mut sinks,
            &args,
            query_idx,
            query,
            scan,
            reader.count,
            reader.residues,
        );
    }
    let table_cmdline = normalize_table_cmdline(&cmdline, &["hmmsearch", "search"]);
    if let Some(f) = sinks.tblout.as_mut() {
        write_table_footer(
            f,
            "hmmsearch",
            "SEARCH",
            &args.hmmfile,
            &args.seqdb,
            &table_cmdline,
        );
    }
    if let Some(f) = sinks.domtblout.as_mut() {
        write_table_footer(
            f,
            "hmmsearch",
            "SEARCH",
            &args.hmmfile,
            &args.seqdb,
            &table_cmdline,
        );
    }
    if let Some(f) = sinks.pfamtblout.as_mut() {
        write_table_footer(
            f,
            "hmmsearch",
            "SEARCH",
            &args.hmmfile,
            &args.seqdb,
            &table_cmdline,
        );
    }

    writeln!(sinks.out, "[ok]").unwrap();
    sinks.out.flush().unwrap();
    if let Some(f) = sinks.tblout.as_mut() {
        f.flush().unwrap();
    }
    if let Some(f) = sinks.domtblout.as_mut() {
        f.flush().unwrap();
    }
    if let Some(f) = sinks.pfamtblout.as_mut() {
        f.flush().unwrap();
    }
    if let Some(f) = sinks.ali.as_mut() {
        f.flush().unwrap();
    }
    std::process::ExitCode::SUCCESS
}

fn configure<'a>(hmm: &'a tongs::Hmm, args: &Args) -> Query<'a> {
    let mut query = Query::new(hmm);
    let pli = &mut query.pli;

    let effective_f1 = if args.max { 1.0 } else { args.f1 };
    let effective_f2 = if args.max { 1.0 } else { args.f2 };
    let effective_f3 = if args.max { 1.0 } else { args.f3 };
    let effective_nobias = args.max || args.nobias;

    // Filter thresholds
    pli.f1 = effective_f1;
    pli.f2 = effective_f2;
    pli.f3 = effective_f3;
    pli.do_max = args.max;
    if effective_nobias {
        pli.do_biasfilter = false;
    }
    if args.nonull2 {
        pli.do_null2 = false;
    }
    pli.seed = args.seed;

    // Reporting thresholds
    pli.e_value_threshold = args.e_value;
    pli.dom_e_value_threshold = args.dom_e;
    pli.inc_e = args.inc_e;
    pli.inc_dome = args.inc_dome;

    // Score-based thresholds (override E-value based)
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

    // Model-specific cutoffs (override both E-value and score thresholds)
    if args.cut_ga {
        pli.use_bit_cutoffs = tongs::pipeline::BitCutoff::GA;
    } else if args.cut_tc {
        pli.use_bit_cutoffs = tongs::pipeline::BitCutoff::TC;
    } else if args.cut_nc {
        pli.use_bit_cutoffs = tongs::pipeline::BitCutoff::NC;
    }
    if let Err(e) = pli.new_model_thresholds(&hmm.cutoff) {
        eprintln!("Error: {} for model {}", e, hmm.name);
        std::process::exit(1);
    }

    // Database size overrides
    if let Some(z) = args.z_value {
        pli.z = z;
        pli.z_setby = tongs::pipeline::ZSetBy::Option;
    }
    if let Some(dz) = args.domz_value {
        pli.domz = dz;
        pli.domz_setby = tongs::pipeline::ZSetBy::Option;
    }
    pli.do_alignment = true;
    pli.do_alignment_display = !args.noali || args.ali_outfile.is_some();
    query
}

#[cfg(test)]
#[path = "../../../tests/unit/subcmd/hmmsearch_test.rs"]
mod tests;
