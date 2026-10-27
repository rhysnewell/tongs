use std::io::{BufReader, Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use clap::Parser;

use tongs::alphabet::{Alphabet, AlphabetType, DSQ_SENTINEL};
use tongs::bg::Bg;
use tongs::builder::Builder;
use tongs::builder::{self, DEFAULT_WINDOW_BETA};
use tongs::calibrate::CalibrationConfig;
use tongs::fm_index::{FmIndex, FmInterval, MmapBytes};
use tongs::hmm::{self as p7hmm, Hmm};
use tongs::logsum;
use tongs::msa;
use tongs::output::{fmt_evalue, fmt_g, fmt_g3};
use tongs::pipeline::Pipeline;
use tongs::prior::PriorStrategy;
use tongs::profile::{self, Profile, P7_LOCAL};
use tongs::sequence::{self, Sequence};
use tongs::simd::oprofile::OProfile;
use tongs::subcmd::cli::{
    command_line_has_option, normalize_table_cmdline, parse_positive_f64, parse_positive_usize,
    parse_textw, parse_window_beta, parse_window_length,
};
use tongs::subcmd::inputs::read_hmms;
use tongs::subcmd::outputs::create_file_or_exit;
use tongs::tophits::TopHits;
use tongs::util::cmath::c_logf_to_f32;
use tongs::util::cmath::{c_exp_f64, c_log_f64};

mod options;
use options::*;
mod header;
use header::*;
mod input;
use input::*;
mod query_msa;
use query_msa::*;
mod target;
pub(crate) use target::*;
mod fm_records;
use fm_records::*;
mod c_stream;
use c_stream::*;
mod fm_windows;
pub(crate) use fm_windows::*;
mod fm_extend;
use fm_extend::*;
mod ssv_windows;
use ssv_windows::*;
mod seed_query;
use seed_query::*;
mod seed_strings;
use seed_strings::*;
mod model_seeds;
use model_seeds::*;
mod score_seeds;
use score_seeds::*;
mod search;
pub(crate) use search::*;
mod pipeline;
use pipeline::*;
mod thresholds;
pub(crate) use thresholds::*;
mod report;
use report::*;
mod tables;
pub(crate) use tables::*;

const MAKEHMMERDB_INDEX_MAGIC: &[u8] = b"HMMERDB_INDEXES\0";
const NHMMER_FM_SEED_MAX_DEPTH: usize = 15;
const NHMMER_FM_SEED_SCORE_THRESHOLD_BITS: f32 = 14.0;
const NHMMER_FM_SEED_SCORE_DENSITY_BITS: f32 = 0.75;
const NHMMER_FM_SEED_DROP_MAX_LEN: usize = 4;
const NHMMER_FM_SEED_DROP_LIMIT_BITS: f32 = 0.3;
const NHMMER_FM_SEED_CONSEC_POS_REQ: usize = 5;
const NHMMER_FM_SCORE_SEED_LIMIT: usize = 256;
const NHMMER_FM_SCORE_TRIE_NORMAL_START_LIMIT: usize = 512;
const NHMMER_FM_SCORE_TRIE_NORMAL_TEXT_LIMIT: usize = 4096;
const NHMMER_FM_CONSENSUS_MATCH_REQ: usize = 11;
const NHMMER_FM_SEED_SSV_LENGTH: usize = 100;
const NHMMER_DEFAULT_BLOCK_LENGTH: usize = 1024 * 256;
const MAKEHMMERDB_INDEX_MIN_RECORD_BYTES: usize = 64 + 4 + 24 + 256 * 8;
const MAKEHMMERDB_C_MIN_SEQUENCE_META_BYTES: usize = 4 + 8 + 4 + 4 + 2 + 2 + 2 + 2 + 4;
const MAKEHMMERDB_C_AMBIGUITY_BYTES: usize = 8;
const MAKEHMMERDB_MAX_IN_MEMORY_BYTES: u64 = 16 * 1024 * 1024 * 1024;
#[cfg(test)]
const NHMMER_FM_WINDOW_LIMIT_REGRESSION_SIZE: usize = 4096;
const NHMMER_FM_WINDOW_SOURCE_LIMIT: usize = 4096;
const NHMMER_FM_MIN_EXTENDED_DIAG_SCORE_BITS: f32 = 0.0;

pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let cmdline = args.join(" ");
    let table_cmdline = normalize_table_cmdline(&cmdline, &["nhmmer"]);
    let requested = Requested::from_args(&args);
    let args = tongs::util::apply_hmmer_ncpu_env_default(args);
    let args = Args::parse_from(&args);

    validate_args(&args);

    logsum::p7_flogsuminit();

    if args.cpu > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.cpu)
            .start_handler(|_| tongs::util::simd_env::init())
            .build_global()
            .ok();
    }

    if args.hmmfile.as_path() == Path::new("-") && args.seqdb.as_path() == Path::new("-") {
        eprintln!("Error: Either <query file> or <seqdb> may be '-' but not both");
        std::process::exit(1);
    }

    let hmms = read_query_hmms(&args).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    if hmms.len() > 1 && args.seqdb.as_path() == Path::new("-") {
        eprintln!(
            "Error: target sequence file - isn't rewindable; can't search it with multiple queries"
        );
        std::process::exit(1);
    }

    // Output destination
    let outfile_handle;
    let stdout;
    let mut out: Box<dyn std::io::Write> = if let Some(ref path) = args.outfile {
        outfile_handle = create_file_or_exit(path, "output file");
        Box::new(std::io::BufWriter::new(outfile_handle))
    } else {
        stdout = std::io::stdout();
        Box::new(stdout.lock())
    };

    output_header(&mut out, &args, &requested, &cmdline);

    // Alignment display line width. Mirrors C nhmmer.c:571-572:
    // `textw=0` means unlimited (single-block display), otherwise --textw.
    let linewidth: usize = if args.notextw { 0 } else { args.textw };

    // Strand selection
    let do_watson = !args.crick; // top strand unless --crick only
    let do_crick = !args.watson; // bottom strand unless --watson only

    let mut tblout_file = args
        .tblout
        .as_ref()
        .map(|p| create_file_or_exit(p, "tblout file"));
    let mut dfamtblout_file = args
        .dfamtblout
        .as_ref()
        .map(|p| create_file_or_exit(p, "dfamtblout file"));
    let mut ali_file = args
        .ali_outfile
        .as_ref()
        .map(|p| create_file_or_exit(p, "alignment output file"));
    let mut aliscores_file = args
        .aliscoresout
        .as_ref()
        .map(|p| create_file_or_exit(p, "alignment scores output file"));
    let mut hmmout_file = args
        .hmmout
        .as_ref()
        .map(|p| create_file_or_exit(p, "hmm output file"));
    for (query_idx, hmm) in hmms.iter().enumerate() {
        let abc = query_alphabet(&args, hmm);
        let mut bg = Bg::new(&abc);
        if let Some(path) = &args.bgfile {
            if let Err(e) = bg.read_file(&abc, path) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        // C's nhmmer.c calls p7_bg_SetFilter here.
        // The default fhmm values give nonsense long-target bias scores.
        bg.set_filter(hmm.m, &hmm.compo);

        let mut gm = Profile::new(hmm.m, &abc);
        profile::profile_config(hmm, &bg, &mut gm, 400, P7_LOCAL);
        let om = OProfile::convert(&gm);

        let mut pli = Pipeline::new();
        pli.new_model(&gm);
        // --max forces only F2 and F3, since C makes it exclusive with --F1 (nhmmer.c:1020-1029).
        // F1 is resolved below, once the target type is known.
        let effective_f2 = if args.max { 1.0 } else { args.f2 };
        let effective_f3 = if args.max { 1.0 } else { args.f3 };
        let effective_nobias = args.max || args.nobias;

        pli.f2 = effective_f2;
        pli.f3 = effective_f3;
        pli.e_value_threshold = args.e_value;
        pli.inc_e = args.inc_e;
        pli.do_max = args.max;
        pli.seed = args.seed;
        if effective_nobias {
            pli.do_biasfilter = false;
        }
        if args.nonull2 {
            pli.do_null2 = false;
        }

        if let Some(t) = args.score_threshold {
            pli.t = Some(t);
            pli.by_e = false;
        }
        if let Some(t) = args.inc_t {
            pli.inc_t = Some(t);
            pli.inc_by_e = false;
        }

        // Model-specific cutoffs
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

        // Database size override (nhmmer uses Megabases)
        if let Some(z) = args.z_value {
            pli.z = z * 1_000_000.0; // convert Mb to bases
            pli.z_setby = tongs::pipeline::ZSetBy::Option;
        }

        writeln!(out, "Query:       {}  [M={}]", hmm.name, hmm.m).unwrap();
        if let Some(acc) = hmm.acc.as_deref() {
            if !acc.is_empty() {
                writeln!(out, "Accession:   {}", acc).unwrap();
            }
        }
        if let Some(desc) = hmm.desc.as_deref() {
            if !desc.is_empty() {
                writeln!(out, "Description: {}", desc).unwrap();
            }
        }
        out.flush().unwrap_or_else(|e| {
            eprintln!("Error writing output: {}", e);
            std::process::exit(1);
        });

        let target_db = read_target_database(&args, &abc).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        });
        let sequences = &target_db.sequences;
        if sequences.is_empty() {
            eprintln!("Error: no sequences found in {}", args.seqdb.display());
            std::process::exit(1);
        }

        use rayon::prelude::*;

        let max_length = nhmmer_max_length(hmm, args.w_length, args.w_beta);
        let mut search_hmm = hmm.clone();
        search_hmm.max_length = max_length;
        let hmm = &search_hmm;
        if let Some(ref mut f) = hmmout_file {
            tongs::hmmfile::write_hmm(f, hmm).unwrap_or_else(|e| {
                eprintln!("Error writing hmm output: {}", e);
                std::process::exit(1);
            });
        }
        // C `nhmmer.c:1020-1029`: when --F1 isn't on the command line, F1 is the
        // method default (0.03 for FM-index targets, 0.02 for FASTA), regardless
        // of --max.
        let f1 = if command_line_has_option(&cmdline, "--F1") {
            args.f1
        } else if !target_db.fm_index_records.is_empty() {
            0.03
        } else {
            0.02
        };
        pli.f1 = f1;
        let f2 = effective_f2;
        let f3 = effective_f3;
        let do_max = args.max;
        let nobias = effective_nobias;
        let nonull2 = args.nonull2;
        let seed = args.seed;
        let bias_windows = NhmmerBiasWindowLengths {
            b1: args.b1,
            b2: args.b2,
            b3: args.b3,
        };
        let threshold_config = NhmmerThresholdConfig::from_pipeline(&pli);
        let fm_seed_config = NhmmerFmSeedConfig::from_args(&args);
        let effective_block_length = args.block_length.unwrap_or(NHMMER_DEFAULT_BLOCK_LENGTH);
        let fm_window_cache = precompute_c_faithful_fm_windows(
            &target_db,
            sequences,
            hmm,
            &bg,
            max_length,
            f1,
            fm_seed_config,
        );

        // Atomic counters for pipeline filter residues, aggregated across
        // threads and per-window sub-pipelines.
        let msv_counter = AtomicU64::new(0);
        let bias_counter = AtomicU64::new(0);
        let vit_counter = AtomicU64::new(0);
        let fwd_counter = AtomicU64::new(0);

        let search = TargetSearch {
            do_watson,
            do_crick,
            fm_window_cache: &fm_window_cache,
            target_db: &target_db,
            hmm,
            gm: &gm,
            om: &om,
            bg: &bg,
            abc: &abc,
            max_length,
            f1,
            f2,
            f3,
            do_max,
            nobias,
            nonull2,
            seed,
            bias_windows,
            threshold_config,
            fm_seed_config,
            effective_block_length,
            msv_counter: &msv_counter,
            bias_counter: &bias_counter,
            vit_counter: &vit_counter,
            fwd_counter: &fwd_counter,
        };

        let all_hits: Vec<Result<Vec<tongs::tophits::Hit>, String>> = sequences
            .par_iter()
            .enumerate()
            .map(|(seq_idx, sq)| search_target(&search, seq_idx, sq))
            .collect();
        let all_hits: Vec<tongs::tophits::Hit> = all_hits
            .into_iter()
            .flat_map(|result: Result<Vec<_>, String>| {
                result.unwrap_or_else(|err| {
                    eprintln!("Error: {err}");
                    std::process::exit(1);
                })
            })
            .collect();

        let mut th = TopHits::new();
        th.hits = all_hits;
        pli.n_targets = sequences.len() as u64;

        // nhmmer E-value space is in residues, not sequences. Count each strand
        // searched once (C HMMER reports this as "residues searched").
        let total_residues: usize = sequences.iter().map(|s| s.n).sum();
        let strand_multiplier = if target_db.fm_index_records.is_empty() {
            (do_watson as usize) + (do_crick as usize)
        } else {
            // C's FM-index path keeps the full top+bottom database-size
            // denominator even for --watson/--crick-only searches.
            2
        };
        let residues_searched = (total_residues * strand_multiplier.max(1)) as f64;
        let evalue_residues = args
            .z_value
            .map(|z_mb| {
                let strands = if !target_db.fm_index_records.is_empty() || (do_watson && do_crick) {
                    2.0
                } else {
                    1.0
                };
                z_mb * 1_000_000.0 * strands
            })
            .unwrap_or(residues_searched);

        // C folds the database size into lnP as log(N/W), with W = om->max_length.
        // So the E-value is exp(lnP) (p7_tophits_ComputeNhmmerEvalues).
        let window_length = max_length as f64;
        // C divides in float and sets dcl[0].lnP rather than adding (p7_tophits.c:796).
        // sortkey is lnP, not C's -lnP, because sort_by_sortkey sorts ascending.
        let nw_ratio = (evalue_residues as f32) / (window_length as f32);
        let nhmmer_ln_nw = c_log_f64(nw_ratio as f64);
        for hit in &mut th.hits {
            hit.lnp += nhmmer_ln_nw;
            hit.sortkey = hit.lnp;
            if let Some(dom0) = hit.dcl.first_mut() {
                dom0.lnp = hit.lnp;
            }
        }

        // After nhmmer adjustment, evalue = exp(lnp) directly (no × Z).
        let z = 1.0_f64;
        pli.z = residues_searched; // preserve residue count for stats output

        // Mirror C nhmmer.c flow: SortBySeqidxAndAlipos → RemoveDuplicates
        // → sort by sortkey → threshold. The positional sort lets
        // RemoveDuplicates detect adjacent-duplicate pairs.
        th.sort_by_seqidx_and_alipos();
        th.remove_duplicates();
        th.sort_by_sortkey();
        th.threshold(&pli, z, z);

        write_hit_scores(&mut out, &th, &args);

        write_hit_annotations(&mut out, &th, hmm, &args, linewidth);

        write_pipeline_statistics(
            &mut out,
            hmm,
            sequences,
            residues_searched,
            &th,
            &pli,
            &msv_counter,
            &bias_counter,
            &vit_counter,
            &fwd_counter,
        );

        // Tblout
        if let Some(ref mut f) = tblout_file {
            write_nhmmer_tblout(
                f,
                "nhmmer",
                "SEARCH",
                &hmm.name,
                hmm.acc.as_deref(),
                &th,
                &args.hmmfile,
                &args.seqdb,
                &table_cmdline,
                query_idx == 0,
                query_idx + 1 == hmms.len(),
                1.0,
                " sq len",
            );
        }
        if let Some(ref mut f) = dfamtblout_file {
            let qacc = search_dfam_query_accession(hmm.acc.as_deref());
            write_nhmmer_dfamtblout(f, &hmm.name, Some(qacc), "sq-len", &th, 1.0);
        }
        if let Some(ref mut f) = aliscores_file {
            write_nhmmer_aliscoresout(f, &hmm.name, &th, &gm, &abc);
        }
        if let Some(ref mut f) = ali_file {
            write_included_alignment(&mut out, f, &th, &abc, hmm, &args);
        }
        writeln!(out, "//").unwrap();
    }

    writeln!(out, "[ok]").unwrap();
    std::process::ExitCode::SUCCESS
}

#[cfg(test)]
#[path = "../../../tests/unit/subcmd/nhmmer_test.rs"]
mod tests;
