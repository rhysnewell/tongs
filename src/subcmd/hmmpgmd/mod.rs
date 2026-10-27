use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Read, Write};
use std::net::{SocketAddrV4, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use clap::Parser;
use socket2::{Domain as SocketDomain, Protocol, SockAddr, Socket, Type};

use tongs::alphabet::{Alphabet, AlphabetType};
use tongs::bg::Bg;
use tongs::hmm::{Hmm, P7H_CA, P7H_CONS, P7H_CS, P7H_MAP, P7H_MMASK, P7H_RF};
use tongs::hmmfile;
use tongs::logsum;
use tongs::pipeline::{BitCutoff, Pipeline};
use tongs::profile::{self, Profile, P7_LOCAL};
use tongs::seqmodel;
use tongs::sequence::{self, Sequence};
use tongs::simd::oprofile::OProfile;
use tongs::subcmd::cli::parse_positive_usize;
use tongs::tophits::{AliDisplay, Domain, Hit, TopHits};

mod db;
use db::*;
mod master;
use master::*;
mod query;
use query::*;
mod shard;
use shard::*;
mod requests;
use requests::*;
mod merge;
use merge::*;
mod payload;
use payload::*;
mod commands;
use commands::*;
mod options;
use options::*;
mod serialize;
use serialize::*;
mod worker;
use worker::*;

#[derive(Parser)]
#[command(name = "hmmpgmd", about = "HMMER search daemon")]
struct Args {
    /// Run as the master server
    #[arg(long = "master", conflicts_with = "worker")]
    master: bool,

    /// Run as a worker connected to master host
    #[arg(long = "worker", value_name = "HOST", conflicts_with = "master")]
    worker: Option<String>,

    /// HMM database file to serve
    ///
    /// C `hmmpgmd.c` declares `--hmmdb` with `incomp = "--worker"` only; it is
    /// NOT mutually exclusive with `--seqdb` (the master may cache both DBs at
    /// once — the canonical `--master --seqdb X --hmmdb Y` invocation). This
    /// port additionally lets a worker preload a DB from the CLI (the C worker
    /// instead receives its DB over the wire), so the C `incomp "--worker"`
    /// constraint is intentionally not applied here.
    #[arg(long = "hmmdb")]
    hmmdb: Option<PathBuf>,

    /// Protein sequence database file to serve
    ///
    /// C `hmmpgmd.c` declares `--seqdb` with `incomp = "--worker"` only; it is
    /// NOT mutually exclusive with `--hmmdb`. See `--hmmdb` for why the
    /// `incomp "--worker"` constraint is not applied in this port.
    #[arg(long = "seqdb")]
    seqdb: Option<PathBuf>,

    /// Legacy alias for --cport
    #[arg(long = "port", value_parser = parse_port)]
    port: Option<u16>,

    /// Port to use for client/server communication
    #[arg(long = "cport", default_value = "51371", value_parser = parse_port, conflicts_with = "worker")]
    cport: u16,

    /// Port to use for server/worker communication
    #[arg(long = "wport", default_value = "51372", value_parser = parse_port)]
    wport: u16,

    /// File to write the process id to
    #[arg(long = "pid")]
    pid: Option<PathBuf>,

    /// Maximum client-side listen backlog for master mode
    #[arg(long = "ccncts", default_value = "16", value_parser = parse_positive_usize, conflicts_with = "worker")]
    ccncts: usize,

    /// Maximum worker-side listen backlog for master mode
    #[arg(long = "wcncts", default_value = "32", value_parser = parse_positive_usize, conflicts_with = "worker")]
    wcncts: usize,

    /// Number of parallel worker CPU threads
    #[arg(long = "cpu", default_value = "2", value_parser = parse_positive_usize, conflicts_with = "master")]
    cpu: usize,
}

impl Args {
    fn client_port(&self) -> u16 {
        self.port.unwrap_or(self.cport)
    }
}

// C checks 49151<n<65536 on a signed int (hmmpgmd.c --cport/--wport).
// So a value outside u16 gets the range message, not a type error.
fn parse_port(s: &str) -> Result<u16, String> {
    let value = s
        .parse::<i64>()
        .map_err(|_| format!("takes integer arg in range 49151<n<65536; got {s}"))?;
    if value > 49151 && value < 65536 {
        Ok(value as u16)
    } else {
        Err(format!("takes integer arg in range 49151<n<65536; got {s}"))
    }
}

const HMMD_CMD_SEARCH: u32 = 10001;
const HMMD_CMD_SCAN: u32 = 10002;
const HMMD_CMD_INIT: u32 = 10003;
const HMMD_CMD_SHUTDOWN: u32 = 10004;
const HMMD_HEADER_SIZE: usize = 12;
const HMMD_INIT_BODY_SIZE: usize = 88;
const HMMD_INIT_RESPONSE_SIZE: usize = 96;
const HMMD_SEARCH_CMD_FIXED_SIZE: usize = 28;
const C_HMMD_COMMAND_SEARCH_PADDING_SIZE: usize = 60;
const HMMD_SEARCH_STATS_SIZE: usize = 122;
const C_P7_HMM_SHELL_SIZE: usize = 296;
const P7_HIT_BASE_SIZE: usize = 109;
const P7_DOMAIN_BASE_SIZE: usize = 92;
const P7_ALIDISPLAY_BASE_SIZE: usize = 45;
const HMMD_SEQUENCE: u32 = 101;
const HMMD_HMM: u32 = 102;
const P7_HIT_ACC_PRESENT: u8 = 1 << 0;
const P7_HIT_DESC_PRESENT: u8 = 1 << 1;
const P7_ALIDISPLAY_RFLINE_PRESENT: u8 = 1 << 0;
const P7_ALIDISPLAY_MMLINE_PRESENT: u8 = 1 << 1;
const P7_ALIDISPLAY_CSLINE_PRESENT: u8 = 1 << 2;
const P7_ALIDISPLAY_PPLINE_PRESENT: u8 = 1 << 3;
const P7_ALIDISPLAY_ASEQ_PRESENT: u8 = 1 << 4;
const P7_ALIDISPLAY_NTSEQ_PRESENT: u8 = 1 << 5;
const MAX_HMMD_HMM_QUERY_LENGTH: usize = 1_000_000;
const MAX_HMMPGMD_INIT_BODY: usize = 16 * 1024 * 1024;
const MAX_HMMPGMD_SEARCH_BODY: usize = 64 * 1024 * 1024;
const MAX_HMMPGMD_SEARCH_RESPONSE_BODY: usize = 512 * 1024 * 1024;
const MAX_HMMPGMD_SHUTDOWN_BODY: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug)]
struct HmmdHeader {
    length: u32,
    command: u32,
    status: u32,
}

#[derive(Debug)]
struct ClientCommand {
    db_mode: DbMode,
    query: ClientQuery,
    options: SearchOptions,
    seqdb_ranges: Option<Vec<DbRange>>,
    db_slice: Option<DbSlice>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DbMode {
    Hmm,
    Seq,
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum ClientQuery {
    Sequence {
        name: String,
        desc: String,
        seq: String,
    },
    Hmm(Hmm),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DbRange {
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug)]
struct DbSlice {
    start: usize,
    count: usize,
}

#[derive(Clone, Copy, Debug)]
struct SearchOptions {
    e_value_threshold: f64,
    dom_e_value_threshold: f64,
    inc_e: f64,
    inc_dome: f64,
    t: Option<f64>,
    dom_t: Option<f64>,
    inc_t: Option<f64>,
    inc_dom_t: Option<f64>,
    bit_cutoff: BitCutoff,
    max: bool,
    f1: f64,
    f2: f64,
    f3: f64,
    nobias: bool,
    nonull2: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            e_value_threshold: 10.0,
            dom_e_value_threshold: 10.0,
            inc_e: 0.01,
            inc_dome: 0.01,
            t: None,
            dom_t: None,
            inc_t: None,
            inc_dom_t: None,
            bit_cutoff: BitCutoff::None,
            max: false,
            f1: 0.02,
            f2: 1e-3,
            f3: 1e-5,
            nobias: false,
            nonull2: false,
        }
    }
}

#[derive(Debug)]
struct SearchHit {
    name: String,
    acc: String,
    desc: String,
    window_length: i32,
    sortkey: f64,
    score: f32,
    pre_score: f32,
    sum_score: f32,
    lnp: f64,
    pre_lnp: f64,
    sum_lnp: f64,
    nexpected: f32,
    nregions: i32,
    nclustered: i32,
    noverlaps: i32,
    nenvelopes: i32,
    flags: u32,
    nreported: i32,
    nincluded: i32,
    best_domain: i32,
    seqidx: i64,
    subseq_start: i64,
    hmm_name: String,
    hmm_acc: String,
    hmm_desc: String,
    seq_name: String,
    seq_acc: String,
    seq_desc: String,
    model_length: i32,
    sequence_length: i64,
    domains: Vec<Domain>,
}

enum ServerState {
    Hmm(HmmDbState),
    Seq(SeqDbState),
}

type WorkerPool = Arc<Mutex<Vec<WorkerConnection>>>;

struct WorkerConnection {
    stream: TcpStream,
}

#[derive(Clone, Copy, Default)]
struct SearchRuntime<'a> {
    thread_pool: Option<&'a rayon::ThreadPool>,
}

impl SearchRuntime<'_> {
    fn is_parallel(self) -> bool {
        self.thread_pool.is_some()
    }
}

struct HmmDbState {
    path: PathBuf,
    hmms: Vec<Hmm>,
    abc: Alphabet,
    bg: Bg,
    profiles: Vec<(Profile, OProfile)>,
}

struct SeqDbState {
    path: PathBuf,
    cache_id: String,
    sequences: Vec<Sequence>,
    abc: Alphabet,
    bg: Bg,
}

// The legacy one-line sequence protocol stays for existing tests and manual use.
pub fn run(args: Vec<String>) -> ExitCode {
    let args = Args::parse_from(&args);

    if let Some(pid) = &args.pid {
        if let Err(e) = std::fs::write(pid, format!("{}\n", std::process::id())) {
            eprintln!(
                "Unable to open PID file {} for writing: {}",
                pid.display(),
                e
            );
            std::process::exit(1);
        }
    }

    if let Some(host) = &args.worker {
        let state = match (&args.hmmdb, &args.seqdb) {
            (Some(hmmdb), None) => Some(load_hmmdb(hmmdb)),
            (None, Some(seqdb)) => Some(load_seqdb(seqdb)),
            (None, None) => None,
            (Some(_), Some(_)) => unreachable!("clap rejects conflicting served databases"),
        };
        return run_worker(host, args.wport, args.cpu, state);
    }

    if !args.master {
        eprintln!("hmmpgmd compatibility mode: treating invocation as --master");
    }

    match (&args.hmmdb, &args.seqdb) {
        (None, None) => {
            eprintln!("hmmpgmd requires --hmmdb or --seqdb");
            std::process::exit(1);
        }
        (Some(_), Some(_)) => {
            eprintln!("hmmpgmd accepts one served database at a time: use --hmmdb or --seqdb");
            std::process::exit(1);
        }
        (Some(hmmdb), None) => run_master(
            load_hmmdb(hmmdb),
            args.client_port(),
            args.wport,
            args.ccncts,
            args.wcncts,
        ),
        (None, Some(seqdb)) => run_master(
            load_seqdb(seqdb),
            args.client_port(),
            args.wport,
            args.ccncts,
            args.wcncts,
        ),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/subcmd/hmmpgmd_test.rs"]
mod tests;
