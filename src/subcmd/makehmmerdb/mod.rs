//! makehmmerdb — create an FM-index database for nhmmer.

use std::io::{Read, Write};
use std::path::PathBuf;

use clap::Parser;
use divsufsort::sort_in_place;

use tongs::alphabet::Alphabet;
use tongs::fm_index::FmIndex;
use tongs::sequence::{self, Sequence};

mod input;
use input::*;
mod types;
use types::*;
mod blocks;
use blocks::*;
mod write;
use write::*;
mod c_stream;
use c_stream::*;

const META_MAGIC: &[u8] = b"HMMERDB_META\0";
const META_VERSION: u32 = 1;
const C_META_MAGIC: &[u8] = b"HMMERDB_C_META\0";
const C_META_VERSION: u32 = 1;
const C_STREAM_MAGIC: &[u8] = b"HMMERDB_C_STREAM\0";
const C_STREAM_VERSION: u32 = 1;
const INDEX_MAGIC: &[u8] = b"HMMERDB_INDEXES\0";
const INDEX_VERSION: u32 = 1;
const INDEX_KIND_FORWARD_STRAND: u32 = 0;
const INDEX_KIND_REVERSE_STRAND: u32 = 1;
const FM_BLOCK_OVERLAP: usize = 20_000;
const FM_DNA_ALPH_TYPE: u8 = 0;
const FM_DNA_ALPH_SIZE: u8 = 4;
const FM_DNA_CHAR_BITS: u8 = 2;
const FM_AMINO_ALPH_TYPE: u8 = 4;
const FM_AMINO_ALPH_SIZE: u8 = 26;
const FM_AMINO_CHAR_BITS: u8 = 5;
const FM_FREQ_CNT_SB: u32 = 65_536;
const FM_AMINO_ALPHABET: &[u8; FM_AMINO_ALPH_SIZE as usize] = b"ACDEFGHIKLMNPQRSTVWYBJZOUX";

#[derive(Parser)]
#[command(name = "makehmmerdb", about = "Create an FM-index database for nhmmer")]
struct Args {
    /// Input sequence file (FASTA)
    seqfile: PathBuf,

    /// Output database file
    binaryfile: PathBuf,

    /// Specify input file format
    #[arg(long = "informat")]
    informat: Option<String>,

    /// Bin length (power of 2; 32<=n<=4096)
    #[arg(long = "bin_length", default_value = "256")]
    bin_length: usize,

    /// Suffix array sample rate (power of 2)
    #[arg(long = "sa_freq", default_value = "8")]
    sa_freq: usize,

    /// Input sequence block size in Mbases
    #[arg(long = "block_size", default_value = "50")]
    block_size: usize,

    /// Assert input alphabet is amino acid
    #[arg(long = "amino", conflicts_with_all = ["dna", "rna"])]
    amino: bool,

    /// Assert input alphabet is DNA
    #[arg(long = "dna", conflicts_with_all = ["amino", "rna"])]
    dna: bool,

    /// Assert input alphabet is RNA
    #[arg(long = "rna", conflicts_with_all = ["amino", "dna"])]
    rna: bool,

    /// Build a forward-strand-only database
    #[arg(long = "fwd_only")]
    fwd_only: bool,

    /// Write the C-layout FM stream directly instead of the Rust HMMERDB container
    #[arg(long = "cstream")]
    cstream: bool,

    /// Write the legacy Rust HMMERDB container instead of the native C stream
    #[arg(long = "container")]
    container: bool,
}

/// Entry point for `makehmmerdb`: turn a (DNA) FASTA into an FM-index database
/// consumable by `nhmmer`'s SSV pre-filter.
///
/// Concatenates every sequence into a single `$`-separated text, partitions it
/// into FM-index blocks via `build_readblock_like_records()` (which mirrors C
/// `esl_sqio_ReadBlock()` per-sequence windowing — see that function), builds
/// BWT + suffix array records, and by default serializes the native C FM-index
/// stream as the top-level file (byte-faithful `FM_METADATA` header from
/// `fwrite()` in makehmmerdb.c:726-763 followed by per-block `FM_DATA` records
/// from buildAndWriteFMIndex(), makehmmerdb.c:315-340). `--container` preserves
/// the older Rust `HMMERDB` wrapper with custom per-block FM indexes, metadata
/// for descriptions/accessions, block windows, overlap, ambiguity ranges, and
/// the same C-layout metadata/FM-record extensions appended after it.
///
/// Faithfulness notes (what matches C byte-for-byte vs. what still differs):
///   * MATCHES C: the metadata header field order/widths (fwd_only, alph_type,
///     alph_size, charBits, freq_SA, freq_cnt_sb, freq_cnt_b, block_count(u16),
///     seq_count, ambig count, char_count(u64)); per-sequence metadata
///     (target_id, target_start(u64), fm_start, length, name/acc/source/desc
///     lengths(u16) and NUL-terminated strings written in name,acc,source,desc
///     order); ambiguity ranges (int lower/upper); and each block's FM record
///     (N, term_loc, seq_offset, ambig_offset, overlap, seq_cnt, ambig_cnt,
///     then packed T (forward record only), packed BWT, SA samples (forward
///     only), occCnts_b, occCnts_sb). Within a block, sequences are
///     concatenated with NO inter-sequence separator and one trailing `$`/0.
///   * MATCHES C: per-sequence block windowing — short sequences read whole,
///     long sequences split with FM_BLOCK_OVERLAP leading context, the
///     `max(block_size-size, block_size*0.05)` request-size shortening.
///   * MATCHES C: degenerate (ambiguous) DNA residues are replaced by a
///     canonical base drawn from a faithful MT19937 port of `esl_random()`
///     (see `Mt19937`), seeded 42 exactly as `esl_randomness_Create(42)`
///     (makehmmerdb.c:426) and consumed in block order at block-build time
///     (makehmmerdb.c:656-674), with overlap residues of split sequences
///     re-drawn just as C re-reads them into the next block. The substituted
///     bases — and therefore the BWT/SA of degenerate blocks — are intended to
///     be bit-identical to a C build. (The MT19937 stream and the per-block
///     draw order are reproduced exactly; this has been unit-tested for
///     determinism and ordering but not yet diffed against a live C build.)
///   * MATCHES C: alphabet auto-guess. When no --amino/--dna/--rna is asserted,
///     the alphabet is guessed by a faithful port of
///     `esl_sqfile_GuessAlphabet` -> `esl_abc_GuessAlphabet` over the first
///     sequence's leading 4000-residue window (see `guess_alphabet`), rather
///     than blindly defaulting to DNA. Unguessable input is rejected as C does.
///   * DIFFERS from C (build mechanism only, NOT output bytes): the FM-index
///     *temp-file two-pass* (esl_tmpfile + rewind/copy, makehmmerdb.c:561,
///     767-837) is not reproduced; the metadata and per-block FM records are
///     assembled in memory and streamed directly. C's two-pass exists purely
///     because it must know block_count/seq_count/ambig_count for the metadata
///     header before it can write the FM records that follow — it buffers the
///     records in a temp file during pass 1, then writes header + replays the
///     temp file in pass 2. We compute the same counts up front and emit header
///     then records in one pass, producing byte-identical output without the
///     temp file. This is an internal implementation detail with no observable
///     effect on the output file.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);

    if let Some(ref informat) = args.informat {
        if !informat.eq_ignore_ascii_case("fasta") {
            eprintln!("{informat} is not a recognized input sequence file format");
            return std::process::ExitCode::FAILURE;
        }
    }
    if let Err(msg) = validate_bin_length(args.bin_length) {
        eprintln!("{msg}");
        return std::process::ExitCode::FAILURE;
    }
    if let Err(msg) = validate_sa_freq(args.sa_freq) {
        eprintln!("{msg}");
        return std::process::ExitCode::FAILURE;
    }
    if args.block_size == 0 {
        eprintln!("Invalid block size: --block_size must be > 0");
        return std::process::ExitCode::FAILURE;
    }
    if args.seqfile == std::path::Path::new("-") && args.binaryfile == std::path::Path::new("-") {
        eprintln!("Either <seqfile> or <binaryfile> can be - but not both.");
        return std::process::ExitCode::FAILURE;
    }
    if args.cstream && args.container {
        eprintln!("makehmmerdb --cstream and --container are mutually exclusive");
        return std::process::ExitCode::FAILURE;
    }
    // Determine the alphabet. C makehmmerdb (makehmmerdb.c:492-515) honours an
    // explicit --amino/--dna/--rna assertion, and otherwise calls
    // `esl_sqfile_GuessAlphabet()` on the open file, failing if it cannot guess.
    // To reproduce the auto-guess we slurp the whole input once, run the same
    // heuristic on the first sequence's leading window, then parse sequences
    // from the in-memory copy. (Reading once and reusing the buffer also lets us
    // guess on stdin, which C handles via its rewindable recording buffer.)
    let raw_input: Vec<u8> = if args.seqfile == std::path::Path::new("-") {
        let mut buf = Vec::new();
        if let Err(e) = std::io::stdin().lock().read_to_end(&mut buf) {
            eprintln!("Error reading stdin: {}", e);
            return std::process::ExitCode::FAILURE;
        }
        buf
    } else {
        match read_seqfile_bytes(&args.seqfile) {
            Ok(buf) => buf,
            Err(e) => {
                eprintln!("Error: {}", e);
                return std::process::ExitCode::FAILURE;
            }
        }
    };

    let guessed_amino = if args.amino {
        true
    } else if args.dna || args.rna {
        false
    } else {
        match guess_alphabet(&raw_input) {
            Some(GuessedAlphabet::Amino) => true,
            Some(GuessedAlphabet::Dna) | Some(GuessedAlphabet::Rna) => false,
            None => {
                eprintln!("Unable to guess alphabet. Try '--dna' or '--amino'");
                return std::process::ExitCode::FAILURE;
            }
        }
    };

    let abc = if guessed_amino {
        Alphabet::amino()
    } else if args.rna {
        Alphabet::rna()
    } else {
        Alphabet::dna()
    };
    let fm_alphabet = if guessed_amino {
        FmAlphabet::amino()
    } else {
        FmAlphabet::dna()
    };
    let fwd_only = args.fwd_only || guessed_amino;

    let mut all_text = Vec::new();
    let mut seq_data = Vec::new();
    let mut ambig_ranges = Vec::new();

    let sqf = sequence::SeqFile::new(std::io::Cursor::new(&raw_input), abc.clone());
    read_sequences(
        sqf,
        &abc,
        !guessed_amino,
        &mut all_text,
        &mut seq_data,
        &mut ambig_ranges,
    );
    if seq_data.is_empty() {
        eprintln!("Error: no sequences found in {}", args.seqfile.display());
        return std::process::ExitCode::FAILURE;
    }

    let input_seq_count = seq_data.len();
    // C's `total_char_count` (makehmmerdb.c:679) counts each input residue
    // exactly once: `if (j > block->list[i].C) total_char_count++` excludes the
    // overlap *context* re-read into continuation blocks. That equals the sum of
    // the ORIGINAL (pre-windowing) sequence lengths, captured here before
    // `build_readblock_like_records` re-partitions `seq_data` into overlapping
    // windows (whose lengths would otherwise double-count overlap regions).
    let input_residue_count: usize = seq_data.iter().map(|seq| seq.length).sum();
    let char_count = input_residue_count;
    let block_size_bases = args.block_size.saturating_mul(1_000_000);
    let (seq_data, mut blocks) =
        build_readblock_like_records(&seq_data, block_size_bases, FM_BLOCK_OVERLAP);
    assign_ambiguities_to_blocks(&seq_data, &ambig_ranges, &mut blocks);

    let stderr = std::io::stderr();
    let mut err = stderr.lock();

    writeln!(
        err,
        "Read {} sequences ({} residues total)",
        input_seq_count, input_residue_count
    )
    .unwrap();

    writeln!(err, "Building FM-index blocks...").unwrap();

    let outpath = args.binaryfile;
    let mut out: Box<dyn Write> = if outpath == std::path::Path::new("-") {
        Box::new(std::io::stdout())
    } else {
        Box::new(std::fs::File::create(&outpath).unwrap_or_else(|e| {
            eprintln!("Error creating output: {}", e);
            std::process::exit(1);
        }))
    };

    if !args.container {
        write_native_c_stream(
            &mut out,
            fwd_only,
            fm_alphabet,
            args.sa_freq,
            args.bin_length,
            char_count,
            &seq_data,
            &blocks,
            &ambig_ranges,
            &all_text,
        )
        .unwrap();
        writeln!(
            err,
            "FM-index built: {} block(s), {} C-stream record(s)",
            blocks.len(),
            blocks.len() * if fwd_only { 1 } else { 2 }
        )
        .unwrap();
        writeln!(err, "Database written to: {}", outpath.display()).unwrap();
        return std::process::ExitCode::SUCCESS;
    }

    let index_records = build_index_records(&all_text, &blocks, !fwd_only);
    writeln!(
        err,
        "FM-index built: {} block(s), {} index record(s)",
        blocks.len(),
        index_records.len()
    )
    .unwrap();

    // Write header
    out.write_all(b"HMMERDB\0").unwrap();
    out.write_all(&(seq_data.len() as u64).to_le_bytes())
        .unwrap();
    out.write_all(&(all_text.len() as u64).to_le_bytes())
        .unwrap();

    // Write sequence names and offsets
    for seq in &seq_data {
        out.write_all(&(seq.name.len() as u32).to_le_bytes())
            .unwrap();
        out.write_all(seq.name.as_bytes()).unwrap();
        out.write_all(&(seq.fm_start as u64).to_le_bytes()).unwrap();
    }

    write_index_extension(&mut out, fwd_only, &index_records).unwrap();

    write_metadata_extension(
        &mut out,
        args.block_size,
        FM_BLOCK_OVERLAP,
        &seq_data,
        &blocks,
        &ambig_ranges,
    )
    .unwrap();

    write_c_metadata_extension(
        &mut out,
        fwd_only,
        fm_alphabet,
        args.sa_freq,
        args.bin_length,
        char_count,
        &seq_data,
        &blocks,
        &ambig_ranges,
    )
    .unwrap();

    write_c_stream_extension(
        &mut out,
        fwd_only,
        fm_alphabet,
        args.sa_freq,
        args.bin_length,
        char_count,
        &seq_data,
        &blocks,
        &ambig_ranges,
        &all_text,
    )
    .unwrap();

    writeln!(err, "Database written to: {}", outpath.display()).unwrap();
    std::process::ExitCode::SUCCESS
}

fn write_string<W: Write + ?Sized>(out: &mut W, value: &str) -> std::io::Result<()> {
    out.write_all(&(value.len() as u32).to_le_bytes())?;
    out.write_all(value.as_bytes())
}

fn write_c_string<W: Write + ?Sized>(out: &mut W, value: &str) -> std::io::Result<()> {
    out.write_all(value.as_bytes())?;
    out.write_all(&[0])
}

fn reject_embedded_nul(field: &str, value: &str) -> std::io::Result<()> {
    if value.as_bytes().contains(&0) {
        Err(invalid_data(format!(
            "{field} contains an embedded NUL byte"
        )))
    } else {
        Ok(())
    }
}

fn checked_u16(value: usize, field: &str) -> std::io::Result<u16> {
    u16::try_from(value).map_err(|_| invalid_data(format!("{field} exceeds C uint16_t range")))
}

fn checked_u32(value: usize, field: &str) -> std::io::Result<u32> {
    u32::try_from(value).map_err(|_| invalid_data(format!("{field} exceeds C uint32_t range")))
}

fn checked_u64(value: usize, field: &str) -> std::io::Result<u64> {
    u64::try_from(value).map_err(|_| invalid_data(format!("{field} exceeds C uint64_t range")))
}

fn checked_i32(value: usize, field: &str) -> std::io::Result<i32> {
    i32::try_from(value).map_err(|_| invalid_data(format!("{field} exceeds C int range")))
}

fn invalid_data(message: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}

/// Validate `--bin_length` exactly as C makehmmerdb does in code
/// (makehmmerdb.c:458): reject unless it is a power of two with
/// `32 <= b <= 4096`. The C getopts table range is NULL, so this is the only
/// gate; without it out-of-range / non-power-of-2 values are silently accepted.
fn validate_bin_length(bin_length: usize) -> Result<(), String> {
    if !(32..=4096).contains(&bin_length) || !bin_length.is_power_of_two() {
        Err("Invalid bin length: --bin_length must be a power of 2 between 32 and 4096".to_string())
    } else {
        Ok(())
    }
}

/// Validate `--sa_freq` as C makehmmerdb does in code (makehmmerdb.c:462):
/// it must be a power of two. (C tests only `freq_SA & (freq_SA - 1)`, which
/// also passes 0 and then crashes downstream; we additionally reject 0 with a
/// clean error rather than reproducing that segfault.)
fn validate_sa_freq(sa_freq: usize) -> Result<(), String> {
    if sa_freq == 0 || !sa_freq.is_power_of_two() {
        Err("Invalid suffix array sample rate: --sa_freq must be a power of 2".to_string())
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/subcmd/makehmmerdb_test.rs"]
mod tests;
