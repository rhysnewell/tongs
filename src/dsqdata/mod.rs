//! Easel dsqdata format: a binary sequence database for fast reading.
//!
//! This is a faithful port of the **on-disk byte format** of Easel's
//! `esl_dsqdata.c` (HMMER/Easel 3.4). Files written here are byte-compatible
//! with Easel's `esl_dsqdata_Write()`, and files written by Easel can be read
//! back by [`read_dsqdata`]. See `hmmer/easel/esl_dsqdata.c` for the
//! source-of-truth.

#![allow(clippy::needless_late_init)]
//!
//! ## On-disk layout (four files)
//!
//! A dsqdata "database" is a stub file `<basename>` plus three binary files:
//!   - `<basename>.dsqi` — index file
//!   - `<basename>.dsqm` — metadata file
//!   - `<basename>.dsqs` — packed sequence file
//!
//! The four files are linked by a 32-bit `uniquetag`.
//!
//! ### Stub file (`<basename>`, text)
//! First line is machine-parsed:
//! ```text
//! Easel dsqdata v1 x<uniquetag>
//! ```
//! followed by human-readable lines (original file/format, type, counts).
//!
//! ### Index file (`.dsqi`)
//! Header: 7 × `uint32` then 3 × `uint64`, little-endian:
//!   `magic, uniquetag, alphatype, flags, max_namelen, max_acclen, max_desclen,`
//!   `max_seqlen, nseq, nres`.
//! Then `nseq` records of `ESL_DSQDATA_RECORD` = two `int64`:
//!   `metadata_end`, `psq_end` — cumulative END offsets (inclusive, i.e. the
//!   offset of this record's *last* element). `metadata_end` is a byte offset
//!   into `.dsqm` (after the 8-byte header); `psq_end` is a *packet* (uint32)
//!   offset into `.dsqs` (after the 8-byte header). Both can be -1 for the
//!   first record if it is empty.
//!
//! ### Metadata file (`.dsqm`)
//! Header: 2 × `uint32` (`magic, uniquetag`). Then per sequence:
//!   `name\0 acc\0 desc\0` followed by an `int32` taxid (-1 = none).
//!
//! ### Sequence file (`.dsqs`)
//! Header: 2 × `uint32` (`magic, uniquetag`). Then packed residues as a stream
//! of `uint32` packets (see packing below).
//!
//! ## Packing
//! Each packet is a `uint32`:
//! ```text
//!  [31]=EOD  [30]=5bit  [29..0]= 6×5-bit residues OR 15×2-bit residues
//! ```
//! 2-bit packing is used for nucleic alphabets (k<=4) where 15 consecutive
//! canonical residues are available; 5-bit packing is used otherwise (and
//! always for amino). The EOD (end-of-data) packet terminates a sequence and
//! may be a partial 5-bit packet whose unused residue slots are 0x1f. A
//! zero-length sequence is a single EOD packet `0xFFFFFFFF`.
//!
//! ## Uniquetag
//! The C `esl_dsqdata_Write()` generates the uniquetag via `esl_randomness_Create(0)`
//! (which seeds Mersenne Twister from `time()+pid+clock()`) followed by one call
//! to `esl_random_uint32()` — so C's uniquetag is **non-deterministic**. Any nonzero
//! uint32 is a valid tag as long as all four files agree. This Rust port instead
//! derives a **deterministic** tag by hashing the dataset's basic statistics
//! (`nseq`, `nres`, `max_seqlen`) with a fixed seed. Databases written by Rust and
//! by C are interoperable regardless of the differing tag derivation methods.
//!
//! ## Taxid
//! The metadata format stores a per-sequence `int32` taxid field (`-1` = none),
//! matching C `esl_dsqdata`'s `chu->taxid[i]`. It is read into and written from
//! [`Sequence::taxid`], so a write→read round-trip preserves it.
//!
//! ## Threaded loader
//! Easel's reader uses a loader thread (disk I/O) and several unpacker threads
//! (CPU decompression) running in a producer/consumer pipeline. This port
//! implements a **two-thread pipeline**: a background loader thread streams raw
//! packed bytes off disk in fixed-size chunks and sends them over a channel;
//! the main thread receives each raw chunk and unpacks it while the loader
//! prefetches the next one. The bytes processed and the sequences produced are
//! format-identical to the synchronous approach; only the overlap of I/O and
//! CPU work differs.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::mpsc;

use crate::alphabet::{Alphabet, AlphabetType, Dsq, DSQ_SENTINEL};
use crate::errors::{HmmerError, HmmerResult};
use crate::sequence::Sequence;

mod pack;
use pack::*;
mod write;
pub use write::*;
mod read;
pub use read::*;

/// Maximum number of sequences per chunk in the threaded loader, matching
/// `eslDSQDATA_CHUNK_MAXSEQ` from the C implementation.
const CHUNK_MAXSEQ: usize = 4096;

/// A raw (still-packed) chunk sent from the loader thread to the main thread.
/// Contains the raw packed uint32 words, raw metadata bytes, and the index
/// records for each sequence in the chunk so the unpacker knows the boundaries.
struct RawChunk {
    /// Packed sequence data (uint32 packets) for this chunk's sequences.
    packed: Vec<u32>,
    /// Raw metadata bytes (name\0 acc\0 desc\0 taxid×4) for this chunk's sequences.
    metadata: Vec<u8>,
    /// Index records for each sequence in this chunk. Each record carries
    /// the *chunk-relative* cumulative end offsets (0-based within this chunk).
    records: Vec<IndexRecord>,
}

/// "dsq1" + 0x80808080. Detects format and byte order (port of `eslDSQDATA_MAGIC_V1`).
const MAGIC_V1: u32 = 0xc4d3d1b1;
/// Byte-swapped magic (port of `eslDSQDATA_MAGIC_V1SWAP`); read support unimplemented (matches C).
const MAGIC_V1SWAP: u32 = 0xb1d1d3c4;

/// Control bit: last packet in a packed sequence (port of `eslDSQDATA_EOD`).
const EOD_BIT: u32 = 1 << 31;
/// Control bit: packet is 5-bit packed (port of `eslDSQDATA_5BIT`).
const FIVEBIT_BIT: u32 = 1 << 30;
const DSQDATA_INDEX_HEADER_LEN: u64 = 7 * 4 + 3 * 8;
const DSQDATA_SIDECAR_HEADER_LEN: u64 = 8;
const DSQDATA_INDEX_RECORD_LEN: u64 = 16;
const MAX_DSQDATA_STUB_LINE_LEN: usize = 4096;

#[inline]
fn is_eod(v: u32) -> bool {
    v & EOD_BIT != 0
}
#[inline]
fn is_5bit(v: u32) -> bool {
    v & FIVEBIT_BIT != 0
}

/// One `.dsqi` index record: cumulative inclusive END offsets.
/// `metadata_end` is a byte offset into `.dsqm`; `psq_end` is a uint32-packet
/// offset into `.dsqs`. Stored on disk as two little-endian `int64`s, in the
/// order (metadata_end, psq_end) matching the C struct field order.
#[derive(Clone, Copy, Debug)]
struct IndexRecord {
    metadata_end: i64,
    psq_end: i64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn with_ext(base: &Path, ext: &str) -> std::path::PathBuf {
    let mut s = base.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    std::path::PathBuf::from(s)
}

fn create(path: &Path) -> HmmerResult<std::fs::File> {
    std::fs::File::create(path).map_err(HmmerError::Io)
}
fn open(path: &Path) -> HmmerResult<std::fs::File> {
    std::fs::File::open(path).map_err(HmmerError::Io)
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> HmmerResult<()> {
    w.write_all(&v.to_le_bytes()).map_err(HmmerError::Io)
}
fn write_i32<W: Write>(w: &mut W, v: i32) -> HmmerResult<()> {
    w.write_all(&v.to_le_bytes()).map_err(HmmerError::Io)
}
fn write_u64<W: Write>(w: &mut W, v: u64) -> HmmerResult<()> {
    w.write_all(&v.to_le_bytes()).map_err(HmmerError::Io)
}
fn write_i64<W: Write>(w: &mut W, v: i64) -> HmmerResult<()> {
    w.write_all(&v.to_le_bytes()).map_err(HmmerError::Io)
}
/// Write a NUL-terminated C string; returns total bytes written (len+1).
fn write_cstr<W: Write>(w: &mut W, s: &[u8]) -> HmmerResult<usize> {
    w.write_all(s).map_err(HmmerError::Io)?;
    w.write_all(&[0u8]).map_err(HmmerError::Io)?;
    Ok(s.len() + 1)
}

fn read_u32<R: Read>(r: &mut R) -> HmmerResult<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(HmmerError::Io)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R) -> HmmerResult<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(HmmerError::Io)?;
    Ok(u64::from_le_bytes(b))
}
fn read_i64<R: Read>(r: &mut R) -> HmmerResult<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(HmmerError::Io)?;
    Ok(i64::from_le_bytes(b))
}

/// Split off a NUL-terminated C string from the front of `buf`. Returns the
/// string bytes (excluding the NUL) and the remainder (after the NUL).
fn take_cstr(buf: &[u8]) -> HmmerResult<(&[u8], &[u8])> {
    match buf.iter().position(|&b| b == 0) {
        Some(i) => Ok((&buf[..i], &buf[i + 1..])),
        None => Err(HmmerError::Format(
            "dsqdata: metadata string not NUL-terminated".into(),
        )),
    }
}

fn decode_type(t: AlphabetType) -> &'static str {
    match t {
        AlphabetType::Rna => "RNA",
        AlphabetType::Dna => "DNA",
        AlphabetType::Amino => "amino",
        AlphabetType::Unknown => "unknown",
    }
}

/// Derive a deterministic uniquetag from dataset statistics.
///
/// The C `esl_dsqdata_Write()` generates the tag by calling
/// `esl_randomness_Create(0)` (which seeds a Mersenne Twister from
/// `time() + pid + clock()`, making it non-deterministic / different on every
/// run) and then drawing one `uint32_t`. Any nonzero `uint32` is valid as a
/// tag, provided all four files agree. This Rust implementation instead
/// produces a **deterministic** tag by mixing the dataset's basic statistics
/// with a fixed nonzero starting constant. Files written by Rust and files
/// written by C are interoperable regardless of the different tag derivation
/// methods, because the reader only checks that all four files carry the same
/// tag — it does not mandate a specific derivation method.
fn derive_uniquetag(nseq: u64, nres: u64, max_seqlen: u64) -> u32 {
    let mut h = 0x9E3779B9u32;
    for v in [nseq, nres, max_seqlen] {
        h = h
            .wrapping_mul(2654435761)
            .wrapping_add((v as u32) ^ ((v >> 32) as u32));
    }
    h | 1 // ensure nonzero
}

#[cfg(test)]
#[path = "../../tests/unit/dsqdata_test.rs"]
mod tests;
