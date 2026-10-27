//! HMM file I/O: reading and writing HMMER3 format HMM files.
//! Direct port of p7_hmmfile.c.
#![allow(clippy::large_enum_variant, clippy::while_let_loop)]

use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

use crate::alphabet::{Alphabet, AlphabetType};
use crate::errors::{HmmerError, HmmerResult};
use crate::hmm::*;
use crate::output::{fmt_fixed1, fmt_fixed2, fmt_fixed4, fmt_fixed5, fmt_fixed6, fmt_hmm_prob};
use crate::util::cmath::{c_expf_to_f32, c_log_f64, c_logf_to_f32};

mod read_ascii;
use read_ascii::*;
mod parse;
use parse::*;
mod write;
pub use write::*;
mod write_h2;
pub use write_h2::*;

const MAX_HMM_MODEL_LENGTH: usize = 1_000_000;
pub(crate) const MAX_ASCII_HMM_LINE_LEN: usize = 1 << 20;

pub(crate) fn read_capped_text_line<B: BufRead>(
    reader: &mut B,
    limit: usize,
) -> HmmerResult<Option<String>> {
    let mut bytes = Vec::new();
    loop {
        let available = reader.fill_buf().map_err(HmmerError::Io)?;
        if available.is_empty() {
            if bytes.is_empty() {
                return Ok(None);
            }
            break;
        }

        let take = available
            .iter()
            .position(|&b| b == b'\n')
            .map(|pos| pos + 1)
            .unwrap_or(available.len());
        if bytes.len().saturating_add(take) > limit {
            return Err(HmmerError::Format(format!(
                "Input line exceeds maximum supported length of {limit} bytes"
            )));
        }
        bytes.extend_from_slice(&available[..take]);
        reader.consume(take);
        if bytes.last() == Some(&b'\n') {
            break;
        }
    }

    String::from_utf8(bytes)
        .map(Some)
        .map_err(|_| HmmerError::Format("Input line is not valid UTF-8".to_string()))
}

struct CappedLineReader<B: BufRead> {
    reader: B,
}

impl<B: BufRead> CappedLineReader<B> {
    fn new(reader: B) -> Self {
        Self { reader }
    }

    fn next_line(&mut self) -> HmmerResult<Option<String>> {
        read_capped_text_line(&mut self.reader, MAX_ASCII_HMM_LINE_LEN)
    }
}

/// Open an HMM save file and read every HMM contained in it.
///
/// Wrapper over [`read_hmms`] that opens `path` first. Returns all HMMs
/// in the file as a `Vec`; errors propagate I/O and format failures.
pub fn read_hmm_file(path: &Path) -> HmmerResult<Vec<Hmm>> {
    let reader = BufReader::new(open_hmm_reader(path)?);
    read_hmms(reader)
}

/// Open an HMM save file and read every HMM, auto-dispatching ASCII vs binary
/// from the leading magic bytes instead of relying on the filename.
pub fn read_hmm_file_auto(path: &Path) -> HmmerResult<Vec<Hmm>> {
    if path.extension().is_some_and(|ext| ext == "gz") {
        return read_hmm_file(path);
    }
    if crate::hmmfile_binary::looks_like_binary_hmm_file(path)? {
        crate::hmmfile_binary::read_binary_hmm_file(path)
    } else {
        read_hmm_file(path)
    }
}

/// Open an HMM save file and read records without enforcing a single ASCII
/// HMMER3 format tag across the whole file. This is only for legacy utility
/// paths that operate record-by-record rather than validating a database.
pub fn read_hmm_file_auto_allow_mixed_formats(path: &Path) -> HmmerResult<Vec<Hmm>> {
    if path.extension().is_some_and(|ext| ext == "gz") {
        return read_hmms_allow_mixed_formats(BufReader::new(open_hmm_reader(path)?));
    }
    if crate::hmmfile_binary::looks_like_binary_hmm_file(path)? {
        crate::hmmfile_binary::read_binary_hmm_file(path)
    } else {
        read_hmms_allow_mixed_formats(BufReader::new(open_hmm_reader(path)?))
    }
}

/// Open an HMM save file and read only the first HMM record.
pub fn read_first_hmm_file_auto(path: &Path) -> HmmerResult<Hmm> {
    if path.extension().is_some_and(|ext| ext == "gz") {
        return read_first_hmm(BufReader::new(open_hmm_reader(path)?));
    }
    if crate::hmmfile_binary::looks_like_binary_hmm_file(path)? {
        return crate::hmmfile_binary::read_binary_hmm_file(path)?
            .into_iter()
            .next()
            .ok_or_else(|| HmmerError::Format("No HMM records found".to_string()));
    }
    read_first_hmm(BufReader::new(open_hmm_reader(path)?))
}

/// Seek to a record offset from an SSI index and read exactly one HMM record.
pub fn read_hmm_file_record_at(path: &Path, offset: u64) -> HmmerResult<Hmm> {
    let mut file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    file.seek(SeekFrom::Start(offset)).map_err(HmmerError::Io)?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).map_err(HmmerError::Io)?;
    file.seek(SeekFrom::Start(offset)).map_err(HmmerError::Io)?;
    if crate::hmmfile_binary::is_binary_hmm_magic(u32::from_ne_bytes(magic)) {
        return crate::hmmfile_binary::read_binary_hmm(&mut BufReader::new(file))?
            .ok_or_else(|| HmmerError::Format(format!("No binary HMM record at offset {offset}")));
    }

    let mut reader = BufReader::new(file);
    let mut record = String::new();
    loop {
        let Some(line) = read_capped_text_line(&mut reader, MAX_ASCII_HMM_LINE_LEN)? else {
            break;
        };
        record.push_str(&line);
        if line.trim() == "//" {
            break;
        }
    }
    if record.is_empty() {
        return Err(HmmerError::Format(format!(
            "No ASCII HMM record at offset {offset}"
        )));
    }
    if !record.trim_end().ends_with("//") {
        return Err(HmmerError::Format(format!(
            "Unterminated ASCII HMM record at offset {offset}"
        )));
    }
    let mut hmms = read_hmms(BufReader::new(Cursor::new(record.into_bytes())))?;
    if hmms.len() != 1 {
        return Err(HmmerError::Format(format!(
            "Expected one HMM record at offset {offset}, found {}",
            hmms.len()
        )));
    }
    Ok(hmms.remove(0))
}

/// Read all HMMs from an open HMM save file stream (Rust port of `p7_hmmfile_Read`).
///
/// Loops calling `read_one_hmm` until the reader hits EOF, collecting each
/// parsed `Hmm` into a vector. The C entry point reads one HMM at a time via
/// a parser dispatch (`read_asc30hmm` / `read_bin30hmm`); here we expose the
/// "read everything" idiom and a single ASCII parser.
pub fn read_hmms<R: Read>(reader: BufReader<R>) -> HmmerResult<Vec<Hmm>> {
    let mut hmms = Vec::new();
    let mut lines = CappedLineReader::new(reader);
    let mut expected_abc = None;
    let mut expected_format: Option<String> = None;

    loop {
        match read_one_hmm_with_format(&mut lines)? {
            Some((hmm, format_version)) => {
                if let Some(expected) = expected_format.as_deref() {
                    if format_version != expected {
                        return Err(HmmerError::Format(format!(
                            "ASCII HMM file contains mixed HMMER3 format versions: first record is {expected}, record {} is {format_version}",
                            hmms.len() + 1
                        )));
                    }
                } else {
                    expected_format = Some(format_version.clone());
                }
                if let Some(expected) = expected_abc {
                    if hmm.abc_type != expected {
                        return Err(HmmerError::Format(format!(
                            "ASCII HMM file contains mixed alphabets: first record is {:?}, record {} is {:?}",
                            expected,
                            hmms.len() + 1,
                            hmm.abc_type
                        )));
                    }
                } else {
                    expected_abc = Some(hmm.abc_type);
                }
                hmms.push(hmm);
            }
            None => break,
        }
    }

    Ok(hmms)
}

/// Outcome of a sequential single-key fetch over an HMM file.
pub enum FetchOutcome {
    /// The requested key was found; this is the matching HMM record.
    Found(Hmm),
    /// The requested key was not present in the file.
    NotFound,
}

/// Fetch a single HMM by name or accession from an HMM file, mirroring C's
/// `onefetch` no-SSI scan loop (`hmmfetch.c:309-326`).
///
/// C reads records one at a time via `p7_hmmfile_Read`, threading a single
/// `ESL_ALPHABET` object: the first record fixes the alphabet, and any later
/// record with a different alphabet raises `eslEINCOMPAT` *before* it can be
/// matched. C `break`s on the first name/accession match, so it never reads
/// past the matched record. This means fetching the first (e.g. amino) model
/// from a mixed amino+DNA file succeeds, but fetching a later model (or a
/// missing key) fails with "contains different alphabets" once the scan reaches
/// the differing record. This function reproduces that alphabet-stateful,
/// break-on-match behavior exactly, for both ASCII and binary files.
pub fn fetch_one_sequential(path: &Path, key: &str) -> HmmerResult<FetchOutcome> {
    if path.extension().is_some_and(|ext| ext == "gz") {
        let mut lines = CappedLineReader::new(BufReader::new(open_hmm_reader(path)?));
        let mut expected_abc = None;
        while let Some((hmm, _fmt)) = read_one_hmm_with_format(&mut lines)? {
            check_shared_alphabet(&mut expected_abc, hmm.abc_type)?;
            if hmm.name == key || hmm.acc.as_deref() == Some(key) {
                return Ok(FetchOutcome::Found(hmm));
            }
        }
        return Ok(FetchOutcome::NotFound);
    }
    if crate::hmmfile_binary::looks_like_binary_hmm_file(path)? {
        let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
        let mut reader = BufReader::new(file);
        let mut expected_abc = None;
        while let Some(hmm) = crate::hmmfile_binary::read_binary_hmm(&mut reader)? {
            check_shared_alphabet(&mut expected_abc, hmm.abc_type)?;
            if hmm.name == key || hmm.acc.as_deref() == Some(key) {
                return Ok(FetchOutcome::Found(hmm));
            }
        }
        return Ok(FetchOutcome::NotFound);
    }

    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut lines = CappedLineReader::new(BufReader::new(file));
    let mut expected_abc = None;
    while let Some((hmm, _fmt)) = read_one_hmm_with_format(&mut lines)? {
        check_shared_alphabet(&mut expected_abc, hmm.abc_type)?;
        if hmm.name == key || hmm.acc.as_deref() == Some(key) {
            return Ok(FetchOutcome::Found(hmm));
        }
    }
    Ok(FetchOutcome::NotFound)
}

fn open_hmm_reader(path: &Path) -> HmmerResult<Box<dyn Read>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    if path.extension().is_some_and(|ext| ext == "gz") {
        Ok(Box::new(flate2::read::GzDecoder::new(file)))
    } else {
        Ok(Box::new(file))
    }
}

/// Thread a single alphabet across sequentially read records, matching C's
/// shared `ESL_ALPHABET` semantics: the first record sets it; a later record
/// with a different alphabet is the `eslEINCOMPAT` case.
fn check_shared_alphabet(
    expected_abc: &mut Option<AlphabetType>,
    abc_type: AlphabetType,
) -> HmmerResult<()> {
    match expected_abc {
        Some(expected) if *expected != abc_type => Err(HmmerError::Format(
            "contains different alphabets".to_string(),
        )),
        Some(_) => Ok(()),
        None => {
            *expected_abc = Some(abc_type);
            Ok(())
        }
    }
}

pub fn read_hmms_allow_mixed_formats<R: Read>(reader: BufReader<R>) -> HmmerResult<Vec<Hmm>> {
    let mut hmms = Vec::new();
    let mut lines = CappedLineReader::new(reader);
    let mut expected_abc = None;

    loop {
        match read_one_hmm_with_format(&mut lines)? {
            Some((hmm, _format_version)) => {
                if let Some(expected) = expected_abc {
                    if hmm.abc_type != expected {
                        return Err(HmmerError::Format(format!(
                            "ASCII HMM file contains mixed alphabets: first record is {:?}, record {} is {:?}",
                            expected,
                            hmms.len() + 1,
                            hmm.abc_type
                        )));
                    }
                } else {
                    expected_abc = Some(hmm.abc_type);
                }
                hmms.push(hmm);
            }
            None => break,
        }
    }

    Ok(hmms)
}

pub fn read_first_hmm<R: Read>(reader: BufReader<R>) -> HmmerResult<Hmm> {
    let mut lines = CappedLineReader::new(reader);
    read_one_hmm_with_format(&mut lines)?
        .map(|(hmm, _format_version)| hmm)
        .ok_or_else(|| HmmerError::Format("No HMM records found".to_string()))
}

/// Read all HMMs from an open stream, auto-dispatching ASCII vs binary from
/// the first four bytes. The prefix is chained back into the selected parser.
pub fn read_hmms_auto<R: Read>(mut reader: BufReader<R>) -> HmmerResult<Vec<Hmm>> {
    let mut prefix = Vec::with_capacity(4);
    while prefix.len() < 4 {
        let mut byte = [0u8; 1];
        match reader.read(&mut byte).map_err(HmmerError::Io)? {
            0 => break,
            1 => prefix.push(byte[0]),
            _ => unreachable!(),
        }
    }

    let is_binary = prefix.len() == 4
        && crate::hmmfile_binary::is_binary_hmm_magic(u32::from_ne_bytes([
            prefix[0], prefix[1], prefix[2], prefix[3],
        ]));
    let chained = Cursor::new(prefix).chain(reader);
    if !is_binary {
        return read_hmms(BufReader::new(chained));
    }

    let mut reader = BufReader::new(chained);
    let mut hmms = Vec::new();
    let mut expected_abc = None;
    loop {
        match crate::hmmfile_binary::read_binary_hmm(&mut reader)? {
            Some(hmm) => {
                if let Some(expected) = expected_abc {
                    if hmm.abc_type != expected {
                        return Err(HmmerError::Format(format!(
                            "Binary HMM file contains mixed alphabets: first record is {:?}, record {} is {:?}",
                            expected,
                            hmms.len() + 1,
                            hmm.abc_type
                        )));
                    }
                } else {
                    expected_abc = Some(hmm.abc_type);
                }
                hmms.push(hmm);
            }
            None => break,
        }
    }
    Ok(hmms)
}

/// Write a profile HMM as a HMMER3 ASCII save file (port of `p7_hmmfile_WriteASCII`).
///
/// Emits the HMMER3 header block (NAME/ACC/DESC/LENG/MAXL/ALPH, annotation
/// flag lines, DATE/NSEQ/EFFN/CKSUM, STATS LOCAL MSV/VITERBI/FORWARD), then
/// the alphabet line, transition label legend, optional COMPO line, node-0
/// insert/transition rows, and one match/insert/transition triplet per node
/// 1..M, terminated by `//`. Probabilities are encoded as `-ln(p)` with `*`
/// for zero, matching the C writer's `logf`-based output exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HmmAsciiFormat {
    Hmmer3a,
    Hmmer3b,
    Hmmer3c,
    Hmmer3d,
    Hmmer3e,
    Hmmer3f,
}

impl HmmAsciiFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "3/a" => Some(Self::Hmmer3a),
            "3/b" => Some(Self::Hmmer3b),
            "3/c" => Some(Self::Hmmer3c),
            "3/d" => Some(Self::Hmmer3d),
            "3/e" => Some(Self::Hmmer3e),
            "3/f" => Some(Self::Hmmer3f),
            _ => None,
        }
    }

    fn code(self) -> char {
        match self {
            Self::Hmmer3a => 'a',
            Self::Hmmer3b => 'b',
            Self::Hmmer3c => 'c',
            Self::Hmmer3d => 'd',
            Self::Hmmer3e => 'e',
            Self::Hmmer3f => 'f',
        }
    }

    fn is_at_least(self, other: Self) -> bool {
        self >= other
    }
}

#[cfg(test)]
#[path = "../../tests/unit/hmmfile_test.rs"]
mod tests;
