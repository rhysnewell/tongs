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

/// Parse one HMMER3 ASCII HMM record from a line iterator (port of `read_asc30hmm`).
///
/// Handles every header key/value pair (NAME, LENG, ALPH, RF, MM, CONS, CS, MAP,
/// DATE, COM, NSEQ, EFFN, CKSUM, STATS, GA, TC, NC), then reads the alphabet
/// header, an optional COMPO line, node-0 insert/transitions, and the M
/// node blocks of match/insert/transition lines, terminated by `//`.
/// File values are stored as `-ln(p)` and are exponentiated back into
/// probabilities on the fly. Returns `Ok(None)` on clean EOF.
fn read_one_hmm_with_format<B: BufRead>(
    lines: &mut CappedLineReader<B>,
) -> HmmerResult<Option<(Hmm, String)>> {
    // Find the format header line
    let header = loop {
        match lines.next_line()? {
            None => return Ok(None),
            Some(line) => {
                let trimmed = line.trim();
                if trimmed.starts_with("HMMER3/") {
                    break trimmed.to_string();
                }
                if trimmed.starts_with("HMMER2.") || trimmed.starts_with("HMMER2/") {
                    return Err(HmmerError::Format(
                        "HMMER2 ASCII input is intentionally unsupported".to_string(),
                    ));
                }
                // Skip blank lines between HMMs
                if !trimmed.is_empty() && !trimmed.starts_with('#') {
                    return Err(HmmerError::Format(format!(
                        "Expected HMMER3/ header, got: {}",
                        trimmed
                    )));
                }
            }
        }
    };

    let format_version = parse_hmmer3_ascii_magic(&header)?;
    // Parse header key-value pairs
    let mut name: Option<String> = None;
    let mut acc: Option<String> = None;
    let mut desc: Option<String> = None;
    let mut m: usize = 0;
    let mut max_length: i32 = -1;
    let mut abc_type = AlphabetType::Unknown;
    let mut rf_flag = false;
    let mut mm_flag = false;
    let mut cons_flag = false;
    let mut cs_flag = false;
    let mut map_flag = false;
    let mut nseq: i32 = -1;
    let mut eff_nseq: f32 = -1.0;
    let mut ctime: Option<String> = None;
    let mut comlog: Option<String> = None;
    let mut checksum: u32 = 0;
    // checksum presence is tracked via P7H_CHKSUM flag
    let mut evparam = [EVPARAM_UNSET; NEVPARAM];
    let mut cutoff = [CUTOFF_UNSET; NCUTOFFS];
    let mut flags: u32 = 0;
    let mut stat_msv = false;
    let mut stat_viterbi = false;
    let mut stat_forward = false;

    // Read header lines until "HMM" line
    loop {
        let line = lines
            .next_line()?
            .ok_or_else(|| HmmerError::Format("Unexpected EOF in HMM header".to_string()))?;
        let trimmed = line.trim();

        if trimmed.starts_with("HMM ") || trimmed == "HMM" {
            validate_alphabet_header(trimmed, abc_type)?;
            break;
        }

        let (key, value) = match trimmed.split_once(char::is_whitespace) {
            Some((k, v)) => (k, v.trim()),
            None => continue,
        };

        match key {
            "NAME" => name = Some(value.to_string()),
            "ACC" => {
                acc = Some(value.to_string());
                flags |= P7H_ACC;
            }
            "DESC" => {
                desc = Some(value.to_string());
                flags |= P7H_DESC;
            }
            "LENG" => {
                m = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad LENG".to_string()))?;
                if m == 0 || m > MAX_HMM_MODEL_LENGTH {
                    return Err(HmmerError::Format(format!(
                        "Invalid model length {value} on LENG line"
                    )));
                }
            }
            "MAXL" => {
                max_length = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad MAXL".to_string()))?;
                if max_length == 0 {
                    return Err(HmmerError::Format(format!(
                        "Invalid max length {value} on MAXL line"
                    )));
                }
            }
            "ALPH" => {
                abc_type = if value.eq_ignore_ascii_case("amino") {
                    AlphabetType::Amino
                } else if value.eq_ignore_ascii_case("DNA") {
                    AlphabetType::Dna
                } else if value.eq_ignore_ascii_case("RNA") {
                    AlphabetType::Rna
                } else {
                    return Err(HmmerError::Format(format!("Unknown alphabet: {}", value)));
                };
            }
            "RF" => rf_flag = parse_hmm_yes_no("RF", value)?,
            "MM" => mm_flag = parse_hmm_yes_no("MM", value)? && format_version == "3f",
            "CONS" => cons_flag = parse_hmm_yes_no("CONS", value)?,
            "CS" => cs_flag = parse_hmm_yes_no("CS", value)?,
            "MAP" => map_flag = parse_hmm_yes_no("MAP", value)?,
            "DATE" => ctime = Some(value.to_string()),
            "COM" => {
                // COM lines may be numbered: "COM   [1] hmmbuild ..."
                let cmd = if value.starts_with('[') {
                    value
                        .split_once(']')
                        .map(|(_, r)| r.trim())
                        .unwrap_or(value)
                } else {
                    value
                };
                match &mut comlog {
                    Some(existing) => {
                        existing.push('\n');
                        existing.push_str(cmd);
                    }
                    None => comlog = Some(cmd.to_string()),
                }
            }
            "NSEQ" => {
                nseq = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad NSEQ".to_string()))?;
                if nseq == 0 {
                    return Err(HmmerError::Format(format!(
                        "Invalid nseq on NSEQ line: should be integer, not {value}"
                    )));
                }
            }
            "EFFN" => {
                eff_nseq = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad EFFN".to_string()))?;
                if eff_nseq <= 0.0 {
                    return Err(HmmerError::Format(format!(
                        "Invalid eff_nseq on EFFN line: should be a real number, not {value}"
                    )));
                }
            }
            "CKSUM" => {
                checksum = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad CKSUM".to_string()))?;
                flags |= P7H_CHKSUM;
            }
            "STATS" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if format_version == "3a" {
                    // HMMER3/a (reverse compatibility): 3-token form
                    //   "LOCAL VLAMBDA <v>", "LOCAL VMU <v>", "LOCAL FTAU <v>"
                    if parts.len() < 3 {
                        return Err(HmmerError::Format(
                            "Too few fields on STATS line".to_string(),
                        ));
                    }
                    if !parts[0].eq_ignore_ascii_case("LOCAL") {
                        return Err(HmmerError::Format(format!(
                            "Failed to parse STATS, {} unrecognized as field 2",
                            parts[0]
                        )));
                    }
                    let v: f32 = parts[2]
                        .parse()
                        .map_err(|_| HmmerError::Format("Bad STATS value".to_string()))?;
                    if parts[1].eq_ignore_ascii_case("VLAMBDA") {
                        evparam[P7_MLAMBDA] = v;
                        evparam[P7_VLAMBDA] = v;
                        evparam[P7_FLAMBDA] = v;
                        stat_msv = true;
                    } else if parts[1].eq_ignore_ascii_case("VMU") {
                        evparam[P7_MMU] = v;
                        evparam[P7_VMU] = v;
                        stat_viterbi = true;
                    } else if parts[1].eq_ignore_ascii_case("FTAU") {
                        evparam[P7_FTAU] = v;
                        stat_forward = true;
                    } else {
                        return Err(HmmerError::Format(format!(
                            "Failed to parse STATS, {} unrecognized as field 3",
                            parts[1]
                        )));
                    }
                } else {
                    // HMMER3/b+ : 4-token form "LOCAL MSV -6.4582 0.72049"
                    if parts.len() < 4 {
                        return Err(HmmerError::Format(
                            "Too few fields on STATS line".to_string(),
                        ));
                    }
                    if !parts[0].eq_ignore_ascii_case("LOCAL") {
                        return Err(HmmerError::Format(format!(
                            "Failed to parse STATS, {} unrecognized as field 2",
                            parts[0]
                        )));
                    }
                    let v1: f32 = parts[2]
                        .parse()
                        .map_err(|_| HmmerError::Format("Bad STATS value".to_string()))?;
                    let v2: f32 = parts[3]
                        .parse()
                        .map_err(|_| HmmerError::Format("Bad STATS value".to_string()))?;
                    if parts[1].eq_ignore_ascii_case("MSV") {
                        evparam[P7_MMU] = v1;
                        evparam[P7_MLAMBDA] = v2;
                        stat_msv = true;
                    } else if parts[1].eq_ignore_ascii_case("VITERBI") {
                        evparam[P7_VMU] = v1;
                        evparam[P7_VLAMBDA] = v2;
                        stat_viterbi = true;
                    } else if parts[1].eq_ignore_ascii_case("FORWARD") {
                        evparam[P7_FTAU] = v1;
                        evparam[P7_FLAMBDA] = v2;
                        stat_forward = true;
                    } else {
                        return Err(HmmerError::Format(format!(
                            "Failed to parse STATS, {} unrecognized as field 3",
                            parts[1]
                        )));
                    }
                }
            }
            "GA" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if let Some(v) = parts.first() {
                    cutoff[P7_GA1] = parse_cutoff_value("GA", v)?;
                } else {
                    return Err(HmmerError::Format("Missing GA cutoff value".to_string()));
                }
                if let Some(v) = parts.get(1) {
                    cutoff[P7_GA2] = parse_cutoff_value("GA", v)?;
                } else {
                    cutoff[P7_GA2] = cutoff[P7_GA1];
                }
                flags |= P7H_GA;
            }
            "TC" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if let Some(v) = parts.first() {
                    cutoff[P7_TC1] = parse_cutoff_value("TC", v)?;
                } else {
                    return Err(HmmerError::Format("Missing TC cutoff value".to_string()));
                }
                if let Some(v) = parts.get(1) {
                    cutoff[P7_TC2] = parse_cutoff_value("TC", v)?;
                } else {
                    cutoff[P7_TC2] = cutoff[P7_TC1];
                }
                flags |= P7H_TC;
            }
            "NC" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if let Some(v) = parts.first() {
                    cutoff[P7_NC1] = parse_cutoff_value("NC", v)?;
                } else {
                    return Err(HmmerError::Format("Missing NC cutoff value".to_string()));
                }
                if let Some(v) = parts.get(1) {
                    cutoff[P7_NC2] = parse_cutoff_value("NC", v)?;
                } else {
                    cutoff[P7_NC2] = cutoff[P7_NC1];
                }
                flags |= P7H_NC;
            }
            _ => {} // Ignore unknown header fields
        }
    }

    let name = name.ok_or_else(|| HmmerError::Format("Missing NAME in HMM".to_string()))?;
    if m == 0 {
        return Err(HmmerError::Format("Missing or zero LENG".to_string()));
    }
    if abc_type == AlphabetType::Unknown {
        return Err(HmmerError::Format("Missing ALPH in HMM".to_string()));
    }
    if stat_msv || stat_viterbi || stat_forward {
        if !(stat_msv && stat_viterbi && stat_forward) {
            return Err(HmmerError::Format(
                "Incomplete STATS block in HMM header".to_string(),
            ));
        }
        flags |= P7H_STATS;
    }

    let abc = crate::alphabet::Alphabet::new(abc_type);
    let k = abc.k;

    // Skip the transition label line ("m->m m->i ...")
    // Skip the transition label line ("m->m m->i ...")
    lines
        .next_line()?
        .ok_or_else(|| HmmerError::Format("Missing transition label line".to_string()))?;

    // Create the HMM
    let mut hmm = Hmm::new(m, abc_type, k);
    hmm.name = name;
    hmm.acc = acc;
    hmm.desc = desc;
    hmm.nseq = nseq;
    hmm.eff_nseq = eff_nseq;
    hmm.max_length = max_length;
    hmm.ctime = ctime;
    hmm.comlog = comlog;
    hmm.checksum = checksum;
    hmm.evparam = evparam;
    hmm.cutoff = cutoff;
    hmm.flags = flags;

    // Initialize optional annotation arrays
    if rf_flag {
        let mut rf = vec![b' '; m + 2];
        rf[0] = b' ';
        hmm.rf = Some(rf);
        hmm.flags |= P7H_RF;
    }
    if mm_flag {
        let mut mm = vec![b' '; m + 2];
        mm[0] = b' ';
        hmm.mm = Some(mm);
        hmm.flags |= P7H_MMASK;
    }
    if cons_flag {
        let mut cons = vec![b' '; m + 2];
        cons[0] = b' ';
        hmm.consensus = Some(cons);
        hmm.flags |= P7H_CONS;
    }
    if cs_flag {
        let mut cs = vec![b' '; m + 2];
        cs[0] = b' ';
        hmm.cs = Some(cs);
        hmm.flags |= P7H_CS;
    }
    if map_flag {
        hmm.map = Some(vec![0i32; m + 1]);
        hmm.flags |= P7H_MAP;
    }

    // Read COMPO line (optional, node 0 match emissions)
    let compo_line = lines
        .next_line()?
        .ok_or_else(|| HmmerError::Format("Missing COMPO/insert line".to_string()))?;

    let compo_trimmed = compo_line.trim();
    if compo_trimmed.starts_with("COMPO") {
        // Parse composition values
        let parts: Vec<&str> = compo_trimmed.split_whitespace().collect();
        if parts.len() < k + 1 {
            return Err(HmmerError::Format(format!(
                "COMPO line has too few fields: expected {k}, got {}",
                parts.len().saturating_sub(1)
            )));
        }
        for i in 0..k.min(MAXABET) {
            hmm.compo[i] = parse_hmm_value(parts[i + 1])?;
        }
        hmm.flags |= P7H_COMPO;

        // Read node 0 insert emissions
        let ins_line = lines
            .next_line()?
            .ok_or_else(|| HmmerError::Format("Missing node 0 insert line".to_string()))?;
        parse_emission_line(&ins_line, k, &mut hmm.ins[0])?;
    } else {
        // No COMPO line — this line IS the node 0 insert emissions
        parse_emission_line(compo_trimmed, k, &mut hmm.ins[0])?;
    }

    // Read node 0 transitions
    let trans_line = lines
        .next_line()?
        .ok_or_else(|| HmmerError::Format("Missing node 0 transition line".to_string()))?;
    parse_transition_line(&trans_line, &mut hmm.t[0])?;

    // Read nodes 1..M
    for node in 1..=m {
        // Match emission line: "  k  <K values> <map> <cons> <rf> <mm/cs>"
        let match_line = lines
            .next_line()?
            .ok_or_else(|| HmmerError::Format(format!("Missing node {} match line", node)))?;
        let parts: Vec<&str> = match_line.split_whitespace().collect();

        // Parse node number (first field)
        let node_num: usize = parts
            .first()
            .ok_or_else(|| HmmerError::Format("Empty match line".to_string()))?
            .parse()
            .map_err(|_| HmmerError::Format("Bad node number".to_string()))?;
        if node_num != node {
            return Err(HmmerError::Format(format!(
                "Expected node {}, got {}",
                node, node_num
            )));
        }

        // Parse K emission values
        if parts.len() < k + 1 {
            return Err(HmmerError::Format(format!(
                "Node {} match emission line has too few fields",
                node
            )));
        }
        for i in 0..k {
            hmm.mat[node][i] = parse_hmm_value(parts[i + 1])?;
        }

        let mut annot_idx = k + 1;
        let map_val = parts.get(annot_idx).ok_or_else(|| {
            HmmerError::Format(format!("Node {node} match line missing MAP column"))
        })?;
        annot_idx += 1;
        if map_flag {
            if let Some(map) = &mut hmm.map {
                map[node] = map_val
                    .parse()
                    .map_err(|_| HmmerError::Format(format!("Bad MAP value: {map_val}")))?;
            }
        }

        let has_cons_column = matches!(format_version.as_str(), "3e" | "3f");
        if has_cons_column {
            let cons_val = parts.get(annot_idx).ok_or_else(|| {
                HmmerError::Format(format!("Node {node} match line missing CONS column"))
            })?;
            annot_idx += 1;
            if cons_flag {
                if let Some(cons) = &mut hmm.consensus {
                    cons[node] = annotation_byte("CONS", cons_val)?;
                }
            }
        }

        let rf_val = parts.get(annot_idx).ok_or_else(|| {
            HmmerError::Format(format!("Node {node} match line missing RF column"))
        })?;
        annot_idx += 1;
        if rf_flag {
            if let Some(rf) = &mut hmm.rf {
                rf[node] = annotation_byte("RF", rf_val)?;
            }
        }

        if format_version == "3f" {
            let mm_val = parts.get(annot_idx).ok_or_else(|| {
                HmmerError::Format(format!("Node {node} match line missing MM column"))
            })?;
            annot_idx += 1;
            if mm_flag {
                if let Some(mm) = &mut hmm.mm {
                    mm[node] = annotation_byte("MM", mm_val)?;
                }
            }
        }

        let cs_val = parts.get(annot_idx).ok_or_else(|| {
            HmmerError::Format(format!("Node {node} match line missing CS column"))
        })?;
        if cs_flag {
            if let Some(cs) = &mut hmm.cs {
                cs[node] = annotation_byte("CS", cs_val)?;
            }
        }

        // Insert emission line
        let ins_line = lines
            .next_line()?
            .ok_or_else(|| HmmerError::Format(format!("Missing node {} insert line", node)))?;
        parse_emission_line(&ins_line, k, &mut hmm.ins[node])?;

        // Transition line
        let trans_line = lines
            .next_line()?
            .ok_or_else(|| HmmerError::Format(format!("Missing node {} transition line", node)))?;
        parse_transition_line(&trans_line, &mut hmm.t[node])?;
    }

    // Read end-of-record marker "//"
    let mut saw_terminator = false;
    loop {
        match lines.next_line()? {
            None => break,
            Some(line) => {
                let trimmed = line.trim();
                if trimmed == "//" {
                    saw_terminator = true;
                    break;
                } else if !trimmed.is_empty() {
                    return Err(HmmerError::Format(format!(
                        "Expected end-of-record marker //, got: {trimmed}"
                    )));
                }
            }
        }
    }
    if !saw_terminator {
        return Err(HmmerError::Format(
            "Unexpected EOF before HMM end-of-record marker //".to_string(),
        ));
    }

    // Convert from -ln(prob) to probability
    // In the file, values are stored as -ln(p). Convert to p.
    for node in 0..=m {
        for i in 0..k {
            hmm.mat[node][i] = c_expf_to_f32(-hmm.mat[node][i]);
            hmm.ins[node][i] = c_expf_to_f32(-hmm.ins[node][i]);
        }
        for i in 0..NTRANSITIONS {
            if hmm.t[node][i] != f32::INFINITY {
                hmm.t[node][i] = c_expf_to_f32(-hmm.t[node][i]);
            } else {
                hmm.t[node][i] = 0.0;
            }
        }
    }

    // Convert compo from -ln(prob) to prob
    for i in 0..k.min(MAXABET) {
        if hmm.compo[i] != COMPO_UNSET {
            hmm.compo[i] = c_expf_to_f32(-hmm.compo[i]);
        }
    }

    Ok(Some((hmm, format_version)))
}

fn parse_hmmer3_ascii_magic(header: &str) -> HmmerResult<String> {
    let tag = header.split_whitespace().next().unwrap_or("");
    let version = match tag {
        "HMMER3/a" => "3a",
        "HMMER3/b" => "3b",
        "HMMER3/c" => "3c",
        "HMMER3/d" => "3d",
        "HMMER3/e" => "3e",
        "HMMER3/f" => "3f",
        _ => {
            return Err(HmmerError::Format(format!(
                "Unsupported HMMER3 ASCII magic tag: {tag}"
            )))
        }
    };
    Ok(version.to_string())
}

fn parse_hmm_yes_no(key: &str, value: &str) -> HmmerResult<bool> {
    if value.eq_ignore_ascii_case("yes") {
        Ok(true)
    } else if value.eq_ignore_ascii_case("no") {
        Ok(false)
    } else {
        Err(HmmerError::Format(format!(
            "Bad {key} value in HMM header: expected yes or no, got {value}"
        )))
    }
}

/// Parse one HMM value: a float, or `"*"` meaning zero probability (stored as `+inf` in -ln space).
fn parse_hmm_value(s: &str) -> HmmerResult<f32> {
    if s == "*" {
        Ok(f32::INFINITY)
    } else {
        s.parse()
            .map_err(|_| HmmerError::Format(format!("Bad HMM probability value: {s}")))
    }
}

fn parse_cutoff_value(key: &str, s: &str) -> HmmerResult<f32> {
    s.trim_end_matches(';')
        .parse()
        .map_err(|_| HmmerError::Format(format!("Bad {key} cutoff value: {s}")))
}

fn validate_alphabet_header(line: &str, abc_type: AlphabetType) -> HmmerResult<()> {
    if abc_type == AlphabetType::Unknown {
        return Err(HmmerError::Format(
            "HMM alphabet header appeared before ALPH".to_string(),
        ));
    }
    let abc = Alphabet::new(abc_type);
    let fields: Vec<&str> = line.split_whitespace().skip(1).collect();
    if fields.len() != abc.k {
        return Err(HmmerError::Format(format!(
            "HMM alphabet header has {} symbols, expected {}",
            fields.len(),
            abc.k
        )));
    }
    for (idx, (got, expected)) in fields.iter().zip(abc.sym.iter()).enumerate() {
        if got.len() != 1 || got.as_bytes()[0] != *expected {
            return Err(HmmerError::Format(format!(
                "HMM alphabet header symbol {} is {}, expected {}",
                idx + 1,
                got,
                *expected as char
            )));
        }
    }
    Ok(())
}

fn annotation_byte(label: &str, s: &str) -> HmmerResult<u8> {
    s.as_bytes()
        .first()
        .copied()
        .ok_or_else(|| HmmerError::Format(format!("Empty {label} annotation")))
}

/// Parse an emission row of `K` whitespace-separated `-ln(p)` values into `values`.
fn parse_emission_line(line: &str, k: usize, values: &mut [f32]) -> HmmerResult<()> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < k {
        return Err(HmmerError::Format(format!(
            "Emission line has too few fields: expected {k}, got {}",
            parts.len()
        )));
    }
    for i in 0..k {
        values[i] = parse_hmm_value(parts[i])?;
    }
    Ok(())
}

/// Parse a transition row of `NTRANSITIONS` whitespace-separated `-ln(p)` values.
fn parse_transition_line(line: &str, values: &mut [f32; NTRANSITIONS]) -> HmmerResult<()> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < NTRANSITIONS {
        return Err(HmmerError::Format(format!(
            "Transition line has too few fields: expected {NTRANSITIONS}, got {}",
            parts.len()
        )));
    }
    for i in 0..NTRANSITIONS {
        values[i] = parse_hmm_value(parts[i])?;
    }
    Ok(())
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

pub fn write_hmm<W: std::io::Write>(w: &mut W, hmm: &Hmm) -> HmmerResult<()> {
    write_hmm_with_format(w, hmm, HmmAsciiFormat::Hmmer3f)
}

pub fn write_hmm_with_format<W: std::io::Write>(
    w: &mut W,
    hmm: &Hmm,
    format: HmmAsciiFormat,
) -> HmmerResult<()> {
    let k = hmm.abc_k;

    if format == HmmAsciiFormat::Hmmer3f {
        writeln!(w, "HMMER3/f [3.4 | Aug 2023]").map_err(HmmerError::Io)?;
    } else {
        writeln!(
            w,
            "HMMER3/{} [3.4 | Aug 2023; reverse compatibility mode]",
            format.code()
        )
        .map_err(HmmerError::Io)?;
    }
    writeln!(w, "NAME  {}", hmm.name).map_err(HmmerError::Io)?;
    if let Some(ref acc) = hmm.acc {
        writeln!(w, "ACC   {}", acc).map_err(HmmerError::Io)?;
    }
    if let Some(ref desc) = hmm.desc {
        writeln!(w, "DESC  {}", desc).map_err(HmmerError::Io)?;
    }
    writeln!(w, "LENG  {}", hmm.m).map_err(HmmerError::Io)?;
    if format.is_at_least(HmmAsciiFormat::Hmmer3c) && hmm.max_length > 0 {
        writeln!(w, "MAXL  {}", hmm.max_length).map_err(HmmerError::Io)?;
    }
    let alph = match hmm.abc_type {
        AlphabetType::Amino => "amino",
        AlphabetType::Dna => "DNA",
        AlphabetType::Rna => "RNA",
        AlphabetType::Unknown => "unknown",
    };
    writeln!(w, "ALPH  {}", alph).map_err(HmmerError::Io)?;
    writeln!(
        w,
        "RF    {}",
        if hmm.flags & P7H_RF != 0 { "yes" } else { "no" }
    )
    .map_err(HmmerError::Io)?;
    if format.is_at_least(HmmAsciiFormat::Hmmer3f) {
        writeln!(
            w,
            "MM    {}",
            if hmm.flags & P7H_MMASK != 0 {
                "yes"
            } else {
                "no"
            }
        )
        .map_err(HmmerError::Io)?;
    }
    if format.is_at_least(HmmAsciiFormat::Hmmer3e) {
        writeln!(
            w,
            "CONS  {}",
            if hmm.flags & P7H_CONS != 0 {
                "yes"
            } else {
                "no"
            }
        )
        .map_err(HmmerError::Io)?;
    }
    writeln!(
        w,
        "CS    {}",
        if hmm.flags & P7H_CS != 0 { "yes" } else { "no" }
    )
    .map_err(HmmerError::Io)?;
    writeln!(
        w,
        "MAP   {}",
        if hmm.flags & P7H_MAP != 0 {
            "yes"
        } else {
            "no"
        }
    )
    .map_err(HmmerError::Io)?;
    if let Some(ref ctime) = hmm.ctime {
        writeln!(w, "DATE  {}", ctime).map_err(HmmerError::Io)?;
    }
    if let Some(ref comlog) = hmm.comlog {
        for (idx, command) in comlog.lines().enumerate() {
            writeln!(w, "COM   [{}] {}", idx + 1, command).map_err(HmmerError::Io)?;
        }
    }
    if hmm.nseq >= 0 {
        writeln!(w, "NSEQ  {}", hmm.nseq).map_err(HmmerError::Io)?;
    }
    if hmm.eff_nseq >= 0.0 {
        writeln!(w, "EFFN  {}", fmt_fixed6(hmm.eff_nseq as f64)).map_err(HmmerError::Io)?;
    }
    if hmm.flags & P7H_CHKSUM != 0 {
        writeln!(w, "CKSUM {}", hmm.checksum).map_err(HmmerError::Io)?;
    }
    // C (p7_hmmfile.c:546-554): nucleic-acid models emit a single cutoff value;
    // only amino emits both the per-sequence and per-domain cutoffs.
    let nucleic = matches!(hmm.abc_type, AlphabetType::Dna | AlphabetType::Rna);
    if hmm.flags & P7H_GA != 0 {
        if nucleic {
            writeln!(w, "GA    {}", fmt_fixed2(hmm.cutoff[P7_GA1] as f64))
                .map_err(HmmerError::Io)?;
        } else {
            writeln!(
                w,
                "GA    {} {}",
                fmt_fixed2(hmm.cutoff[P7_GA1] as f64),
                fmt_fixed2(hmm.cutoff[P7_GA2] as f64)
            )
            .map_err(HmmerError::Io)?;
        }
    }
    if hmm.flags & P7H_TC != 0 {
        if nucleic {
            writeln!(w, "TC    {}", fmt_fixed2(hmm.cutoff[P7_TC1] as f64))
                .map_err(HmmerError::Io)?;
        } else {
            writeln!(
                w,
                "TC    {} {}",
                fmt_fixed2(hmm.cutoff[P7_TC1] as f64),
                fmt_fixed2(hmm.cutoff[P7_TC2] as f64)
            )
            .map_err(HmmerError::Io)?;
        }
    }
    if hmm.flags & P7H_NC != 0 {
        if nucleic {
            writeln!(w, "NC    {}", fmt_fixed2(hmm.cutoff[P7_NC1] as f64))
                .map_err(HmmerError::Io)?;
        } else {
            writeln!(
                w,
                "NC    {} {}",
                fmt_fixed2(hmm.cutoff[P7_NC1] as f64),
                fmt_fixed2(hmm.cutoff[P7_NC2] as f64)
            )
            .map_err(HmmerError::Io)?;
        }
    }
    if hmm.flags & P7H_STATS != 0 {
        if format == HmmAsciiFormat::Hmmer3a {
            writeln!(
                w,
                "STATS LOCAL     VLAMBDA {}",
                fmt_fixed6(hmm.evparam[P7_MLAMBDA] as f64)
            )
            .map_err(HmmerError::Io)?;
            writeln!(
                w,
                "STATS LOCAL         VMU {}",
                fmt_fixed6(hmm.evparam[P7_MMU] as f64)
            )
            .map_err(HmmerError::Io)?;
            writeln!(
                w,
                "STATS LOCAL        FTAU {}",
                fmt_fixed6(hmm.evparam[P7_FTAU] as f64)
            )
            .map_err(HmmerError::Io)?;
        } else {
            // C: fprintf(fp, "STATS LOCAL MSV      %8.4f %8.5f\n", ...)
            // (label + spaces, then a width-8 mu, one space, width-8 lambda).
            // `fmt_fixed4`/`fmt_fixed5` reproduce C's `%.4f`/`%.5f`; right-
            // justifying in width 8 reproduces the `%8.4f`/`%8.5f` field width.
            writeln!(
                w,
                "STATS LOCAL MSV      {:>8} {:>8}",
                fmt_fixed4(hmm.evparam[P7_MMU] as f64),
                fmt_fixed5(hmm.evparam[P7_MLAMBDA] as f64)
            )
            .map_err(HmmerError::Io)?;
            writeln!(
                w,
                "STATS LOCAL VITERBI  {:>8} {:>8}",
                fmt_fixed4(hmm.evparam[P7_VMU] as f64),
                fmt_fixed5(hmm.evparam[P7_VLAMBDA] as f64)
            )
            .map_err(HmmerError::Io)?;
            writeln!(
                w,
                "STATS LOCAL FORWARD  {:>8} {:>8}",
                fmt_fixed4(hmm.evparam[P7_FTAU] as f64),
                fmt_fixed5(hmm.evparam[P7_FLAMBDA] as f64)
            )
            .map_err(HmmerError::Io)?;
        }
    }

    // Alphabet header
    let abc = crate::alphabet::Alphabet::new(hmm.abc_type);
    write!(w, "HMM     ").map_err(HmmerError::Io)?;
    for i in 0..k {
        write!(w, "     {}   ", abc.sym[i] as char).map_err(HmmerError::Io)?;
    }
    writeln!(w).map_err(HmmerError::Io)?;

    // Transition label line
    writeln!(
        w,
        "            m->m     m->i     m->d     i->m     i->i     d->m     d->d"
    )
    .map_err(HmmerError::Io)?;

    // COMPO line
    if hmm.flags & P7H_COMPO != 0 {
        write!(w, "  COMPO ").map_err(HmmerError::Io)?;
        for i in 0..k {
            write!(w, " {}", fmt_prob(hmm.compo[i])).map_err(HmmerError::Io)?;
        }
        writeln!(w).map_err(HmmerError::Io)?;
    }

    // Node 0 insert emissions
    write!(w, "        ").map_err(HmmerError::Io)?;
    for i in 0..k {
        write!(w, " {}", fmt_prob(hmm.ins[0][i])).map_err(HmmerError::Io)?;
    }
    writeln!(w).map_err(HmmerError::Io)?;

    // Node 0 transitions
    write!(w, "        ").map_err(HmmerError::Io)?;
    for i in 0..NTRANSITIONS {
        write!(w, " {}", fmt_prob(hmm.t[0][i])).map_err(HmmerError::Io)?;
    }
    writeln!(w).map_err(HmmerError::Io)?;

    // Nodes 1..M
    for node in 1..=hmm.m {
        // Match emission line: C writes " %6d " (leading + trailing space).
        write!(w, " {:>6} ", node).map_err(HmmerError::Io)?;
        for i in 0..k {
            write!(w, " {}", fmt_prob(hmm.mat[node][i])).map_err(HmmerError::Io)?;
        }
        if let Some(ref map) = hmm.map {
            write!(w, " {:>6}", map[node]).map_err(HmmerError::Io)?;
        } else {
            write!(w, " {:>6}", "-").map_err(HmmerError::Io)?;
        }
        let cons_ch = hmm
            .consensus
            .as_ref()
            .map(|cons| cons[node] as char)
            .unwrap_or('-');
        let rf_ch = hmm.rf.as_ref().map(|rf| rf[node] as char).unwrap_or('-');
        let mm_ch = hmm.mm.as_ref().map(|mm| mm[node] as char).unwrap_or('-');
        let cs_ch = hmm.cs.as_ref().map(|cs| cs[node] as char).unwrap_or('-');
        if format.is_at_least(HmmAsciiFormat::Hmmer3e) {
            write!(w, " {}", cons_ch).map_err(HmmerError::Io)?;
        }
        write!(w, " {}", rf_ch).map_err(HmmerError::Io)?;
        if format.is_at_least(HmmAsciiFormat::Hmmer3f) {
            write!(w, " {}", mm_ch).map_err(HmmerError::Io)?;
        }
        write!(w, " {}", cs_ch).map_err(HmmerError::Io)?;
        writeln!(w).map_err(HmmerError::Io)?;

        // Insert emission line
        write!(w, "        ").map_err(HmmerError::Io)?;
        for i in 0..k {
            write!(w, " {}", fmt_prob(hmm.ins[node][i])).map_err(HmmerError::Io)?;
        }
        writeln!(w).map_err(HmmerError::Io)?;

        // Transition line
        write!(w, "        ").map_err(HmmerError::Io)?;
        for i in 0..NTRANSITIONS {
            write!(w, " {}", fmt_prob(hmm.t[node][i])).map_err(HmmerError::Io)?;
        }
        writeln!(w).map_err(HmmerError::Io)?;
    }

    writeln!(w, "//").map_err(HmmerError::Io)?;
    Ok(())
}

/// Write an H3 HMM in HMMER2 ASCII save-file format (`hmmconvert -2`).
///
/// Mirrors `p7_h2io_WriteASCII` in `hmmer/src/h2_io.c`. HMMER2 stored the null
/// model and the search configuration (local vs glocal) in the HMM file; H3
/// only stores the core HMM, so the output is emitted for HMMER2's default
/// "ls mode" (glocal) with default null-model and special-state (NECJ)
/// transitions. Statistical calibration and the alignment checksum are omitted
/// (H2 and H3 differ too much for those).
pub fn write_hmm_h2_ascii<W: std::io::Write>(w: &mut W, hmm: &Hmm) -> HmmerResult<()> {
    let abc = Alphabet::new(hmm.abc_type);
    let bg = crate::bg::Bg::new(&abc);
    let k = hmm.abc_k;

    // Default H2 null-model / special-state transitions: amino vs nucleic only.
    // (h2_io.c:96-97 — computed as double then narrowed to float.)
    let (pmove, ploop): (f32, f32) = if hmm.abc_type == AlphabetType::Amino {
        ((1.0_f64 / 351.0) as f32, (350.0_f64 / 351.0) as f32)
    } else {
        ((1.0_f64 / 1001.0) as f32, (1000.0_f64 / 1001.0) as f32)
    };

    // magic header (HMMER_VERSION == "3.4")
    writeln!(w, "HMMER2.0  [converted from 3.4]").map_err(HmmerError::Io)?;
    writeln!(w, "NAME  {}", hmm.name).map_err(HmmerError::Io)?;
    if let Some(ref acc) = hmm.acc {
        writeln!(w, "ACC   {}", acc).map_err(HmmerError::Io)?;
    }
    if let Some(ref desc) = hmm.desc {
        writeln!(w, "DESC  {}", desc).map_err(HmmerError::Io)?;
    }
    writeln!(w, "LENG  {}", hmm.m).map_err(HmmerError::Io)?;
    match hmm.abc_type {
        AlphabetType::Amino => writeln!(w, "ALPH  Amino").map_err(HmmerError::Io)?,
        AlphabetType::Dna | AlphabetType::Rna => {
            writeln!(w, "ALPH  Nucleic").map_err(HmmerError::Io)?
        }
        AlphabetType::Unknown => {
            return Err(HmmerError::InvalidArg(
                "Only protein, DNA, RNA HMMs can be saved in H2 format".to_string(),
            ));
        }
    }
    writeln!(
        w,
        "RF    {}",
        if hmm.flags & P7H_RF != 0 { "yes" } else { "no" }
    )
    .map_err(HmmerError::Io)?;
    writeln!(
        w,
        "CS    {}",
        if hmm.flags & P7H_CS != 0 { "yes" } else { "no" }
    )
    .map_err(HmmerError::Io)?;
    writeln!(
        w,
        "MAP   {}",
        if hmm.flags & P7H_MAP != 0 {
            "yes"
        } else {
            "no"
        }
    )
    .map_err(HmmerError::Io)?;

    if let Some(ref comlog) = hmm.comlog {
        h2_multiline(w, "COM   ", comlog)?;
    }
    if hmm.nseq != -1 {
        writeln!(w, "NSEQ  {}", hmm.nseq).map_err(HmmerError::Io)?;
    }
    if let Some(ref ctime) = hmm.ctime {
        writeln!(w, "DATE  {}", ctime).map_err(HmmerError::Io)?;
    }
    // Checksum is intentionally not written (H2 and H3 use different algorithms).

    if hmm.flags & P7H_GA != 0 {
        writeln!(
            w,
            "GA    {} {}",
            fmt_fixed1(hmm.cutoff[P7_GA1] as f64),
            fmt_fixed1(hmm.cutoff[P7_GA2] as f64)
        )
        .map_err(HmmerError::Io)?;
    }
    if hmm.flags & P7H_TC != 0 {
        writeln!(
            w,
            "TC    {} {}",
            fmt_fixed1(hmm.cutoff[P7_TC1] as f64),
            fmt_fixed1(hmm.cutoff[P7_TC2] as f64)
        )
        .map_err(HmmerError::Io)?;
    }
    if hmm.flags & P7H_NC != 0 {
        writeln!(
            w,
            "NC    {} {}",
            fmt_fixed1(hmm.cutoff[P7_NC1] as f64),
            fmt_fixed1(hmm.cutoff[P7_NC2] as f64)
        )
        .map_err(HmmerError::Io)?;
    }

    // XT: special-state transitions (NB NN EC EJ CT CC JB JJ), default ls config.
    write!(w, "XT     ").map_err(HmmerError::Io)?;
    h2_printprob(w, 6, pmove, 1.0)?; // NB
    h2_printprob(w, 6, ploop, 1.0)?; // NN
    h2_printprob(w, 6, 0.5, 1.0)?; // EC
    h2_printprob(w, 6, 0.5, 1.0)?; // EJ
    h2_printprob(w, 6, pmove, 1.0)?; // CT
    h2_printprob(w, 6, ploop, 1.0)?; // CC
    h2_printprob(w, 6, pmove, 1.0)?; // JB
    h2_printprob(w, 6, ploop, 1.0)?; // JJ
    writeln!(w).map_err(HmmerError::Io)?;

    // NULT: default H2 null-model transitions (NOT H3's).
    write!(w, "NULT   ").map_err(HmmerError::Io)?;
    h2_printprob(w, 6, ploop, 1.0)?; // 1-p1
    h2_printprob(w, 6, pmove, 1.0)?; // p1
    writeln!(w).map_err(HmmerError::Io)?;

    // NULE: H3 null-model emissions (these really are H3's).
    write!(w, "NULE   ").map_err(HmmerError::Io)?;
    let nule_null = (1.0_f64 / k as f64) as f32;
    for x in 0..k {
        h2_printprob(w, 6, bg.f[x], nule_null)?;
    }
    writeln!(w).map_err(HmmerError::Io)?;

    // Main model section.
    write!(w, "HMM      ").map_err(HmmerError::Io)?;
    for x in 0..k {
        write!(w, "  {}    ", abc.sym[x] as char).map_err(HmmerError::Io)?;
    }
    writeln!(w).map_err(HmmerError::Io)?;
    writeln!(
        w,
        "       {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "m->m", "m->i", "m->d", "i->m", "i->i", "d->m", "d->d", "b->m", "m->e"
    )
    .map_err(HmmerError::Io)?;

    // BEGIN node (k=0) transition line: m->m = 1 - t[0][MD], m->i = '*', m->d = t[0][MD].
    write!(w, "      ").map_err(HmmerError::Io)?;
    h2_printprob(w, 6, 1.0 - hmm.t[0][MD], 1.0)?;
    write!(w, " {:>6}", "*").map_err(HmmerError::Io)?;
    h2_printprob(w, 6, hmm.t[0][MD], 1.0)?;
    writeln!(w).map_err(HmmerError::Io)?;

    for node in 1..=hmm.m {
        // Line 1: node index, match emissions, optional map.
        write!(w, " {:>5} ", node).map_err(HmmerError::Io)?;
        for x in 0..k {
            h2_printprob(w, 6, hmm.mat[node][x], bg.f[x])?;
        }
        if hmm.flags & P7H_MAP != 0 {
            let mapval = hmm.map.as_ref().map(|m| m[node]).unwrap_or(0);
            write!(w, " {:>5}", mapval).map_err(HmmerError::Io)?;
        }
        writeln!(w).map_err(HmmerError::Io)?;

        // Line 2: RF char + insert emissions (0.0 at the final node).
        let rf_char = if hmm.flags & P7H_RF != 0 {
            hmm.rf.as_ref().map(|r| r[node]).unwrap_or(b'-') as char
        } else {
            '-'
        };
        write!(w, " {:>5} ", rf_char).map_err(HmmerError::Io)?;
        for x in 0..k {
            let p = if node < hmm.m { hmm.ins[node][x] } else { 0.0 };
            h2_printprob(w, 6, p, bg.f[x])?;
        }
        writeln!(w).map_err(HmmerError::Io)?;

        // Line 3: CS char + transitions (b->m only at node 1; m->e only at the last node).
        let cs_char = if hmm.flags & P7H_CS != 0 {
            hmm.cs.as_ref().map(|c| c[node]).unwrap_or(b'-') as char
        } else {
            '-'
        };
        write!(w, " {:>5} ", cs_char).map_err(HmmerError::Io)?;
        for ts in 0..7 {
            let p = if node < hmm.m { hmm.t[node][ts] } else { 0.0 };
            h2_printprob(w, 6, p, 1.0)?;
        }
        let bm = if node == 1 { hmm.t[0][MM] } else { 0.0 };
        h2_printprob(w, 6, bm, 1.0)?;
        let me = if node < hmm.m { 0.0 } else { 1.0 };
        h2_printprob(w, 6, me, 1.0)?;
        writeln!(w).map_err(HmmerError::Io)?;
    }
    writeln!(w, "//").map_err(HmmerError::Io)?;
    Ok(())
}

/// Print one probability as an H2-format integer log-odds score, with a leading
/// space, right-justified in `fieldwidth`. Mirrors `printprob()` in
/// `hmmer/src/h2_io.c`:
///   - `p == 0.0`                  → `" %*s"` with `"*"`
///   - `null == 1.0 && p == 1.0`   → `" %*d"` with `0`
///   - else                        → `" %*d"` with `floor(0.5 + 1442.695 * log(p/null))`
fn h2_printprob<W: std::io::Write>(
    w: &mut W,
    fieldwidth: usize,
    p: f32,
    null: f32,
) -> HmmerResult<()> {
    if p == 0.0 {
        write!(w, " {:>width$}", "*", width = fieldwidth).map_err(HmmerError::Io)
    } else if null == 1.0 && p == 1.0 {
        write!(w, " {:>width$}", 0, width = fieldwidth).map_err(HmmerError::Io)
    } else {
        // C: (int) floor(0.5 + 1442.695 * log(p/null)). p/null is computed in
        // f32 (both are float), promoted to f64 for log() (libc, via c_log_f64).
        let score = (0.5 + 1442.695 * c_log_f64((p / null) as f64)).floor() as i32;
        write!(w, " {:>width$}", score, width = fieldwidth).map_err(HmmerError::Io)
    }
}

/// Print a multi-line record (e.g. the command log) with a fixed prefix, one
/// line per `\n`-separated segment. Mirrors `h2_multiline()` in `h2_io.c`
/// (note: H2 records bare commands, with no `[n]` numbering — the Rust comlog is
/// already stored without the numbering).
fn h2_multiline<W: std::io::Write>(w: &mut W, pfx: &str, s: &str) -> HmmerResult<()> {
    let bytes = s.as_bytes();
    let mut start = 0usize;
    loop {
        match bytes[start..].iter().position(|&c| c == b'\n') {
            Some(rel) => {
                let end = start + rel;
                write!(w, "{} ", pfx).map_err(HmmerError::Io)?;
                w.write_all(&bytes[start..end]).map_err(HmmerError::Io)?;
                writeln!(w).map_err(HmmerError::Io)?;
                start = end + 1;
                if start >= bytes.len() {
                    break; // C loop ends when *sptr == '\0' after the final '\n'.
                }
            }
            None => {
                writeln!(w, "{} {}", pfx, &s[start..]).map_err(HmmerError::Io)?;
                break;
            }
        }
    }
    Ok(())
}

/// Format a probability `p` as `-ln(p)` (or `*` if zero) using single-precision
/// `logf`, matching the C HMMER ASCII writer's field width and digits.
///
/// Mirrors `printprob(fp, 8, p)` from `hmmer/src/p7_hmmfile.c:2091`:
///   - `p == 0.0` → `" %*s"` with `fieldwidth=8` and `"*"` → `"       *"` (7 spaces + `*`)
///   - `p == 1.0` → `" %*.5f"` with `fieldwidth=8` and `0.0` → `" 0.00000"` (8 chars)
///   - else       → `" %*.5f"` with `fieldwidth=8` and `-logf(p)` → 8-char float string
///
/// Returns an 8-wide string; the caller prepends one space via `" {}"` to
/// reproduce C's `fprintf(fp, " %*.5f", 8, ...)` = 9 bytes per probability value.
fn fmt_prob(p: f32) -> String {
    // C's `printprob` uses `%8s` / `%8.5f` (8-wide field).
    if p <= 0.0 {
        // C: fprintf(fp, " %*s", 8, "*") -> "       *" (7 spaces + '*', 8 chars)
        "       *".to_string()
    } else if p == 1.0 {
        // C: fprintf(fp, " %*.5f", 8, 0.0) -> " 0.00000" (8 chars)
        fmt_hmm_prob(0.0)
    } else {
        // HMMER's C writer uses logf(), not double-precision log().
        // fmt_hmm_prob is now %8.5f, directly matching C's fieldwidth=8.
        fmt_hmm_prob(-c_logf_to_f32(p) as f64)
    }
}

#[cfg(test)]
#[path = "../tests/unit/hmmfile_test.rs"]
mod tests;
