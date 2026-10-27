//! Biological sequence type and FASTA I/O.
//! Simplified port of esl_sq and esl_sqio_ascii for FASTA format.

#![allow(clippy::manual_strip)]

use crate::alphabet::{Alphabet, Dsq, DSQ_IGNORED, DSQ_ILLEGAL, DSQ_SENTINEL};
use crate::errors::{HmmerError, HmmerResult};
use crate::msa;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Cursor, Read};

const MAX_SEQUENCE_LINE_LEN: usize = 1 << 20;

/// A biological sequence (digital or text mode).
#[derive(Debug, Clone)]
pub struct Sequence {
    pub name: String,
    pub acc: String,
    pub desc: String,
    /// Digital sequence, 1-based: `dsq[0]` = SENTINEL, `dsq[1..=n]` = seq, `dsq[n+1]` = SENTINEL
    pub dsq: Vec<Dsq>,
    /// Length of the sequence
    pub n: usize,
    /// Full source length (same as n for complete sequences)
    pub l: usize,
    /// NCBI taxonomy id (dsqdata metadata); `-1` = unset, matching C's
    /// `esl_dsqdata` sentinel.
    pub taxid: i32,
}

impl Sequence {
    /// Create a new empty digital `Sequence` (port of `esl_sq_Create`).
    ///
    /// The dsq starts with a single `DSQ_SENTINEL`; subsequent reads append
    /// residues at positions `1..=n` and add a trailing sentinel.
    pub fn new() -> Self {
        Sequence {
            name: String::new(),
            acc: String::new(),
            desc: String::new(),
            dsq: vec![DSQ_SENTINEL],
            n: 0,
            l: 0,
            taxid: -1,
        }
    }

    /// Reset the object so a new sequence may be read into it
    /// (port of `esl_sq_Reuse`).
    ///
    /// Clears metadata and the digital sequence buffer (re-seeding it with
    /// the leading sentinel) without freeing capacity, so a hot read loop
    /// avoids reallocation between records.
    pub fn reuse(&mut self) {
        self.name.clear();
        self.acc.clear();
        self.desc.clear();
        self.dsq.clear();
        self.dsq.push(DSQ_SENTINEL);
        self.n = 0;
        self.l = 0;
        self.taxid = -1;
    }
}

impl Default for Sequence {
    fn default() -> Self {
        Self::new()
    }
}

/// Unaligned sequence file formats that the Rust reader can parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceFormat {
    Fasta,
    UniProt,
    GenBank,
    Embl,
    Ddbj,
    Stockholm,
}

impl SequenceFormat {
    pub fn from_name(name: &str) -> Option<Self> {
        if name.eq_ignore_ascii_case("fasta") {
            Some(Self::Fasta)
        } else if name.eq_ignore_ascii_case("uniprot") {
            Some(Self::UniProt)
        } else if name.eq_ignore_ascii_case("genbank") {
            Some(Self::GenBank)
        } else if name.eq_ignore_ascii_case("embl") {
            Some(Self::Embl)
        } else if name.eq_ignore_ascii_case("ddbj") {
            Some(Self::Ddbj)
        } else if name.eq_ignore_ascii_case("stockholm")
            || name.eq_ignore_ascii_case("sto")
            || name.eq_ignore_ascii_case("pfam")
        {
            Some(Self::Stockholm)
        } else {
            None
        }
    }

    fn matches_record_start(self, trimmed_line: &str) -> bool {
        match self {
            Self::Fasta => trimmed_line.starts_with('>'),
            Self::UniProt | Self::Embl => trimmed_line.starts_with("ID "),
            Self::GenBank | Self::Ddbj => trimmed_line.starts_with("LOCUS "),
            Self::Stockholm => trimmed_line.starts_with("# STOCKHOLM"),
        }
    }
}

/// Sequence file reader.
pub struct SeqFile<R: Read> {
    reader: BufReader<R>,
    abc: Alphabet,
    asserted_format: Option<SequenceFormat>,
    /// Buffered line for look-ahead
    pending_header: Option<String>,
    pending_msa_sequences: VecDeque<Sequence>,
    at_eof: bool,
}

impl<R: Read> SeqFile<R> {
    /// Wrap a `Read` source as a buffered sequence file using alphabet `abc`.
    /// Format autodetection happens on the first line of [`Self::read`].
    pub fn new(reader: R, abc: Alphabet) -> Self {
        SeqFile {
            reader: BufReader::new(reader),
            abc,
            asserted_format: None,
            pending_header: None,
            pending_msa_sequences: VecDeque::new(),
            at_eof: false,
        }
    }

    /// Require FASTA input instead of using sequence format autodetection.
    pub fn with_fasta_only(mut self) -> Self {
        self.asserted_format = Some(SequenceFormat::Fasta);
        self
    }

    /// Require a specific input format instead of using sequence format autodetection.
    pub fn with_format(mut self, format: SequenceFormat) -> Self {
        self.asserted_format = Some(format);
        self
    }

    /// Read the next sequence record into `sq` (port of the common
    /// `esl_sqio_Read` entry point).
    ///
    /// Auto-detects FASTA (`>name`), UniProt/Swiss-Prot (`ID … // SQ …`), or
    /// GenBank (`LOCUS … // ORIGIN …`) format from the first non-empty line
    /// and parses one record. Residues are digitised through the alphabet,
    /// preserving ignored whitespace and rejecting illegal sequence symbols.
    /// Returns `Ok(false)` at EOF.
    pub fn read(&mut self, sq: &mut Sequence) -> HmmerResult<bool> {
        if self.at_eof {
            if let Some(next) = self.pending_msa_sequences.pop_front() {
                *sq = next;
                return Ok(true);
            }
            return Ok(false);
        }

        sq.reuse();
        if let Some(next) = self.pending_msa_sequences.pop_front() {
            *sq = next;
            return Ok(true);
        }

        // Find the start of a sequence record
        let first_line = if let Some(h) = self.pending_header.take() {
            h
        } else {
            let mut line = String::new();
            loop {
                line.clear();
                let n = read_capped_line(&mut self.reader, &mut line)?;
                if n == 0 {
                    self.at_eof = true;
                    return Ok(false);
                }
                let trimmed = line.trim();
                if self.is_accepted_record_start(trimmed) {
                    break;
                }
                if !trimmed.is_empty() && !trimmed.starts_with('#') {
                    return Err(HmmerError::Format(format!(
                        "unrecognized sequence file record start: {}",
                        trimmed
                    )));
                }
            }
            line
        };

        let trimmed = first_line.trim();
        let mut allow_zero_length = false;

        if trimmed.starts_with("# STOCKHOLM") {
            let reader = Cursor::new(first_line.into_bytes()).chain(&mut self.reader);
            self.at_eof = true;
            let msas = msa::read_stockholm_from_reader(BufReader::new(reader))?;
            for alignment in msas {
                for idx in 0..alignment.nseq {
                    self.pending_msa_sequences
                        .push_back(self.sequence_from_msa_row(&alignment, idx)?);
                }
            }
            if let Some(next) = self.pending_msa_sequences.pop_front() {
                *sq = next;
                return Ok(true);
            }
            return Ok(false);
        } else if trimmed.starts_with('>') {
            // FASTA format
            allow_zero_length = true;
            let after_gt = trimmed[1..].trim_start();
            if after_gt.is_empty() {
                return Err(HmmerError::Format("no FASTA name found".to_string()));
            }
            let parts: Vec<&str> = after_gt.splitn(2, char::is_whitespace).collect();
            sq.name = parts[0].to_string();
            if let Some(desc) = parts.get(1) {
                sq.desc = desc
                    .as_bytes()
                    .split(|&b| b == 0x01)
                    .next()
                    .map(|bytes| String::from_utf8_lossy(bytes).trim().to_string())
                    .unwrap_or_default();
            }

            let mut line = String::new();
            loop {
                line.clear();
                let n = read_capped_line(&mut self.reader, &mut line)?;
                if n == 0 {
                    self.at_eof = true;
                    break;
                }
                let trimmed = line.trim();
                if trimmed.starts_with('>') {
                    self.pending_header = Some(line.clone());
                    break;
                }
                for &ch in trimmed.as_bytes() {
                    if let Some(code) = self.digitize_fasta_residue(ch, &sq.name)? {
                        sq.dsq.push(code);
                    }
                }
            }
        } else if trimmed.starts_with("ID ") {
            // UniProt/SwissProt format
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                sq.name = parts[1].to_string();
            }

            // Read until SQ line, collecting DE and AC
            let mut line = String::new();
            let mut in_seq = false;
            loop {
                line.clear();
                let n = read_capped_line(&mut self.reader, &mut line)?;
                if n == 0 {
                    self.at_eof = true;
                    break;
                }
                let trimmed = line.trim();

                if trimmed == "//" {
                    break; // end of record
                }

                if line.starts_with("DE   ") {
                    // Match Easel esl_sqio_ascii.c read_uniprot exactly:
                    //   s = buf + 5;               (strip the 5-char "DE   " prefix)
                    //   esl_strchop(s, nc-5);       (chop trailing whitespace only)
                    //   esl_sq_AppendDesc(sq, s);   (join with a single space, but
                    //                                preserve the continuation line's
                    //                                leading whitespace)
                    // Crucially, leading whitespace of continuation lines is *not*
                    // trimmed, so the original column spacing is retained.
                    let de = line
                        .get(5..)
                        .unwrap_or("")
                        .trim_end_matches(|c: char| c.is_ascii_whitespace());
                    if sq.desc.is_empty() {
                        sq.desc.push_str(de);
                    } else {
                        // esl_sq_AppendDesc always inserts one space before the new
                        // text when the existing description is non-empty.
                        sq.desc.push(' ');
                        sq.desc.push_str(de);
                    }
                } else if trimmed.starts_with("AC ") && sq.acc.is_empty() {
                    let acc = trimmed[5..].trim().trim_end_matches(';');
                    sq.acc = acc.split(';').next().unwrap_or("").trim().to_string();
                } else if trimmed.starts_with("SQ ") {
                    in_seq = true;
                    continue;
                }

                if in_seq {
                    // Sequence data: letters and spaces, ending with //
                    for &ch in trimmed.as_bytes() {
                        if ch.is_ascii_alphabetic()
                            || matches!(ch, b'*' | b'~' | b'-' | b'.' | b'_')
                        {
                            if let Some(code) =
                                self.digitize_sequence_residue(ch, "UniProt sequence", &sq.name)?
                            {
                                sq.dsq.push(code);
                            }
                        }
                    }
                }
            }
        }

        if trimmed.starts_with("LOCUS ") {
            // GenBank format
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                sq.name = parts[1].to_string();
            }

            let mut line = String::new();
            let mut in_seq = false;
            let mut in_definition = false;
            loop {
                line.clear();
                let n = read_capped_line(&mut self.reader, &mut line)?;
                if n == 0 {
                    self.at_eof = true;
                    break;
                }
                let trimmed = line.trim();

                if trimmed == "//" {
                    break;
                }

                if in_definition
                    && !in_seq
                    && (line.starts_with(' ') || line.starts_with('\t'))
                    && !trimmed.is_empty()
                {
                    if !sq.desc.is_empty() {
                        sq.desc.push(' ');
                    }
                    sq.desc.push_str(trimmed);
                    continue;
                }
                in_definition = false;

                if trimmed.starts_with("DEFINITION") && sq.desc.is_empty() {
                    sq.desc = trimmed[10..].trim().to_string();
                    in_definition = true;
                } else if trimmed.starts_with("ACCESSION") && sq.acc.is_empty() {
                    sq.acc = trimmed[9..]
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .to_string();
                } else if trimmed == "ORIGIN" {
                    in_seq = true;
                    continue;
                }

                if in_seq {
                    for &ch in trimmed.as_bytes() {
                        if ch.is_ascii_alphabetic()
                            || matches!(ch, b'*' | b'~' | b'-' | b'.' | b'_')
                        {
                            if let Some(code) =
                                self.digitize_sequence_residue(ch, "GenBank sequence", &sq.name)?
                            {
                                sq.dsq.push(code);
                            }
                        }
                    }
                }
            }
        }

        sq.dsq.push(DSQ_SENTINEL);
        sq.n = sq.dsq.len() - 2;
        sq.l = sq.n;

        if sq.n == 0 && !allow_zero_length {
            return Err(HmmerError::Format(format!(
                "zero-length sequence record '{}'",
                sq.name
            )));
        }
        Ok(true)
    }

    fn digitize_fasta_residue(&self, ch: u8, seq_name: &str) -> HmmerResult<Option<Dsq>> {
        self.digitize_sequence_residue(ch, "FASTA sequence", seq_name)
    }

    fn is_accepted_record_start(&self, trimmed_line: &str) -> bool {
        if let Some(format) = self.asserted_format {
            return format.matches_record_start(trimmed_line);
        }
        trimmed_line.starts_with('>')
            || trimmed_line.starts_with("# STOCKHOLM")
            || trimmed_line.starts_with("ID ")
            || trimmed_line.starts_with("LOCUS ")
    }

    fn sequence_from_msa_row(&self, alignment: &msa::Msa, idx: usize) -> HmmerResult<Sequence> {
        let mut sq = Sequence::new();
        sq.name = alignment.sqname[idx].clone();
        sq.desc = alignment.sqdesc.get(idx).cloned().unwrap_or_default();
        for &ch in &alignment.aseq[idx] {
            if matches!(ch, b'-' | b'.' | b'_' | b'~') || ch.is_ascii_whitespace() {
                continue;
            }
            if let Some(code) =
                self.digitize_sequence_residue(ch, "Stockholm sequence", &sq.name)?
            {
                sq.dsq.push(code);
            }
        }
        sq.dsq.push(DSQ_SENTINEL);
        sq.n = sq.dsq.len() - 2;
        sq.l = sq.n;
        if sq.n == 0 {
            return Err(HmmerError::Format(format!(
                "zero-length sequence record '{}'",
                sq.name
            )));
        }
        Ok(sq)
    }

    fn digitize_sequence_residue(
        &self,
        ch: u8,
        format_name: &str,
        seq_name: &str,
    ) -> HmmerResult<Option<Dsq>> {
        let code = self.abc.digitize_symbol(ch);
        if code == DSQ_IGNORED {
            return Ok(None);
        }
        // C's per-format inmaps (`inmap_fasta`/`inmap_embl`, esl_sqio_ascii.c)
        // inherit the alphabet inmap, which `SetEquiv`s '.' and '_' to the gap
        // code, and then override only '-' -> ILLEGAL. So in digital read mode
        // '.' and '_' are accepted as gap residues (stored in the dsq, counted
        // toward sequence length) while '-' is rejected. Match that exactly:
        // accept the gap code only when it originated from '.' or '_'.
        if self.abc.is_gap(code) && (ch == b'.' || ch == b'_') {
            return Ok(Some(code));
        }
        if code == DSQ_ILLEGAL || (!self.abc.is_residue(code) && code != self.abc.nonresidue_code())
        {
            let display = if ch.is_ascii_graphic() || ch == b' ' {
                (ch as char).to_string()
            } else {
                format!("\\x{ch:02x}")
            };
            let where_text = if seq_name.is_empty() {
                format_name.to_string()
            } else {
                format!("{format_name} '{seq_name}'")
            };
            return Err(HmmerError::Format(format!(
                "Illegal symbol '{display}' in {where_text}"
            )));
        }
        Ok(Some(code))
    }
}

fn read_capped_line<R: BufRead>(reader: &mut R, line: &mut String) -> HmmerResult<usize> {
    line.clear();
    let mut total = 0usize;
    loop {
        let available = reader.fill_buf().map_err(HmmerError::Io)?;
        if available.is_empty() {
            return Ok(total);
        }
        let take = available
            .iter()
            .position(|&b| b == b'\n')
            .map_or(available.len(), |idx| idx + 1);
        total = total
            .checked_add(take)
            .ok_or_else(|| HmmerError::Format("sequence file line length overflow".to_string()))?;
        if total > MAX_SEQUENCE_LINE_LEN {
            return Err(HmmerError::Format(format!(
                "sequence file line exceeds {} bytes",
                MAX_SEQUENCE_LINE_LEN
            )));
        }
        let chunk = &available[..take];
        let text = std::str::from_utf8(chunk)
            .map_err(|e| HmmerError::Format(format!("invalid UTF-8 in sequence file: {e}")))?;
        line.push_str(text);
        let at_line_end = chunk.ends_with(b"\n");
        reader.consume(take);
        if at_line_end {
            return Ok(total);
        }
    }
}

/// Open a sequence file for reading with `abc` (port of `esl_sqfile_Open`).
///
/// Transparently wraps `.gz` files with a `flate2` decompressor; the
/// underlying reader is type-erased into `Box<dyn Read>`. Returns
/// [`HmmerError::Io`] if the path cannot be opened.
pub fn open_seq_file(
    path: &std::path::Path,
    abc: &Alphabet,
) -> HmmerResult<SeqFile<Box<dyn Read>>> {
    open_seq_file_inner(path, abc, None)
}

/// Open a sequence file and require FASTA records.
pub fn open_fasta_seq_file(
    path: &std::path::Path,
    abc: &Alphabet,
) -> HmmerResult<SeqFile<Box<dyn Read>>> {
    open_seq_file_inner(path, abc, Some(SequenceFormat::Fasta))
}

/// Open a sequence file and require a specific supported format.
pub fn open_seq_file_with_format(
    path: &std::path::Path,
    abc: &Alphabet,
    format: SequenceFormat,
) -> HmmerResult<SeqFile<Box<dyn Read>>> {
    open_seq_file_inner(path, abc, Some(format))
}

fn open_seq_file_inner(
    path: &std::path::Path,
    abc: &Alphabet,
    asserted_format: Option<SequenceFormat>,
) -> HmmerResult<SeqFile<Box<dyn Read>>> {
    if path == std::path::Path::new("-") {
        let reader: Box<dyn Read> = Box::new(std::io::stdin());
        let sqf = SeqFile::new(reader, abc.clone());
        return Ok(if let Some(format) = asserted_format {
            sqf.with_format(format)
        } else {
            sqf
        });
    }
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let reader: Box<dyn Read> = if path.extension().is_some_and(|e| e == "gz") {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let sqf = SeqFile::new(reader, abc.clone());
    Ok(if let Some(format) = asserted_format {
        sqf.with_format(format)
    } else {
        sqf
    })
}

#[cfg(test)]
#[path = "../tests/unit/sequence_test.rs"]
mod tests;
