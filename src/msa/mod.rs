//! Multiple Sequence Alignment (MSA) I/O.

use crate::alphabet::{Alphabet, Dsq, DSQ_ILLEGAL};
use crate::errors::{HmmerError, HmmerResult};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

mod parse;
use parse::*;
mod phylip;
use phylip::*;
mod rows;
use rows::*;
mod stockholm;
pub use stockholm::*;
mod stockholm_block;
use stockholm_block::*;
mod full;
pub use full::*;
mod write_full;
pub use write_full::*;
mod write;
pub use write::*;

const MAX_TEXT_MSA_BYTES: usize = 512 * 1024 * 1024;

fn read_text_msa_to_string<R: Read>(reader: &mut R) -> HmmerResult<String> {
    read_text_msa_to_string_with_limit(reader, MAX_TEXT_MSA_BYTES)
}

fn read_text_msa_to_string_with_limit<R: Read>(
    reader: &mut R,
    limit: usize,
) -> HmmerResult<String> {
    let mut bytes = Vec::new();
    let max_read = u64::try_from(limit).unwrap_or(u64::MAX).saturating_add(1);
    reader
        .take(max_read)
        .read_to_end(&mut bytes)
        .map_err(HmmerError::Io)?;
    if bytes.len() > limit {
        return Err(HmmerError::Format(format!(
            "MSA input exceeds {} bytes",
            limit
        )));
    }
    String::from_utf8(bytes)
        .map_err(|e| HmmerError::Format(format!("invalid UTF-8 in MSA input: {e}")))
}

/// Stockholm model-specific bit score cutoffs from `#=GF GA/TC/NC`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StockholmCutoffs {
    pub ga: Option<[f32; 2]>,
    pub tc: Option<[f32; 2]>,
    pub nc: Option<[f32; 2]>,
}

/// A multiple sequence alignment.
#[derive(Debug, Clone)]
pub struct Msa {
    /// Alignment name (from #=GF ID)
    pub name: String,
    /// Alignment accession (from #=GF AC)
    pub acc: Option<String>,
    /// Alignment description (from #=GF DE)
    pub desc: Option<String>,
    /// Alignment author/provenance (from #=GF AU)
    pub author: Option<String>,
    /// Sequence names
    pub sqname: Vec<String>,
    /// Per-sequence accessions (`#=GS <seq> AC`)
    pub sqacc: Vec<String>,
    /// Per-sequence descriptions (`#=GS <seq> DE`)
    pub sqdesc: Vec<String>,
    /// Per-sequence relative weights (`#=GS <seq> WT`) if supplied.
    pub weights: Option<Vec<f64>>,
    /// Aligned sequences (text, with gap characters)
    pub aseq: Vec<Vec<u8>>,
    /// Per-sequence posterior probability annotation (`#=GR <seq> PP`)
    pub pp: Vec<Option<Vec<u8>>>,
    /// Number of sequences
    pub nseq: usize,
    /// Alignment length (columns)
    pub alen: usize,
    /// Reference annotation (#=GC RF)
    pub rf: Option<Vec<u8>>,
    /// Model mask annotation (#=GC MM)
    pub mm: Option<Vec<u8>>,
    /// Consensus secondary structure annotation (#=GC SS_cons)
    pub ss_cons: Option<Vec<u8>>,
    /// Consensus surface accessibility annotation (#=GC SA_cons)
    pub sa_cons: Option<Vec<u8>>,
    /// Consensus posterior probability annotation (#=GC PP_cons)
    pub pp_cons: Option<Vec<u8>>,
}

/// A parsed Stockholm alignment plus the original body lines needed for
/// metadata-preserving round trips.
#[derive(Debug, Clone)]
pub struct StockholmMsa {
    pub msa: Msa,
    pub cutoffs: StockholmCutoffs,
    pub body_lines: Vec<String>,
}

impl Msa {
    /// Validate that every stored alignment character can be digitized by
    /// `abc`. Gap characters are accepted explicitly because aligned MSA text
    /// maps them to the alphabet gap code before normal symbol digitization.
    pub fn validate_digitizable(&self, abc: &Alphabet) -> HmmerResult<()> {
        for (idx, seq) in self.aseq.iter().enumerate() {
            for &ch in seq {
                if ch == b'-' || ch == b'.' {
                    continue;
                }
                let code = abc.digitize_symbol(ch);
                if code == DSQ_ILLEGAL {
                    let display = if ch.is_ascii_graphic() || ch == b' ' {
                        (ch as char).to_string()
                    } else {
                        format!("\\x{ch:02x}")
                    };
                    let name = self.sqname.get(idx).map(String::as_str).unwrap_or("?");
                    return Err(HmmerError::Format(format!(
                        "Stockholm sequence {name} contains illegal symbol '{display}'"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Digitize a text-mode alignment into Easel-style digital rows
    /// (port of `esl_msa_Digitize`, returning the rows instead of
    /// mutating in place).
    ///
    /// Each row is 1-based with `DSQ_SENTINEL` bytes flanking the
    /// alignment columns. Aligned gap characters (`-` or `.`) are mapped
    /// to the alphabet's gap code; symbols that the alphabet ignores
    /// (e.g. whitespace) are skipped silently.
    pub fn digitize(&self, abc: &Alphabet) -> Vec<Vec<Dsq>> {
        let gap = abc.gap_code();
        self.aseq
            .iter()
            .map(|seq| {
                let mut dsq = Vec::with_capacity(self.alen + 2);
                dsq.push(crate::alphabet::DSQ_SENTINEL);
                for &ch in seq {
                    if ch == b'-' || ch == b'.' {
                        dsq.push(gap);
                    } else {
                        let code = abc.digitize_symbol(ch);
                        if code != crate::alphabet::DSQ_IGNORED {
                            dsq.push(code);
                        }
                    }
                }
                dsq.push(crate::alphabet::DSQ_SENTINEL);
                dsq
            })
            .collect()
    }
}

/// Compute the Easel 32-bit alignment checksum (`esl_msa_Checksum`).
///
/// Considers only alignment data (digital symbols, columns 1..alen of every
/// sequence), so two MSAs with identical columns but different annotation
/// hash the same. Used to verify that an alignment matches a known
/// reference, e.g. when `hmmalign --mapali` is mapping new sequences onto
/// the seed alignment an HMM was built from. Implements the variant of
/// Jenkins' hash from `esl_keyhash`.
pub fn checksum(msa: &Msa, abc: &Alphabet) -> u32 {
    let mut val = 0u32;
    for row in msa.digitize(abc) {
        for &sym in row.iter().skip(1).take(msa.alen) {
            val = val.wrapping_add(sym as u32);
            val = val.wrapping_add(val << 10);
            val ^= val >> 6;
        }
    }
    val = val.wrapping_add(val << 3);
    val ^= val >> 11;
    val = val.wrapping_add(val << 15);
    val
}

/// Read every Stockholm alignment in `path` (convenience wrapper that opens
/// the file and dispatches to [`read_stockholm_from_reader`]).
pub fn read_stockholm(path: &Path) -> HmmerResult<Vec<Msa>> {
    Ok(read_stockholm_preserved(path)?
        .into_iter()
        .map(|stockholm| stockholm.msa)
        .collect())
}

/// Read every Stockholm alignment in `path`, preserving original body lines
/// alongside parsed MSA fields.
pub fn read_stockholm_preserved(path: &Path) -> HmmerResult<Vec<StockholmMsa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let reader = BufReader::new(file);
    read_stockholm_preserved_from_reader(reader)
}

/// Read one aligned FASTA/AFA alignment from `path`.
pub fn read_afa(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    read_afa_from_reader(&mut reader, alignment_name_from_path(path))
}

/// Read one aligned FASTA/AFA alignment from an open reader.
pub fn read_afa_from_reader<R: Read>(
    reader: &mut R,
    alignment_name: String,
) -> HmmerResult<Vec<Msa>> {
    let text = read_text_msa_to_string(reader)?;
    Ok(vec![parse_afa_alignment(&text, alignment_name)?])
}

/// Read one A2M alignment from `path`.
pub fn read_a2m(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    read_a2m_from_reader(&mut reader, alignment_name_from_path(path))
}

/// Read one A2M alignment from an open reader.
///
/// A2M marks consensus columns with uppercase residues or `-`; lowercase
/// residues are inserts, and `.` insert-column placeholders are ignored.
/// Rows may therefore have unequal raw lengths as long as they contain the
/// same number of consensus columns.
pub fn read_a2m_from_reader<R: Read>(
    reader: &mut R,
    alignment_name: String,
) -> HmmerResult<Vec<Msa>> {
    let text = read_text_msa_to_string(reader)?;
    Ok(vec![parse_a2m_alignment(&text, alignment_name)?])
}

/// Read one PSIBLAST alignment from `path`.
pub fn read_psiblast(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    read_psiblast_from_reader(&mut reader, alignment_name_from_path(path))
}

/// Read one PSIBLAST alignment from an open reader.
pub fn read_psiblast_from_reader<R: Read>(
    reader: &mut R,
    alignment_name: String,
) -> HmmerResult<Vec<Msa>> {
    let text = read_text_msa_to_string(reader)?;
    Ok(vec![parse_psiblast_alignment(&text, alignment_name)?])
}

/// Read one CLUSTAL/CLUSTAL-like alignment from `path`.
pub fn read_clustal(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    read_clustal_from_reader(&mut reader, alignment_name_from_path(path))
}

/// Read one CLUSTAL/CLUSTAL-like alignment from an open reader.
pub fn read_clustal_from_reader<R: Read>(
    reader: &mut R,
    alignment_name: String,
) -> HmmerResult<Vec<Msa>> {
    let text = read_text_msa_to_string(reader)?;
    Ok(vec![parse_clustal_alignment(&text, alignment_name)?])
}

/// Read one SELEX alignment from `path`.
pub fn read_selex(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    read_selex_from_reader(&mut reader, alignment_name_from_path(path))
}

/// Read one SELEX alignment from an open reader.
pub fn read_selex_from_reader<R: Read>(
    reader: &mut R,
    alignment_name: String,
) -> HmmerResult<Vec<Msa>> {
    let text = read_text_msa_to_string(reader)?;
    Ok(vec![parse_selex_alignment(&text, alignment_name)?])
}

/// Read one interleaved PHYLIP alignment from `path`.
pub fn read_phylip(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    read_phylip_from_reader(&mut reader, alignment_name_from_path(path))
}

/// Read one interleaved PHYLIP alignment from an open reader.
pub fn read_phylip_from_reader<R: Read>(
    reader: &mut R,
    alignment_name: String,
) -> HmmerResult<Vec<Msa>> {
    let text = read_text_msa_to_string(reader)?;
    Ok(vec![parse_interleaved_phylip_alignment(
        &text,
        alignment_name,
    )?])
}

/// Read one sequential PHYLIP alignment from `path`.
pub fn read_phylips(path: &Path) -> HmmerResult<Vec<Msa>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    read_phylips_from_reader(&mut reader, alignment_name_from_path(path))
}

/// Read one sequential PHYLIP alignment from an open reader.
pub fn read_phylips_from_reader<R: Read>(
    reader: &mut R,
    alignment_name: String,
) -> HmmerResult<Vec<Msa>> {
    let text = read_text_msa_to_string(reader)?;
    Ok(vec![parse_sequential_phylip_alignment(
        &text,
        alignment_name,
    )?])
}

#[cfg(test)]
#[path = "../../tests/unit/msa_test.rs"]
mod tests;
