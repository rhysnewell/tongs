use super::*;
pub(super) use tongs::util::random::Mt19937;

#[derive(Debug, Clone)]
pub(super) struct SequenceMetadata {
    pub(super) target_id: usize,
    pub(super) target_start: usize,
    pub(super) fm_start: usize,
    pub(super) length: usize,
    pub(super) block_id: usize,
    pub(super) block_offset: usize,
    pub(super) overlap_bases: usize,
    pub(super) name: String,
    pub(super) acc: String,
    pub(super) desc: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AmbiguityRange {
    pub(super) lower: usize,
    pub(super) upper: usize,
}

#[derive(Debug, Clone)]
pub(super) struct BlockRecord {
    pub(super) id: usize,
    pub(super) fm_start: usize,
    pub(super) length: usize,
    pub(super) seq_offset: usize,
    pub(super) seq_count: usize,
    pub(super) ambig_offset: usize,
    pub(super) ambig_count: usize,
    pub(super) overlap_bases: usize,
}

pub(super) struct IndexRecord {
    pub(super) block_id: usize,
    pub(super) kind: u32,
    pub(super) text_start: usize,
    pub(super) text_len: usize,
    pub(super) seq_offset: usize,
    pub(super) seq_count: usize,
    pub(super) ambig_offset: usize,
    pub(super) ambig_count: usize,
    pub(super) overlap_bases: usize,
    pub(super) fm: FmIndex,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct FmAlphabet {
    pub(super) alph_type: u8,
    pub(super) alph_size: u8,
    pub(super) char_bits: u8,
    pub(super) amino: bool,
}

impl FmAlphabet {
    pub(super) fn dna() -> Self {
        Self {
            alph_type: FM_DNA_ALPH_TYPE,
            alph_size: FM_DNA_ALPH_SIZE,
            char_bits: FM_DNA_CHAR_BITS,
            amino: false,
        }
    }

    pub(super) fn amino() -> Self {
        Self {
            alph_type: FM_AMINO_ALPH_TYPE,
            alph_size: FM_AMINO_ALPH_SIZE,
            char_bits: FM_AMINO_CHAR_BITS,
            amino: true,
        }
    }

    pub(super) fn encode_text_without_separators(
        self,
        text: &[u8],
        rng: &mut Mt19937,
    ) -> std::io::Result<Vec<u8>> {
        if self.amino {
            c_amino_text_without_separators(text)
        } else {
            c_dna_text_without_separators(text, rng)
        }
    }

    pub(super) fn pack(self, values: &[u8]) -> Vec<u8> {
        if self.amino {
            values.to_vec()
        } else {
            pack_dna_quads(values)
        }
    }
}

pub(super) fn dna_replacement_code(rng: &mut Mt19937) -> u8 {
    (rng.next_f64() * 4.0) as u8
}

/// Seed used by C makehmmerdb for degenerate-residue replacement
/// (`esl_randomness_Create(42)`, makehmmerdb.c:426).
pub(super) const MAKEHMMERDB_RNG_SEED: u32 = 42;
