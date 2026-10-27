use super::*;

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

/// Faithful port of Easel's 32-bit Mersenne Twister (MT19937), the generator
/// behind `esl_random()` for a randomness object created with
/// `esl_randomness_Create(seed)` (`r->type == eslRND_MERSENNE`; see
/// `hmmer/easel/esl_random.c` `mersenne_twister`/`mersenne_seed_table`/
/// `mersenne_fill_table`).
///
/// NOTE on why this lives here rather than in `src/util/random.rs`: that module
/// owns the RNG API but its `MersenneTwister` type is actually Easel's *fast*
/// Knuth LCG (`esl_randomness_CreateFast`), and its `Rand64` is the 64-bit
/// MT19937-64 (`esl_rand64`). Neither matches the 32-bit MT19937 stream that
/// `esl_random()` produces for makehmmerdb's `esl_randomness_Create(42)` RNG, so
/// a faithful, byte-identical degenerate-residue replacement needs this exact
/// generator. It is kept private to makehmmerdb to avoid disturbing the shared
/// module that another part of the codebase depends on.
pub(super) struct Mt19937 {
    pub(super) mt: [u32; 624],
    pub(super) mti: usize,
}

impl Mt19937 {
    /// Seed exactly as Easel `mersenne_seed_table`: `mt[0]=seed`, then
    /// `mt[z] = 69069 * mt[z-1]` (32-bit wrapping). `mti` is set to 624 so the
    /// first draw triggers a table fill, matching `esl_randomness_Init`'s
    /// `r->mti = 0` followed by the `if (r->mti >= 624) mersenne_fill_table`
    /// guard on the first `mersenne_twister` call (Easel initializes mti=0 but
    /// fills lazily; starting at 624 reproduces the identical first fill).
    pub(super) fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for z in 1..624 {
            mt[z] = 69069u32.wrapping_mul(mt[z - 1]);
        }
        Mt19937 { mt, mti: 624 }
    }

    fn fill_table(&mut self) {
        const MAG01: [u32; 2] = [0x0, 0x9908b0df];
        for z in 0..227 {
            let y = (self.mt[z] & 0x80000000) | (self.mt[z + 1] & 0x7fffffff);
            self.mt[z] = self.mt[z + 397] ^ (y >> 1) ^ MAG01[(y & 0x1) as usize];
        }
        for z in 227..623 {
            let y = (self.mt[z] & 0x80000000) | (self.mt[z + 1] & 0x7fffffff);
            self.mt[z] = self.mt[z - 227] ^ (y >> 1) ^ MAG01[(y & 0x1) as usize];
        }
        let y = (self.mt[623] & 0x80000000) | (self.mt[0] & 0x7fffffff);
        self.mt[623] = self.mt[396] ^ (y >> 1) ^ MAG01[(y & 0x1) as usize];
        self.mti = 0;
    }

    /// Draw a tempered 32-bit variate (Easel `mersenne_twister`).
    pub(super) fn next_u32(&mut self) -> u32 {
        if self.mti >= 624 {
            self.fill_table();
        }
        let mut x = self.mt[self.mti];
        self.mti += 1;
        x ^= x >> 11;
        x ^= (x << 7) & 0x9d2c5680;
        x ^= (x << 15) & 0xefc60000;
        x ^= x >> 18;
        x
    }

    /// `esl_random(r)`: uniform double on [0,1) = `next_u32() / 2^32`.
    fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / 4294967296.0
    }

    /// Reproduce C's `(int)(esl_random(r) * 4)` -> a canonical DNA code in 0..3.
    /// makehmmerdb.c:661 picks `meta->alph[(int)(esl_random(r)*4)]` from
    /// `"ACGT"` and stores `inv_alph[that]`, which equals the drawn index, so
    /// the FM code is exactly this 0..3 value (A=0,C=1,G=2,T=3).
    pub(super) fn dna_replacement_code(&mut self) -> u8 {
        (self.next_f64() * 4.0) as u8
    }
}

/// Seed used by C makehmmerdb for degenerate-residue replacement
/// (`esl_randomness_Create(42)`, makehmmerdb.c:426).
pub(super) const MAKEHMMERDB_RNG_SEED: u32 = 42;
