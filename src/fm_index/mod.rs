//! FM-index for fast pattern searching in DNA sequences.
//! Used by nhmmer for long-target scanning.

use divsufsort::sort_in_place;
use memmap2::Mmap;
use std::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

mod tables;
use tables::*;
mod build;
mod parts;
mod query;

/// Superblock sampling interval for the cumulative occurrence counts (C
/// `FM_METADATA.freq_cnt_sb`; makehmmerdb default 65536).
const FREQ_CNT_SB: usize = 65536;
/// Block sampling interval for the within-superblock occurrence counts (C
/// `FM_METADATA.freq_cnt_b`; makehmmerdb default 256).
const FREQ_CNT_B: usize = 256;
const DEFAULT_SA_SAMPLE_RATE: usize = 32;

const PACKED_BYTE_EQ_COUNTS: [[u8; 256]; 4] = build_packed_byte_eq_counts();
const PACKED_BYTE_LT_COUNTS: [[u8; 256]; 4] = build_packed_byte_lt_counts();

const fn build_packed_byte_eq_counts() -> [[u8; 256]; 4] {
    let mut table = [[0u8; 256]; 4];
    let mut code = 0usize;
    while code < 4 {
        let mut byte = 0usize;
        while byte < 256 {
            let b = byte as u8;
            table[code][byte] = (((b >> 6) & 0b11) == code as u8) as u8
                + (((b >> 4) & 0b11) == code as u8) as u8
                + (((b >> 2) & 0b11) == code as u8) as u8
                + ((b & 0b11) == code as u8) as u8;
            byte += 1;
        }
        code += 1;
    }
    table
}

const fn build_packed_byte_lt_counts() -> [[u8; 256]; 4] {
    let mut table = [[0u8; 256]; 4];
    let mut code = 0usize;
    while code < 4 {
        let mut byte = 0usize;
        while byte < 256 {
            let b = byte as u8;
            table[code][byte] = (((b >> 6) & 0b11) < code as u8) as u8
                + (((b >> 4) & 0b11) < code as u8) as u8
                + (((b >> 2) & 0b11) < code as u8) as u8
                + ((b & 0b11) < code as u8) as u8;
            byte += 1;
        }
        code += 1;
    }
    table
}

pub struct MmapBytes {
    storage: MmapStorage,
}

enum MmapStorage {
    Mmap(Mmap),
    Empty,
}

impl MmapBytes {
    pub fn open(path: &Path, max_len: u64) -> Result<Arc<Self>, String> {
        let file = File::open(path).map_err(|e| {
            format!(
                "failed to open FM-index target database {}: {e}",
                path.display()
            )
        })?;
        let len = file
            .metadata()
            .map_err(|e| {
                format!(
                    "failed to stat FM-index target database {}: {e}",
                    path.display()
                )
            })?
            .len();
        if len > max_len {
            return Err(format!(
                "FM-index target database {} is too large for the current in-memory reader ({} bytes > {} bytes)",
                path.display(),
                len,
                max_len
            ));
        }
        let len = usize::try_from(len)
            .map_err(|_| "FM-index target database size overflows usize".to_string())?;

        if len == 0 {
            Ok(Arc::new(Self {
                storage: MmapStorage::Empty,
            }))
        } else {
            let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
                format!(
                    "failed to mmap FM-index target database {}: {e}",
                    path.display()
                )
            })?;
            Ok(Arc::new(Self {
                storage: MmapStorage::Mmap(mmap),
            }))
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        match &self.storage {
            MmapStorage::Mmap(mmap) => mmap,
            MmapStorage::Empty => &[],
        }
    }
}

impl fmt::Debug for MmapBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmapBytes")
            .field("len", &self.as_slice().len())
            .finish()
    }
}

/// A simple FM-index for DNA sequences. Field layout mirrors a reduced C
/// `FM_DATA`: the BWT, sampled suffix array, cumulative `C[]` table, and the
/// two-level sampled occurrence-count rank (`occ`, = C `occCnts_sb`/`occCnts_b`).
#[derive(Debug)]
pub struct FmIndex {
    /// Burrows-Wheeler transform
    pub bwt: Vec<u8>,
    mapped_bwt: Option<MappedBwt>,
    packed_dna_bwt: Option<PackedDnaBwt>,
    /// Suffix array
    pub sa: Vec<i32>,
    /// Sparse suffix-array samples as `(BWT row, text position)`.
    sa_samples: Vec<(i32, i32)>,
    regular_sa_samples: Option<RegularSaSamples>,
    mapped_sa: Option<MappedSa>,
    /// Count of each character in BWT prefix (C array)
    pub c: [usize; 256],
    /// Length of the indexed text
    pub n: usize,
    /// Sampled rank tables (C `FM_DATA.occCnts_sb`/`occCnts_b`).
    occ_rank: OccRank,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FmInterval {
    pub lo: usize,
    pub hi: usize,
}

impl FmInterval {
    pub fn len(self) -> usize {
        self.hi.saturating_sub(self.lo)
    }

    pub fn is_empty(self) -> bool {
        self.lo >= self.hi
    }
}

impl FmIndex {
    /// Return the full suffix-array interval before any pattern characters are
    /// applied.
    pub fn bwt_len(&self) -> usize {
        if let Some(mapped) = &self.mapped_bwt {
            mapped.len
        } else if let Some(packed) = &self.packed_dna_bwt {
            packed.len
        } else {
            self.bwt.len()
        }
    }

    fn bwt_slice(&self) -> &[u8] {
        if let Some(mapped) = &self.mapped_bwt {
            &mapped.bytes.as_slice()[mapped.offset..mapped.offset + mapped.len]
        } else {
            &self.bwt
        }
    }

    fn bwt_at(&self, pos: usize) -> Option<u8> {
        if let Some(packed) = &self.packed_dna_bwt {
            if pos >= packed.len {
                return None;
            }
            if pos == packed.term_loc {
                return Some(0);
            }
            let byte = packed.byte_at(pos / 4)?;
            return Some(match (byte >> (6 - ((pos & 0x3) * 2))) & 0b11 {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                3 => b'T',
                _ => unreachable!(),
            });
        }
        self.bwt_slice().get(pos).copied()
    }

    pub fn root_interval(&self) -> FmInterval {
        FmInterval {
            lo: 0,
            hi: self.bwt_len(),
        }
    }

    /// Apply one backward-search step to an existing interval.
    ///
    /// This is the primitive C's FM trie search uses while recursively adding
    /// bases and pruning empty suffix-array intervals before exploring deeper
    /// seeds.
    pub fn prepend_interval(&self, interval: FmInterval, ch: u8) -> Option<FmInterval> {
        if ch == 0 || interval.is_empty() || interval.hi > self.bwt_len() {
            return None;
        }

        let lo = self.c[ch as usize].checked_add(self.occ(ch, interval.lo))?;
        let hi = self.c[ch as usize].checked_add(self.occ(ch, interval.hi))?;
        (lo < hi).then_some(FmInterval { lo, hi })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/fm_index_test.rs"]
mod tests;
