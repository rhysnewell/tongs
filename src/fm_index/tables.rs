use super::*;

#[derive(Debug)]
pub(super) struct MappedSa {
    pub(super) bytes: Arc<MmapBytes>,
    pub(super) offset: usize,
    pub(super) len: usize,
}

#[derive(Debug)]
pub(super) struct MappedBwt {
    pub(super) bytes: Arc<MmapBytes>,
    pub(super) offset: usize,
    pub(super) len: usize,
}

#[derive(Debug)]
pub(super) struct PackedDnaBwt {
    pub(super) bytes: Vec<u8>,
    pub(super) mapped_bytes: Option<Arc<MmapBytes>>,
    pub(super) mapped_offset: usize,
    pub(super) byte_len: usize,
    pub(super) len: usize,
    pub(super) term_loc: usize,
}

impl PackedDnaBwt {
    #[inline]
    pub(super) fn byte_at(&self, idx: usize) -> Option<u8> {
        if let Some(bytes) = &self.mapped_bytes {
            bytes
                .as_slice()
                .get(self.mapped_offset.checked_add(idx)?)
                .copied()
        } else {
            self.bytes.get(idx).copied()
        }
    }

    pub(super) fn validate_span(&self) -> Result<(), String> {
        if self.byte_len != self.len.div_ceil(4) {
            return Err("FM-index packed BWT has unexpected length".to_string());
        }
        if let Some(bytes) = &self.mapped_bytes {
            self.mapped_offset
                .checked_add(self.byte_len)
                .filter(|&end| end <= bytes.as_slice().len())
                .ok_or_else(|| "FM-index mapped packed BWT is outside mapped file".to_string())?;
        } else if self.bytes.len() != self.byte_len {
            return Err("FM-index packed BWT has unexpected length".to_string());
        }
        Ok(())
    }

    #[inline]
    fn code_at(&self, pos: usize) -> u8 {
        (self
            .byte_at(pos / 4)
            .expect("validated packed BWT position must be in range")
            >> (6 - ((pos & 0x3) * 2)))
            & 0b11
    }

    #[inline]
    fn terminal_in_range(&self, start: usize, end: usize) -> usize {
        usize::from(start <= self.term_loc && self.term_loc < end)
    }

    #[inline]
    fn byte_count_eq(byte: u8, code: u8) -> usize {
        PACKED_BYTE_EQ_COUNTS[code as usize][byte as usize] as usize
    }

    #[inline]
    fn byte_count_lt(byte: u8, code: u8) -> usize {
        PACKED_BYTE_LT_COUNTS[code as usize][byte as usize] as usize
    }

    #[inline]
    fn count_code_eq(&self, mut start: usize, end: usize, code: u8) -> usize {
        let mut count = 0usize;
        while start < end && (start & 0x3) != 0 {
            count += usize::from(self.code_at(start) == code);
            start += 1;
        }
        while start + 4 <= end {
            count += Self::byte_count_eq(
                self.byte_at(start / 4)
                    .expect("validated packed BWT position must be in range"),
                code,
            );
            start += 4;
        }
        while start < end {
            count += usize::from(self.code_at(start) == code);
            start += 1;
        }
        count
    }

    #[inline]
    fn count_code_lt(&self, mut start: usize, end: usize, code: u8) -> usize {
        let mut count = 0usize;
        while start < end && (start & 0x3) != 0 {
            count += usize::from(self.code_at(start) < code);
            start += 1;
        }
        while start + 4 <= end {
            count += Self::byte_count_lt(
                self.byte_at(start / 4)
                    .expect("validated packed BWT position must be in range"),
                code,
            );
            start += 4;
        }
        while start < end {
            count += usize::from(self.code_at(start) < code);
            start += 1;
        }
        count
    }

    #[inline]
    pub(super) fn count_eq(&self, start: usize, end: usize, ch: u8) -> usize {
        match ch {
            0 => self.terminal_in_range(start, end),
            b'A' => self.count_code_eq(start, end, 0) - self.terminal_in_range(start, end),
            b'C' => self.count_code_eq(start, end, 1),
            b'G' => self.count_code_eq(start, end, 2),
            b'T' => self.count_code_eq(start, end, 3),
            _ => 0,
        }
    }

    #[inline]
    pub(super) fn count_lt(&self, start: usize, end: usize, ch: u8) -> usize {
        match ch {
            0 => 0,
            b'A' => self.terminal_in_range(start, end),
            b'C' => self.count_code_lt(start, end, 1),
            b'G' => self.count_code_lt(start, end, 2),
            b'T' => self.count_code_lt(start, end, 3),
            _ => end - start,
        }
    }
}

#[derive(Debug)]
pub(super) enum RegularSaSamples {
    InMemory {
        positions: Vec<i32>,
        freq_sa: usize,
    },
    Mapped {
        bytes: Arc<MmapBytes>,
        offset: usize,
        count: usize,
        freq_sa: usize,
        text_len: usize,
    },
}

impl RegularSaSamples {
    pub(super) fn position_for_row(&self, row: usize) -> Option<usize> {
        match self {
            RegularSaSamples::InMemory { positions, freq_sa } => {
                if !row.is_multiple_of(*freq_sa) {
                    return None;
                }
                positions.get(row / *freq_sa).map(|&pos| pos as usize)
            }
            RegularSaSamples::Mapped {
                bytes,
                offset,
                count,
                freq_sa,
                text_len,
            } => {
                if !row.is_multiple_of(*freq_sa) {
                    return None;
                }
                let idx = row / *freq_sa;
                if idx >= *count {
                    return None;
                }
                let offset = offset.checked_add(idx.checked_mul(4)?)?;
                let chunk = bytes.as_slice().get(offset..offset + 4)?;
                let pos = i32::from_le_bytes(chunk.try_into().ok()?);
                Some(if pos < 0 { *text_len } else { pos as usize })
            }
        }
    }
}

/// Two-level sampled rank tables over the BWT, mirroring C `FM_DATA`'s
/// `occCnts_sb` (superblock cumulative) and `occCnts_b` (u16 within-superblock
/// block cumulative). They make `Occ(c, pos)` O(`FREQ_CNT_B`) instead of O(pos).
/// Counts are kept only for the symbols actually present in the BWT.
#[derive(Debug)]
pub(super) struct OccRank {
    /// Distinct BWT symbols, in byte order.
    pub(super) symbols: Vec<u8>,
    /// byte -> index into `symbols`, or -1 if the byte never occurs.
    pub(super) sym_of: Vec<i16>,
    /// `occ_sb[sb * n_sym + s]` = count of symbol `s` in `bwt[0 .. sb*FREQ_CNT_SB]`.
    pub(super) occ_sb: Vec<usize>,
    /// `occ_b[b * n_sym + s]` = count of symbol `s` in `bwt[sb_start .. b*FREQ_CNT_B]`
    /// where `sb_start` is the start of the superblock containing block `b`.
    pub(super) occ_b: Vec<u16>,
}

impl OccRank {
    pub(super) fn build(bwt: &[u8]) -> Self {
        let mut present = [false; 256];
        for &b in bwt {
            present[b as usize] = true;
        }
        let mut sym_of = vec![-1i16; 256];
        let mut symbols = Vec::new();
        for (b, &p) in present.iter().enumerate() {
            if p {
                sym_of[b] = symbols.len() as i16;
                symbols.push(b as u8);
            }
        }
        let ns = symbols.len().max(1);
        let nb = bwt.len() / FREQ_CNT_B + 1;
        let nsb = bwt.len() / FREQ_CNT_SB + 1;
        let mut occ_sb = vec![0usize; nsb * ns];
        let mut occ_b = vec![0u16; nb * ns];
        let mut running = vec![0usize; ns];
        let mut sb_base = vec![0usize; ns];
        // Iterate inclusively to `bwt.len()` so the trailing sample row(s) at an
        // exact block/superblock boundary are written. Without this, when
        // `bwt.len()` is a multiple of FREQ_CNT_B the last block row stays 0 and
        // `occ(ch, bwt.len())` (used by the very first backward-search step from
        // the root interval) wrongly returns 0 — making every FM search return
        // no hits for texts of length ≡ 255 (mod 256).
        for pos in 0..=bwt.len() {
            // Record samples for the count over bwt[0..pos] (exclusive of pos),
            // matching the half-open Occ semantics used everywhere else.
            if pos % FREQ_CNT_SB == 0 {
                let sb = pos / FREQ_CNT_SB;
                for s in 0..ns {
                    occ_sb[sb * ns + s] = running[s];
                    sb_base[s] = running[s];
                }
            }
            if pos % FREQ_CNT_B == 0 {
                let b = pos / FREQ_CNT_B;
                for s in 0..ns {
                    occ_b[b * ns + s] = (running[s] - sb_base[s]) as u16;
                }
            }
            if pos < bwt.len() {
                let s = sym_of[bwt[pos] as usize];
                running[s as usize] += 1;
            }
        }
        OccRank {
            symbols,
            sym_of,
            occ_sb,
            occ_b,
        }
    }

    #[inline]
    pub(super) fn n_sym(&self) -> usize {
        self.symbols.len().max(1)
    }

    pub(super) fn from_makehmmerdb_dna_tables(
        bwt_len: usize,
        term_loc: usize,
        freq_cnt_b: usize,
        occ_b_codes: &[u16],
        occ_sb_codes: &[u32],
    ) -> Result<Self, String> {
        if freq_cnt_b == 0 {
            return Err("FM-index occurrence sampling frequency is zero".to_string());
        }
        let nb = 1 + bwt_len.div_ceil(freq_cnt_b);
        let nsb = 1 + bwt_len.div_ceil(FREQ_CNT_SB);
        if occ_b_codes.len() != nb * 4 {
            return Err("FM-index occurrence block table has unexpected length".to_string());
        }
        if occ_sb_codes.len() != nsb * 4 {
            return Err("FM-index occurrence superblock table has unexpected length".to_string());
        }

        let symbols = vec![0, b'A', b'C', b'G', b'T'];
        let mut sym_of = vec![-1i16; 256];
        for (idx, &sym) in symbols.iter().enumerate() {
            sym_of[sym as usize] = idx as i16;
        }

        let ns = symbols.len();
        let mut occ_b = vec![0u16; nb * ns];
        let mut occ_sb = vec![0usize; nsb * ns];

        for row in 0..nb {
            let pos = (row * freq_cnt_b).min(bwt_len);
            let sb_start = (pos / FREQ_CNT_SB) * FREQ_CNT_SB;
            let terminal_in_block = sb_start <= term_loc && term_loc < pos;
            occ_b[row * ns] = terminal_in_block as u16;
            let code0 = occ_b_codes[row * 4];
            occ_b[row * ns + 1] = code0.saturating_sub(terminal_in_block as u16);
            occ_b[row * ns + 2] = occ_b_codes[row * 4 + 1];
            occ_b[row * ns + 3] = occ_b_codes[row * 4 + 2];
            occ_b[row * ns + 4] = occ_b_codes[row * 4 + 3];
        }

        for row in 0..nsb {
            let pos = (row * FREQ_CNT_SB).min(bwt_len);
            let terminal_in_prefix = term_loc < pos;
            occ_sb[row * ns] = terminal_in_prefix as usize;
            let code0 = occ_sb_codes[row * 4] as usize;
            occ_sb[row * ns + 1] = code0.saturating_sub(terminal_in_prefix as usize);
            occ_sb[row * ns + 2] = occ_sb_codes[row * 4 + 1] as usize;
            occ_sb[row * ns + 3] = occ_sb_codes[row * 4 + 2] as usize;
            occ_sb[row * ns + 4] = occ_sb_codes[row * 4 + 3] as usize;
        }

        Ok(Self {
            symbols,
            sym_of,
            occ_sb,
            occ_b,
        })
    }
}
