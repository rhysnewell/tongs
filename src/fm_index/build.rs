use super::*;

impl FmIndex {
    /// Build an FM-index over `text` (typically uppercase ACGT, plus appended
    /// sentinel). Constructs the suffix array via divsufsort, derives the BWT
    /// from it, and tabulates the cumulative C[] array used for backward
    /// search. Counterpart to C's `fm_FM_read()` / makehmmerdb construction.
    pub fn build(text: &[u8]) -> Self {
        // Append sentinel
        let mut text_s = text.to_vec();
        text_s.push(0); // sentinel (smallest byte)
        let n = text_s.len();

        // Build suffix array using divsufsort
        let mut sa = vec![0i32; n];
        sort_in_place(&text_s, &mut sa);

        // Build BWT from suffix array
        let mut bwt = vec![0u8; n];
        for i in 0..n {
            let pos = sa[i] as usize;
            bwt[i] = if pos == 0 { 0 } else { text_s[pos - 1] };
        }

        // Build C array (count of characters < c in the text)
        let mut char_count = [0usize; 256];
        for &ch in &text_s {
            char_count[ch as usize] += 1;
        }
        let mut c = [0usize; 256];
        let mut cumsum = 0;
        for i in 0..256 {
            c[i] = cumsum;
            cumsum += char_count[i];
        }

        let orig_n = text.len();
        let occ_rank = OccRank::build(&bwt);
        FmIndex {
            bwt,
            mapped_bwt: None,
            packed_dna_bwt: None,
            sa,
            sa_samples: Vec::new(),
            regular_sa_samples: None,
            mapped_sa: None,
            c,
            n: orig_n,
            occ_rank,
        }
    }

    pub fn from_makehmmerdb_raw_parts(
        bwt: Vec<u8>,
        raw_sa_samples: Vec<i32>,
        term_loc: usize,
        freq_sa: usize,
        freq_cnt_b: usize,
        occ_b_codes: Vec<u16>,
        occ_sb_codes: Vec<u32>,
    ) -> Result<Self, String> {
        let bwt_len = bwt.len();
        if bwt_len == 0 {
            return Err("FM-index raw BWT is empty".to_string());
        }
        let text_len = bwt_len - 1;
        let mut char_count = [0usize; 256];
        for &ch in &bwt {
            char_count[ch as usize] += 1;
        }
        let mut c = [0usize; 256];
        let mut cumsum = 0usize;
        for i in 0..256 {
            c[i] = cumsum;
            cumsum += char_count[i];
        }
        Self::validate_parts_common(&bwt, c, text_len, bwt_len)?;

        let regular_sa_samples =
            Self::make_regular_sa_samples(raw_sa_samples, freq_sa, text_len, bwt_len)?;

        let occ_rank = OccRank::from_makehmmerdb_dna_tables(
            bwt_len,
            term_loc,
            freq_cnt_b,
            &occ_b_codes,
            &occ_sb_codes,
        )?;
        Ok(FmIndex {
            bwt,
            mapped_bwt: None,
            packed_dna_bwt: None,
            sa: Vec::new(),
            sa_samples: Vec::new(),
            regular_sa_samples: Some(regular_sa_samples),
            mapped_sa: None,
            c,
            n: text_len,
            occ_rank,
        })
    }

    fn make_regular_sa_samples(
        raw_sa_samples: Vec<i32>,
        freq_sa: usize,
        text_len: usize,
        bwt_len: usize,
    ) -> Result<RegularSaSamples, String> {
        if freq_sa == 0 {
            return Err("FM-index suffix-array sampling frequency is zero".to_string());
        }
        let mut positions = Vec::with_capacity(raw_sa_samples.len());
        for (idx, &pos) in raw_sa_samples.iter().enumerate() {
            let row = idx
                .checked_mul(freq_sa)
                .ok_or_else(|| "FM-index suffix-array sample row overflows usize".to_string())?;
            if row >= bwt_len {
                continue;
            }
            let pos = if pos < 0 {
                i32::try_from(text_len)
                    .map_err(|_| "FM-index suffix-array sample exceeds i32 range".to_string())?
            } else {
                pos
            };
            if pos as usize >= bwt_len {
                return Err("FM-index suffix-array sample is outside indexed text".to_string());
            }
            positions.push(pos);
        }
        Ok(RegularSaSamples::InMemory { positions, freq_sa })
    }

    fn make_mapped_regular_sa_samples(
        mapped_bytes: Arc<MmapBytes>,
        raw_sa_offset: usize,
        raw_sa_count: usize,
        freq_sa: usize,
        text_len: usize,
        bwt_len: usize,
    ) -> Result<RegularSaSamples, String> {
        if freq_sa == 0 {
            return Err("FM-index suffix-array sampling frequency is zero".to_string());
        }
        let raw_sa_bytes = raw_sa_count
            .checked_mul(4)
            .ok_or_else(|| "FM-index suffix-array sample span overflows usize".to_string())?;
        let slice = mapped_bytes
            .as_slice()
            .get(raw_sa_offset..raw_sa_offset + raw_sa_bytes)
            .ok_or_else(|| "FM-index suffix-array samples are outside mapped file".to_string())?;
        for (idx, chunk) in slice.as_chunks::<4>().0.iter().enumerate() {
            let row = idx
                .checked_mul(freq_sa)
                .ok_or_else(|| "FM-index suffix-array sample row overflows usize".to_string())?;
            if row >= bwt_len {
                continue;
            }
            let pos = i32::from_le_bytes(*chunk);
            let pos = if pos < 0 {
                i32::try_from(text_len)
                    .map_err(|_| "FM-index suffix-array sample exceeds i32 range".to_string())?
            } else {
                pos
            };
            if pos as usize >= bwt_len {
                return Err("FM-index suffix-array sample is outside indexed text".to_string());
            }
        }
        Ok(RegularSaSamples::Mapped {
            bytes: mapped_bytes,
            offset: raw_sa_offset,
            count: raw_sa_count,
            freq_sa,
            text_len,
        })
    }

    // One argument per table makehmmerdb writes, in file order.
    #[allow(clippy::too_many_arguments)]
    pub fn from_makehmmerdb_packed_dna_parts(
        packed_bwt: Vec<u8>,
        bwt_len: usize,
        raw_sa_samples: Vec<i32>,
        term_loc: usize,
        freq_sa: usize,
        freq_cnt_b: usize,
        occ_b_codes: Vec<u16>,
        occ_sb_codes: Vec<u32>,
    ) -> Result<Self, String> {
        if freq_cnt_b == 0 {
            return Err("FM-index occurrence sampling frequency is zero".to_string());
        }
        if bwt_len == 0 {
            return Err("FM-index raw BWT is empty".to_string());
        }
        let packed_bwt = PackedDnaBwt {
            byte_len: packed_bwt.len(),
            bytes: packed_bwt,
            mapped_bytes: None,
            mapped_offset: 0,
            len: bwt_len,
            term_loc,
        };
        packed_bwt.validate_span()?;
        if term_loc >= bwt_len {
            return Err("FM-index terminal location is outside BWT".to_string());
        }
        let text_len = bwt_len - 1;
        let nsb = 1 + bwt_len.div_ceil(FREQ_CNT_SB);
        let final_occ_sb_row = (nsb - 1) * 4;
        if occ_sb_codes.len() < final_occ_sb_row + 4 {
            return Err("FM-index occurrence superblock table has unexpected length".to_string());
        }
        let code0_total = occ_sb_codes[final_occ_sb_row] as usize;
        if code0_total == 0 {
            return Err("FM-index occurrence table is missing terminal symbol".to_string());
        }
        let counts = [
            1usize,
            code0_total - 1,
            occ_sb_codes[final_occ_sb_row + 1] as usize,
            occ_sb_codes[final_occ_sb_row + 2] as usize,
            occ_sb_codes[final_occ_sb_row + 3] as usize,
        ];
        if counts.iter().sum::<usize>() != bwt_len {
            return Err("FM-index occurrence table totals do not match BWT length".to_string());
        }
        let mut c = [0usize; 256];
        c[0] = 0;
        c[b'A' as usize] = counts[0];
        c[b'C' as usize] = counts[0] + counts[1];
        c[b'G' as usize] = counts[0] + counts[1] + counts[2];
        c[b'T' as usize] = counts[0] + counts[1] + counts[2] + counts[3];

        let regular_sa_samples =
            Self::make_regular_sa_samples(raw_sa_samples, freq_sa, text_len, bwt_len)?;

        let occ_rank = OccRank::from_makehmmerdb_dna_tables(
            bwt_len,
            term_loc,
            freq_cnt_b,
            &occ_b_codes,
            &occ_sb_codes,
        )?;
        Ok(FmIndex {
            bwt: Vec::new(),
            mapped_bwt: None,
            packed_dna_bwt: Some(packed_bwt),
            sa: Vec::new(),
            sa_samples: Vec::new(),
            regular_sa_samples: Some(regular_sa_samples),
            mapped_sa: None,
            c,
            n: text_len,
            occ_rank,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_makehmmerdb_mapped_packed_dna_parts_mapped_sa(
        mapped_packed_bwt: Arc<MmapBytes>,
        packed_bwt_offset: usize,
        packed_bwt_bytes: usize,
        bwt_len: usize,
        mapped_sa_bytes: Arc<MmapBytes>,
        raw_sa_offset: usize,
        raw_sa_count: usize,
        term_loc: usize,
        freq_sa: usize,
        freq_cnt_b: usize,
        occ_b_codes: Vec<u16>,
        occ_sb_codes: Vec<u32>,
    ) -> Result<Self, String> {
        let packed_bwt = PackedDnaBwt {
            bytes: Vec::new(),
            mapped_bytes: Some(mapped_packed_bwt),
            mapped_offset: packed_bwt_offset,
            byte_len: packed_bwt_bytes,
            len: bwt_len,
            term_loc,
        };
        packed_bwt.validate_span()?;
        if freq_cnt_b == 0 {
            return Err("FM-index occurrence sampling frequency is zero".to_string());
        }
        if bwt_len == 0 {
            return Err("FM-index raw BWT is empty".to_string());
        }
        if term_loc >= bwt_len {
            return Err("FM-index terminal location is outside BWT".to_string());
        }
        let text_len = bwt_len - 1;
        let nsb = 1 + bwt_len.div_ceil(FREQ_CNT_SB);
        let final_occ_sb_row = (nsb - 1) * 4;
        if occ_sb_codes.len() < final_occ_sb_row + 4 {
            return Err("FM-index occurrence superblock table has unexpected length".to_string());
        }
        let code0_total = occ_sb_codes[final_occ_sb_row] as usize;
        if code0_total == 0 {
            return Err("FM-index occurrence table is missing terminal symbol".to_string());
        }
        let counts = [
            1usize,
            code0_total - 1,
            occ_sb_codes[final_occ_sb_row + 1] as usize,
            occ_sb_codes[final_occ_sb_row + 2] as usize,
            occ_sb_codes[final_occ_sb_row + 3] as usize,
        ];
        if counts.iter().sum::<usize>() != bwt_len {
            return Err("FM-index occurrence table totals do not match BWT length".to_string());
        }
        let mut c = [0usize; 256];
        c[0] = 0;
        c[b'A' as usize] = counts[0];
        c[b'C' as usize] = counts[0] + counts[1];
        c[b'G' as usize] = counts[0] + counts[1] + counts[2];
        c[b'T' as usize] = counts[0] + counts[1] + counts[2] + counts[3];

        let regular_sa_samples = Self::make_mapped_regular_sa_samples(
            mapped_sa_bytes,
            raw_sa_offset,
            raw_sa_count,
            freq_sa,
            text_len,
            bwt_len,
        )?;
        let occ_rank = OccRank::from_makehmmerdb_dna_tables(
            bwt_len,
            term_loc,
            freq_cnt_b,
            &occ_b_codes,
            &occ_sb_codes,
        )?;
        Ok(FmIndex {
            bwt: Vec::new(),
            mapped_bwt: None,
            packed_dna_bwt: Some(packed_bwt),
            sa: Vec::new(),
            sa_samples: Vec::new(),
            regular_sa_samples: Some(regular_sa_samples),
            mapped_sa: None,
            c,
            n: text_len,
            occ_rank,
        })
    }
}
