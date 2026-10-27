use super::*;

impl FmIndex {
    /// Reconstruct an FM-index from serialized components.
    ///
    /// `text_len` is the length of the indexed text before the internal
    /// sentinel. Serialized makehmmerdb container indexes store BWT/SA/C arrays
    /// separately, so nhmmer uses this constructor when loading those records.
    pub fn from_parts(
        bwt: Vec<u8>,
        sa: Vec<i32>,
        c: [usize; 256],
        text_len: usize,
    ) -> Result<Self, String> {
        Self::from_parts_inner(bwt, sa, c, text_len, None)
    }

    /// Reconstruct an FM-index while retaining only sampled suffix-array rows.
    ///
    /// This preserves FM locate semantics and trades up to `sample_rate - 1`
    /// LF steps per located row for substantially lower resident memory when
    /// loading makehmmerdb container records.
    pub fn from_parts_sampled(
        bwt: Vec<u8>,
        sa: Vec<i32>,
        c: [usize; 256],
        text_len: usize,
    ) -> Result<Self, String> {
        Self::from_parts_inner(bwt, sa, c, text_len, Some(DEFAULT_SA_SAMPLE_RATE))
    }

    pub fn from_mapped_file_parts(
        mapped_bytes: Arc<MmapBytes>,
        bwt_offset: usize,
        bwt_len: usize,
        sa_offset: usize,
        sa_len: usize,
        c: [usize; 256],
        text_len: usize,
    ) -> Result<Self, String> {
        Self::from_mapped_file_parts_with_occ(
            mapped_bytes,
            bwt_offset,
            bwt_len,
            sa_offset,
            sa_len,
            c,
            text_len,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_mapped_file_parts_with_makehmmerdb_occ(
        mapped_bytes: Arc<MmapBytes>,
        bwt_offset: usize,
        bwt_len: usize,
        sa_offset: usize,
        sa_len: usize,
        c: [usize; 256],
        text_len: usize,
        term_loc: usize,
        freq_cnt_b: usize,
        occ_b_offset: usize,
        occ_b_count: usize,
        occ_sb_offset: usize,
        occ_sb_count: usize,
    ) -> Result<Self, String> {
        let mapped_slice = mapped_bytes.as_slice();
        let occ_b_bytes = occ_b_count
            .checked_mul(2)
            .ok_or_else(|| "FM-index occurrence block table span overflows usize".to_string())?;
        let occ_sb_bytes = occ_sb_count.checked_mul(4).ok_or_else(|| {
            "FM-index occurrence superblock table span overflows usize".to_string()
        })?;
        let occ_b_slice = mapped_slice
            .get(occ_b_offset..occ_b_offset + occ_b_bytes)
            .ok_or_else(|| "FM-index occurrence block table is outside mapped file".to_string())?;
        let occ_sb_slice = mapped_slice
            .get(occ_sb_offset..occ_sb_offset + occ_sb_bytes)
            .ok_or_else(|| {
                "FM-index occurrence superblock table is outside mapped file".to_string()
            })?;
        let mut occ_b = Vec::with_capacity(occ_b_count);
        for chunk in occ_b_slice.as_chunks::<2>().0 {
            occ_b.push(u16::from_le_bytes(*chunk));
        }
        let mut occ_sb = Vec::with_capacity(occ_sb_count);
        for chunk in occ_sb_slice.as_chunks::<4>().0 {
            occ_sb.push(u32::from_le_bytes(*chunk));
        }
        let occ_rank =
            OccRank::from_makehmmerdb_dna_tables(bwt_len, term_loc, freq_cnt_b, &occ_b, &occ_sb)?;
        Self::from_mapped_file_parts_with_occ(
            mapped_bytes,
            bwt_offset,
            bwt_len,
            sa_offset,
            sa_len,
            c,
            text_len,
            Some(occ_rank),
        )
    }

    // One argument per region of the mapped file, in file order.
    #[allow(clippy::too_many_arguments)]
    fn from_mapped_file_parts_with_occ(
        mapped_bytes: Arc<MmapBytes>,
        bwt_offset: usize,
        bwt_len: usize,
        sa_offset: usize,
        sa_len: usize,
        c: [usize; 256],
        text_len: usize,
        occ_rank: Option<OccRank>,
    ) -> Result<Self, String> {
        let expected_len = text_len
            .checked_add(1)
            .ok_or_else(|| "FM-index text length overflows usize".to_string())?;
        let mapped_slice = mapped_bytes.as_slice();
        let bwt = mapped_slice
            .get(bwt_offset..bwt_offset + bwt_len)
            .ok_or_else(|| "FM-index mapped BWT is outside mapped file".to_string())?;
        Self::validate_parts_common(bwt, c, text_len, expected_len)?;
        if sa_len != bwt_len {
            return Err(format!(
                "FM-index suffix-array length {} does not match BWT length {}",
                sa_len, bwt_len
            ));
        }
        let sa_bytes = sa_len
            .checked_mul(4)
            .ok_or_else(|| "FM-index mapped suffix-array span overflows usize".to_string())?;
        sa_offset
            .checked_add(sa_bytes)
            .filter(|&end| end <= mapped_slice.len())
            .ok_or_else(|| "FM-index mapped suffix-array is outside mapped file".to_string())?;
        let occ_rank = occ_rank.unwrap_or_else(|| OccRank::build(bwt));
        Ok(FmIndex {
            bwt: Vec::new(),
            mapped_bwt: Some(MappedBwt {
                bytes: Arc::clone(&mapped_bytes),
                offset: bwt_offset,
                len: bwt_len,
            }),
            packed_dna_bwt: None,
            sa: Vec::new(),
            sa_samples: Vec::new(),
            regular_sa_samples: None,
            mapped_sa: Some(MappedSa {
                bytes: mapped_bytes,
                offset: sa_offset,
                len: sa_len,
            }),
            c,
            n: text_len,
            occ_rank,
        })
    }

    pub(super) fn from_parts_inner(
        bwt: Vec<u8>,
        sa: Vec<i32>,
        c: [usize; 256],
        text_len: usize,
        sample_rate: Option<usize>,
    ) -> Result<Self, String> {
        let expected_len = text_len
            .checked_add(1)
            .ok_or_else(|| "FM-index text length overflows usize".to_string())?;
        Self::validate_parts_common(&bwt, c, text_len, expected_len)?;
        if sa.len() != bwt.len() {
            return Err(format!(
                "FM-index suffix-array length {} does not match BWT length {}",
                sa.len(),
                bwt.len()
            ));
        }
        if sa
            .iter()
            .any(|&pos| pos < 0 || pos as usize >= expected_len)
        {
            return Err("FM-index suffix-array entry is outside indexed text".to_string());
        }
        #[cfg(debug_assertions)]
        {
            let mut seen = vec![false; expected_len];
            for &pos in &sa {
                let pos = pos as usize;
                if seen[pos] {
                    return Err(format!(
                        "FM-index suffix-array is not a permutation: duplicate position {pos}"
                    ));
                }
                seen[pos] = true;
            }
        }

        let sa_samples = if let Some(sample_rate) = sample_rate {
            if sample_rate == 0 {
                return Err("FM-index suffix-array sample rate is zero".to_string());
            }
            let mut samples = Vec::with_capacity(sa.len().div_ceil(sample_rate));
            for (row, &pos) in sa.iter().enumerate() {
                if row % sample_rate == 0 {
                    samples.push((
                        i32::try_from(row)
                            .map_err(|_| "FM-index row exceeds i32 range".to_string())?,
                        pos,
                    ));
                }
            }
            samples
        } else {
            Vec::new()
        };
        let sa = if sample_rate.is_some() {
            Vec::new()
        } else {
            sa
        };
        let occ_rank = OccRank::build(&bwt);
        Ok(FmIndex {
            bwt,
            mapped_bwt: None,
            packed_dna_bwt: None,
            sa,
            sa_samples,
            regular_sa_samples: None,
            mapped_sa: None,
            c,
            n: text_len,
            occ_rank,
        })
    }

    pub(super) fn validate_parts_common(
        bwt: &[u8],
        c: [usize; 256],
        text_len: usize,
        expected_len: usize,
    ) -> Result<(), String> {
        if bwt.len() != expected_len {
            return Err(format!(
                "FM-index BWT length {} does not match text length {} plus sentinel",
                bwt.len(),
                text_len
            ));
        }
        if c[0] != 0 {
            return Err("FM-index C table must start at zero".to_string());
        }
        if c.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err("FM-index C table is not monotonic".to_string());
        }
        let mut counts = [0usize; 256];
        for &ch in bwt {
            counts[ch as usize] += 1;
        }
        if counts[0] != 1 {
            return Err(format!(
                "FM-index BWT must contain exactly one sentinel byte, found {}",
                counts[0]
            ));
        }
        let mut cumulative = 0usize;
        for ch in 0..256 {
            if c[ch] != cumulative {
                return Err(format!(
                    "FM-index C table entry {ch} is {}, expected {cumulative}",
                    c[ch]
                ));
            }
            cumulative = cumulative
                .checked_add(counts[ch])
                .ok_or_else(|| "FM-index C table cumulative count overflows usize".to_string())?;
        }
        if cumulative != bwt.len() {
            return Err(format!(
                "FM-index C table covers {cumulative} symbols, expected {}",
                bwt.len()
            ));
        }
        Ok(())
    }
}
