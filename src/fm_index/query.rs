use super::*;

impl FmIndex {
    /// Count exact occurrences of `pattern` in the indexed text by BWT
    /// backward search. Returns 0 when the pattern does not occur.
    /// Analog of C's `getSARangeReverse()` interval-shrinking loop.
    pub fn count(&self, pattern: &[u8]) -> usize {
        if pattern.is_empty() || self.n == 0 || pattern.contains(&0) {
            return 0;
        }

        let bwt_len = self.bwt_len();
        let mut lo = 0usize;
        let mut hi = bwt_len;

        // Backward search
        for &ch in pattern.iter().rev() {
            // Count occurrences of ch in bwt[0..lo] and bwt[0..hi]
            let occ_lo = self.occ(ch, lo);
            let occ_hi = self.occ(ch, hi);

            lo = self.c[ch as usize] + occ_lo;
            hi = self.c[ch as usize] + occ_hi;

            if lo >= hi {
                return 0;
            }
        }

        hi - lo
    }

    /// Locate all 0-based starting positions of `pattern` in the indexed text.
    /// Same backward search as `count()`, then maps the resulting suffix-array
    /// interval back to text positions. Analog of C's `fm_getOriginalPosition()`
    /// applied across the matching SA range.
    pub fn locate(&self, pattern: &[u8]) -> Vec<usize> {
        if pattern.is_empty() || self.n == 0 || pattern.contains(&0) {
            return Vec::new();
        }

        let mut interval = self.root_interval();

        for &ch in pattern.iter().rev() {
            let Some(next) = self.prepend_interval(interval, ch) else {
                return Vec::new();
            };
            interval = next;
        }

        self.locate_interval(interval)
    }

    /// Locate all text positions in an already-computed suffix-array interval.
    pub fn locate_interval(&self, interval: FmInterval) -> Vec<usize> {
        if interval.is_empty() || interval.hi > self.bwt_len() {
            return Vec::new();
        }

        (interval.lo..interval.hi)
            .filter_map(|i| self.suffix_position(i))
            .filter(|&pos| pos < self.n)
            .collect()
    }

    /// Locate at most `limit` text positions in an already-computed suffix-array
    /// interval. This mirrors `locate_interval()` order without materializing
    /// very large FM intervals that callers are going to truncate anyway.
    pub fn locate_interval_limited(&self, interval: FmInterval, limit: usize) -> Vec<usize> {
        if limit == 0 || interval.is_empty() || interval.hi > self.bwt_len() {
            return Vec::new();
        }

        let mut positions = Vec::new();
        for row in interval.lo..interval.hi {
            if let Some(pos) = self.suffix_position(row).filter(|&pos| pos < self.n) {
                positions.push(pos);
                if positions.len() >= limit {
                    break;
                }
            }
        }
        positions
    }

    /// Faithful public counterpart of C `FM_backtrackSeed()` for one BWT row.
    ///
    /// `suffix_position()` already performs the same LF backtracking to a
    /// sampled suffix-array row when only samples are resident, and returns the
    /// text position represented by this row.
    pub fn backtrack_seed_position(&self, row: usize) -> Option<usize> {
        self.suffix_position(row).filter(|&pos| pos < self.n)
    }

    fn suffix_position(&self, row: usize) -> Option<usize> {
        if !self.sa.is_empty() {
            return self.sa.get(row).map(|&pos| pos as usize);
        }
        if let Some(mapped) = &self.mapped_sa {
            return self.mapped_suffix_position(mapped, row);
        }

        let mut row = row;
        let mut steps = 0usize;
        let limit = self.bwt_len();
        while steps <= limit {
            if let Some(samples) = &self.regular_sa_samples {
                if let Some(sampled_pos) = samples.position_for_row(row) {
                    return Some((sampled_pos + steps) % self.bwt_len());
                }
            }
            let row_i32 = i32::try_from(row).ok()?;
            if let Ok(idx) = self
                .sa_samples
                .binary_search_by_key(&row_i32, |&(sample_row, _)| sample_row)
            {
                let sampled_pos = self.sa_samples[idx].1 as usize;
                return Some((sampled_pos + steps) % self.bwt_len());
            }
            let ch = self.bwt_at(row)?;
            row = self.c[ch as usize].checked_add(self.occ(ch, row))?;
            steps += 1;
        }
        None
    }

    fn mapped_suffix_position(&self, mapped: &MappedSa, row: usize) -> Option<usize> {
        if row >= mapped.len {
            return None;
        }
        let offset = mapped.offset.checked_add(row.checked_mul(4)?)?;
        let bytes = mapped.bytes.as_slice().get(offset..offset + 4)?;
        let pos = i32::from_le_bytes(bytes.try_into().ok()?);
        (pos >= 0).then_some(pos as usize)
    }

    /// `Occ(ch, pos)`: counts `ch` in `bwt[0..pos]`, using the two-level sampled
    /// rank tables (C `fm_getOccCount`): superblock cumulative + within-superblock
    /// block cumulative + a short scan of the trailing partial block. O(`FREQ_CNT_B`).
    pub(super) fn occ(&self, ch: u8, pos: usize) -> usize {
        let rank = &self.occ_rank;
        let s = rank.sym_of[ch as usize];
        if s < 0 {
            return 0;
        }
        let s = s as usize;
        let ns = rank.n_sym();
        let sb = pos / FREQ_CNT_SB;
        let b = pos / FREQ_CNT_B;
        let mut cnt = rank.occ_sb[sb * ns + s];
        if b != sb * (FREQ_CNT_SB / FREQ_CNT_B) {
            cnt += rank.occ_b[b * ns + s] as usize;
        }
        let block_start = b * FREQ_CNT_B;
        if let Some(packed) = &self.packed_dna_bwt {
            return cnt + packed.count_eq(block_start, pos, ch);
        }
        for idx in block_start..pos {
            if self.bwt_at(idx) == Some(ch) {
                cnt += 1;
            }
        }
        cnt
    }

    /// `OccLT(ch, pos)`: number of BWT symbols *strictly less than* `ch` in
    /// `bwt[0..pos]` (the sentinel byte 0 and any smaller bases all count). This
    /// is the second value C's `fm_getOccCountLT()` returns and is the quantity
    /// the bi-directional forward interval update needs. Uses the same sampled
    /// rank tables, summed over the present symbols `< ch`.
    fn occ_lt(&self, ch: u8, pos: usize) -> usize {
        let rank = &self.occ_rank;
        let ns = rank.n_sym();
        let sb = pos / FREQ_CNT_SB;
        let b = pos / FREQ_CNT_B;
        let mut cnt = 0usize;
        for (s, &sym) in rank.symbols.iter().enumerate() {
            if sym < ch {
                cnt += rank.occ_sb[sb * ns + s];
                if b != sb * (FREQ_CNT_SB / FREQ_CNT_B) {
                    cnt += rank.occ_b[b * ns + s] as usize;
                }
            }
        }
        let block_start = b * FREQ_CNT_B;
        if let Some(packed) = &self.packed_dna_bwt {
            return cnt + packed.count_lt(block_start, pos, ch);
        }
        for idx in block_start..pos {
            if self.bwt_at(idx).is_some_and(|x| x < ch) {
                cnt += 1;
            }
        }
        cnt
    }

    /// Total occurrences of `ch` across the whole BWT (= count of `ch` in the
    /// indexed text). Used to seed a single-character interval.
    pub(super) fn occ_total(&self, ch: u8) -> usize {
        self.occ(ch, self.bwt_len())
    }

    /// Half-open interval over the suffix array for a single character `ch`
    /// (the starting point of a forward or backward search). Empty if `ch`
    /// does not occur.
    pub fn char_interval(&self, ch: u8) -> Option<FmInterval> {
        if ch == 0 {
            return None;
        }
        let lo = self.c[ch as usize];
        let hi = lo.checked_add(self.occ_total(ch))?;
        (lo < hi).then_some(FmInterval { lo, hi })
    }

    /// Apply one *forward* (left-to-right) search step of the bi-directional
    /// BWT, faithful port of C `fm_updateIntervalForward()` (Algorithm 4 of
    /// Simpson 2010, `hmmer/src/fm_general.c`).
    ///
    /// Maintains the synchronized pair of intervals over a single index:
    /// `bk` is the SA-range of `reverse(W)` (the mirror, updated by a normal
    /// backward-search step) and `fwd` is the SA-range of the forward pattern
    /// `W` itself (updated via the `OccLT` offset). Appending `ch` to the
    /// pattern (`W -> W·ch`) returns the new `(bk, fwd)` pair, where `fwd` is
    /// directly locatable via [`locate_interval`]. Returns `None` when the
    /// extended pattern does not occur.
    ///
    /// C uses inclusive `[lower, upper]` ranges; this port uses the crate's
    /// half-open `[lo, hi)` convention throughout (verified in the unit tests
    /// against a brute-force forward search).
    pub fn update_interval_forward(
        &self,
        bk: FmInterval,
        fwd: FmInterval,
        ch: u8,
    ) -> Option<(FmInterval, FmInterval)> {
        if ch == 0
            || bk.is_empty()
            || fwd.is_empty()
            || bk.hi > self.bwt_len()
            || fwd.hi > self.bwt_len()
        {
            return None;
        }
        let occ_lo = self.occ(ch, bk.lo);
        let occ_hi = self.occ(ch, bk.hi);
        // New mirror (backward) interval: normal backward-search step on `ch`.
        let new_bk_lo = self.c[ch as usize].checked_add(occ_lo)?;
        let new_bk_hi = self.c[ch as usize].checked_add(occ_hi)?;
        if new_bk_lo >= new_bk_hi {
            return None;
        }
        // New forward interval: shift `fwd.lo` by the number of suffixes in the
        // current mirror range that carry a symbol strictly smaller than `ch`,
        // then take the same size as the new mirror range.
        let occ_lt_lo = self.occ_lt(ch, bk.lo);
        let occ_lt_hi = self.occ_lt(ch, bk.hi);
        let fwd_delta = occ_lt_hi.checked_sub(occ_lt_lo)?;
        let interval_len = new_bk_hi.checked_sub(new_bk_lo)?;
        let new_fwd_lo = fwd.lo.checked_add(fwd_delta)?;
        let new_fwd_hi = new_fwd_lo.checked_add(interval_len)?;
        if new_fwd_hi > self.bwt_len() {
            return None;
        }
        Some((
            FmInterval {
                lo: new_bk_lo,
                hi: new_bk_hi,
            },
            FmInterval {
                lo: new_fwd_lo,
                hi: new_fwd_hi,
            },
        ))
    }
}
