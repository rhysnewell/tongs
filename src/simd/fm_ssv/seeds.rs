use super::*;

/// Faithful Rust counterpart of C `FM_backtrackSeed()`.
///
/// The underlying `FmIndex` method performs the LF walk to a suffix-array
/// sample, matching C's per-row seed backtracking contract.
fn fm_backtrack_seed(fm: &FmIndex, row: usize) -> Option<usize> {
    fm.backtrack_seed_position(row)
}

/// Faithful Rust counterpart of C `fm_newSeed()`: append one seed slot and
/// return it to the caller for field assignment.
fn fm_new_seed(seeds: &mut Vec<FmDiag>) -> &mut FmDiag {
    seeds.push(FmDiag {
        n: 0,
        k: 0,
        length: 0,
        score: 0.0,
        complementarity: Complementarity::NoComplement,
    });
    seeds.last_mut().expect("just pushed FM seed")
}

/// Result of `FM_getPassingDiags`: each surviving FM interval entry becomes a
/// seed. Faithful to C's coordinate computation.
#[allow(clippy::too_many_arguments)]
fn fm_get_passing_diags(
    fm: &FmIndex,
    k: i32,
    depth: i32,
    model_direction: ModelDirection,
    complementarity: Complementarity,
    interval: FmInterval,
    seeds: &mut Vec<FmDiag>,
) {
    // C uses `fmf->N` = the BWT length including the sentinel = text_len + 1.
    let fm_n = fm.n as i64 + 1;
    for row in interval.lo..interval.hi {
        let Some(backtrack) = fm_backtrack_seed(fm, row) else {
            continue;
        };
        let backtrack = backtrack as i64;
        let seed = fm_new_seed(seeds);
        seed.k = k;
        seed.length = depth;
        // C `FM_getPassingDiags`:
        //   NOCOMPLEMENT: n = N - backtrack - depth - 1
        //   COMPLEMENT:   n = backtrack
        seed.n = if complementarity == Complementarity::NoComplement {
            fm_n - backtrack - depth as i64 - 1
        } else {
            backtrack
        };
        seed.complementarity = complementarity;
        // C: if model_direction == fm_forward, seed->k -= (depth - 1)
        if model_direction == ModelDirection::Forward {
            seed.k -= depth - 1;
        }
    }
}

/// Heart of the kernel: faithful port of C `FM_Recurse` restricted to the
/// `fm_direction == fm_forward` sweep (single backward-search FM index).
///
/// `dp_pairs` is the shared scratch vector C indexes with `first..=last`;
/// surviving children are appended past `last` and recursed on. We mirror that
/// exactly using indices into a growable Vec.
/// Look up `opt_ext[idx][j]`, returning `None` (i.e. do not prune) when the
/// indices are outside the valid/computed range.
#[inline]
fn opt_ext_lookahead(table: &[[f32; 10]], idx: i32, j: i32) -> Option<f32> {
    if idx < 1 || !(0..10).contains(&j) {
        return None;
    }
    table.get(idx as usize).map(|row| row[j as usize])
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fm_recurse(
    depth: i32,
    fmf: &FmIndex,
    fmb: &FmIndex,
    direction: FmDirection,
    alph: &FmAlphabet,
    scores: &SsvScores,
    consensus: &[usize],
    sc_thresh_fm: f32,
    config: &FmSsvConfig,
    opt_ext: &OptExt,
    dp_pairs: &mut Vec<FmDpPair>,
    first: usize,
    last: usize,
    // Forward sweep: the locatable `fmf` reverse-search interval.
    // Backward sweep: the `fmb` mirror interval.
    interval_1: FmInterval,
    // Backward sweep only: the `fmf`-coordinate forward interval (locatable).
    interval_2: Option<FmInterval>,
    seeds: &mut Vec<FmDiag>,
) {
    let m = scores.m;

    for c in 0..alph.alph_size {
        let base = fm_code_to_base(c);
        let mut step_cache: Option<Option<(FmInterval, FmInterval)>> = None;

        // dppos tracks the last index appended for this character's column.
        let mut dppos = last;

        for i in first..=last {
            let pair = dp_pairs[i];
            let k = match pair.model_direction {
                ModelDirection::Forward => pair.pos + 1,
                ModelDirection::Backward => pair.pos - 1,
            };

            // next_score and consensus character, accounting for complement.
            let (next_score, cons_c) = match pair.complementarity {
                Complementarity::Complement => {
                    let cc = alph.compl_alph[c];
                    (scores.get(k, cc), alph.compl_alph[consensus[k as usize]])
                }
                Complementarity::NoComplement => (scores.get(k, c), consensus[k as usize]),
            };

            let sc = pair.score + next_score;
            let positive_run: i16 = if next_score > 0.0 {
                pair.consec_pos + 1
            } else {
                0
            };
            let consec_consensus: i16 = if c == cons_c {
                pair.consec_consensus + 1
            } else {
                0
            };

            if sc >= sc_thresh_fm
                || (config.consensus_match_req > 0
                    && consec_consensus as usize == config.consensus_match_req)
            {
                // A seed to extend: emit all matching target positions from the
                // fmf-locatable interval for this character.
                let step = *step_cache.get_or_insert_with(|| match direction {
                    FmDirection::Forward => {
                        fmf.prepend_interval(interval_1, base).map(|iv| (iv, iv))
                    }
                    FmDirection::Backward => {
                        let iv2 = interval_2.expect("backward sweep requires interval_2");
                        fmb.update_interval_forward(interval_1, iv2, base)
                    }
                });
                if let Some((_, locatable)) = step {
                    fm_get_passing_diags(
                        fmf,
                        k,
                        depth,
                        pair.model_direction,
                        pair.complementarity,
                        locatable,
                        seeds,
                    );
                }
            } else if sc <= 0.0
                || depth == config.max_depth as i32
                || (pair.model_direction == ModelDirection::Forward && k == m)
                || (pair.model_direction == ModelDirection::Backward && k == 1)
                || depth == pair.score_peak_len + config.drop_max_len as i32
                || (depth > 4
                    && depth > consec_consensus as i32
                    && (sc / depth as f32) < config.score_density_req)
                || ((pair.max_consec_pos as usize) < config.consec_pos_req
                    && config.consec_pos_req.saturating_sub(positive_run as usize)
                        == (config.max_depth as i32 - depth + 1) as usize)
                // C's optimal-extension look-ahead prune: near the depth limit,
                // drop a diagonal that can't reach threshold even with the best
                // possible extension. `j = max_depth - depth - 1` (C indexes the
                // size-10 opt_ext arrays); guarded to a valid j and model range.
                || (pair.model_direction == ModelDirection::Forward
                    && depth > config.max_depth as i32 - 10
                    && opt_ext_lookahead(&opt_ext.fwd, k, config.max_depth as i32 - depth - 1)
                        .is_some_and(|best| sc + best < sc_thresh_fm))
                || (pair.model_direction == ModelDirection::Backward
                    && depth > config.max_depth as i32 - 10
                    && opt_ext_lookahead(&opt_ext.rev, k - 1, config.max_depth as i32 - depth - 1)
                        .is_some_and(|best| sc + best < sc_thresh_fm))
            {
                // pruned - do nothing.
            } else {
                // Extendable: append a new DP pair past `last`.
                dppos += 1;
                let (max_score, score_peak_len) = if sc > pair.max_score {
                    (sc, depth)
                } else if sc >= pair.max_score - config.drop_lim {
                    (pair.max_score, depth)
                } else {
                    (pair.max_score, pair.score_peak_len)
                };
                let new_pair = FmDpPair {
                    pos: k,
                    score: sc,
                    max_score,
                    score_peak_len,
                    consec_pos: positive_run,
                    max_consec_pos: positive_run.max(pair.max_consec_pos),
                    consec_consensus,
                    complementarity: pair.complementarity,
                    model_direction: pair.model_direction,
                };
                // dp_pairs grows; index dppos may be one past current end.
                if dppos < dp_pairs.len() {
                    dp_pairs[dppos] = new_pair;
                } else {
                    debug_assert_eq!(dppos, dp_pairs.len());
                    dp_pairs.push(new_pair);
                }
            }
        }

        if dppos > last {
            // At least one extendable diagonal: descend the trie on c.
            let step = *step_cache.get_or_insert_with(|| match direction {
                FmDirection::Forward => fmf.prepend_interval(interval_1, base).map(|iv| (iv, iv)),
                FmDirection::Backward => {
                    let iv2 = interval_2.expect("backward sweep requires interval_2");
                    fmb.update_interval_forward(interval_1, iv2, base)
                }
            });
            let Some((new_iv1, locatable)) = step else {
                continue;
            };
            let next_iv2 = match direction {
                FmDirection::Forward => None,
                FmDirection::Backward => Some(locatable),
            };
            fm_recurse(
                depth + 1,
                fmf,
                fmb,
                direction,
                alph,
                scores,
                consensus,
                sc_thresh_fm,
                config,
                opt_ext,
                dp_pairs,
                last + 1,
                dppos,
                new_iv1,
                next_iv2,
                seeds,
            );
        }
    }
}

/// Map an FM alphabet code (0=A,1=C,2=G,3=T) to the byte the Rust `FmIndex`
/// expects. The Rust FM indexes are built over ASCII uppercase ACGT text.
#[inline]
pub(super) fn fm_code_to_base(c: usize) -> u8 {
    match c {
        0 => b'A',
        1 => b'C',
        2 => b'G',
        3 => b'T',
        _ => 0,
    }
}

/// Faithful port of C `FM_getSeeds` (forward sweep only). Kickstarts a DP
/// column for each starting character `c`, seeding forward-on-model and
/// reverse-on-model diagonals (and their complement variants), then recurses.
///
/// `strands`: when `top_only`/`bottom_only` are both false, both strands are
/// considered (C `p7_STRAND_BOTH`).
/// Faithful port of C `FM_getSeeds`: for each starting character it builds the
/// forward-sweep and backward-sweep DP columns and runs both `FM_Recurse`
/// sweeps, then merges. `fmf` is the BWT of the reversed block text (Rust
/// `kind=0`); `fmb` is the BWT of the forward block text (Rust `kind=1`).
#[allow(clippy::too_many_arguments)]
pub fn fm_get_seeds(
    fmf: &FmIndex,
    fmb: &FmIndex,
    alph: &FmAlphabet,
    scores: &SsvScores,
    consensus: &[usize],
    sc_thresh_fm: f32,
    config: &FmSsvConfig,
    top_only: bool,
    bottom_only: bool,
) -> Vec<FmDiag> {
    let m = scores.m;
    let mut seeds: Vec<FmDiag> = Vec::new();
    let opt_ext = OptExt::build(scores);

    for c in 0..alph.alph_size {
        let base = fm_code_to_base(c);
        // Single-character interval. C inits all three intervals (f1, f2, bk)
        // to `fmf->C[c]..fmf->C[c+1]-1`; the C[] arrays of fmf and fmb agree
        // (same character multiset), so this value seeds both sweeps.
        let Some(iv0) = fmf.char_interval(base) else {
            continue;
        };

        // C `dp_pairs_fwd` (forward sweep) and `dp_pairs_rev` (backward sweep).
        let mut dp_fwd: Vec<FmDpPair> = Vec::new();
        let mut dp_rev: Vec<FmDpPair> = Vec::new();

        for k in 1..=m {
            if !bottom_only {
                let sc = scores.get(k, c);
                if sc > 0.0 {
                    // fwd-on-model -> forward sweep; rev-on-model -> backward sweep.
                    if k < m - 3 {
                        dp_fwd.push(initial_pair(
                            k,
                            sc,
                            c == consensus[k as usize],
                            Complementarity::NoComplement,
                            ModelDirection::Forward,
                        ));
                    }
                    if k > 4 {
                        dp_rev.push(initial_pair(
                            k,
                            sc,
                            c == consensus[k as usize],
                            Complementarity::NoComplement,
                            ModelDirection::Backward,
                        ));
                    }
                }
            }
            if !top_only {
                let cc = alph.compl_alph[c];
                let sc = scores.get(k, cc);
                if sc > 0.0 {
                    // complement rev-on-model -> forward sweep;
                    // complement fwd-on-model -> backward sweep.
                    if k > 4 {
                        dp_fwd.push(initial_pair(
                            k,
                            sc,
                            c == consensus[k as usize],
                            Complementarity::Complement,
                            ModelDirection::Backward,
                        ));
                    }
                    if k < m - 3 {
                        dp_rev.push(initial_pair(
                            k,
                            sc,
                            c == consensus[k as usize],
                            Complementarity::Complement,
                            ModelDirection::Forward,
                        ));
                    }
                }
            }
        }

        if !dp_fwd.is_empty() {
            let last = dp_fwd.len() - 1;
            fm_recurse(
                2,
                fmf,
                fmb,
                FmDirection::Forward,
                alph,
                scores,
                consensus,
                sc_thresh_fm,
                config,
                &opt_ext,
                &mut dp_fwd,
                0,
                last,
                iv0,
                None,
                &mut seeds,
            );
        }
        if !dp_rev.is_empty() {
            let last = dp_rev.len() - 1;
            fm_recurse(
                2,
                fmf,
                fmb,
                FmDirection::Backward,
                alph,
                scores,
                consensus,
                sc_thresh_fm,
                config,
                &opt_ext,
                &mut dp_rev,
                0,
                last,
                iv0,
                Some(iv0),
                &mut seeds,
            );
        }
    }

    // C passes `fmf->N`, the BWT length including the terminal sentinel. Rust
    // `FmIndex::n` is the biological text length, so use `bwt_len()` here.
    fm_merge_seeds(
        &mut seeds,
        fmf.bwt_len() as i64,
        m,
        config.ssv_length as i32,
    );
    seeds
}
