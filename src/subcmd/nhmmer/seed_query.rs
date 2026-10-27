use super::*;

pub(super) struct NhmmerFmSeedQuery<'a> {
    pub(super) fm_bases: Vec<u8>,
    pub(super) seed: &'a NhmmerConsensusSeed,
    pub(super) score_bits: f32,
}

pub(super) fn fm_seed_queries_for_record<'a>(
    seeds: &'a [NhmmerConsensusSeed],
    record: &NhmmerFmIndexRecord,
    seq_len: usize,
    is_complement: bool,
    hmm: &Hmm,
    bg: &Bg,
) -> Vec<NhmmerFmSeedQuery<'a>> {
    let mut queries = Vec::with_capacity(seeds.len());
    for seed in seeds {
        if seed.bases.len() > record.text_len || seed.bases.len() > seq_len {
            continue;
        }
        let fm_bases = if is_complement {
            reverse_complement_dna_seed(&seed.bases)
        } else {
            let mut reversed_seed = seed.bases.clone();
            reversed_seed.reverse();
            reversed_seed
        };
        queries.push(NhmmerFmSeedQuery {
            fm_bases,
            seed,
            score_bits: fm_seed_model_lod_bits(hmm, bg, seed),
        });
    }
    queries
}

pub(super) fn fm_locate_seed_queries<'a>(
    fm: &FmIndex,
    queries: &'a [NhmmerFmSeedQuery<'a>],
    limit: usize,
) -> Vec<(&'a NhmmerConsensusSeed, usize)> {
    if limit == 0 {
        return Vec::new();
    }
    let mut indexed_queries: Vec<usize> = (0..queries.len()).collect();
    indexed_queries.sort_by(|&a, &b| {
        queries[a]
            .fm_bases
            .iter()
            .rev()
            .cmp(queries[b].fm_bases.iter().rev())
            .then_with(|| queries[a].fm_bases.cmp(&queries[b].fm_bases))
    });

    let mut located = Vec::new();
    fm_locate_seed_queries_recurse(
        fm,
        queries,
        &indexed_queries,
        0,
        fm.root_interval(),
        &mut located,
        limit,
    );
    located
}

fn fm_locate_seed_queries_recurse<'a>(
    fm: &FmIndex,
    queries: &'a [NhmmerFmSeedQuery<'a>],
    query_ids: &[usize],
    depth: usize,
    interval: FmInterval,
    located: &mut Vec<(&'a NhmmerConsensusSeed, usize)>,
    limit: usize,
) {
    if located.len() >= limit {
        return;
    }
    let mut child_groups = Vec::new();
    let mut child_start = 0usize;
    while child_start < query_ids.len() {
        let query = &queries[query_ids[child_start]];
        if depth == query.fm_bases.len() {
            let mut terminal_end = child_start + 1;
            while terminal_end < query_ids.len()
                && depth == queries[query_ids[terminal_end]].fm_bases.len()
            {
                terminal_end += 1;
            }
            let mut terminals: Vec<usize> = query_ids[child_start..terminal_end].to_vec();
            terminals.sort_by(|&a, &b| {
                queries[b]
                    .score_bits
                    .total_cmp(&queries[a].score_bits)
                    .then_with(|| queries[a].seed.model_end.cmp(&queries[b].seed.model_end))
                    .then_with(|| queries[a].seed.bases.cmp(&queries[b].seed.bases))
            });
            let positions = fm.locate_interval_limited(interval, limit - located.len());
            for query_id in terminals {
                let terminal_query = &queries[query_id];
                for &fm_pos in &positions {
                    located.push((terminal_query.seed, fm_pos));
                    if located.len() >= limit {
                        return;
                    }
                }
            }
            child_start = terminal_end;
            continue;
        }

        let ch = query.fm_bases[query.fm_bases.len() - 1 - depth];
        let mut child_end = child_start + 1;
        while child_end < query_ids.len() {
            let child_query = &queries[query_ids[child_end]];
            if depth == child_query.fm_bases.len()
                || child_query.fm_bases[child_query.fm_bases.len() - 1 - depth] != ch
            {
                break;
            }
            child_end += 1;
        }

        let max_score = query_ids[child_start..child_end]
            .iter()
            .map(|&query_id| queries[query_id].score_bits)
            .fold(f32::NEG_INFINITY, f32::max);
        child_groups.push((child_start, child_end, ch, max_score));
        child_start = child_end;
    }

    child_groups.sort_by(|a, b| {
        b.3.total_cmp(&a.3)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    });
    for (child_start, child_end, ch, _) in child_groups {
        if let Some(child_interval) = fm.prepend_interval(interval, ch) {
            fm_locate_seed_queries_recurse(
                fm,
                queries,
                &query_ids[child_start..child_end],
                depth + 1,
                child_interval,
                located,
                limit,
            );
        }
        if located.len() >= limit {
            return;
        }
    }
}

pub(super) fn fm_extend_seed_diagonal(
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
    seed: &NhmmerConsensusSeed,
    seq_seed_start0: usize,
    ssv_length: usize,
) -> (usize, usize, usize, f32) {
    let seed_len = seed.bases.len();
    let seed_model_start = seed.model_end.saturating_sub(seed_len).saturating_add(1);
    if seed_len == 0 || seed_model_start == 0 || seed.model_end > hmm.m {
        return (seq_seed_start0, seed_len, seed.model_end.min(hmm.m), 0.0);
    }

    let extend = 10usize.max(ssv_length.saturating_sub(seed_len));
    // NOTE: this is the seed-then-rescore approximation's extension window, NOT a
    // 1:1 port of C `FM_extendSeed`; the downstream window extension + MSV
    // rescore recompute final coordinates. Adding C's `model_start = k-extend+1`
    // "+ 1" here was tried and is observably neutral on the current fixtures, so
    // the simpler long-standing no-`+1` form is kept.
    let mut model_start = seed_model_start.saturating_sub(extend).max(1);
    let mut model_end = (seed_model_start + seed_len + extend - 1).min(hmm.m);
    let mut target_start = seq_seed_start0 as isize - (seed_model_start - model_start) as isize;
    let mut target_end = seq_seed_start0 as isize + (model_end - seed_model_start) as isize;

    if target_start < 0 {
        let shift = (-target_start) as usize;
        model_start = model_start.saturating_add(shift);
        target_start = 0;
    }
    let last_target = seq.n.saturating_sub(1) as isize;
    if target_end > last_target {
        let shift = (target_end - last_target) as usize;
        model_end = model_end.saturating_sub(shift);
        target_end = last_target;
    }
    if seq.n == 0 || model_start > model_end || target_start > target_end {
        return (seq_seed_start0, seed_len, seed.model_end, 0.0);
    }

    let mut score = 0.0_f32;
    let mut best_score = 0.0_f32;
    let mut hit_start = 0usize;
    let mut best_start = 0usize;
    let mut best_end = 0usize;
    let scan_len = (model_end - model_start + 1).min((target_end - target_start + 1) as usize);
    for offset in 0..scan_len {
        let model_pos = model_start + offset;
        let seq_pos = target_start as usize + offset + 1;
        let Some(delta) = fm_match_lod_bits(seq, hmm, bg, model_pos, seq_pos) else {
            score = 0.0;
            hit_start = offset + 1;
            continue;
        };
        score += delta;
        if score < 0.0 {
            score = 0.0;
            hit_start = offset + 1;
        } else if score > best_score {
            best_score = score;
            best_start = hit_start;
            best_end = offset;
        }
    }

    if best_score <= 0.0 {
        return (seq_seed_start0, seed_len, seed.model_end, best_score);
    }

    let window_start0 = target_start as usize + best_start;
    let window_len = best_end - best_start + 1;
    let model_end = model_start + best_end;
    (window_start0, window_len, model_end, best_score)
}

pub(super) fn fm_diagonal_lod_bits(
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
    model_start: usize,
    seq_start0: usize,
    len: usize,
) -> f32 {
    let mut score = 0.0_f32;
    for offset in 0..len {
        let Some(delta) =
            fm_match_lod_bits(seq, hmm, bg, model_start + offset, seq_start0 + 1 + offset)
        else {
            break;
        };
        score += delta;
    }
    score
}

pub(super) fn fm_seed_model_lod_bits(hmm: &Hmm, bg: &Bg, seed: &NhmmerConsensusSeed) -> f32 {
    let seed_len = seed.bases.len();
    let seed_model_start = seed.model_end.saturating_sub(seed_len).saturating_add(1);
    if seed_len == 0 || seed_model_start == 0 || seed.model_end > hmm.m {
        return 0.0;
    }

    let abc = Alphabet::new(hmm.abc_type);
    let mut score = 0.0_f32;
    for (offset, &base) in seed.bases.iter().enumerate() {
        let Some(x) = abc.inmap.get(base as usize).copied() else {
            return score;
        };
        let x = x as usize;
        if x >= bg.f.len() {
            return score;
        }
        let model_pos = seed_model_start + offset;
        if model_pos == 0 || model_pos > hmm.m || x >= hmm.mat[model_pos].len() {
            return score;
        }
        let p = hmm.mat[model_pos][x];
        let q = bg.f[x];
        if p <= 0.0 || q <= 0.0 {
            return score;
        }
        score += ((p as f64) / (q as f64)).log2() as f32;
    }
    score
}

pub(super) fn fm_match_lod_bits(
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
    model_pos: usize,
    seq_pos: usize,
) -> Option<f32> {
    let x = *seq.dsq.get(seq_pos)? as usize;
    if model_pos == 0 || model_pos > hmm.m || x >= bg.f.len() || x >= hmm.mat[model_pos].len() {
        return None;
    }
    let p = hmm.mat[model_pos][x];
    let q = bg.f[x];
    (p > 0.0 && q > 0.0).then(|| ((p as f64) / (q as f64)).log2() as f32)
}

pub(super) fn fm_seed_is_wholly_in_leading_overlap(
    record: &NhmmerFmIndexRecord,
    seed_end: usize,
) -> bool {
    record.overlap_bases > 0 && seed_end <= record.overlap_bases
}

pub(super) fn fm_seed_overlaps_ambiguity(
    ambiguities: &[(usize, usize)],
    ambig_offset: usize,
    ambig_count: usize,
    seed_start: usize,
    seed_end: usize,
) -> bool {
    let Some(ambig_end) = ambig_offset.checked_add(ambig_count) else {
        return true;
    };
    if ambig_end > ambiguities.len() {
        return true;
    }
    for &(lower, upper) in &ambiguities[ambig_offset..ambig_end] {
        let ambiguity_end = upper.saturating_add(1);
        if seed_start < ambiguity_end && seed_end > lower {
            return true;
        }
    }
    false
}
