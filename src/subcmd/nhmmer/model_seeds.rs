use super::*;

#[cfg(test)]
pub(super) fn fm_locate_model_score_seeds(
    fm: &FmIndex,
    hmm: &Hmm,
    bg: &Bg,
    is_complement: bool,
    text_len: usize,
    seq_len: usize,
    limit: usize,
) -> Vec<(NhmmerConsensusSeed, usize)> {
    fm_locate_model_score_seeds_with_start_limit(
        fm,
        hmm,
        bg,
        is_complement,
        text_len,
        seq_len,
        limit,
        usize::MAX,
        NhmmerFmSeedConfig::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fm_locate_model_score_seeds_with_start_limit(
    fm: &FmIndex,
    hmm: &Hmm,
    bg: &Bg,
    is_complement: bool,
    text_len: usize,
    seq_len: usize,
    limit: usize,
    start_limit: usize,
    config: NhmmerFmSeedConfig,
) -> Vec<(NhmmerConsensusSeed, usize)> {
    let Some((abc, max_depth, lod_bits, best_suffix, best_prefix, consensus_codes)) =
        nhmmer_fm_lod_tables(hmm, bg, config)
    else {
        return Vec::new();
    };
    if text_len == 0 || seq_len == 0 || limit == 0 {
        return Vec::new();
    }

    let mut located = Vec::new();
    let mut starts: Vec<(bool, usize, f32)> = (1..=hmm.m)
        .filter_map(|start| {
            let max_len = max_depth.min(hmm.m - start + 1).min(text_len).min(seq_len);
            (max_len > 0).then_some((false, start, best_suffix[start][max_len]))
        })
        .collect();
    starts.extend((1..=hmm.m).filter_map(|end| {
        let max_len = max_depth.min(end).min(text_len).min(seq_len);
        (max_len > 0).then_some((true, end, best_prefix[end][max_len]))
    }));
    starts.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.1.cmp(&b.1))
    });
    starts.truncate(start_limit);
    for (reverse, anchor, _) in starts {
        if located.len() >= limit {
            break;
        }
        if reverse {
            let max_len = max_depth.min(anchor).min(text_len).min(seq_len);
            let mut current_rev = Vec::with_capacity(max_len);
            fm_model_score_seed_reverse_dfs(
                fm,
                anchor,
                anchor,
                max_len,
                is_complement,
                &mut current_rev,
                &lod_bits,
                &best_prefix,
                &consensus_codes,
                &abc,
                &mut located,
                limit,
                NhmmerFmScoreState::new(),
                config,
            );
        } else {
            let max_len = max_depth.min(hmm.m - anchor + 1).min(text_len).min(seq_len);
            let mut current = Vec::with_capacity(max_len);
            fm_model_score_seed_dfs(
                fm,
                anchor,
                anchor,
                max_len,
                fm.root_interval(),
                is_complement,
                &mut current,
                &lod_bits,
                &best_suffix,
                &consensus_codes,
                &abc,
                &mut located,
                limit,
                NhmmerFmScoreState::new(),
                config,
            );
        }
    }
    located
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fm_model_score_seed_dfs(
    fm: &FmIndex,
    start: usize,
    node: usize,
    max_len: usize,
    interval: FmInterval,
    is_complement: bool,
    current: &mut Vec<u8>,
    lod_bits: &[[f32; 4]],
    best_suffix: &[Vec<f32>],
    consensus_codes: &[Option<usize>],
    abc: &Alphabet,
    located: &mut Vec<(NhmmerConsensusSeed, usize)>,
    limit: usize,
    state: NhmmerFmScoreState,
    config: NhmmerFmSeedConfig,
) {
    if current.len() == max_len || located.len() >= limit {
        return;
    }

    let depth = current.len() + 1;
    let remaining_after = max_len - depth;
    if state.score_bits + best_suffix[node][remaining_after + 1] < config.score_threshold_bits
        && !state.can_reach_consensus_seed(remaining_after + 1, config)
    {
        return;
    }

    let mut choices: Vec<(usize, f32)> = (0..abc.k.min(4))
        .filter_map(|x| {
            let sc = lod_bits[node][x];
            sc.is_finite().then_some((x, sc))
        })
        .collect();
    choices.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    for (x, delta) in choices {
        let next_state = state.extend(delta, depth, consensus_codes, node, x, config);
        if next_state.should_prune_below_threshold(depth, max_len, config)
            && !next_state.reaches_seed(config)
        {
            continue;
        }
        let tail_len = max_len - depth;
        if next_state.score_bits + best_suffix[node + 1][tail_len] < config.score_threshold_bits
            && !next_state.can_reach_consensus_seed(tail_len, config)
        {
            continue;
        }

        let ch = match abc.sym[x].to_ascii_uppercase() {
            b'U' => b'T',
            ch => ch,
        };
        let fm_ch = if is_complement {
            complement_dna_base(ch)
        } else {
            ch
        };
        let Some(child_interval) = fm.prepend_interval(interval, fm_ch) else {
            continue;
        };

        current.push(ch);
        let len = current.len();
        if next_state.reaches_seed(config) {
            let seed = NhmmerConsensusSeed {
                bases: current.clone(),
                model_end: start + len - 1,
            };
            for fm_pos in fm.locate_interval_limited(child_interval, limit - located.len()) {
                located.push((seed.clone(), fm_pos));
                if located.len() >= limit {
                    current.pop();
                    return;
                }
            }
        }

        fm_model_score_seed_dfs(
            fm,
            start,
            node + 1,
            max_len,
            child_interval,
            is_complement,
            current,
            lod_bits,
            best_suffix,
            consensus_codes,
            abc,
            located,
            limit,
            next_state,
            config,
        );
        current.pop();
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fm_model_score_seed_reverse_dfs(
    fm: &FmIndex,
    end: usize,
    node: usize,
    max_len: usize,
    is_complement: bool,
    current_rev: &mut Vec<u8>,
    lod_bits: &[[f32; 4]],
    best_prefix: &[Vec<f32>],
    consensus_codes: &[Option<usize>],
    abc: &Alphabet,
    located: &mut Vec<(NhmmerConsensusSeed, usize)>,
    limit: usize,
    state: NhmmerFmScoreState,
    config: NhmmerFmSeedConfig,
) {
    if current_rev.len() == max_len || located.len() >= limit || node == 0 {
        return;
    }

    let depth = current_rev.len() + 1;
    let remaining_after = max_len - depth;
    if state.score_bits + best_prefix[node][remaining_after + 1] < config.score_threshold_bits
        && !state.can_reach_consensus_seed(remaining_after + 1, config)
    {
        return;
    }

    let mut choices: Vec<(usize, f32)> = (0..abc.k.min(4))
        .filter_map(|x| {
            let sc = lod_bits[node][x];
            sc.is_finite().then_some((x, sc))
        })
        .collect();
    choices.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    for (x, delta) in choices {
        let next_state = state.extend(delta, depth, consensus_codes, node, x, config);
        if next_state.should_prune_below_threshold(depth, max_len, config)
            && !next_state.reaches_seed(config)
        {
            continue;
        }
        let tail_len = max_len - depth;
        if next_state.score_bits + best_prefix[node - 1][tail_len] < config.score_threshold_bits
            && !next_state.can_reach_consensus_seed(tail_len, config)
        {
            continue;
        }

        let ch = match abc.sym[x].to_ascii_uppercase() {
            b'U' => b'T',
            ch => ch,
        };
        current_rev.push(ch);
        let Some(child_interval) =
            fm_interval_for_reverse_model_prefix(fm, current_rev, is_complement)
        else {
            current_rev.pop();
            continue;
        };

        if next_state.reaches_seed(config) {
            let seed = NhmmerConsensusSeed {
                bases: current_rev.iter().rev().copied().collect(),
                model_end: end,
            };
            for fm_pos in fm.locate_interval_limited(child_interval, limit - located.len()) {
                located.push((seed.clone(), fm_pos));
                if located.len() >= limit {
                    current_rev.pop();
                    return;
                }
            }
        }

        fm_model_score_seed_reverse_dfs(
            fm,
            end,
            node - 1,
            max_len,
            is_complement,
            current_rev,
            lod_bits,
            best_prefix,
            consensus_codes,
            abc,
            located,
            limit,
            next_state,
            config,
        );
        current_rev.pop();
    }
}

#[cfg(test)]
pub(super) fn fm_interval_for_model_seed(
    fm: &FmIndex,
    bases: &[u8],
    is_complement: bool,
) -> Option<FmInterval> {
    let mut interval = fm.root_interval();
    for &ch in bases {
        let fm_ch = if is_complement {
            complement_dna_base(ch)
        } else {
            ch
        };
        interval = fm.prepend_interval(interval, fm_ch)?;
    }
    Some(interval)
}

pub(super) fn fm_interval_for_reverse_model_prefix(
    fm: &FmIndex,
    current_rev: &[u8],
    is_complement: bool,
) -> Option<FmInterval> {
    let mut interval = fm.root_interval();
    for &ch in current_rev.iter().rev() {
        let fm_ch = if is_complement {
            complement_dna_base(ch)
        } else {
            ch
        };
        interval = fm.prepend_interval(interval, fm_ch)?;
    }
    Some(interval)
}
