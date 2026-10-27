use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn nhmmer_score_seed_dfs(
    start: usize,
    node: usize,
    max_len: usize,
    current: &mut Vec<u8>,
    lod_bits: &[[f32; 4]],
    best_suffix: &[Vec<f32>],
    consensus_codes: &[Option<usize>],
    abc: &Alphabet,
    seeds: &mut Vec<NhmmerConsensusSeed>,
    seed_limit: usize,
    state: NhmmerFmScoreState,
    config: NhmmerFmSeedConfig,
) {
    if current.len() == max_len || seeds.len() >= seed_limit {
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
        current.push(ch);

        let len = current.len();
        if next_state.reaches_seed(config) {
            let model_end = start + len - 1;
            let seed = NhmmerConsensusSeed {
                bases: current.clone(),
                model_end,
            };
            if nhmmer_push_unique_score_seed(seeds, seed, seed_limit) {
                current.pop();
                return;
            }
        }

        nhmmer_score_seed_dfs(
            start,
            node + 1,
            max_len,
            current,
            lod_bits,
            best_suffix,
            consensus_codes,
            abc,
            seeds,
            seed_limit,
            next_state,
            config,
        );
        current.pop();
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn nhmmer_score_seed_reverse_dfs(
    end: usize,
    node: usize,
    max_len: usize,
    current_rev: &mut Vec<u8>,
    lod_bits: &[[f32; 4]],
    best_prefix: &[Vec<f32>],
    consensus_codes: &[Option<usize>],
    abc: &Alphabet,
    seeds: &mut Vec<NhmmerConsensusSeed>,
    seed_limit: usize,
    state: NhmmerFmScoreState,
    config: NhmmerFmSeedConfig,
) {
    if current_rev.len() == max_len || seeds.len() >= seed_limit || node == 0 {
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

        if next_state.reaches_seed(config) {
            let seed = NhmmerConsensusSeed {
                bases: current_rev.iter().rev().copied().collect(),
                model_end: end,
            };
            if nhmmer_push_unique_score_seed(seeds, seed, seed_limit) {
                current_rev.pop();
                return;
            }
        }

        nhmmer_score_seed_reverse_dfs(
            end,
            node - 1,
            max_len,
            current_rev,
            lod_bits,
            best_prefix,
            consensus_codes,
            abc,
            seeds,
            seed_limit,
            next_state,
            config,
        );
        current_rev.pop();
    }
}

pub(super) fn nhmmer_fm_consensus_codes(hmm: &Hmm, abc: &Alphabet) -> Vec<Option<usize>> {
    let consensus = nhmmer_consensus_bytes(hmm);
    let mut codes = vec![None; hmm.m + 1];
    for node in 1..=hmm.m.min(consensus.len()) {
        let ch = match consensus[node - 1].to_ascii_uppercase() {
            b'U' => b'T',
            ch => ch,
        };
        let code = abc.inmap[ch as usize];
        if code < abc.k as u8 && (code as usize) < 4 {
            codes[node] = Some(code as usize);
        }
    }
    codes
}

pub(super) fn nhmmer_next_consec_consensus(
    consensus_codes: &[Option<usize>],
    node: usize,
    x: usize,
    previous: usize,
) -> usize {
    if consensus_codes.get(node).copied().flatten() == Some(x) {
        previous + 1
    } else {
        0
    }
}

fn nhmmer_push_unique_score_seed(
    seeds: &mut Vec<NhmmerConsensusSeed>,
    seed: NhmmerConsensusSeed,
    seed_limit: usize,
) -> bool {
    if !seeds.iter().any(|existing| {
        existing.model_end == seed.model_end && existing.bases.as_slice() == seed.bases.as_slice()
    }) {
        seeds.push(seed);
    }
    seeds.len() >= seed_limit
}

pub(super) fn nhmmer_consensus_bytes(hmm: &Hmm) -> Vec<u8> {
    if let Some(consensus) = hmm.consensus.as_ref() {
        let bytes: Vec<u8> = consensus
            .iter()
            .skip(1)
            .take(hmm.m)
            .map(|&ch| ch.to_ascii_uppercase())
            .map(|ch| if ch == b'U' { b'T' } else { ch })
            .collect();
        if bytes
            .iter()
            .any(|&ch| matches!(ch, b'A' | b'C' | b'G' | b'T'))
        {
            return bytes;
        }
    }

    let abc = Alphabet::new(hmm.abc_type);
    (1..=hmm.m)
        .map(|node| {
            let best = hmm.mat[node]
                .iter()
                .take(abc.k)
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let ch = abc.sym[best].to_ascii_uppercase();
            if ch == b'U' {
                b'T'
            } else {
                ch
            }
        })
        .collect()
}
