use super::*;

pub(super) fn reverse_complement_dna_seed(seed: &[u8]) -> Vec<u8> {
    seed.iter()
        .rev()
        .map(|&ch| match ch {
            b'A' => b'T',
            b'C' => b'G',
            b'G' => b'C',
            b'T' | b'U' => b'A',
            _ => ch,
        })
        .collect()
}

pub(super) fn complement_dna_base(ch: u8) -> u8 {
    match ch {
        b'A' => b'T',
        b'C' => b'G',
        b'G' => b'C',
        b'T' | b'U' => b'A',
        _ => ch,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NhmmerConsensusSeed {
    pub(super) bases: Vec<u8>,
    pub(super) model_end: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct NhmmerFmScoreState {
    pub(super) score_bits: f32,
    pub(super) max_score_bits: f32,
    pub(super) score_peak_len: usize,
    pub(super) consec_pos: usize,
    pub(super) max_consec_pos: usize,
    pub(super) consec_consensus: usize,
}

impl NhmmerFmScoreState {
    pub(super) fn new() -> Self {
        Self {
            score_bits: 0.0,
            max_score_bits: 0.0,
            score_peak_len: 0,
            consec_pos: 0,
            max_consec_pos: 0,
            consec_consensus: 0,
        }
    }

    pub(super) fn extend(
        self,
        delta_bits: f32,
        depth: usize,
        consensus_codes: &[Option<usize>],
        node: usize,
        code: usize,
        config: NhmmerFmSeedConfig,
    ) -> Self {
        let score_bits = self.score_bits + delta_bits;
        let consec_pos = if delta_bits > 0.0 {
            self.consec_pos + 1
        } else {
            0
        };
        let consec_consensus =
            nhmmer_next_consec_consensus(consensus_codes, node, code, self.consec_consensus);
        let (max_score_bits, score_peak_len) = if score_bits > self.max_score_bits {
            (score_bits, depth)
        } else if score_bits >= self.max_score_bits - config.drop_lim_bits {
            (self.max_score_bits, depth)
        } else {
            (self.max_score_bits, self.score_peak_len)
        };
        Self {
            score_bits,
            max_score_bits,
            score_peak_len,
            consec_pos,
            max_consec_pos: self.max_consec_pos.max(consec_pos),
            consec_consensus,
        }
    }

    pub(super) fn reaches_seed(self, config: NhmmerFmSeedConfig) -> bool {
        self.score_bits >= config.score_threshold_bits
            || (config.consensus_match_req > 0
                && self.consec_consensus == config.consensus_match_req)
    }

    pub(super) fn can_reach_consensus_seed(
        self,
        remaining_including_next: usize,
        config: NhmmerFmSeedConfig,
    ) -> bool {
        config.consensus_match_req > 0
            && self.consec_consensus + remaining_including_next >= config.consensus_match_req
    }

    pub(super) fn should_prune_below_threshold(
        self,
        depth: usize,
        max_len: usize,
        config: NhmmerFmSeedConfig,
    ) -> bool {
        self.score_bits <= 0.0
            || depth == max_len
            || (config.drop_max_len > 0 && depth == self.score_peak_len + config.drop_max_len)
            || (self.score_bits < config.score_threshold_bits
                && depth > 4
                && depth > self.consec_consensus
                && self.score_bits / (depth as f32) < config.score_density_bits)
            || (config.consec_pos_req > 0
                && self.max_consec_pos < config.consec_pos_req
                && config.consec_pos_req.saturating_sub(self.consec_pos)
                    == max_len.saturating_sub(depth).saturating_add(1))
    }
}

pub(super) fn nhmmer_seed_strings_with_config(
    hmm: &Hmm,
    bg: &Bg,
    config: NhmmerFmSeedConfig,
) -> Vec<NhmmerConsensusSeed> {
    let consensus = nhmmer_consensus_bytes(hmm);
    let seed_len = consensus.len().min(config.max_depth).max(1);
    let mut seeds = Vec::new();
    nhmmer_push_consensus_seeds(&consensus, seed_len, &mut seeds);

    // C's FM trie also accepts a seed when it accumulates
    // `--seed_consens_match` consecutive consensus matches (default 10), even
    // before the short diagonal reaches the score threshold. Add that concrete
    // seed tier so FM-index targets do not fall back solely because no exact
    // max-depth consensus seed occurs in the target.
    if config.consensus_match_req > 0 && seed_len > config.consensus_match_req {
        nhmmer_push_consensus_seeds(&consensus, config.consensus_match_req, &mut seeds);
    }
    nhmmer_push_score_threshold_seeds(hmm, bg, &mut seeds, config);
    seeds
}

fn nhmmer_push_consensus_seeds(
    consensus: &[u8],
    seed_len: usize,
    seeds: &mut Vec<NhmmerConsensusSeed>,
) {
    if seed_len == 0 {
        return;
    }
    for (start0, window) in consensus.windows(seed_len).enumerate() {
        if !window
            .iter()
            .all(|&ch| matches!(ch, b'A' | b'C' | b'G' | b'T'))
        {
            continue;
        }
        let model_end = start0 + seed_len;
        if !seeds
            .iter()
            .any(|seed| seed.model_end == model_end && seed.bases == window)
        {
            seeds.push(NhmmerConsensusSeed {
                bases: window.to_vec(),
                model_end,
            });
            if seeds.len() >= 64 {
                break;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fm_locate_model_consensus_seeds(
    fm: &FmIndex,
    hmm: &Hmm,
    bg: &Bg,
    is_complement: bool,
    text_len: usize,
    seq_len: usize,
    limit: usize,
    config: NhmmerFmSeedConfig,
) -> Vec<(NhmmerConsensusSeed, usize)> {
    let consensus = nhmmer_consensus_bytes(hmm);
    if consensus.is_empty() || text_len == 0 || seq_len == 0 || limit == 0 {
        return Vec::new();
    }

    let full_seed_len = consensus
        .len()
        .min(config.max_depth)
        .min(text_len)
        .min(seq_len);
    let short_seed_len = if config.consensus_match_req > 0 {
        config
            .consensus_match_req
            .min(consensus.len())
            .min(text_len)
            .min(seq_len)
    } else {
        full_seed_len
    };
    if full_seed_len == 0 {
        return Vec::new();
    }

    let mut located = Vec::new();
    let mut starts: Vec<(usize, f32)> = (0..consensus.len())
        .map(|start0| {
            let seed_len = full_seed_len.min(consensus.len() - start0);
            (
                start0,
                fm_consensus_seed_model_lod_bits(&consensus, start0, seed_len, hmm, bg),
            )
        })
        .collect();
    starts.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (start0, _) in starts {
        if located.len() >= limit {
            break;
        }
        let remaining = consensus.len() - start0;
        let max_len = full_seed_len.min(remaining);
        if max_len == 0 {
            continue;
        }
        fm_model_consensus_seed_dfs(
            fm,
            &consensus,
            start0,
            max_len,
            full_seed_len,
            short_seed_len,
            fm.root_interval(),
            is_complement,
            &mut Vec::with_capacity(max_len),
            &mut located,
            limit,
        );
    }
    located
}

fn fm_consensus_seed_model_lod_bits(
    consensus: &[u8],
    start0: usize,
    len: usize,
    hmm: &Hmm,
    bg: &Bg,
) -> f32 {
    let Some(seed_end) = start0.checked_add(len) else {
        return f32::NEG_INFINITY;
    };
    if seed_end > consensus.len() || seed_end > hmm.m {
        return f32::NEG_INFINITY;
    }
    let seed = NhmmerConsensusSeed {
        bases: consensus[start0..seed_end].to_vec(),
        model_end: seed_end,
    };
    fm_seed_model_lod_bits(hmm, bg, &seed)
}

#[allow(clippy::too_many_arguments)]
fn fm_model_consensus_seed_dfs(
    fm: &FmIndex,
    consensus: &[u8],
    start0: usize,
    max_len: usize,
    full_seed_len: usize,
    short_seed_len: usize,
    interval: FmInterval,
    is_complement: bool,
    current: &mut Vec<u8>,
    located: &mut Vec<(NhmmerConsensusSeed, usize)>,
    limit: usize,
) {
    if current.len() == max_len || located.len() >= limit {
        return;
    }

    let node0 = start0 + current.len();
    let ch = consensus[node0];
    if !matches!(ch, b'A' | b'C' | b'G' | b'T') {
        return;
    }
    let fm_ch = if is_complement {
        complement_dna_base(ch)
    } else {
        ch
    };
    let Some(child_interval) = fm.prepend_interval(interval, fm_ch) else {
        return;
    };

    current.push(ch);
    let len = current.len();
    if len == full_seed_len || (full_seed_len > short_seed_len && len == short_seed_len) {
        let seed = NhmmerConsensusSeed {
            bases: current.clone(),
            model_end: start0 + len,
        };
        for fm_pos in fm.locate_interval_limited(child_interval, limit - located.len()) {
            located.push((seed.clone(), fm_pos));
            if located.len() >= limit {
                current.pop();
                return;
            }
        }
    }

    fm_model_consensus_seed_dfs(
        fm,
        consensus,
        start0,
        max_len,
        full_seed_len,
        short_seed_len,
        child_interval,
        is_complement,
        current,
        located,
        limit,
    );
    current.pop();
}

fn nhmmer_push_score_threshold_seeds(
    hmm: &Hmm,
    bg: &Bg,
    seeds: &mut Vec<NhmmerConsensusSeed>,
    config: NhmmerFmSeedConfig,
) {
    let Some((abc, max_depth, lod_bits, best_suffix, best_prefix, consensus_codes)) =
        nhmmer_fm_lod_tables(hmm, bg, config)
    else {
        return;
    };

    let before = seeds.len();
    let seed_limit = before + NHMMER_FM_SCORE_SEED_LIMIT;
    let mut starts: Vec<(bool, usize, f32)> = (1..=hmm.m)
        .filter_map(|start| {
            let max_len = max_depth.min(hmm.m - start + 1);
            (max_len > 0).then_some((false, start, best_suffix[start][max_len]))
        })
        .collect();
    starts.extend((1..=hmm.m).filter_map(|end| {
        let max_len = max_depth.min(end);
        (max_len > 0).then_some((true, end, best_prefix[end][max_len]))
    }));
    starts.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.1.cmp(&b.1))
    });

    for (reverse, anchor, _) in starts {
        if seeds.len() >= before + NHMMER_FM_SCORE_SEED_LIMIT {
            break;
        }
        if reverse {
            let max_len = max_depth.min(anchor);
            let mut current_rev = Vec::with_capacity(max_len);
            nhmmer_score_seed_reverse_dfs(
                anchor,
                anchor,
                max_len,
                &mut current_rev,
                &lod_bits,
                &best_prefix,
                &consensus_codes,
                &abc,
                seeds,
                seed_limit,
                NhmmerFmScoreState::new(),
                config,
            );
        } else {
            let max_len = max_depth.min(hmm.m - anchor + 1);
            let mut current = Vec::with_capacity(max_len);
            nhmmer_score_seed_dfs(
                anchor,
                anchor,
                max_len,
                &mut current,
                &lod_bits,
                &best_suffix,
                &consensus_codes,
                &abc,
                seeds,
                seed_limit,
                NhmmerFmScoreState::new(),
                config,
            );
        }
    }
}

#[allow(clippy::type_complexity)]
pub(super) fn nhmmer_fm_lod_tables(
    hmm: &Hmm,
    bg: &Bg,
    config: NhmmerFmSeedConfig,
) -> Option<(
    Alphabet,
    usize,
    Vec<[f32; 4]>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Option<usize>>,
)> {
    if !matches!(hmm.abc_type, AlphabetType::Dna | AlphabetType::Rna) || hmm.m == 0 {
        return None;
    }

    let abc = Alphabet::new(hmm.abc_type);
    let max_depth = hmm.m.min(config.max_depth);
    if max_depth == 0 {
        return None;
    }

    let mut lod_bits = vec![[f32::NEG_INFINITY; 4]; hmm.m + 1];
    let mut best_suffix = vec![vec![0.0_f32; max_depth + 1]; hmm.m + 2];
    let mut best_prefix = vec![vec![0.0_f32; max_depth + 1]; hmm.m + 1];
    for (node, lod_row) in lod_bits.iter_mut().enumerate().take(hmm.m + 1).skip(1) {
        for (x, lod_cell) in lod_row.iter_mut().enumerate().take(abc.k.min(4)) {
            if hmm.mat[node][x] > 0.0 && bg.f[x] > 0.0 {
                *lod_cell = ((hmm.mat[node][x] as f64) / (bg.f[x] as f64)).log2() as f32;
            }
        }
    }
    for node in (1..=hmm.m).rev() {
        let best_here = lod_bits[node]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            .max(0.0);
        for len in 1..=max_depth {
            best_suffix[node][len] = best_here + best_suffix[node + 1][len - 1];
        }
    }
    for node in 1..=hmm.m {
        let best_here = lod_bits[node]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            .max(0.0);
        for len in 1..=max_depth {
            best_prefix[node][len] = best_here + best_prefix[node - 1][len - 1];
        }
    }

    let consensus_codes = nhmmer_fm_consensus_codes(hmm, &abc);

    Some((
        abc,
        max_depth,
        lod_bits,
        best_suffix,
        best_prefix,
        consensus_codes,
    ))
}
