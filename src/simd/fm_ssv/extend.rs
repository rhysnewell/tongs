use super::*;

#[inline]
pub(super) fn initial_pair(
    k: i32,
    sc: f32,
    is_consensus: bool,
    complementarity: Complementarity,
    model_direction: ModelDirection,
) -> FmDpPair {
    FmDpPair {
        pos: k,
        score: sc,
        max_score: sc,
        score_peak_len: 1,
        consec_pos: 1,
        max_consec_pos: 1,
        consec_consensus: if is_consensus { 1 } else { 0 },
        complementarity,
        model_direction,
    }
}

/// Faithful port of C `FM_mergeSeeds`: sort by (strand, diagonal, model pos),
/// then merge overlapping/nearby diagonals on the same diagonal.
pub fn fm_merge_seeds(seeds: &mut Vec<FmDiag>, fm_n: i64, m: i32, ssv_length: i32) {
    if seeds.is_empty() {
        return;
    }

    seeds.sort_by(|a, b| {
        a.sortkey(fm_n, m)
            .partial_cmp(&b.sortkey(fm_n, m))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let curr = seeds[0];
    let mut write_pos = 0usize;
    let mut curr_complement = curr.complementarity == Complementarity::Complement;
    let mut curr_n = curr.n;
    let mut curr_k = curr.k;
    let mut curr_len = curr.length;
    let mut curr_end = curr_n + curr_len as i64 - 1;
    let mut curr_diagval = curr.n - curr.k as i64;

    for read_pos in 1..seeds.len() {
        let next = seeds[read_pos];
        let next_complement = next.complementarity == Complementarity::Complement;
        if next_complement == curr_complement
            && (next.n - next.k as i64) == curr_diagval
            && next.n + (next.length as i64) < curr_n + curr_len as i64 + ssv_length as i64
        {
            let tmp = next.n + next.length as i64 - 1;
            if tmp > curr_end {
                curr_end = tmp;
                curr_len = (curr_end - curr_n + 1) as i32;
            }
        } else {
            seeds[write_pos] = FmDiag {
                n: curr_n,
                k: curr_k,
                length: (curr_end - curr_n + 1) as i32,
                score: 0.0,
                complementarity: if curr_complement {
                    Complementarity::Complement
                } else {
                    Complementarity::NoComplement
                },
            };
            write_pos += 1;
            curr_n = next.n;
            curr_k = next.k;
            curr_len = next.length;
            curr_end = curr_n + curr_len as i64 - 1;
            curr_diagval = next.n - next.k as i64;
            curr_complement = next_complement;
        }
    }
    seeds[write_pos] = FmDiag {
        n: curr_n,
        k: curr_k,
        length: (curr_end - curr_n + 1) as i32,
        score: 0.0,
        complementarity: if curr_complement {
            Complementarity::Complement
        } else {
            Complementarity::NoComplement
        },
    };
    write_pos += 1;

    seeds.truncate(write_pos);
}

/// Faithful port of C `FM_extendSeed`: extend the seed in both directions
/// within a bounded window and keep the best-scoring sub-diagonal.
///
/// `target` is the WHOLE (already strand-resolved) target text in alphabet
/// codes, 1-indexed so `target[1]` is FM position 0 (index 0 is a sentinel),
/// matching C's `fm_convertRange2DSQ` extraction of `[target_start, target_end]`
/// into `tmp_sq->dsq[1..]`. `fm_n` is the FM text length `fm->N`.
pub fn fm_extend_seed_core(
    diag: &mut FmDiag,
    target: &[usize],
    scores: &SsvScores,
    config: &FmSsvConfig,
    fm_n: i64,
) {
    let m = scores.m;
    let kp = scores.kp;
    let extend = (config.ssv_length as i32 - diag.length).max(10);

    // C: model_start = max(1, diag->k - extend + 1); the "+1" is load-bearing.
    let mut model_start = (diag.k - extend + 1).max(1);
    let mut model_end = (diag.k + diag.length + extend - 1).min(m);
    let mut target_start: i64 = diag.n - (diag.k as i64 - model_start as i64);
    let mut target_end: i64 = diag.n + (model_end as i64 - diag.k as i64);

    if target_start < 0 {
        model_start -= target_start as i32;
        target_start = 0;
    }
    if target_end > fm_n - 2 {
        model_end -= (target_end - (fm_n - 2)) as i32;
        target_end = fm_n - 2;
    }
    if model_start > model_end || target_start > target_end {
        return;
    }

    // Walk model_start..=model_end against the extracted target range. C reads
    // tmp_sq->dsq[n] with n starting at 1, where tmp_sq holds the range
    // [target_start, target_end] extracted by fm_convertRange2DSQ. Here `target`
    // is the WHOLE strand-resolved text, 1-indexed (target[1] = FM position 0),
    // so position `target_start + (n-1)` (0-based) maps to absolute 1-index
    // `target_start + n`. `n` is the 1-based offset within the extracted range
    // and is what max_hit_start/end are expressed in, exactly like C.
    let mut k = model_start;
    let mut n: usize = 1;
    let mut sc = 0.0_f32;
    let mut hit_start: i64 = n as i64;
    let mut max_sc = 0.0_f32;
    let mut max_hit_start: i64 = n as i64;
    let mut max_hit_end: i64 = n as i64;

    while k <= model_end {
        let abs_idx = (target_start + n as i64) as usize;
        let cidx = (k as usize) * kp + target[abs_idx];
        let delta = scores.scores[cidx];
        sc += delta;
        if sc < 0.0 {
            sc = 0.0;
            hit_start = n as i64 + 1;
        } else if sc > max_sc {
            max_sc = sc;
            max_hit_start = hit_start;
            max_hit_end = n as i64;
        }
        k += 1;
        n += 1;
    }

    diag.n = target_start + max_hit_start - 1;
    diag.k = model_start + (max_hit_start - 1) as i32;
    diag.length = (max_hit_end - max_hit_start + 1) as i32;
    diag.score = max_sc;
}
