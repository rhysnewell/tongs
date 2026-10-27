use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NhmmerCandidateWindow {
    pub(super) n: usize,
    pub(super) length: usize,
    pub(super) k: usize,
    pub(super) fm_frame: Option<NhmmerCandidateFmFrame>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NhmmerCandidateFmFrame {
    pub(super) fm_n: i64,
    pub(super) id: i32,
    pub(super) target_len: usize,
    pub(super) fm_start: i64,
    pub(super) fm_bwt_len: i64,
}

#[derive(Clone, Debug)]
pub(super) struct NhmmerScoredCandidateWindow {
    pub(super) window: NhmmerCandidateWindow,
    pub(super) score_bits: f32,
}

pub(super) fn precompute_c_faithful_fm_windows(
    target_db: &NhmmerTargetDb,
    sequences: &[Sequence],
    hmm: &Hmm,
    bg: &Bg,
    max_length: i32,
    f1: f64,
    config: NhmmerFmSeedConfig,
) -> Option<Vec<[Option<Vec<NhmmerCandidateWindow>>; 2]>> {
    if target_db.fm_index_records.is_empty() {
        return None;
    }
    if !target_db
        .fm_index_records
        .iter()
        .all(NhmmerFmIndexRecord::has_text)
    {
        return None;
    }

    let mut scored_by_seq: Vec<[Vec<NhmmerScoredCandidateWindow>; 2]> = (0..sequences.len())
        .map(|_| [Vec::new(), Vec::new()])
        .collect();

    for record in &target_db.fm_index_records {
        let strand_idx = record.kind as usize;
        if strand_idx > 1 {
            continue;
        }
        let sibling_kind = if record.kind == 0 { 1 } else { 0 };
        let Some(sibling) = target_db
            .fm_index_records
            .iter()
            .find(|r| r.block_id == record.block_id && r.kind == sibling_kind)
        else {
            continue;
        };
        let is_complement = record.kind == 1;
        let (fmf, fmb) = if is_complement {
            (&sibling.fm, &record.fm)
        } else {
            (&record.fm, &sibling.fm)
        };
        let Some(block_windows) = fm_ssv_collect_block_windows(
            fmf,
            fmb,
            record,
            &target_db.fm_ambiguities,
            &target_db.fm_sequence_meta,
            hmm,
            bg,
            config,
            is_complement,
            max_length,
            f1,
        ) else {
            continue;
        };
        for (seq_idx, windows) in block_windows.into_iter().enumerate() {
            if seq_idx < scored_by_seq.len() && !windows.is_empty() {
                scored_by_seq[seq_idx][strand_idx].extend(windows);
            }
        }
    }

    let mut out = Vec::with_capacity(sequences.len());
    for (seq_idx, [watson, crick]) in scored_by_seq.into_iter().enumerate() {
        let watson = fm_finalize_seed_windows(watson, 1, &sequences[seq_idx], hmm, bg, true);
        let crick = fm_finalize_seed_windows(crick, 1, &sequences[seq_idx], hmm, bg, true);
        out.push([watson, crick]);
    }
    Some(out)
}

#[cfg(test)]
pub(super) fn fm_seed_candidate_windows(
    target_db: &NhmmerTargetDb,
    seq_idx: usize,
    hmm: &Hmm,
    bg: &Bg,
    max_length: i32,
    is_complement: bool,
) -> Option<Vec<NhmmerCandidateWindow>> {
    fm_seed_candidate_windows_with_config(
        target_db,
        seq_idx,
        hmm,
        bg,
        max_length,
        0.02, // default --F1
        is_complement,
        NhmmerFmSeedConfig::default(),
    )
}

// FM-index candidate-window generation. Two complementary mechanisms produce
// the candidate windows handed to `p7_pipeline_longtarget`:
//
//   1. A seed-then-rescore stage (the consensus/score-threshold tries below):
//      enumerates candidate model k-mers, locates them in the FM index, extends
//      and merges. Fast but only finds windows around *exact* high-scoring
//      model k-mers.
//   2. `fm_ssv_augment_windows`: the faithful two-sweep port of C
//      `p7_SSVFM_longlarget` / `FM_Recurse` (`src/simd/fm_ssv.rs`), which walks
//      the FM trie scoring SSV diagonals directly over FM intervals and so
//      recovers weak diagonals with no exact k-mer match — closing the
//      sensitivity gap vs C. Its extended diagonals are filtered by C's
//      Gumbel-derived `sc_thresh` (matching `p7_SSVFM_longlarget`) before
//      windowing. Both strands are covered.
//
// The union of windows is deduped/merged in `fm_finalize_seed_windows`, and the
// real SSV/MSV/Viterbi/Forward scoring re-runs downstream in
// `p7_pipeline_longtarget`, which derives the final hit coordinates.
#[allow(clippy::too_many_arguments)]
pub(super) fn fm_seed_candidate_windows_with_config(
    target_db: &NhmmerTargetDb,
    seq_idx: usize,
    hmm: &Hmm,
    bg: &Bg,
    max_length: i32,
    f1: f64,
    is_complement: bool,
    config: NhmmerFmSeedConfig,
) -> Option<Vec<NhmmerCandidateWindow>> {
    if target_db.fm_index_records.is_empty() {
        return None;
    }
    let seq_meta = target_db.fm_sequence_meta.get(seq_idx)?;
    let seq = target_db.sequences.get(seq_idx)?;
    let seeds = nhmmer_seed_strings_with_config(hmm, bg, config);
    if seeds.is_empty() {
        return None;
    }

    let desired_len = config.ssv_length.max(1).min(seq.n);
    if desired_len == 0 {
        return None;
    }

    let mut windows = Vec::new();
    // When the FM-augment runs in its C-faithful `FM_extendSeed` mode, its
    // windows alone reproduce C `p7_SSVFM_longlarget`'s list; the Rust-only seed
    // sources below are then dropped (and the seed pre-merge / window gate
    // skipped) so the FM residue counters match C. See `fm_ssv_augment_windows`.
    let mut augment_c_faithful = false;
    for record in &target_db.fm_index_records {
        let expected_kind = if is_complement { 1 } else { 0 };
        if record.kind != expected_kind
            || seq_idx < record.seq_offset
            || seq_idx >= record.seq_offset + record.seq_count
        {
            continue;
        }

        let sibling_kind = if is_complement { 0 } else { 1 };
        let sibling = target_db
            .fm_index_records
            .iter()
            .find(|r| r.block_id == record.block_id && r.kind == sibling_kind);
        if let Some(sibling) = sibling {
            let (fmf, fmb) = if is_complement {
                (&sibling.fm, &record.fm)
            } else {
                (&record.fm, &sibling.fm)
            };
            let mut augment_windows = Vec::new();
            let c_faithful = fm_ssv_augment_windows(
                fmf,
                fmb,
                record,
                &target_db.fm_ambiguities,
                &target_db.fm_sequence_meta,
                seq_idx,
                seq,
                hmm,
                bg,
                config,
                is_complement,
                max_length,
                f1,
                desired_len,
                &mut augment_windows,
            );
            if c_faithful && !augment_windows.is_empty() {
                augment_c_faithful = true;
                windows = augment_windows;
                continue;
            }
            windows.extend(augment_windows);
        }

        if seq.dsq.len() <= 2 && seq.n > 0 {
            continue;
        }

        let seed_queries =
            fm_seed_queries_for_record(&seeds, record, seq.n, is_complement, hmm, bg);
        let located =
            fm_locate_seed_queries(&record.fm, &seed_queries, NHMMER_FM_WINDOW_SOURCE_LIMIT);
        for (seed, fm_pos) in located {
            fm_push_seed_window(
                &mut windows,
                &target_db.fm_ambiguities,
                seq_meta,
                seq,
                record,
                hmm,
                bg,
                is_complement,
                seed,
                fm_pos,
                desired_len,
            );
        }
        for (seed, fm_pos) in fm_locate_model_consensus_seeds(
            &record.fm,
            hmm,
            bg,
            is_complement,
            record.text_len,
            seq.n,
            NHMMER_FM_WINDOW_SOURCE_LIMIT,
            config,
        ) {
            fm_push_seed_window(
                &mut windows,
                &target_db.fm_ambiguities,
                seq_meta,
                seq,
                record,
                hmm,
                bg,
                is_complement,
                &seed,
                fm_pos,
                desired_len,
            );
        }
        let score_trie_limit = if record.text_len <= NHMMER_FM_SCORE_TRIE_NORMAL_TEXT_LIMIT
            && seq.n <= NHMMER_FM_SCORE_TRIE_NORMAL_TEXT_LIMIT
        {
            NHMMER_FM_SCORE_SEED_LIMIT
                .min(NHMMER_FM_WINDOW_SOURCE_LIMIT.saturating_sub(windows.len()))
        } else {
            0
        };
        if score_trie_limit > 0 {
            for (seed, fm_pos) in fm_locate_model_score_seeds_with_start_limit(
                &record.fm,
                hmm,
                bg,
                is_complement,
                record.text_len,
                seq.n,
                score_trie_limit,
                NHMMER_FM_SCORE_TRIE_NORMAL_START_LIMIT,
                config,
            ) {
                fm_push_seed_window(
                    &mut windows,
                    &target_db.fm_ambiguities,
                    seq_meta,
                    seq,
                    record,
                    hmm,
                    bg,
                    is_complement,
                    &seed,
                    fm_pos,
                    desired_len,
                );
            }
        }
    }

    if windows.is_empty() {
        return None;
    }
    fm_finalize_seed_windows(windows, desired_len, seq, hmm, bg, augment_c_faithful)
}

/// Reduce the raw FM seed windows to the candidate-window list handed to the
/// long-target pipeline.
///
/// `c_faithful` is `true` when the windows came from the C-faithful FM-augment
/// (`FM_getSeeds` + `FM_extendSeed`); in that case the windows already match C
/// `p7_SSVFM_longlarget`'s extended-diagonal list, so the Rust-specific seed
/// pre-merge (`fm_merge_seed_windows`) and the lenient window gate
/// (`fm_extended_diagonal_passes_window_gate`) are skipped — the downstream
/// `p7_pli_extend_and_merge_windows(.., 0.0)` does C's
/// `p7_pli_ExtendAndMergeWindows(.., 0)` merge, which makes the FM residue
/// counters match C. For the fallback seed-then-rescore path (`c_faithful ==
/// false`) those Rust-only guards stay in place to keep the hit set correct.
pub(super) fn fm_finalize_seed_windows(
    windows: Vec<NhmmerScoredCandidateWindow>,
    desired_len: usize,
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
    c_faithful: bool,
) -> Option<Vec<NhmmerCandidateWindow>> {
    let mut windows = if c_faithful {
        windows
    } else {
        fm_merge_seed_windows(windows, desired_len, seq, hmm, bg)
    };
    if !c_faithful {
        windows.retain(|scored| {
            fm_extended_diagonal_passes_window_gate(seq, hmm, bg, &scored.window, scored.score_bits)
        });
    }
    if windows.is_empty() {
        return None;
    }
    windows.sort_by(|a, b| {
        (a.window.n, a.window.length, a.window.k)
            .cmp(&(b.window.n, b.window.length, b.window.k))
            .then_with(|| b.score_bits.total_cmp(&a.score_bits))
    });
    windows.dedup_by(|next, prev| next.window == prev.window);
    windows.sort_by(|a, b| {
        b.score_bits
            .total_cmp(&a.score_bits)
            .then_with(|| a.window.n.cmp(&b.window.n))
            .then_with(|| a.window.length.cmp(&b.window.length))
            .then_with(|| a.window.k.cmp(&b.window.k))
    });
    windows.sort_by_key(|w| (w.window.n, w.window.length, w.window.k));
    Some(windows.into_iter().map(|w| w.window).collect())
}

pub(super) fn fm_merge_seed_windows(
    mut windows: Vec<NhmmerScoredCandidateWindow>,
    ssv_length: usize,
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
) -> Vec<NhmmerScoredCandidateWindow> {
    if windows.len() < 2 {
        return windows;
    }

    windows.sort_by(|a, b| {
        fm_window_diag_key(&a.window)
            .cmp(&fm_window_diag_key(&b.window))
            .then_with(|| a.window.n.cmp(&b.window.n))
            .then_with(|| fm_window_model_start(&a.window).cmp(&fm_window_model_start(&b.window)))
    });

    let mut merged = Vec::with_capacity(windows.len());
    let mut current = windows[0].window.clone();
    for scored in windows.into_iter().skip(1) {
        let next = scored.window;
        let current_diag = fm_window_diag_key(&current);
        let next_diag = fm_window_diag_key(&next);
        let current_start0 = current.n - 1;
        let current_model_start = fm_window_model_start(&current);
        let current_end0 = current.n - 1 + current.length - 1;
        let next_end0 = next.n - 1 + next.length - 1;
        if current_diag == next_diag
            && next.n - 1 + next.length < current.n - 1 + current.length + ssv_length
        {
            if next_end0 > current_end0 {
                current.length = next_end0 - current_start0 + 1;
                current.k = current_model_start + current.length - 1;
            }
        } else {
            merged.push(fm_rescore_seed_window(current, seq, hmm, bg));
            current = next;
        }
    }
    merged.push(fm_rescore_seed_window(current, seq, hmm, bg));
    merged
}

fn fm_window_model_start(window: &NhmmerCandidateWindow) -> usize {
    window.k.saturating_sub(window.length).saturating_add(1)
}

fn fm_window_diag_key(window: &NhmmerCandidateWindow) -> isize {
    (window.n as isize - 1) - fm_window_model_start(window) as isize
}

fn fm_rescore_seed_window(
    window: NhmmerCandidateWindow,
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
) -> NhmmerScoredCandidateWindow {
    let model_start = fm_window_model_start(&window);
    let score_bits = fm_diagonal_lod_bits(seq, hmm, bg, model_start, window.n - 1, window.length);
    NhmmerScoredCandidateWindow { window, score_bits }
}

pub(super) fn fm_extended_diagonal_passes_window_gate(
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
    window: &NhmmerCandidateWindow,
    score_bits: f32,
) -> bool {
    if !score_bits.is_finite() {
        return false;
    }
    if score_bits > NHMMER_FM_MIN_EXTENDED_DIAG_SCORE_BITS {
        return true;
    }
    !fm_diagonal_has_complete_lod(seq, hmm, bg, window)
}

fn fm_diagonal_has_complete_lod(
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
    window: &NhmmerCandidateWindow,
) -> bool {
    let model_start = fm_window_model_start(window);
    (0..window.length).all(|offset| {
        fm_match_lod_bits(seq, hmm, bg, model_start + offset, window.n + offset).is_some()
    })
}
