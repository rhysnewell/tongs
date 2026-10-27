use super::*;

/// Faithful Rust counterpart of C `fm_convertRange2DSQ()`.
///
/// `first` is a 0-based position in FM->T for Watson, or in revcomp(FM->T) for
/// Crick. The returned DSQ is 1-based with sentinels at both ends.
pub(super) fn fm_convert_range_to_dsq(
    record: &NhmmerFmIndexRecord,
    ambiguities: &[(usize, usize)],
    first: usize,
    length: usize,
    is_complement: bool,
    fix_ambiguities: bool,
) -> Option<Vec<u8>> {
    let mut first = first;
    if is_complement {
        first = record
            .fm
            .bwt_len()
            .checked_sub(first.checked_add(length)?)?
            .checked_sub(1)?;
    }
    let end = first.checked_add(length)?;
    if end > record.text_len {
        return None;
    }

    let mut dsq = Vec::with_capacity(length + 2);
    dsq.push(DSQ_SENTINEL);
    if let Some(packed_text) = &record.mapped_packed_text {
        for pos in first..end {
            dsq.push(packed_dna_quad_dsq_at_mapped(packed_text, pos)?);
        }
    } else if !record.packed_text.is_empty() {
        for pos in first..end {
            dsq.push(packed_dna_quad_dsq_at(&record.packed_text, pos)?);
        }
    } else {
        for &base in &record.text[first..end] {
            dsq.push(match base {
                b'A' | b'a' => 0,
                b'C' | b'c' => 1,
                b'G' | b'g' => 2,
                b'T' | b't' | b'U' | b'u' => 3,
                _ => 0,
            });
        }
    }
    dsq.push(DSQ_SENTINEL);

    if fix_ambiguities {
        let range_end = end.saturating_sub(1);
        for &(lower, upper) in ambiguities
            .iter()
            .skip(record.ambig_offset)
            .take(record.ambig_count)
        {
            if upper < first {
                continue;
            }
            if lower > range_end {
                break;
            }
            let start = lower.max(first);
            let stop = upper.min(range_end);
            for pos in start..=stop {
                dsq[pos - first + 1] = 15; // DNA unknown/N = Kp-3 in Easel/HMMER.
            }
        }
    }

    if is_complement {
        let last = dsq.len().saturating_sub(2);
        for i in 1..=(last / 2) {
            let j = last - i + 1;
            let a = complement_dna_dsq(dsq[i]);
            let b = complement_dna_dsq(dsq[j]);
            dsq[i] = b;
            dsq[j] = a;
        }
        if last % 2 == 1 {
            let mid = last.div_ceil(2);
            dsq[mid] = complement_dna_dsq(dsq[mid]);
        }
    }

    Some(dsq)
}

/// Faithful Rust counterpart of C `fm_computeSequenceOffset()`.
fn fm_compute_sequence_offset(
    record: &NhmmerFmIndexRecord,
    seq_meta: &[NhmmerFmSequenceMeta],
    pos: usize,
) -> Option<usize> {
    if record.seq_count == 0 {
        return None;
    }
    let mut lo = record.seq_offset;
    let mut hi = lo.checked_add(record.seq_count)?.checked_sub(1)?;
    if hi >= seq_meta.len() {
        return None;
    }
    if lo == hi {
        return Some(lo);
    }
    if seq_meta[hi].fm_start <= pos {
        return Some(hi);
    }
    loop {
        let mid = (lo + hi).div_ceil(2);
        if seq_meta[mid].fm_start <= pos {
            lo = mid;
        } else if mid == 0 || seq_meta[mid - 1].fm_start > pos {
            hi = mid;
        } else {
            return Some(mid - 1);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NhmmerFmOriginalPosition {
    pub(super) segment_id: usize,
    pub(super) seg_pos: usize,
    pub(super) crosses_segment: bool,
}

/// Faithful Rust counterpart of C `fm_getOriginalPosition()`.
pub(super) fn fm_get_original_position(
    record: &NhmmerFmIndexRecord,
    seq_meta: &[NhmmerFmSequenceMeta],
    length: usize,
    is_complement: bool,
    fm_pos: usize,
) -> Option<NhmmerFmOriginalPosition> {
    let mut fm_pos = fm_pos;
    if is_complement {
        fm_pos = record.fm.bwt_len().checked_sub(fm_pos)?.checked_sub(1)?;
    }
    let segment_id = fm_compute_sequence_offset(record, seq_meta, fm_pos)?;
    let segment = seq_meta.get(segment_id)?;
    let mut seg_pos = fm_pos.checked_sub(segment.fm_start)?.checked_add(1)?;
    if is_complement {
        seg_pos = segment.length.checked_sub(seg_pos)?.checked_add(1)?;
    }
    let crosses_segment = seg_pos
        .checked_add(length)
        .and_then(|value| value.checked_sub(1))
        .is_none_or(|end| end > segment.length);
    Some(NhmmerFmOriginalPosition {
        segment_id,
        seg_pos,
        crosses_segment,
    })
}

fn complement_dna_dsq(code: u8) -> u8 {
    match code {
        0 => 3,
        1 => 2,
        2 => 1,
        3 => 0,
        _ => code,
    }
}

pub(super) fn packed_dna_quad_dsq_at(packed: &[u8], pos: usize) -> Option<u8> {
    let byte = *packed.get(pos / 4)?;
    Some((byte >> (6 - ((pos & 0x3) * 2))) & 0b11)
}

pub(super) fn packed_dna_quad_dsq_at_mapped(packed: &MappedPackedBytes, pos: usize) -> Option<u8> {
    let byte = packed.byte_at(pos / 4)?;
    Some((byte >> (6 - ((pos & 0x3) * 2))) & 0b11)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fm_push_seed_window(
    windows: &mut Vec<NhmmerScoredCandidateWindow>,
    ambiguities: &[(usize, usize)],
    seq_meta: &NhmmerFmSequenceMeta,
    seq: &Sequence,
    record: &NhmmerFmIndexRecord,
    hmm: &Hmm,
    bg: &Bg,
    is_complement: bool,
    seed: &NhmmerConsensusSeed,
    fm_pos: usize,
    ssv_length: usize,
) {
    if fm_pos + seed.bases.len() > record.text_len {
        return;
    }
    let block_seed_start = if is_complement {
        fm_pos
    } else {
        record.text_len - fm_pos - seed.bases.len()
    };
    let Some(block_seed_end) = block_seed_start.checked_add(seed.bases.len()) else {
        return;
    };
    if fm_seed_is_wholly_in_leading_overlap(record, block_seed_end) {
        return;
    }
    if fm_seed_overlaps_ambiguity(
        ambiguities,
        record.ambig_offset,
        record.ambig_count,
        block_seed_start,
        block_seed_end,
    ) {
        return;
    }
    let forward_seq_seed_start = match block_seed_start.checked_sub(seq_meta.fm_start) {
        Some(pos) if pos + seed.bases.len() <= seq_meta.length => pos,
        _ => return,
    };
    let seq_seed_start = if is_complement {
        seq.n - forward_seq_seed_start - seed.bases.len()
    } else {
        forward_seq_seed_start
    };
    let (window_start0, window_len, model_end, score_bits) =
        fm_extend_seed_diagonal(seq, hmm, bg, seed, seq_seed_start, ssv_length);
    let window = NhmmerCandidateWindow {
        n: window_start0 + 1,
        length: window_len,
        k: model_end,
        fm_frame: None,
    };
    if !fm_extended_diagonal_passes_window_gate(seq, hmm, bg, &window, score_bits) {
        return;
    }
    windows.push(NhmmerScoredCandidateWindow { window, score_bits });
}

/// Emit a candidate window for a diagonal that has ALREADY been extended by the
/// faithful `fm_ssv::fm_extend_seed` (C `FM_extendSeed`), rather than re-running
/// the local sequence-space extension. This is the FM-augment path: the kernel
/// already produced an extended diagonal `(diag.n, diag.k, diag.length,
/// diag.score)` in C's coordinate/score frame, so we only do the same
/// FM→sequence coordinate mapping as C `FM_window_from_diag()` and credit the
/// kernel's own bit score. `seed_len` is the extended diagonal length, `fm_pos`
/// is `diag.n`, `model_end` is `diag.k + diag.length - 1`, and `score_bits` is
/// `diag.score`.
#[allow(clippy::too_many_arguments)]
pub(super) fn fm_push_extended_diag_window(
    windows: &mut Vec<NhmmerScoredCandidateWindow>,
    _ambiguities: &[(usize, usize)],
    seq_meta: &[NhmmerFmSequenceMeta],
    seq_idx: usize,
    seq: &Sequence,
    record: &NhmmerFmIndexRecord,
    is_complement: bool,
    seed_len: usize,
    model_end: usize,
    fm_pos: usize,
    score_bits: f32,
) {
    if seed_len == 0 || fm_pos > record.text_len || !score_bits.is_finite() {
        return;
    }
    let diag = crate::simd::fm_ssv::FmDiag {
        n: fm_pos as i64,
        k: model_end.saturating_sub(seed_len).saturating_add(1) as i32,
        length: seed_len as i32,
        score: score_bits,
        complementarity: if is_complement {
            crate::simd::fm_ssv::Complementarity::Complement
        } else {
            crate::simd::fm_ssv::Complementarity::NoComplement
        },
    };
    if let Some(window) = fm_window_from_diag(record, seq_meta, seq_idx, seq.n, &diag) {
        windows.push(window);
    }
}

/// Faithful Rust counterpart of C `FM_window_from_diag()`: convert an
/// FM-coordinate diagonal into the original sequence coordinate frame.
pub(super) fn fm_window_from_diag(
    record: &NhmmerFmIndexRecord,
    seq_meta: &[NhmmerFmSequenceMeta],
    seq_idx: usize,
    seq_len: usize,
    diag: &crate::simd::fm_ssv::FmDiag,
) -> Option<NhmmerScoredCandidateWindow> {
    if diag.length <= 0 || diag.n < 0 || !diag.score.is_finite() {
        return None;
    }
    let is_complement = diag.complementarity == crate::simd::fm_ssv::Complementarity::Complement;
    let seed_len = diag.length as usize;
    let fm_pos = diag.n as usize;
    let model_end = (diag.k + diag.length - 1) as usize;
    let pos = fm_get_original_position(record, seq_meta, seed_len, is_complement, fm_pos)?;
    if pos.segment_id != seq_idx || pos.seg_pos == 0 {
        return None;
    }
    let seq_seed_start = pos.seg_pos - 1;
    if seq_seed_start >= seq_len {
        return None;
    }
    let window = NhmmerCandidateWindow {
        n: seq_seed_start + 1,
        length: seed_len,
        k: model_end,
        fm_frame: Some(NhmmerCandidateFmFrame {
            fm_n: fm_pos as i64,
            id: pos.segment_id as i32,
            target_len: seq_len,
            fm_start: seq_meta
                .get(seq_idx)
                .map(|meta| meta.fm_start as i64)
                .unwrap_or(0),
            fm_bwt_len: record.fm.bwt_len() as i64,
        }),
    };
    Some(NhmmerScoredCandidateWindow {
        window,
        score_bits: diag.score,
    })
}

/// Convert an FM-coordinate diagonal into whichever original sequence segment
/// it belongs to, matching C `FM_window_from_diag()`'s block-level use.
pub(super) fn fm_window_from_diag_any_segment(
    record: &NhmmerFmIndexRecord,
    seq_meta: &[NhmmerFmSequenceMeta],
    diag: &crate::simd::fm_ssv::FmDiag,
) -> Option<(usize, NhmmerScoredCandidateWindow)> {
    if diag.length <= 0 || diag.n < 0 || !diag.score.is_finite() {
        return None;
    }
    let is_complement = diag.complementarity == crate::simd::fm_ssv::Complementarity::Complement;
    let seed_len = diag.length as usize;
    let fm_pos = diag.n as usize;
    let model_end = (diag.k + diag.length - 1) as usize;
    let pos = fm_get_original_position(record, seq_meta, seed_len, is_complement, fm_pos)?;
    if pos.seg_pos == 0 {
        return None;
    }
    let seq_len = seq_meta.get(pos.segment_id)?.length;
    let seq_seed_start = pos.seg_pos - 1;
    if seq_seed_start >= seq_len {
        return None;
    }
    let window = NhmmerCandidateWindow {
        n: seq_seed_start + 1,
        length: seed_len,
        k: model_end,
        fm_frame: Some(NhmmerCandidateFmFrame {
            fm_n: fm_pos as i64,
            id: pos.segment_id as i32,
            target_len: seq_len,
            fm_start: seq_meta
                .get(pos.segment_id)
                .map(|meta| meta.fm_start as i64)
                .unwrap_or(0),
            fm_bwt_len: record.fm.bwt_len() as i64,
        }),
    };
    Some((
        pos.segment_id,
        NhmmerScoredCandidateWindow {
            window,
            score_bits: diag.score,
        },
    ))
}

/// Faithful Rust counterpart of C `FM_extendSeed()`: compute the candidate
/// extension range, extract it through `fm_convertRange2DSQ()`, and keep the
/// best-scoring sub-diagonal.
pub(super) fn fm_extend_seed(
    diag: &mut crate::simd::fm_ssv::FmDiag,
    record: &NhmmerFmIndexRecord,
    ambiguities: &[(usize, usize)],
    scores: &crate::simd::fm_ssv::SsvScores,
    config: &crate::simd::fm_ssv::FmSsvConfig,
    fm_n: i64,
) -> bool {
    let m = scores.m;
    let kp = scores.kp;
    let extend = (config.ssv_length as i32 - diag.length).max(10);

    let mut model_start = (diag.k - extend + 1).max(1);
    let mut model_end = (diag.k + diag.length + extend - 1).min(m);
    let mut target_start = diag.n - (diag.k as i64 - model_start as i64);
    let mut target_end = diag.n + (model_end as i64 - diag.k as i64);

    if target_start < 0 {
        model_start -= target_start as i32;
        target_start = 0;
    }
    if target_end > fm_n - 2 {
        model_end -= (target_end - (fm_n - 2)) as i32;
        target_end = fm_n - 2;
    }
    if model_start > model_end || target_start > target_end {
        return false;
    }

    let is_complement = diag.complementarity == crate::simd::fm_ssv::Complementarity::Complement;
    let Some(tmp_sq) = fm_convert_range_to_dsq(
        record,
        ambiguities,
        target_start as usize,
        (target_end - target_start + 1) as usize,
        is_complement,
        false,
    ) else {
        return false;
    };

    let mut k = model_start;
    let mut n: usize = 1;
    let mut sc = 0.0_f32;
    let mut hit_start: i64 = n as i64;
    let mut max_sc = 0.0_f32;
    let mut max_hit_start: i64 = n as i64;
    let mut max_hit_end: i64 = n as i64;

    while k <= model_end {
        let c = tmp_sq[n] as usize;
        let cidx = (k as usize) * kp + c;
        sc += scores.scores[cidx];
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
    true
}
