use super::*;

pub(super) struct TargetSearch<'a> {
    pub(super) do_watson: bool,
    pub(super) do_crick: bool,
    pub(super) fm_window_cache: &'a Option<Vec<[Option<Vec<NhmmerCandidateWindow>>; 2]>>,
    pub(super) target_db: &'a NhmmerTargetDb,
    pub(super) hmm: &'a Hmm,
    pub(super) gm: &'a Profile,
    pub(super) om: &'a OProfile,
    pub(super) bg: &'a Bg,
    pub(super) abc: &'a Alphabet,
    pub(super) max_length: i32,
    pub(super) f1: f64,
    pub(super) f2: f64,
    pub(super) f3: f64,
    pub(super) do_max: bool,
    pub(super) nobias: bool,
    pub(super) nonull2: bool,
    pub(super) seed: u32,
    pub(super) bias_windows: NhmmerBiasWindowLengths,
    pub(super) threshold_config: NhmmerThresholdConfig,
    pub(super) fm_seed_config: NhmmerFmSeedConfig,
    pub(super) effective_block_length: usize,
    pub(super) msv_counter: &'a AtomicU64,
    pub(super) bias_counter: &'a AtomicU64,
    pub(super) vit_counter: &'a AtomicU64,
    pub(super) fwd_counter: &'a AtomicU64,
}

pub(super) fn search_target(
    search: &TargetSearch,
    seq_idx: usize,
    sq: &Sequence,
) -> Result<Vec<tongs::tophits::Hit>, String> {
    let TargetSearch {
        do_watson,
        do_crick,
        fm_window_cache,
        target_db,
        hmm,
        gm,
        om,
        bg,
        abc,
        max_length,
        f1,
        f2,
        f3,
        do_max,
        nobias,
        nonull2,
        seed,
        bias_windows,
        threshold_config,
        fm_seed_config,
        effective_block_length,
        msv_counter,
        bias_counter,
        vit_counter,
        fwd_counter,
    } = *search;
    let mut hits = Vec::new();

    // Search top strand (watson)
    if do_watson {
        let strand_msv_counter = AtomicU64::new(0);
        let fm_windows = if let Some(cache) = fm_window_cache.as_ref() {
            cache[seq_idx][0].clone()
        } else {
            fm_seed_candidate_windows_with_config(
                target_db,
                seq_idx,
                hmm,
                bg,
                max_length,
                f1,
                false,
                fm_seed_config,
            )
        };
        let initial_windows = if target_db.fm_index_records.is_empty() {
            fm_windows.as_deref()
        } else {
            Some(fm_windows.as_deref().unwrap_or(&[]))
        };
        let fm_context = target_db
            .fm_index_records
            .iter()
            .find(|record| {
                record.kind == 0
                    && seq_idx >= record.seq_offset
                    && seq_idx < record.seq_offset + record.seq_count
            })
            .map(|record| NhmmerSearchFmContext {
                record,
                seq_meta: &target_db.fm_sequence_meta,
                ambiguities: &target_db.fm_ambiguities,
                seq_idx,
            });
        hits.extend(search_sequence(
            sq,
            hmm,
            gm,
            om,
            bg,
            max_length,
            f1,
            f2,
            f3,
            do_max,
            nobias,
            nonull2,
            seed,
            bias_windows,
            threshold_config,
            false,
            &strand_msv_counter,
            bias_counter,
            vit_counter,
            fwd_counter,
            initial_windows,
            fm_context,
        )?);
        let strand_msv_count = strand_msv_counter.load(std::sync::atomic::Ordering::Relaxed);
        let strand_msv_count =
            if target_db.fm_index_records.is_empty() && !do_max && sq.n > effective_block_length {
                c_style_blocked_msv_residue_count(
                    sq,
                    hmm,
                    om,
                    bg,
                    max_length,
                    f1,
                    effective_block_length,
                )
            } else {
                strand_msv_count
            };
        msv_counter.fetch_add(strand_msv_count, std::sync::atomic::Ordering::Relaxed);
    }

    // Search complement strand (crick)
    if do_crick && abc.complement.is_some() {
        let strand_msv_counter = AtomicU64::new(0);
        let rc_dsq = if target_db.fm_index_records.is_empty() {
            let mut rc_dsq = sq.dsq.clone();
            abc.revcomp(&mut rc_dsq, sq.n);
            rc_dsq
        } else {
            vec![DSQ_SENTINEL, DSQ_SENTINEL]
        };
        let rc_sq = Sequence {
            name: sq.name.clone(),
            acc: sq.acc.clone(),
            desc: sq.desc.clone(),
            dsq: rc_dsq,
            n: sq.n,
            l: sq.l,
            taxid: -1,
        };
        let fm_windows = if let Some(cache) = fm_window_cache.as_ref() {
            cache[seq_idx][1].clone()
        } else {
            fm_seed_candidate_windows_with_config(
                target_db,
                seq_idx,
                hmm,
                bg,
                max_length,
                f1,
                true,
                fm_seed_config,
            )
        };
        let initial_windows = if target_db.fm_index_records.is_empty() {
            fm_windows.as_deref()
        } else {
            Some(fm_windows.as_deref().unwrap_or(&[]))
        };
        let fm_context = target_db
            .fm_index_records
            .iter()
            .find(|record| {
                record.kind == 1
                    && seq_idx >= record.seq_offset
                    && seq_idx < record.seq_offset + record.seq_count
            })
            .map(|record| NhmmerSearchFmContext {
                record,
                seq_meta: &target_db.fm_sequence_meta,
                ambiguities: &target_db.fm_ambiguities,
                seq_idx,
            });
        let mut rc_hits = search_sequence(
            &rc_sq,
            hmm,
            gm,
            om,
            bg,
            max_length,
            f1,
            f2,
            f3,
            do_max,
            nobias,
            nonull2,
            seed,
            bias_windows,
            threshold_config,
            true,
            &strand_msv_counter,
            bias_counter,
            vit_counter,
            fwd_counter,
            initial_windows,
            fm_context,
        )?;
        let strand_msv_count = strand_msv_counter.load(std::sync::atomic::Ordering::Relaxed);
        let strand_msv_count =
            if target_db.fm_index_records.is_empty() && !do_max && rc_sq.n > effective_block_length
            {
                c_style_blocked_msv_residue_count(
                    &rc_sq,
                    hmm,
                    om,
                    bg,
                    max_length,
                    f1,
                    effective_block_length,
                )
            } else {
                strand_msv_count
            };
        msv_counter.fetch_add(strand_msv_count, std::sync::atomic::Ordering::Relaxed);
        // Convert complement-strand coordinates back to forward
        // strand. nhmmer convention (C p7_pipeline.c:1449-1456):
        // for minus-strand hits, iali/jali and ienv/jenv are
        // emitted in reverse order (iali > jali) so the `strand`
        // column can be derived as iali < jali ? '+' : '-'.
        //
        // LOW-2 (audit 02-nhmmer-longtarget): C folds the strand
        // flip, `seq_start`, and `window_start` into one expression
        //   iali = seq_start - (window_start + iali) + 2
        // inside p7_pli_postViterbi_LongTarget (p7_pipeline.c:1445).
        // The Rust driver instead reverse-complements the target up
        // front, applies the window->sequence offset inside
        // `p7_pipeline_longtarget` (`+= win_start - 1`), then does the
        // whole-sequence `sq.n - x + 1` flip here. These are
        // algebraically identical to C precisely when
        // `seq_start == sq.start == 1`, which is ALWAYS true for the
        // Rust driver: it reads whole sequences (no blocked
        // `esl_sqio_ReadWindow` streaming — see LOW-3 below), so
        // `seq_start` never differs from 1. Verified bit-identical
        // to C on MADE1/3box and on multi-segment FM DBs, both
        // strands. Adopting C's single expression here would change
        // nothing observable (seq_start is constant 1) while risking
        // the currently-green crick coordinate parity, so the
        // verified-equivalent two-step form is kept deliberately.
        let fm_complement_adjust = i64::from(!target_db.fm_index_records.is_empty());
        for hit in &mut rc_hits {
            for dom in &mut hit.dcl {
                // Both forward coords; ali higher first, lower second.
                let ali_hi = sq.n as i64 - dom.iali + 1 - fm_complement_adjust;
                let ali_lo = sq.n as i64 - dom.jali + 1 - fm_complement_adjust;
                dom.iali = ali_hi;
                dom.jali = ali_lo;
                let env_hi = sq.n as i64 - dom.ienv + 1 - fm_complement_adjust;
                let env_lo = sq.n as i64 - dom.jenv + 1 - fm_complement_adjust;
                dom.ienv = env_hi;
                dom.jenv = env_lo;
                if let Some(ref mut ad) = dom.ad {
                    let ad_hi = sq.n.saturating_sub(ad.sqfrom) + 1 - fm_complement_adjust as usize;
                    let ad_lo = sq.n.saturating_sub(ad.sqto) + 1 - fm_complement_adjust as usize;
                    ad.sqfrom = ad_hi;
                    ad.sqto = ad_lo;
                }
            }
        }
        hits.extend(rc_hits);
    }

    for hit in &mut hits {
        hit.seqidx = seq_idx as i64;
    }
    Ok(hits)
}

/// Port of nhmmer.c:938-940, the "Assign HMM max_length" block.
///
/// `w_length`/`w_beta` are `None` when the user did not pass `--w_length` /
/// `--w_beta`; C represents that as -1, which is what the comparisons below
/// rely on. Note the order: an explicit window length wins, then an explicit
/// beta, and only a model that still carries the -1 sentinel is computed at the
/// default tail mass. A model that already has a window length — from
/// `p7_SingleBuilder()` or from a file's `MAXL` — keeps it.
pub(super) fn nhmmer_max_length(
    hmm: &tongs::hmm::Hmm,
    w_length: Option<i32>,
    w_beta: Option<f64>,
) -> i32 {
    let window_length = w_length.unwrap_or(-1);
    let window_beta = w_beta.unwrap_or(-1.0);

    if window_length > 0 {
        window_length
    } else if window_beta > 0.0 {
        builder::max_length_from_beta(hmm, window_beta)
    } else if hmm.max_length == -1 {
        builder::max_length_from_beta(hmm, DEFAULT_WINDOW_BETA)
    } else {
        hmm.max_length
    }
}

#[derive(Clone, Copy)]
pub(crate) struct NhmmerBiasWindowLengths {
    pub(crate) b1: usize,
    pub(crate) b2: usize,
    pub(crate) b3: usize,
}

impl Default for NhmmerBiasWindowLengths {
    fn default() -> Self {
        Self {
            b1: 110,
            b2: 240,
            b3: 1000,
        }
    }
}

/// Search a single target sequence (one strand) and return its hits.
///
/// Counterpart to one strand-pass inside C `serial_loop`
/// (`hmmer/src/nhmmer.c:1330`), which calls `p7_Pipeline_LongTarget` on every
/// sequence regardless of length. Dispatches to `p7_pipeline_longtarget` when the
/// SSE2 SIMD path is available; fails loudly otherwise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn search_sequence(
    sq: &Sequence,
    hmm: &tongs::hmm::Hmm,
    gm: &Profile,
    om: &OProfile,
    bg: &Bg,
    max_length: i32,
    f1: f64,
    f2: f64,
    f3: f64,
    do_max: bool,
    nobias: bool,
    nonull2: bool,
    seed: u32,
    bias_windows: NhmmerBiasWindowLengths,
    threshold_config: NhmmerThresholdConfig,
    is_complement: bool,
    msv_counter: &std::sync::atomic::AtomicU64,
    bias_counter: &std::sync::atomic::AtomicU64,
    vit_counter: &std::sync::atomic::AtomicU64,
    fwd_counter: &std::sync::atomic::AtomicU64,
    initial_windows: Option<&[NhmmerCandidateWindow]>,
    fm_context: Option<NhmmerSearchFmContext<'_>>,
) -> Result<Vec<tongs::tophits::Hit>, String> {
    // C HMMER's nhmmer calls p7_Pipeline_LongTarget for every sequence regardless of length.
    Ok(p7_pipeline_longtarget(
        sq,
        hmm,
        gm,
        om,
        bg,
        max_length,
        f1,
        f2,
        f3,
        do_max,
        nobias,
        nonull2,
        seed,
        bias_windows,
        threshold_config,
        is_complement,
        msv_counter,
        bias_counter,
        vit_counter,
        fwd_counter,
        initial_windows,
        fm_context,
    ))
}

fn c_style_blocked_msv_residue_count(
    sq: &Sequence,
    hmm: &tongs::hmm::Hmm,
    om: &OProfile,
    bg: &Bg,
    max_length: i32,
    f1: f64,
    block_length: usize,
) -> u64 {
    if sq.n <= block_length {
        return msv_residue_count_for_longtarget_window(sq, hmm, om, bg, max_length, f1);
    }

    let overlap = max_length.max(0) as usize;
    let mut total = 0_u64;
    let mut start = 1_usize;
    loop {
        let end = (start + block_length - 1).min(sq.n);
        let mut block_dsq = Vec::with_capacity(end - start + 3);
        block_dsq.push(DSQ_SENTINEL);
        block_dsq.extend_from_slice(&sq.dsq[start..=end]);
        block_dsq.push(DSQ_SENTINEL);
        let block_sq = Sequence {
            name: sq.name.clone(),
            acc: sq.acc.clone(),
            desc: sq.desc.clone(),
            n: end - start + 1,
            l: end - start + 1,
            dsq: block_dsq,
            taxid: -1,
        };
        total += msv_residue_count_for_longtarget_window(&block_sq, hmm, om, bg, max_length, f1);

        if end == sq.n {
            break;
        }
        start = if overlap == 0 {
            end + 1
        } else {
            end.saturating_sub(overlap).saturating_add(1).max(1)
        };
    }
    total
}

fn msv_residue_count_for_longtarget_window(
    sq: &Sequence,
    hmm: &tongs::hmm::Hmm,
    om: &OProfile,
    bg: &Bg,
    max_length: i32,
    f1: f64,
) -> u64 {
    use tongs::simd::ssv_longtarget;

    let mut om_ssv = om.clone();
    om_ssv.reconfig_msv_length(max_length);
    let om = &om_ssv;

    let mut windows =
        unsafe { ssv_longtarget::p7_ssv_filter_longtarget(&sq.dsq, sq.n, om, bg, f1, max_length) };
    if windows.is_empty() {
        return 0;
    }

    let ml = if max_length > 0 {
        max_length as usize
    } else {
        hmm.m * 4
    };
    let (prefix_lens, suffix_lens) = ssv_longtarget::compute_prefix_suffix_lengths_from_om(om);
    ssv_longtarget::p7_pli_extend_and_merge_windows(
        &mut windows,
        ml,
        sq.n,
        0.0,
        &prefix_lens,
        &suffix_lens,
    );

    let mut total = 0_u64;
    for win in &windows {
        let win_start = win.n;
        let win_end = (win.n + win.length - 1).min(sq.n);
        if win_end < win_start || (win_end - win_start + 1) < hmm.m {
            continue;
        }
        let win_len = win_end - win_start + 1;
        let mut sub_dsq = vec![DSQ_SENTINEL];
        sub_dsq.extend_from_slice(&sq.dsq[win_start..=win_end]);
        sub_dsq.push(DSQ_SENTINEL);

        let mut msv_om = om.clone();
        msv_om.reconfig_msv_length(win_len as i32);
        let mut bg_win = bg.clone();
        bg_win.set_length(win_len);
        let nullsc = bg_win.null_one(win_len);
        let usc_result =
            unsafe { tongs::simd::msv_filter::p7_msv_filter(&sub_dsq, win_len, &msv_om) };
        let usc = match usc_result {
            tongs::simd::msv_filter::MsvResult::Ok(sc) => sc,
            tongs::simd::msv_filter::MsvResult::Overflow => f32::INFINITY,
        };
        let msv_pval = tongs::pipeline::msv_pvalue(usc, nullsc, &hmm.evparam);
        if msv_pval <= f1 {
            total += win_len as u64;
        }
    }
    total
}
