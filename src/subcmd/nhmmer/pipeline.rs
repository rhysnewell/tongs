use super::*;

/// Search one strand of a target sequence using the long-target SSV pipeline.
///
/// Mirrors C `p7_Pipeline_LongTarget` (`hmmer/src/p7_pipeline.c:1763`) end to
/// end: (1) SSV longtarget scan finds candidate windows; (2) windows are
/// extended/merged using per-k prefix/suffix lengths; (3) per-window MSV F1
/// gate; (4) post-SSV bias and Viterbi F2 gates (`p7_pli_postSSV_LongTarget`)
/// produce finer vit sub-windows; (5) long windows (>80 kb) are split with
/// 40 kb overlap for numerical stability; (6) Forward+null2+domain definition
/// runs per window via `Pipeline::run`, producing one `Hit` per emitted
/// domain. Maintains the residue counters (msv/bias/vit/fwd) and the
/// inter-window `fwd_overlap` carry exactly like C. Returned coordinates are
/// in the strand-local (input `sq`) frame; the caller flips them for crick.
#[allow(clippy::too_many_arguments)]
pub(super) fn p7_pipeline_longtarget(
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
) -> Vec<tongs::tophits::Hit> {
    use tongs::simd::ssv_longtarget;

    let _ = bias_windows.b3;

    // Match C p7_Pipeline_LongTarget (p7_pipeline.c:1763): reconfigure MSV
    // length to max_length BEFORE SSV scan. This sets om->tjb_b based on
    // max_length (e.g. 253 for tRNA), not the full sequence length. Without
    // this, Rust's SSV sc_thresh is too high and misses weak peaks that C
    // catches.
    let mut om_ssv = om.clone();
    om_ssv.reconfig_msv_length(max_length);
    let om = &om_ssv;

    // Phase 1: SSV longtarget filter — find candidate windows.
    let mut windows = if let Some(seed_windows) = initial_windows {
        seed_windows
            .iter()
            .map(|w| ssv_longtarget::HmmWindow {
                n: w.n,
                k: w.k,
                length: w.length,
                score: 0.0,
                target_len: w.fm_frame.map_or(sq.n, |frame| frame.target_len),
                complement: is_complement,
                // Multi-segment FM windows carry C's concatenated-FM coordinate
                // (`fm_n >= 0`) so complement windows take C's special
                // extension flip. FASTA and single-segment FM windows keep
                // fm_n=-1, the RC-local/no-flip frame.
                id: w.fm_frame.map_or(0, |frame| frame.id),
                fm_n: w.fm_frame.map_or(-1, |frame| frame.fm_n),
                fm_start: w.fm_frame.map_or(-1, |frame| frame.fm_start),
                fm_bwt_len: w.fm_frame.map_or(0, |frame| frame.fm_bwt_len),
            })
            .collect()
    } else {
        unsafe { ssv_longtarget::p7_ssv_filter_longtarget(&sq.dsq, sq.n, om, bg, f1, max_length) }
    };

    // C p7_Pipeline_LongTarget has NO do_max special-case here: --max sets
    // F1=0.3, F2=F3=1.0 (p7_pipeline.c:355-356) and runs the identical SSV
    // windowing + MSV/Vit/Fwd pipeline. The F2/F3=1.0 gates pass everything,
    // but the F1=0.3 SSV gate still segments windows, and the per-stage
    // counters are still credited. (The previous Rust code collapsed --max to a
    // single full-sequence window and skipped the stages, which both lost C's
    // multi-window hits and left pos_past_msv/bias/vit at 0.)
    if windows.is_empty() {
        return Vec::new();
    }

    // Extend and merge overlapping windows. C uses pct_overlap = 0 for the
    // msv (SSV) stage (any overlap merges) and 0.5 for the vit stage
    // (hmmer/src/p7_pipeline.c:1794 and 1614). We also use per-k
    // prefix_lengths / suffix_lengths from p7_hmm_ScoreDataComputeRest to
    // compute window-specific extension sizes.
    let ml = if max_length > 0 {
        max_length as usize
    } else {
        hmm.m * 4
    };
    let (prefix_lens, suffix_lens) = ssv_longtarget::compute_prefix_suffix_lengths_from_om(om);
    {
        ssv_longtarget::p7_pli_extend_and_merge_windows(
            &mut windows,
            ml,
            sq.n,
            0.0,
            &prefix_lens,
            &suffix_lens,
        );
        if let Some(ctx) = fm_context {
            let mut i = 0;
            while i < windows.len() {
                if windows[i].fm_n >= 0 {
                    let mut keep = true;
                    let mut again = true;
                    while again {
                        again = false;
                        let Some(pos) = fm_get_original_position(
                            ctx.record,
                            ctx.seq_meta,
                            windows[i].length,
                            windows[i].complement,
                            windows[i].fm_n as usize,
                        ) else {
                            keep = false;
                            break;
                        };
                        let Some(segment) = ctx.seq_meta.get(pos.segment_id) else {
                            keep = false;
                            break;
                        };

                        windows[i].id = pos.segment_id as i32;
                        windows[i].target_len = segment.length;
                        windows[i].n = pos.seg_pos;

                        if pos.crosses_segment {
                            let overext = (pos.seg_pos as i64 + windows[i].length as i64)
                                - (segment.target_start as i64 + segment.length as i64 - 1);
                            let use_length = windows[i].length as i64 - overext + 1;
                            if overext <= 0 || use_length <= 0 {
                                keep = false;
                                break;
                            }
                            let overext = overext as usize;
                            let use_length = use_length as usize;
                            if use_length >= 8 && windows[i].length >= 8 {
                                let mut split = windows[i].clone();
                                split.id = if windows[i].complement {
                                    (pos.segment_id as i32) - 1
                                } else {
                                    (pos.segment_id as i32) + 1
                                };
                                split.k = windows[i].k + use_length - 1;
                                split.length = use_length;
                                windows.push(split);
                                windows[i].k += use_length;
                                windows[i].length = overext;
                                again = true;
                            } else if windows[i].length >= 8 {
                                windows[i].k += use_length;
                                windows[i].length = overext;
                            } else {
                                windows[i].length = use_length;
                            }
                        }
                    }
                    if !keep || windows[i].id as usize != ctx.seq_idx {
                        windows.remove(i);
                        continue;
                    }
                }
                i += 1;
            }
        } else {
            for win in &mut windows {
                if win.fm_n >= 0 && win.fm_start >= 0 {
                    let fm_pos = if win.complement {
                        win.fm_bwt_len - win.fm_n - 1
                    } else {
                        win.fm_n
                    };
                    let seq_pos = fm_pos - win.fm_start + 1;
                    let seg_pos = if win.complement {
                        win.target_len as i64 - seq_pos + 1
                    } else {
                        seq_pos
                    };
                    if seg_pos > 0 {
                        win.n = seg_pos as usize;
                    }
                    win.fm_n = -1;
                }
            }
        }
    }

    // Phase 1a: Per-window standard MSV filter (F1 gate). Mirrors C
    // p7_pipeline.c:1862-1866 which calls p7_oprofile_ReconfigMSVLength(om,
    // window->length), p7_MSVFilter, then `if (P > pli->F1) continue;`. This
    // rejects SSV peaks that don't also pass a full MSV p-value threshold.
    // Without this, Rust finds SSV peaks that C rejects at this stage.
    let f1_thresh = f1;
    {
        let mut msv_filtered: Vec<ssv_longtarget::HmmWindow> = Vec::new();
        for win in &windows {
            let (win_len, sub_dsq) = if let Some(ctx) = fm_context.filter(|_| win.fm_n >= 0) {
                let Some(sub_dsq) = fm_convert_range_to_dsq(
                    ctx.record,
                    ctx.ambiguities,
                    win.fm_n as usize,
                    win.length,
                    win.complement,
                    true,
                ) else {
                    continue;
                };
                (win.length, sub_dsq)
            } else {
                let win_start = win.n;
                let win_end = (win.n + win.length - 1).min(sq.n);
                if win_end < win_start || (win_end - win_start + 1) < hmm.m {
                    continue;
                }
                let win_len = win_end - win_start + 1;
                let mut sub_dsq = vec![DSQ_SENTINEL];
                sub_dsq.extend_from_slice(&sq.dsq[win_start..=win_end]);
                sub_dsq.push(DSQ_SENTINEL);
                (win_len, sub_dsq)
            };
            if win_len < hmm.m {
                continue;
            }

            // Reconfig MSV length for this window, compute null, run MSVFilter,
            // check p-value against F1.
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
            if msv_pval > f1_thresh {
                continue;
            }
            // Mirror C p7_pipeline.c:1867 — accumulate residues after MSV F1 gate.
            use std::sync::atomic::Ordering;
            msv_counter.fetch_add(win_len as u64, Ordering::Relaxed);
            // Store the MSV score (usc) on the window so postSSV's F1-bias gate
            // can use it (C passes `usc` to p7_pli_postSSV_LongTarget).
            let mut filtered = win.clone();
            filtered.score = usc;
            msv_filtered.push(filtered);
        }
        windows = msv_filtered;
    }

    // Phase 1b: Vit_longtarget per merged SSV window to produce finer-grained
    // vit sub-windows. Mirrors C p7_pli_postSSV_LongTarget → ViterbiFilter_longtarget.
    // For each SSV window, we run Vit_longtarget on its subseq to find positions
    // where the E-state Vit score exceeds the F2 threshold; those positions get
    // re-extended and merged before going to Forward.
    let f2_p_thresh = f2;
    let max_length_usize = if max_length > 0 {
        max_length as usize
    } else {
        hmm.max_length.max((hmm.m * 4) as i32) as usize
    };
    let mut vit_windows: Vec<ssv_longtarget::HmmWindow> = Vec::new();
    {
        for msv in &windows {
            let msv_start = msv.n;
            let (subseq_len, vit_sub_dsq) = if let Some(ctx) = fm_context.filter(|_| msv.fm_n >= 0)
            {
                let Some(vit_sub_dsq) = fm_convert_range_to_dsq(
                    ctx.record,
                    ctx.ambiguities,
                    msv.fm_n as usize,
                    msv.length,
                    msv.complement,
                    true,
                ) else {
                    continue;
                };
                (msv.length, vit_sub_dsq)
            } else {
                let msv_end = (msv.n + msv.length - 1).min(sq.n);
                if msv_end < msv_start || (msv_end - msv_start + 1) < hmm.m {
                    continue;
                }
                let subseq_len = msv_end - msv_start + 1;
                let mut vit_sub_dsq = vec![DSQ_SENTINEL];
                vit_sub_dsq.extend_from_slice(&sq.dsq[msv_start..=msv_end]);
                vit_sub_dsq.push(DSQ_SENTINEL);
                (subseq_len, vit_sub_dsq)
            };
            if subseq_len < hmm.m {
                continue;
            }

            // Match C p7_pli_postSSV_LongTarget exactly (p7_pipeline.c:1580-1612):
            //   (a) bias filter on full window_len, with F1_L=min(100,win_len) scaling
            //   (b) compute loc_window_len = min(win_len, max_length), recompute nullsc
            //   (c) F2 filtersc = nullsc + bias_delta * F2_L/win_len (F2_L=min(240,win_len))
            //   (d) reconfig profile to loc_window_len
            //   (e) run Vit_longtarget with this filtersc
            // nhmmer-specific defaults from hmmer/src/nhmmer.c:160-162.
            // The generic p7_pipeline defaults are 100/240/1000 (p7_pipeline.c:341-344),
            // but nhmmer overrides `--B1` to 110.
            let f1_l = subseq_len.min(bias_windows.b1);
            let f2_l = subseq_len.min(bias_windows.b2);
            let loc_window_len = subseq_len.min(max_length_usize);

            // Compute bias_delta once (based on full win_len): bias_delta = filter_score - nullsc_win.
            let mut bg_win = bg.clone();
            bg_win.set_length(subseq_len);
            let nullsc_full = bg_win.null_one(subseq_len);
            let bias_delta = if !nobias {
                bg_win.filter_score(&vit_sub_dsq, subseq_len) - nullsc_full
            } else {
                0.0
            };

            // Mirror C p7_pipeline.c:1580-1587 bias F1 gate. Uses the MSV score
            // (usc) stored on the window above. If bias-corrected MSV p-value
            // exceeds F1 = 0.02, skip this SSV window.
            if !nobias {
                let f1_scale = if f1_l == subseq_len {
                    1.0_f32
                } else {
                    f1_l as f32 / subseq_len as f32
                };
                let filtersc_f1 = nullsc_full + bias_delta * f1_scale;
                let usc = msv.score;
                let bias_pval = tongs::pipeline::msv_pvalue(usc, filtersc_f1, &hmm.evparam);
                if bias_pval > f1_thresh {
                    continue;
                }
            }
            // After bias F1 gate: count these residues toward pos_past_bias.
            use std::sync::atomic::Ordering;
            bias_counter.fetch_add(subseq_len as u64, Ordering::Relaxed);

            // Recompute nullsc at loc_window_len (may be shorter than win_len).
            let mut bg_loc = bg.clone();
            bg_loc.set_length(loc_window_len);
            let nullsc_win = bg_loc.null_one(loc_window_len);

            // F2 filtersc: scale bias_delta by F2_L/win_len, add nullsc_loc.
            let f2_scale = if f2_l == subseq_len {
                1.0_f32
            } else {
                f2_l as f32 / subseq_len as f32
            };
            let filtersc = nullsc_win + bias_delta * f2_scale;

            // Reconfig profile to loc_window_len (matches C p7_pipeline.c:1606).
            let mut vit_om = om.clone();
            vit_om.reconfig_length(loc_window_len as i32);
            let mut sub_windows = unsafe {
                tongs::simd::vit_filter::p7_viterbi_filter_longtarget(
                    &vit_sub_dsq,
                    subseq_len,
                    &vit_om,
                    filtersc,
                    f2_p_thresh,
                )
            };
            // Extend and merge Vit windows WITHIN the subseq (target_len =
            // subseq_len), matching C p7_pipeline.c:1614 which does
            // `p7_pli_ExtendAndMergeWindows(om, data, vit_windowlist, 0.5)` with
            // vit_windowlist entries having target_len=window_len (subseq_len).
            // This bounds extension to subseq bounds; without it, extension
            // could reach beyond the SSV window.
            ssv_longtarget::p7_pli_extend_and_merge_windows(
                &mut sub_windows,
                ml,
                subseq_len,
                0.5,
                &prefix_lens,
                &suffix_lens,
            );
            // Mirror C p7_pipeline.c:1637-1641: pos_past_vit += length, subtract
            // overlap with previous vit window. C does NOT clamp the net
            // contribution to 0 — if overlap > length, pos_past_vit can receive
            // a net-negative delta. Rust mirrors this via wrapping u64 add so
            // parallel aggregation still gives the same final value.
            let mut prev_end: Option<usize> = None;
            for w in sub_windows {
                let abs_n = msv_start + w.n - 1;
                let fm_n = if msv.fm_n >= 0 {
                    msv.fm_n + w.n as i64 - 1
                } else {
                    -1
                };
                let mut add = w.length as i64;
                if let Some(pe) = prev_end {
                    if pe > abs_n {
                        add -= (pe - abs_n) as i64;
                    }
                }
                vit_counter.fetch_add(add as u64, Ordering::Relaxed);
                prev_end = Some(abs_n + w.length);
                vit_windows.push(ssv_longtarget::HmmWindow {
                    n: abs_n,
                    k: w.k,
                    length: w.length,
                    score: w.score,
                    target_len: sq.n,
                    complement: is_complement,
                    id: msv.id,
                    fm_n,
                    fm_start: msv.fm_start,
                    fm_bwt_len: msv.fm_bwt_len,
                });
            }
        }

        // Vit windows are already extended+merged within each SSV subseq above
        // (matching C p7_pipeline.c:1614). Just aggregate them.
        windows = vit_windows;
        if windows.is_empty() {
            return Vec::new();
        }
    }

    // Window splitting for numerical stability in Forward: mirror C
    // p7_pli_postSSV_LongTarget (p7_pipeline.c:1620-1634). If a merged window
    // is longer than 80000 residues, split it into overlapping sub-windows of
    // length at most 80000 with overlap = min(40000, max_length).
    // LOW-1: split via the faithful do/while port (see
    // ssv_longtarget::split_long_windows; mirrors p7_pipeline.c:1620-1634).
    const MAX_WINDOW_LEN: usize = 80000;
    let overlap_len = 40000_usize.min(ml);
    windows = ssv_longtarget::split_long_windows(&windows, MAX_WINDOW_LEN, overlap_len);

    // Phase 2: Run full pipeline on each window
    let mut all_hits = Vec::new();
    // Track fwd overlap carried from previous F3-passing window, matching
    // C p7_pipeline.c:1335-1337 + 1649-1654.
    let mut fwd_overlap: usize = 0;

    for (win_idx, win) in windows.iter().enumerate() {
        let win_start = win.n;
        let win_end = (win.n + win.length - 1).min(sq.n);
        let win_len = win_end - win_start + 1;
        if win_len < hmm.m {
            fwd_overlap = 0;
            continue;
        }

        // Create window sub-sequence
        let win_dsq = if let Some(ctx) = fm_context.filter(|_| win.fm_n >= 0) {
            let Some(win_dsq) = fm_convert_range_to_dsq(
                ctx.record,
                ctx.ambiguities,
                win.fm_n as usize,
                win_len,
                win.complement,
                true,
            ) else {
                fwd_overlap = 0;
                continue;
            };
            win_dsq
        } else {
            let mut win_dsq = vec![DSQ_SENTINEL];
            win_dsq.extend_from_slice(&sq.dsq[win_start..=win_end]);
            win_dsq.push(DSQ_SENTINEL);
            win_dsq
        };

        let win_sq = Sequence {
            name: sq.name.clone(),
            acc: sq.acc.clone(),
            desc: sq.desc.clone(),
            dsq: win_dsq,
            n: win_len,
            l: win_len,
            taxid: -1,
        };

        let mut lgm = gm.clone();
        let mut lom = om.clone();
        // Match C postViterbi_LongTarget (p7_pipeline.c:1316): SetLength(bg,
        // window_len) before NullOne. Without this, Pipeline::run's call to
        // `bg.null_one(l)` uses whatever `p1` the outer bg was initialized
        // with (default 350/351), instead of `window_len/(window_len+1)`.
        let mut lb = bg.clone();
        lb.set_length(win_len);
        let mut lpli = Pipeline::new();
        lpli.new_model(&lgm);
        lpli.long_target = true;
        threshold_config.apply_to(&mut lpli);
        // Propagate top-level F3 threshold (nhmmer default 3e-5, not pipeline
        // default 1e-5). Matches C nhmmer.c:114.
        lpli.f3 = f3;
        lpli.do_max = do_max;
        lpli.seed = seed;
        // Propagate --nobias/--nonull2 so the long-target bias scaling and
        // null2 correction inside Pipeline::run honor the user's flags.
        // Without this, Pipeline::new()'s defaults (do_biasfilter=true,
        // do_null2=true) override the user's intent.
        lpli.do_biasfilter = !nobias;
        lpli.do_null2 = !nonull2;
        lpli.long_target_b3 = bias_windows.b3;
        // SSV already ran; still apply Viterbi (F2) and Forward (F3) filters
        // per window so we don't emit one Hit for every SSV candidate.

        let mut lth = TopHits::new();
        let ran = lpli.run(&mut lgm, &mut lom, &lb, hmm, &win_sq, &mut lth);
        // msv/bias/vit counters already updated upstream. fwd_counter
        // += win_len - fwd_overlap (C p7_pipeline.c:1335). On F3 pass, carry
        // overlap with NEXT window; on F3 fail, reset fwd_overlap to 0.
        use std::sync::atomic::Ordering;
        if lpli.n_past_fwd > 0 {
            // C p7_pipeline.c:1335 `pli->pos_past_fwd += window_len - *overlap;`
            // does not clamp to zero. Use wrapping u64 add to allow net-negative
            // contributions, matching C's counter arithmetic exactly.
            let add = (win_len as i64) - (fwd_overlap as i64);
            fwd_counter.fetch_add(add as u64, Ordering::Relaxed);
            // Compute overlap with next window if any.
            if win_idx + 1 < windows.len() {
                let next = &windows[win_idx + 1];
                let this_end = win.n + win.length;
                fwd_overlap = this_end.saturating_sub(next.n);
            } else {
                fwd_overlap = 0;
            }
        } else {
            fwd_overlap = 0;
        }
        if ran {
            // nhmmer (long_target) creates one Hit per domain (C postViterbi
            // line 1433: loop `for (d = 0; d < ddef->ndom; d++)` creates one
            // hit per domain). Our Pipeline::run emits one hit per window
            // with all domains stuffed into hit.dcl, so we split here.
            for src_hit in lth.hits {
                for src_dom in &src_hit.dcl {
                    let mut dom = src_dom.clone();
                    // Adjust coordinates to global sequence position.
                    dom.iali += (win_start - 1) as i64;
                    dom.jali += (win_start - 1) as i64;
                    dom.ienv += (win_start - 1) as i64;
                    dom.jenv += (win_start - 1) as i64;
                    if let Some(ref mut ad) = dom.ad {
                        ad.sqfrom += win_start - 1;
                        ad.sqto += win_start - 1;
                    }
                    all_hits.push(longtarget_domain_hit(sq, &src_hit, dom, threshold_config));
                }
            }
        }
    }

    all_hits
}
