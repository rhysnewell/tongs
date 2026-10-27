use super::*;

/// Augment the FM candidate windows with the exact two-sweep SSV-over-FM kernel
/// (`src/simd/fm_ssv.rs`), a faithful port of C `p7_SSVFM_longlarget` /
/// `FM_Recurse`. The seed-then-rescore loops in `fm_windows` only find *exact* model
/// k-mers in the FM index; this walks the FM index accumulating SSV score over
/// the *actual* target characters, so it recovers weak diagonals with no exact
/// high-scoring k-mer match — the historical sensitivity gap vs C.
///
/// All score-valued inputs are in NATS (`ln(p/q)`; bit thresholds * ln2),
/// mirroring C's `ssv_scores_f` / `fm_cfg` units. The kernel only emits seed
/// *coordinates*; they are turned into windows by the existing validated
/// `fm_push_seed_window` (which re-scores in bits and whose downstream MSV/Vit/
/// Fwd gates filter false positives), so the unit choice stays isolated to the
/// kernel.
///
/// `is_complement` selects the strand. The caller passes `fmf` = the block's
/// kind=0 (reversed-text) index and `fmb` = its kind=1 (forward-text) index for
/// either strand, and `record` = the strand's own record (kind=0 for Watson,
/// kind=1 for Crick) used for coordinate mapping. The Crick coordinate matches
/// C's complement `fm_getOriginalPosition` conversion (validated against
/// bundled C).
///
/// To mirror C `p7_SSVFM_longlarget`, the extended diagonals are filtered by the
/// Gumbel-derived `sc_thresh` (computed from `max_length`, `f1`, and the model's
/// MSV evalue parameters) before becoming windows — without it, the exhaustive
/// kernel's extra raw seeds would survive the lenient seed-then-rescore `>0`
/// window gate and produce hits C never reports.
///
/// Returns `true` when the C-faithful `FM_extendSeed` extension path was taken.
/// In that mode the augment alone reproduces C `p7_SSVFM_longlarget`'s window
/// list, so the caller drops the Rust-only seed sources and skips the seed
/// pre-merge / window gate (`fm_finalize_seed_windows`), matching C's residue
/// counters. Returns `false` for the fallback (seed-then-rescore) path, where
/// those Rust-specific guards stay in place.
#[allow(clippy::too_many_arguments)]
pub(super) fn fm_ssv_collect_block_windows(
    fmf: &FmIndex,
    fmb: &FmIndex,
    record: &NhmmerFmIndexRecord,
    ambiguities: &[(usize, usize)],
    seq_meta: &[NhmmerFmSequenceMeta],
    hmm: &Hmm,
    bg: &Bg,
    config: NhmmerFmSeedConfig,
    is_complement: bool,
    max_length: i32,
    f1: f64,
) -> Option<Vec<Vec<NhmmerScoredCandidateWindow>>> {
    use crate::simd::fm_ssv;
    let m = hmm.m;
    if m == 0 || !record.has_text() {
        return None;
    }
    let kp = 4usize;
    let ln2 = std::f32::consts::LN_2;

    let mut scores = fm_ssv::SsvScores::new(m as i32, kp);
    let mut best_sc_sum = 0.0f64;
    for k in 1..=m {
        let mut best = 0.0f32;
        for x in 0..kp {
            let p = hmm
                .mat
                .get(k)
                .and_then(|r| r.get(x))
                .copied()
                .unwrap_or(0.0);
            let q = bg.f.get(x).copied().unwrap_or(0.0);
            let v = if p > 0.0 && q > 0.0 {
                ((p as f64) / (q as f64)).ln() as f32
            } else {
                -1.0e9
            };
            scores.set(k as i32, x, v);
            if v > best {
                best = v;
            }
        }
        best_sc_sum += best as f64;
    }

    let cons_bytes = nhmmer_consensus_bytes(hmm);
    let mut consensus = vec![0usize; m + 1];
    for (i, &b) in cons_bytes.iter().enumerate().take(m) {
        consensus[i + 1] = match b {
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => 0,
        };
    }

    let best_sc_avg = (best_sc_sum / (m as f64).sqrt()).max(5.0);
    let ratio = (best_sc_avg / 7.0).min(1.0) as f32;
    let sc_thresh_fm = config.score_threshold_bits * ratio * ln2;
    let fm_config = fm_ssv::FmSsvConfig {
        max_depth: config.max_depth,
        drop_max_len: config.drop_max_len,
        drop_lim: config.drop_lim_bits * ln2,
        consec_pos_req: config.consec_pos_req,
        consensus_match_req: config.consensus_match_req,
        score_density_req: config.score_density_bits * ln2,
        ssv_length: config.ssv_length.max(1),
    };

    let sc_thresh_bits = {
        let ml = max_length.max(1) as f32;
        let tloop_total = c_logf_to_f32(ml / (ml + 3.0)) * ml;
        let tmove = c_logf_to_f32(3.0 / (ml + 3.0));
        let tbmk = c_logf_to_f32(2.0 / (m as f32 * (m as f32 + 1.0)));
        let tec = c_logf_to_f32(1.0f32 / 2.0);
        let mut bg_ml = bg.clone();
        bg_ml.set_length(ml as usize);
        let nullsc = bg_ml.null_one(ml as usize);
        let mmu = hmm.evparam[p7hmm::P7_MMU] as f64;
        let mlambda = hmm.evparam[p7hmm::P7_MLAMBDA] as f64;
        let inv_p = tongs::stats::gumbel::invsurv(f1, mmu, mlambda) as f32;
        let sc_thresh_nats = inv_p * ln2 + nullsc - (tmove + tloop_total + tmove + tbmk + tec);
        sc_thresh_nats / ln2
    };

    let want = if is_complement {
        fm_ssv::Complementarity::Complement
    } else {
        fm_ssv::Complementarity::NoComplement
    };
    let alph = fm_ssv::FmAlphabet::dna();
    let diags = fm_ssv::fm_get_seeds(
        fmf,
        fmb,
        &alph,
        &scores,
        &consensus,
        sc_thresh_fm,
        &fm_config,
        !is_complement,
        is_complement,
    );

    let mut out: Vec<Vec<NhmmerScoredCandidateWindow>> =
        (0..seq_meta.len()).map(|_| Vec::new()).collect();
    let fm_n = record.fm.bwt_len() as i64;
    for mut diag in diags {
        if diag.complementarity != want || diag.length <= 0 || diag.k < 1 {
            continue;
        }
        if !fm_extend_seed(&mut diag, record, ambiguities, &scores, &fm_config, fm_n) {
            continue;
        }
        if diag.length <= 0 || diag.score < sc_thresh_bits * ln2 {
            continue;
        }
        let model_end = (diag.k + diag.length - 1) as usize;
        if model_end == 0 || model_end > hmm.m || diag.n < 0 {
            continue;
        }
        if let Some((seq_idx, window)) = fm_window_from_diag_any_segment(record, seq_meta, &diag) {
            if let Some(windows) = out.get_mut(seq_idx) {
                windows.push(window);
            }
        }
    }
    Some(out)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fm_ssv_augment_windows(
    fmf: &FmIndex,
    fmb: &FmIndex,
    record: &NhmmerFmIndexRecord,
    ambiguities: &[(usize, usize)],
    seq_meta: &[NhmmerFmSequenceMeta],
    seq_idx: usize,
    seq: &Sequence,
    hmm: &Hmm,
    bg: &Bg,
    config: NhmmerFmSeedConfig,
    is_complement: bool,
    max_length: i32,
    f1: f64,
    desired_len: usize,
    windows: &mut Vec<NhmmerScoredCandidateWindow>,
) -> bool {
    use crate::simd::fm_ssv;
    let m = hmm.m;
    if m == 0 {
        return false;
    }
    let Some(current_seq_meta) = seq_meta.get(seq_idx) else {
        return false;
    };
    let kp = 4usize; // ACGT
    let ln2 = std::f32::consts::LN_2;

    // ssv_scores_f in NATS: ln(mat[k][x] / bg[x]), per model node k and base x.
    let mut scores = fm_ssv::SsvScores::new(m as i32, kp);
    let mut best_sc_sum = 0.0f64;
    for k in 1..=m {
        let mut best = 0.0f32;
        for x in 0..kp {
            let p = hmm
                .mat
                .get(k)
                .and_then(|r| r.get(x))
                .copied()
                .unwrap_or(0.0);
            let q = bg.f.get(x).copied().unwrap_or(0.0);
            let v = if p > 0.0 && q > 0.0 {
                ((p as f64) / (q as f64)).ln() as f32
            } else {
                -1.0e9
            };
            scores.set(k as i32, x, v);
            if v > best {
                best = v;
            }
        }
        best_sc_sum += best as f64;
    }

    // Consensus codes (0..3) from the model's consensus residues.
    let cons_bytes = nhmmer_consensus_bytes(hmm);
    let mut consensus = vec![0usize; m + 1];
    for (i, &b) in cons_bytes.iter().enumerate().take(m) {
        consensus[i + 1] = match b {
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => 0,
        };
    }

    // sc_thresh_ratio (C nhmmer.c:996-1007): best_sc_avg = sum(max match score)/
    // sqrt(M) in nats, floored at 5; ratio = min(best_sc_avg/7, 1). Then
    // sc_threshFM = user scthreshFM (bits) * ratio, in nats.
    let best_sc_avg = (best_sc_sum / (m as f64).sqrt()).max(5.0);
    let ratio = (best_sc_avg / 7.0).min(1.0) as f32;
    let sc_thresh_fm = config.score_threshold_bits * ratio * ln2;

    let fm_config = fm_ssv::FmSsvConfig {
        max_depth: config.max_depth,
        drop_max_len: config.drop_max_len,
        drop_lim: config.drop_lim_bits * ln2,
        consec_pos_req: config.consec_pos_req,
        consensus_match_req: config.consensus_match_req,
        score_density_req: config.score_density_bits * ln2,
        ssv_length: config.ssv_length.max(1),
    };

    // C `p7_SSVFM_longlarget` window threshold `sc_thresh`: the score an
    // *extended* SSV diagonal must reach to become a window. In nats:
    //   invP = gumbel_invsurv(F1, MMU, MLAMBDA)
    //   sc_thresh = invP*ln2 + nullsc - (tmove + tloop_total + tmove + tbmk + tec)
    // with the length model reconfigured to max_length (nu = 2.0). We compare it
    // against `fm_extend_seed_diagonal`'s bit score, so convert to bits.
    let sc_thresh_bits = {
        let ml = max_length.max(1) as f32;
        let tloop_total = c_logf_to_f32(ml / (ml + 3.0)) * ml;
        let tmove = c_logf_to_f32(3.0 / (ml + 3.0));
        let tbmk = c_logf_to_f32(2.0 / (m as f32 * (m as f32 + 1.0)));
        let tec = c_logf_to_f32(1.0f32 / 2.0); // nu = 2.0
        let mut bg_ml = bg.clone();
        bg_ml.set_length(ml as usize);
        let nullsc = bg_ml.null_one(ml as usize);
        // evparam[0]=P7_MMU, evparam[1]=P7_MLAMBDA.
        let mmu = hmm.evparam[p7hmm::P7_MMU] as f64;
        let mlambda = hmm.evparam[p7hmm::P7_MLAMBDA] as f64;
        let inv_p = tongs::stats::gumbel::invsurv(f1, mmu, mlambda) as f32;
        let sc_thresh_nats = inv_p * ln2 + nullsc - (tmove + tloop_total + tmove + tbmk + tec);
        sc_thresh_nats / ln2
    };

    let want = if is_complement {
        fm_ssv::Complementarity::Complement
    } else {
        fm_ssv::Complementarity::NoComplement
    };
    let alph = fm_ssv::FmAlphabet::dna();
    let mut ssv_windows: Vec<NhmmerScoredCandidateWindow> = Vec::new();
    let diags = fm_ssv::fm_get_seeds(
        fmf,
        fmb,
        &alph,
        &scores,
        &consensus,
        sc_thresh_fm,
        &fm_config,
        /* top_only (Watson / NoComplement) */ !is_complement,
        /* bottom_only (Crick / Complement) */ is_complement,
    );

    let c_faithful = record.has_text();
    let text_len = record.text_len as i64;
    // C's FM_DATA.N is the BWT length, including the terminal symbol. Rust
    // `FmIndex::n` stores only the biological text length, so use the serialized
    // BWT length for faithful FM_extendSeed coordinates.
    let fm_n = record.fm.bwt_len() as i64;
    for mut diag in diags {
        if diag.complementarity != want || diag.length <= 0 {
            continue;
        }
        if diag.k < 1 {
            continue;
        }

        if c_faithful {
            // Faithful C `FM_extendSeed`: extract the diagonal range through
            // `fm_convertRange2DSQ`, extend it with the SSV score table, then
            // window from the extended coordinates.
            if !fm_extend_seed(&mut diag, record, ambiguities, &scores, &fm_config, fm_n) {
                continue;
            }
            // `scores`/`diag.score` are in NATS; `sc_thresh_bits` is in BITS.
            // C compares the extended diagonal's score against `sc_thresh`
            // (nats), so convert the threshold to nats here. Store the window's
            // score in bits to match the seed-then-rescore path.
            if diag.length <= 0 || diag.score < sc_thresh_bits * ln2 {
                continue;
            }
            let model_end = (diag.k + diag.length - 1) as usize;
            if model_end == 0 || model_end > hmm.m {
                continue;
            }
            if diag.n < 0 {
                continue;
            }
            fm_push_extended_diag_window(
                &mut ssv_windows,
                ambiguities,
                seq_meta,
                seq_idx,
                seq,
                record,
                is_complement,
                diag.length as usize,
                model_end,
                diag.n as usize,
                diag.score / ln2,
            );
        } else {
            // Fallback (multi-segment / coordinate frame we don't reconstruct
            // here): the original seed-then-rescore window construction.
            let model_end = (diag.k + diag.length - 1) as usize;
            if model_end == 0 || model_end > hmm.m {
                continue;
            }
            let fm_pos = text_len - diag.n - diag.length as i64;
            if fm_pos < 0 {
                continue;
            }
            let synthetic = NhmmerConsensusSeed {
                bases: vec![0u8; diag.length as usize],
                model_end,
            };
            fm_push_seed_window(
                &mut ssv_windows,
                ambiguities,
                current_seq_meta,
                seq,
                record,
                hmm,
                bg,
                is_complement,
                &synthetic,
                fm_pos as usize,
                desired_len,
            );
        }
    }

    // C filters extended diagonals by `sc_thresh` before windowing. The
    // seed-then-rescore path's lenient `>0` gate (already applied inside
    // `fm_push_seed_window`) is not enough for the exhaustive kernel's extra
    // seeds, so apply the Gumbel-derived bit threshold here.
    ssv_windows.retain(|w| w.score_bits >= sc_thresh_bits);
    windows.extend(ssv_windows);
    c_faithful
}
