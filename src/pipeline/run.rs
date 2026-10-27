use crate::bg::Bg;
#[cfg(not(striped))]
use crate::dp::gmx::Gmx;
use crate::profile::*;
use crate::sequence::Sequence;
#[cfg(striped)]
use crate::simd::msv_filter::MsvResult;
use crate::simd::oprofile::OProfile;
#[cfg(target_arch = "x86_64")]
use crate::simd::vit_filter::p7_viterbi_filter;
#[cfg(striped)]
use crate::simd::vit_filter::VitResult;
use crate::stats;
use crate::tophits::*;
use crate::util::cmath::{c_exp_f64, c_log_f64};

use crate::domaindef::{Definition, Model, Settings, Target};

use super::{
    forward_pvalue, msv_pvalue, nats_to_bits_from_scores, reconstruction_score_nats,
    viterbi_pvalue, BitCutoff, Pipeline,
};

impl Pipeline {
    pub fn run(
        &mut self,
        gm: &mut Profile,
        om: &mut OProfile,
        bg: &Bg,
        hmm: &crate::hmm::Hmm,
        sq: &Sequence,
        th: &mut TopHits,
    ) -> bool {
        if sq.n == 0 {
            return false;
        }
        reconfig_length(gm, sq.n as i32);
        om.reconfig_length(sq.n as i32);
        let null_sc = bg.null_one(sq.n);
        self.run_configured(gm, om, bg, hmm, sq, null_sc, th)
    }

    // The caller has already set every length-dependent term for this sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn run_configured(
        &mut self,
        gm: &Profile,
        om: &OProfile,
        bg: &Bg,
        hmm: &crate::hmm::Hmm,
        sq: &Sequence,
        null_sc: f32,
        th: &mut TopHits,
    ) -> bool {
        let Some(filtersc) = self.filters(gm, om, bg, sq, null_sc) else {
            return false;
        };
        let fwd_sc = self.forward(gm, om, sq);
        self.after_forward(gm, om, bg, hmm, sq, null_sc, filtersc, fwd_sc, th)
    }

    pub(crate) fn filters(
        &mut self,
        gm: &Profile,
        om: &OProfile,
        bg: &Bg,
        sq: &Sequence,
        null_sc: f32,
    ) -> Option<f32> {
        #[cfg(striped)]
        let _ = gm;
        let l = sq.n;
        if l == 0 {
            return None;
        }
        self.n_targets += 1;

        // C HMMER runs the bias filter only after MSV passes; keep it lazy.
        let mut filtersc = null_sc;

        // For long_target, SSV_longtarget and ViterbiFilter_longtarget already
        // ran at the postSSV stage (in nhmmer::p7_pipeline_longtarget); C's
        // postViterbi_LongTarget (p7_pipeline.c:1289) does NOT re-run MSV or
        // Viterbi — it calls p7_ForwardParser directly. Match that here.
        if !self.do_max && !self.long_target {
            let mut filter_p = 0.0_f64;

            // Stage 1: MSV filter
            let usc: f32;
            #[cfg(striped)]
            {
                let msv_result = unsafe {
                    crate::simd::msv_filter::p7_msv_filter_with_scratch_rust_helper(
                        &sq.dsq,
                        l,
                        om,
                        &mut self.msv_dp,
                    )
                };
                usc = match msv_result {
                    MsvResult::Ok(sc) => sc,
                    MsvResult::Overflow => f32::INFINITY,
                };
            }
            #[cfg(not(striped))]
            {
                let mut gx = Gmx::new(gm.m, l);
                usc = crate::dp::generic_msv::g_msv(&sq.dsq, l, gm, &mut gx, 2.0);
            }
            if usc != f32::INFINITY {
                let msv_pval = msv_pvalue(usc, null_sc, &self.evparam);
                if msv_pval > self.f1 {
                    return None;
                }
                filter_p = msv_pval;
            }
            self.n_past_msv += 1;

            // Stage 1b: Bias composition filter
            if self.do_biasfilter {
                filtersc = bg.filter_score(&sq.dsq, l);
                let bias_pval = msv_pvalue(usc, filtersc, &self.evparam);
                if bias_pval > self.f1 {
                    return None;
                }
                filter_p = bias_pval;
            }
            self.n_past_bias += 1;

            // Stage 2: Viterbi filter (uses filtersc as baseline, matching C)
            if filter_p > self.f2 {
                let vit_sc: f32;
                #[cfg(striped)]
                {
                    #[cfg(target_arch = "x86_64")]
                    let vit_result = match om.vit.as_ref() {
                        Some(vp) => unsafe {
                            crate::simd::vit_avx2::viterbi_filter(
                                &sq.dsq,
                                l,
                                om,
                                vp,
                                &mut self.vit_dp,
                            )
                        },
                        None => unsafe { p7_viterbi_filter(&sq.dsq, l, om) },
                    };
                    #[cfg(not(target_arch = "x86_64"))]
                    let vit_result = unsafe {
                        crate::simd::vit_filter::p7_viterbi_filter_with_scratch(
                            &sq.dsq,
                            l,
                            om,
                            &mut self.vit_dp,
                        )
                    };
                    vit_sc = match vit_result {
                        VitResult::Ok(sc) => sc,
                        VitResult::Overflow => f32::INFINITY,
                    };
                }
                #[cfg(not(striped))]
                {
                    let mut gx = Gmx::new(gm.m, l);
                    vit_sc = crate::dp::generic_viterbi::g_viterbi(&sq.dsq, l, gm, &mut gx);
                }

                if vit_sc != f32::INFINITY {
                    let vit_pval = viterbi_pvalue(vit_sc, filtersc, &self.evparam);
                    if vit_pval > self.f2 {
                        return None;
                    }
                }
            }
            self.n_past_vit += 1;
        } else {
            self.n_past_msv += 1;
            self.n_past_bias += 1;
            self.n_past_vit += 1;
        }

        // Long-target bias scaling for Forward stage: match C
        // p7_pli_postViterbi_LongTarget (p7_pipeline.c:1330):
        //   F3_L = min(window_len, B3=1000);
        //   filtersc = nullsc + bias_filtersc * (F3_L == window_len ? 1.0 : F3_L/window_len).
        // This caps the bias contribution at B3=1000 residues, making the F3
        // filter more permissive on long windows.
        if self.long_target && self.do_biasfilter {
            let bias_full = bg.filter_score(&sq.dsq, l);
            let bias_delta = bias_full - null_sc;
            let b3 = self.long_target_b3;
            let f3_l = l.min(b3);
            let scale = if f3_l == l {
                1.0_f32
            } else {
                f3_l as f32 / l as f32
            };
            filtersc = null_sc + bias_delta * scale;
        }
        Some(filtersc)
    }

    pub(crate) fn forward(&mut self, gm: &Profile, om: &OProfile, sq: &Sequence) -> f32 {
        #[cfg(striped)]
        {
            let _ = gm;
            self.fwd_pmx.resize_parser(om.m, sq.n);
            unsafe {
                crate::simd::fwd_filter::forward_parser_pmx(&sq.dsq, sq.n, om, &mut self.fwd_pmx)
            }
        }
        #[cfg(not(striped))]
        {
            let mut gx = Gmx::new(gm.m, sq.n);
            crate::dp::generic_fwdback::g_forward(&sq.dsq, sq.n, gm, &mut gx)
        }
    }

    pub(crate) fn passes_forward(&self, fwd_sc: f32, filtersc: f32) -> bool {
        let pval = forward_pvalue(fwd_sc, filtersc, &self.evparam);
        self.do_max || pval <= self.f3 || pval.is_nan()
    }

    pub(crate) fn survives_forward(&mut self, fwd_sc: f32, filtersc: f32) -> bool {
        let passes = self.passes_forward(fwd_sc, filtersc);
        self.n_past_fwd += passes as u64;
        passes
    }

    pub(crate) fn settings(&self) -> Settings {
        Settings {
            seed: self.seed,
            make_alignment: self.do_alignment,
            make_alignment_display: self.do_alignment_display,
            long_target: self.long_target,
            do_null2: self.do_null2,
        }
    }

    pub(crate) fn define<const N: usize>(
        &mut self,
        model: &Model,
        targets: [&Target; N],
    ) -> [Definition; N] {
        crate::domaindef::define(model, targets, self.settings(), &mut self.domains)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn after_forward(
        &mut self,
        gm: &Profile,
        om: &OProfile,
        bg: &Bg,
        hmm: &crate::hmm::Hmm,
        sq: &Sequence,
        null_sc: f32,
        filtersc: f32,
        fwd_sc: f32,
        th: &mut TopHits,
    ) -> bool {
        if !self.survives_forward(fwd_sc, filtersc) {
            return false;
        }
        #[cfg(striped)]
        {
            self.bck_pmx.resize_parser(om.m, sq.n);
            unsafe {
                crate::simd::bck_filter::backward_parser_pmx(
                    &sq.dsq,
                    sq.n,
                    om,
                    Some(&self.fwd_pmx.row_scale),
                    &mut self.bck_pmx,
                )
            };
        }
        let target = Target {
            dsq: &sq.dsq,
            l: sq.n,
            null_sc,
            #[cfg(striped)]
            xf: om.xf,
            #[cfg(striped)]
            fwd: &self.fwd_pmx,
            #[cfg(striped)]
            bck: &self.bck_pmx,
        };
        let model = Model {
            gm,
            #[cfg(striped)]
            om,
            hmm,
            bg,
        };
        #[cfg(not(striped))]
        let _ = om;
        let [definition] =
            crate::domaindef::define(&model, [&target], self.settings(), &mut self.domains);
        self.report(gm, bg, sq, null_sc, fwd_sc, definition, th)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn report(
        &mut self,
        gm: &Profile,
        bg: &Bg,
        sq: &Sequence,
        null_sc: f32,
        fwd_sc: f32,
        definition: Definition,
        th: &mut TopHits,
    ) -> bool {
        let l = sq.n;
        let Definition {
            mut domains,
            nexpected,
            seq_bias: pipeline_seq_correction_sum,
            stats: domain_stats,
        } = definition;
        if self.long_target {
            domains.retain(|dom| {
                let span = dom
                    .ad
                    .as_ref()
                    .map(|ad| ad.sqfrom.abs_diff(ad.sqto) + 1)
                    .unwrap_or_else(|| (dom.iali - dom.jali).unsigned_abs() as usize + 1);
                span >= 8
            });
        }
        if domains.is_empty() {
            return false;
        }

        // Sequence-level scoring
        // Match C HMMER's two sequence scores:
        // 1. the full-sequence Forward score corrected by summed n2sc[]
        // 2. a reconstruction score from individually rescored domains
        let omega = bg.omega;
        let mut pre_score = nats_to_bits_from_scores(fwd_sc, null_sc);
        let seqbias = if self.do_null2 {
            crate::logsum::p7_flogsum(
                0.0,
                (c_log_f64(omega as f64) + pipeline_seq_correction_sum as f64) as f32,
            )
        } else {
            0.0
        };
        let mut seq_score = nats_to_bits_from_scores(fwd_sc, null_sc + seqbias);

        let mut sum_env = 0.0_f32;
        let mut sum_correction = 0.0_f32;
        let mut ld = 0usize;
        for dom in &domains {
            let use_domain = if self.do_null2 {
                dom.envsc - dom.domcorrection > 0.0
            } else {
                dom.envsc > 0.0
            };
            if use_domain {
                sum_env += dom.envsc;
                sum_correction += dom.domcorrection;
                ld += (dom.jenv - dom.ienv + 1) as usize;
            }
        }
        let sum_score_nats = reconstruction_score_nats(sum_env, l, ld);
        let sum_bias = if self.do_null2 {
            crate::logsum::p7_flogsum(
                0.0,
                (c_log_f64(omega as f64) + sum_correction as f64) as f32,
            )
        } else {
            0.0
        };
        let pre2_score = nats_to_bits_from_scores(sum_score_nats, null_sc);
        let sum_score = nats_to_bits_from_scores(sum_score_nats, null_sc + sum_bias);
        if ld > 0 && sum_score > seq_score {
            seq_score = sum_score;
            pre_score = pre2_score;
        }
        // C stores/prints the raw difference unclamped (p7_tophits.c:1677,1764:
        // `pre_score - score`), which can be slightly negative.
        let seq_bias_bits = pre_score - seq_score;

        if !self.do_null2 && !self.long_target {
            let tau = gm.evparam[crate::hmm::P7_FTAU] as f64;
            let lambda = gm.evparam[crate::hmm::P7_FLAMBDA] as f64;
            for dom in &mut domains {
                let env_len = (dom.jenv - dom.ienv + 1) as usize;
                let length_correction = reconstruction_score_nats(0.0, l, env_len);
                dom.dombias = 0.0;
                dom.bitscore = nats_to_bits_from_scores(dom.envsc + length_correction, null_sc);
                dom.lnp = crate::stats::exponential::logsurv(dom.bitscore as f64, tau, lambda);
            }
        } else if !self.do_null2 {
            for dom in &mut domains {
                dom.dombias = 0.0;
                dom.domcorrection = 0.0;
            }
        }

        // P-values
        let mu_f = self.evparam[crate::hmm::P7_FTAU] as f64;
        let lam_f = self.evparam[crate::hmm::P7_FLAMBDA] as f64;
        let lnp = stats::exponential::logsurv(seq_score as f64, mu_f, lam_f);
        let pre_lnp = stats::exponential::logsurv(pre_score as f64, mu_f, lam_f);

        // nhmmer (long_target) reports the best per-envelope score as hit.score;
        // it does not aggregate across domains the way hmmsearch does.
        // Mirrors C p7_pipeline.c:1462 `hit->sum_score = hit->score = dom_score`.
        let mut long_target_best_domain_idx = None;
        let (hit_score, hit_pre_score, hit_bias, hit_lnp, hit_pre_lnp) = if self.long_target {
            let (best_idx, best) = domains
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.lnp.total_cmp(&b.1.lnp))
                .expect("long_target hit must have at least one domain");
            long_target_best_domain_idx = Some(best_idx);
            let best_bitscore = best.bitscore;
            // dom.dombias is already in bits (see domaindef.rs). C does not clamp
            // the long-target bias (p7_pipeline.c:1455 stores dom_bias raw and
            // p7_tophits.c:1664 prints dcl[0].dombias * eslCONST_LOG2R unclamped),
            // so keep the raw (possibly slightly negative) value to match C.
            let best_bias = best.dombias;
            let pre = best_bitscore + best_bias;
            let pre_lnp_lt = stats::exponential::logsurv(pre as f64, mu_f, lam_f);
            (best_bitscore, pre, best_bias, best.lnp, pre_lnp_lt)
        } else {
            (seq_score, pre_score, seq_bias_bits, lnp, pre_lnp)
        };

        if let Some(best_idx) = long_target_best_domain_idx {
            domains.swap(0, best_idx);
        }

        // Storage gate, matching C p7_pipeline.c:1062: a hit is stored only if it
        // passes p7_pli_TargetReportable against the *current* pli->Z (running
        // target count for Ntargets, fixed -Z for Option). This is a permissive
        // lower bound; final thresholding re-applies with the true Z.
        let apply_early_storage_gate =
            !(self.long_target && self.use_bit_cutoffs == BitCutoff::None);
        if apply_early_storage_gate && !self.target_reportable(hit_score, hit_lnp) {
            return false;
        }

        let hit = th.create_next_hit();
        hit.name = sq.name.clone();
        hit.acc = sq.acc.clone();
        hit.desc = sq.desc.clone();
        hit.n = sq.n;
        hit.score = hit_score;
        hit.bias = hit_bias;
        hit.pre_score = hit_pre_score;
        hit.sum_score = if self.long_target {
            hit_score
        } else {
            sum_score
        };
        hit.lnp = hit_lnp;
        hit.pre_lnp = hit_pre_lnp;
        hit.sum_lnp = if self.long_target {
            hit_lnp
        } else {
            stats::exponential::logsurv(sum_score as f64, mu_f, lam_f)
        };
        hit.sortkey = self.hit_sortkey(hit.score, hit.lnp);
        hit.nexpected = nexpected;
        hit.nregions = domain_stats.nregions;
        hit.nclustered = domain_stats.nclustered;
        hit.noverlaps = domain_stats.noverlaps;
        hit.nenvelopes = domain_stats.nenvelopes;
        hit.ndom = domains.len();
        hit.dcl = domains;

        if self.use_bit_cutoffs != BitCutoff::None {
            let target_reported = if self.by_e {
                (self.z.max(1.0)) * c_exp_f64(hit.lnp) <= self.e_value_threshold
            } else if let Some(t) = self.t {
                hit.score as f64 >= t
            } else {
                (self.z.max(1.0)) * c_exp_f64(hit.lnp) <= self.e_value_threshold
            };
            if target_reported {
                hit.flags |= P7_IS_REPORTED;
                if self.target_includable(hit.score, hit.lnp) {
                    hit.flags |= P7_IS_INCLUDED;
                }
            }

            for dom in &mut hit.dcl {
                let dom_reported = if self.dom_by_e {
                    (self.domz.max(1.0)) * c_exp_f64(dom.lnp) <= self.dom_e_value_threshold
                } else if let Some(t) = self.dom_t {
                    dom.bitscore as f64 >= t
                } else {
                    (self.domz.max(1.0)) * c_exp_f64(dom.lnp) <= self.dom_e_value_threshold
                };
                if dom_reported {
                    dom.is_reported = true;
                    let dom_included = if self.incdom_by_e {
                        (self.domz.max(1.0)) * c_exp_f64(dom.lnp) <= self.inc_dome
                    } else if let Some(t) = self.inc_dom_t {
                        dom.bitscore as f64 >= t
                    } else {
                        (self.domz.max(1.0)) * c_exp_f64(dom.lnp) <= self.inc_dome
                    };
                    if dom_included {
                        dom.is_included = true;
                    }
                }
            }
        }

        true
    }
}
