//! Search pipeline using SIMD MSV + Viterbi + Forward filters.

use crate::profile::Profile;
use crate::stats;
use crate::util::cmath::{c_exp_f64, c_log_f64, ESL_CONST_LOG2};

mod run;

/// How Z/domZ was set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZSetBy {
    /// Automatically counted from number of targets.
    Ntargets,
    /// Explicitly set by user via -Z / --domZ.
    Option,
}

/// Model-specific bit-score cutoff mode (--cut_ga, --cut_nc, --cut_tc).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BitCutoff {
    None,
    GA,
    TC,
    NC,
}

/// Pipeline configuration and state.
pub struct Pipeline {
    // Thresholds
    pub f1: f64, // MSV filter threshold (default 0.02)
    pub f2: f64, // Viterbi filter threshold (default 1e-3)
    pub f3: f64, // Forward filter threshold (default 1e-5)

    // Reporting thresholds (E-value based)
    pub e_value_threshold: f64,     // per-sequence E-value (default 10.0)
    pub dom_e_value_threshold: f64, // per-domain E-value (default 10.0)
    pub inc_e: f64,                 // inclusion E-value (default 0.01)
    pub inc_dome: f64,              // domain inclusion E-value (default 0.01)

    // Reporting thresholds (score based, set by -T/--domT/--incT/--incdomT or --cut_*)
    pub t: Option<f64>,         // per-sequence score threshold
    pub dom_t: Option<f64>,     // per-domain score threshold
    pub inc_t: Option<f64>,     // inclusion score threshold
    pub inc_dom_t: Option<f64>, // domain inclusion score threshold

    // Whether to use E-value or score thresholds
    pub by_e: bool,        // report by E-value (default true)
    pub dom_by_e: bool,    // report domains by E-value (default true)
    pub inc_by_e: bool,    // include by E-value (default true)
    pub incdom_by_e: bool, // include domains by E-value (default true)

    // Model-specific bit cutoffs
    pub use_bit_cutoffs: BitCutoff,

    // Flags
    pub do_biasfilter: bool,
    pub do_null2: bool,
    pub do_max: bool, // if true, skip MSV/Viterbi filters
    pub do_alignment: bool,
    pub do_alignment_display: bool,
    pub seed: u32, // RNG seed for stochastic traceback (default 42, 0 = arbitrary)
    // long_target = nhmmer-style DNA genome search. Switches domain-scoring to
    // reconfigure the profile's rest-length per envelope, matching C
    // `rescore_isolated_domain()`'s `if (long_target) ReconfigRestLength(om, j-i+1)`.
    pub long_target: bool,
    pub long_target_b3: usize,

    // Statistics
    pub n_past_msv: u64,
    pub n_past_bias: u64,
    pub n_past_vit: u64,
    pub n_past_fwd: u64,
    pub n_targets: u64,
    pub z: f64,    // database size (number of sequences) for E-value
    pub domz: f64, // domain database size
    pub z_setby: ZSetBy,
    pub domz_setby: ZSetBy,

    // E-value parameters (from HMM)
    pub evparam: [f32; 6],

    #[cfg(striped)]
    msv_dp: Vec<crate::simd::sse::__m128i>,
    #[cfg(target_arch = "x86_64")]
    vit_dp: Vec<std::arch::x86_64::__m256i>,
    #[cfg(all(striped, not(target_arch = "x86_64")))]
    vit_dp: Vec<crate::simd::sse::__m128i>,
    #[cfg(striped)]
    fwd_pmx: crate::simd::probmx::ProbMx,
    #[cfg(striped)]
    bck_pmx: crate::simd::probmx::ProbMx,
    domains: crate::domaindef::Scratch,
}

impl Pipeline {
    /// Create a new accelerated comparison pipeline with default thresholds.
    /// Port of `p7_pipeline_Create()`: sets reporting/inclusion E-values, F1/F2/F3
    /// filter thresholds, RNG seed, and bias/null2 flags to HMMER 3 defaults.
    pub fn new() -> Self {
        Pipeline {
            f1: 0.02,
            f2: 1e-3,
            f3: 1e-5,
            e_value_threshold: 10.0,
            dom_e_value_threshold: 10.0,
            inc_e: 0.01,
            inc_dome: 0.01,
            t: None,
            dom_t: None,
            inc_t: None,
            inc_dom_t: None,
            by_e: true,
            dom_by_e: true,
            inc_by_e: true,
            incdom_by_e: true,
            use_bit_cutoffs: BitCutoff::None,
            do_biasfilter: true,
            do_null2: true,
            do_max: false,
            do_alignment: true,
            do_alignment_display: true,
            seed: 42,
            long_target: false,
            long_target_b3: 1000,
            n_past_msv: 0,
            n_past_bias: 0,
            n_past_vit: 0,
            n_past_fwd: 0,
            n_targets: 0,
            z: 0.0,
            domz: 0.0,
            z_setby: ZSetBy::Ntargets,
            domz_setby: ZSetBy::Ntargets,
            evparam: [0.0; 6],
            #[cfg(striped)]
            msv_dp: Vec::new(),
            #[cfg(striped)]
            vit_dp: Vec::new(),
            #[cfg(striped)]
            fwd_pmx: Default::default(),
            #[cfg(striped)]
            bck_pmx: Default::default(),
            domains: Default::default(),
        }
    }

    /// Configure the pipeline for a new query/target model.
    /// Port of `p7_pli_NewModel()`: copies the model's E-value calibration
    /// parameters into the pipeline so subsequent filter p-values use them.
    pub fn new_model(&mut self, gm: &Profile) {
        self.evparam = gm.evparam;
    }

    pub fn fork(&self) -> Pipeline {
        Pipeline {
            n_past_msv: 0,
            n_past_bias: 0,
            n_past_vit: 0,
            n_past_fwd: 0,
            n_targets: 0,
            #[cfg(striped)]
            msv_dp: Vec::new(),
            #[cfg(striped)]
            vit_dp: Vec::new(),
            #[cfg(striped)]
            fwd_pmx: Default::default(),
            #[cfg(striped)]
            bck_pmx: Default::default(),
            domains: Default::default(),
            ..*self
        }
    }

    /// True when target ranking should follow bit score instead of P-value.
    /// `TopHits` sorts ascending, so score-based sort keys are stored as
    /// negative scores to keep higher-scoring hits first.
    pub fn score_sort_active(&self) -> bool {
        self.use_bit_cutoffs != BitCutoff::None || (!self.inc_by_e && self.inc_t.is_some())
    }

    pub fn hit_sortkey(&self, score: f32, lnp: f64) -> f64 {
        if self.score_sort_active() {
            -(score as f64)
        } else {
            lnp
        }
    }

    pub fn target_reportable(&self, score: f32, lnp: f64) -> bool {
        if self.by_e {
            if self.long_target {
                c_exp_f64(lnp) <= self.e_value_threshold
            } else {
                c_exp_f64(lnp) * self.current_target_z() <= self.e_value_threshold
            }
        } else if let Some(t) = self.t {
            score as f64 >= t
        } else {
            c_exp_f64(lnp) * self.current_target_z() <= self.e_value_threshold
        }
    }

    pub(crate) fn target_includable(&self, score: f32, lnp: f64) -> bool {
        if self.inc_by_e {
            if self.long_target {
                c_exp_f64(lnp) <= self.inc_e
            } else {
                c_exp_f64(lnp) * self.current_target_z() <= self.inc_e
            }
        } else if let Some(t) = self.inc_t {
            score as f64 >= t
        } else {
            c_exp_f64(lnp) * self.current_target_z() <= self.inc_e
        }
    }

    fn current_target_z(&self) -> f64 {
        match self.z_setby {
            ZSetBy::Option => self.z,
            ZSetBy::Ntargets => self.z.max(self.n_targets as f64).max(1.0),
        }
    }

    /// Apply model-specific bit-score cutoffs (--cut_ga, --cut_nc, --cut_tc).
    /// Called after new_model() when use_bit_cutoffs is set.
    /// Returns Err if the model doesn't have the requested cutoff annotation.
    pub fn new_model_thresholds(
        &mut self,
        cutoffs: &[f32; crate::hmm::NCUTOFFS],
    ) -> Result<(), String> {
        use crate::hmm::*;
        match self.use_bit_cutoffs {
            BitCutoff::GA => {
                if cutoffs[P7_GA1] == CUTOFF_UNSET || cutoffs[P7_GA2] == CUTOFF_UNSET {
                    return Err("GA cutoff not set in model".to_string());
                }
                self.t = Some(cutoffs[P7_GA1] as f64);
                self.inc_t = Some(cutoffs[P7_GA1] as f64);
                self.dom_t = Some(cutoffs[P7_GA2] as f64);
                self.inc_dom_t = Some(cutoffs[P7_GA2] as f64);
            }
            BitCutoff::TC => {
                if cutoffs[P7_TC1] == CUTOFF_UNSET || cutoffs[P7_TC2] == CUTOFF_UNSET {
                    return Err("TC cutoff not set in model".to_string());
                }
                self.t = Some(cutoffs[P7_TC1] as f64);
                self.inc_t = Some(cutoffs[P7_TC1] as f64);
                self.dom_t = Some(cutoffs[P7_TC2] as f64);
                self.inc_dom_t = Some(cutoffs[P7_TC2] as f64);
            }
            BitCutoff::NC => {
                if cutoffs[P7_NC1] == CUTOFF_UNSET || cutoffs[P7_NC2] == CUTOFF_UNSET {
                    return Err("NC cutoff not set in model".to_string());
                }
                self.t = Some(cutoffs[P7_NC1] as f64);
                self.inc_t = Some(cutoffs[P7_NC1] as f64);
                self.dom_t = Some(cutoffs[P7_NC2] as f64);
                self.inc_dom_t = Some(cutoffs[P7_NC2] as f64);
            }
            BitCutoff::None => {}
        }

        if self.use_bit_cutoffs != BitCutoff::None {
            self.by_e = false;
            self.dom_by_e = false;
            self.inc_by_e = false;
            self.incdom_by_e = false;
        }
        Ok(())
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate MSV p-value from raw score.
pub fn msv_pvalue(msv_sc: f32, null_sc: f32, evparam: &[f32; 6]) -> f64 {
    let score = nats_to_bits_from_scores(msv_sc, null_sc);
    let mu = evparam[crate::hmm::P7_MMU] as f64;
    let lambda = evparam[crate::hmm::P7_MLAMBDA] as f64;
    stats::gumbel::surv(score as f64, mu, lambda)
}

/// Calculate Viterbi p-value from raw score.
pub(crate) fn viterbi_pvalue(vit_sc: f32, null_sc: f32, evparam: &[f32; 6]) -> f64 {
    let score = nats_to_bits_from_scores(vit_sc, null_sc);
    let mu = evparam[crate::hmm::P7_VMU] as f64;
    let lambda = evparam[crate::hmm::P7_VLAMBDA] as f64;
    stats::gumbel::surv(score as f64, mu, lambda)
}

/// Calculate Forward p-value from raw score.
pub(crate) fn forward_pvalue(fwd_sc: f32, null_sc: f32, evparam: &[f32; 6]) -> f64 {
    let score = nats_to_bits_from_scores(fwd_sc, null_sc);
    let tau = evparam[crate::hmm::P7_FTAU] as f64;
    let lambda = evparam[crate::hmm::P7_FLAMBDA] as f64;
    stats::exponential::surv(score as f64, tau, lambda)
}

/// Convert a raw score (nats, relative to `baseline`) into bits via division by ln(2).
#[inline]
pub(crate) fn nats_to_bits_from_scores(score: f32, baseline: f32) -> f32 {
    (((score - baseline) as f64) / ESL_CONST_LOG2) as f32
}

/// Sequence reconstruction score in nats: sum of envelope scores plus a
/// length-correction term for residues not covered by any envelope. Used to
/// build the alternative per-sequence score `sum_score` in `p7_Pipeline()`.
#[inline]
pub(crate) fn reconstruction_score_nats(sum_env: f32, l: usize, ld: usize) -> f32 {
    let len_ratio = l as f32 / (l as f32 + 3.0);
    let uncovered = l as i64 - ld as i64;
    (sum_env as f64 + uncovered as f64 * c_log_f64(len_ratio as f64)) as f32
}

#[cfg(test)]
#[path = "../../tests/unit/pipeline_test.rs"]
mod tests;
