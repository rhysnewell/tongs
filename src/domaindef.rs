//! Domain definition using posterior probabilities.
//! Port of p7_domaindef.c p7_domaindef_ByPosteriorHeuristics().
#![cfg_attr(
    not(striped),
    allow(dead_code, unused_imports, unused_mut, unused_variables)
)]

use crate::alphabet::{Alphabet, Dsq};
use crate::bg::Bg;
use crate::dp::generic_backward::g_backward;
use crate::dp::generic_decoding::{domain_decoding, g_decoding};
use crate::dp::generic_fwdback::g_forward;
use crate::dp::generic_null2;
use crate::dp::generic_optacc::{g_oa_trace, g_optimal_accuracy_with_deltas, OptAccTDelta};
use crate::dp::generic_stotrace::g_stochastic_trace;
use crate::dp::gmx::*;
use crate::hmm::Hmm;
use crate::profile::*;
use crate::spensemble::{self, ClusterParams, SegmentPair};
use crate::tophits::{AliDisplay, Domain};
use crate::trace::{State, Trace};
use crate::util::cmath::{c_log_f64, c_logf_to_f32, ESL_CONST_LOG2, ESL_CONST_LOG2R};
use crate::util::random::MersenneTwister;

// Thresholds matching C's p7_domaindef defaults
const RT1: f32 = 0.25; // mocc threshold to trigger a domain region
const RT2: f32 = 0.10; // mocc threshold to end a domain region
const RT3: f32 = 0.20; // threshold for multi-domain region detection
const NSAMPLES: usize = 200; // stochastic tracebacks for clustering

/// True if every residue in `dsq[dsq_offset+1 ..= dsq_offset+l]` is a canonical
/// alphabet code (`< abc_k`), enabling the fast no-degeneracy SIMD paths.
#[cfg(striped)]
#[inline(always)]
fn is_canonical_window(dsq: &[Dsq], dsq_offset: usize, l: usize, abc_k: usize) -> bool {
    let Some(end) = dsq_offset.checked_add(l) else {
        return false;
    };
    if l == 0 || end >= dsq.len() {
        return false;
    }
    unsafe {
        let ptr = dsq.as_ptr().add(dsq_offset);
        for i in 1..=l {
            if *ptr.add(i) as usize >= abc_k {
                return false;
            }
        }
    }
    true
}

/// Plain serial summation matching the C `for`-loop accumulation order;
/// used where bit-exact parity with the original is required.
#[inline(always)]
fn c_fsum(values: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for &value in values {
        sum += value;
    }
    sum
}

/// Kahan-compensated summation matching Easel's `esl_vec_FSum`, used for the
/// numerically-stable per-sequence bias accumulation.
#[inline(always)]
fn esl_vec_fsum(values: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let mut c = 0.0_f32;
    for &value in values {
        let y = value - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DomainDefinitionStats {
    pub nregions: usize,
    pub nclustered: usize,
    pub noverlaps: usize,
    pub nenvelopes: usize,
}

#[cfg(striped)]
pub(crate) struct DomainSimdScratch {
    fwd_dp: Vec<crate::simd::sse::__m128>,
    bck_prev: Vec<crate::simd::sse::__m128>,
    bck_cur: Vec<crate::simd::sse::__m128>,
    fwd_pmx: crate::simd::probmx::ProbMx,
    bck_pmx: crate::simd::probmx::ProbMx,
    pp_pmx: crate::simd::probmx::ProbMx,
    oa_pmx: crate::simd::probmx::ProbMx,
    region_pmx: Option<crate::simd::probmx::ProbMx>,
    pp_cells: Vec<crate::simd::sse::__m128>,
    expected: crate::simd::optacc::Expectation,
    null2: Vec<f32>,
    exp_m: Vec<f32>,
    exp_i: Vec<f32>,
    pp_gmx: Gmx,
    oa_gmx: Gmx,
    trim_pp_gmx: Gmx,
    trim_sub_dsq: Vec<Dsq>,
    btot: Vec<f32>,
    etot: Vec<f32>,
    mocc: Vec<f32>,
}

#[cfg(striped)]
impl DomainSimdScratch {
    /// Allocate an empty SIMD scratch bundle (matrices/buffers grow on first use).
    /// Reused across sequences to amortize allocations.
    pub(crate) fn new() -> Self {
        Self {
            fwd_dp: Vec::new(),
            bck_prev: Vec::new(),
            bck_cur: Vec::new(),
            fwd_pmx: crate::simd::probmx::ProbMx::new_full(0, 0),
            bck_pmx: crate::simd::probmx::ProbMx::new_full(0, 0),
            pp_pmx: crate::simd::probmx::ProbMx::new_full(0, 0),
            oa_pmx: crate::simd::probmx::ProbMx::new_full(0, 0),
            region_pmx: None,
            pp_cells: Vec::new(),
            expected: Default::default(),
            null2: Vec::new(),
            exp_m: Vec::new(),
            exp_i: Vec::new(),
            pp_gmx: Gmx::new(0, 0),
            oa_gmx: Gmx::new(0, 0),
            trim_pp_gmx: Gmx::new(0, 0),
            trim_sub_dsq: Vec::new(),
            btot: Vec::new(),
            etot: Vec::new(),
            mocc: Vec::new(),
        }
    }
}

#[cfg(not(striped))]
pub(crate) struct DomainSimdScratch {
    fwd_gmx: Gmx,
    bck_gmx: Gmx,
    env_fwd_gmx: Gmx,
    env_bck_gmx: Gmx,
    pp_gmx: Gmx,
    oa_gmx: Gmx,
}

#[cfg(not(striped))]
impl DomainSimdScratch {
    /// Allocate a minimal scratch bundle for the non-x86 (generic) path.
    pub(crate) fn new() -> Self {
        Self {
            fwd_gmx: Gmx::new(0, 0),
            bck_gmx: Gmx::new(0, 0),
            env_fwd_gmx: Gmx::new(0, 0),
            env_bck_gmx: Gmx::new(0, 0),
            pp_gmx: Gmx::new(0, 0),
            oa_gmx: Gmx::new(0, 0),
        }
    }
}

/// Region detection state machine extracted from `p7_domaindef_ByPosteriorHeuristics`.
///
/// Walks the per-position B/E totals and match-occupancy posteriors, using
/// thresholds `RT1`/`RT2` to trigger and close domain regions. Returns the
/// inclusive `(i,j)` sequence-coordinate pairs of each detected region.
fn find_domain_regions(btot: &[f32], etot: &[f32], mocc: &[f32], l: usize) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut i: i64 = -1;
    let mut triggered = false;

    for j in 1..=l {
        if !triggered {
            // Looking for the START of a domain region.
            // Reset i when mocc (minus local B contribution) drops below rt2.
            if mocc[j] - (btot[j] - btot[j - 1]) < RT2 || i == -1 {
                i = j as i64;
            }
            // Trigger when mocc rises above rt1
            if mocc[j] >= RT1 {
                triggered = true;
            }
        } else if mocc[j] - (etot[j] - etot[j - 1]) < RT2 {
            // Found the END of a domain region: mocc dropped below rt2
            // (after subtracting local E contribution). Region is i..j.
            if i >= 1 {
                regions.push((i as usize, j));
            }
            i = -1;
            triggered = false;
        }
    }

    regions
}

/// Trigger for handing a region off to deeper stochastic-traceback analysis.
///
/// Mirrors C `is_multidomain_region()`: returns true iff
/// `max_z [ min(E(z), B(z)) ] >= RT3`, where `E(z)` and `B(z)` are expected
/// counts of E states before, and B states after, position `z` in `i..=j`.
/// A high value means the region likely contains an E→B transition and so
/// encompasses two or more domains.
fn is_multidomain_region(btot: &[f32], etot: &[f32], i: usize, j: usize) -> bool {
    let mut max = -1.0_f32;
    for z in i..=j {
        let expected_n = (etot[z] - etot[i - 1]).min(btot[j] - btot[z - 1]);
        max = max.max(expected_n);
    }
    max >= RT3
}

/// SIMD-optimized null2 odds ratios for a trace segment. Port of C
/// `p7_GNull2_ByTrace`, operating on a striped `OProfile` instead of a
/// generic profile/Gmx.
///
/// Counts state usages over `tr[zstart..=zend]` (residue-emitting positions
/// only), normalizes to expected per-residue usage, and produces the
/// length-`om.abc_k` odds vector f'(x)/f(x) used to bias-correct the domain
/// envelope. The model configuration is irrelevant; only emission odds are
/// used. Returns an all-ones vector if the segment emits no residues.
#[cfg(striped)]
fn null2_by_trace_optimized(
    om: &crate::simd::oprofile::OProfile,
    tr: &Trace,
    zstart: usize,
    zend: usize,
) -> Vec<f32> {
    let q = crate::simd::oprofile::nqf(om.m);
    let mut exp_m = vec![0.0_f32; q * 4];
    let mut exp_n = 0.0_f32;
    let mut exp_c = 0.0_f32;
    let mut exp_j = 0.0_f32;
    let mut ld = 0usize;

    for z in zstart..=zend.min(tr.n.saturating_sub(1)) {
        if tr.i[z] == 0 {
            continue;
        }
        ld += 1;
        if tr.k[z] > 0 {
            let k = tr.k[z];
            let qi = (k - 1) % q;
            let lane = (k - 1) / q;
            exp_m[qi * 4 + lane] += 1.0;
        } else {
            match tr.st[z] {
                State::N => exp_n += 1.0,
                State::C => exp_c += 1.0,
                State::J => exp_j += 1.0,
                _ => {}
            }
        }
    }

    if ld == 0 {
        return vec![1.0; om.abc_k];
    }

    let norm = 1.0 / ld as f32;
    for val in &mut exp_m {
        *val *= norm;
    }
    exp_n *= norm;
    exp_c *= norm;
    exp_j *= norm;
    let xfactor = exp_n + exp_c + exp_j;

    let mut null2 = vec![0.0_f32; om.abc_k];
    for (x, null2_x) in null2.iter_mut().enumerate().take(om.abc_k) {
        let mut lanes = [0.0_f32; 4];
        for qi in 0..q {
            let odds = om.rfv[x][qi];
            let base = qi * 4;
            for lane in 0..4 {
                lanes[lane] += exp_m[base + lane] * odds[lane];
            }
        }
        let h01 = lanes[0] + lanes[1];
        let h23 = lanes[2] + lanes[3];
        *null2_x = (h01 + h23) + xfactor;
    }
    null2
}

/// Apply null2 odds ratios across an envelope. For each position in `ienv..=jenv`,
/// look up `log f'(x)/f(x)` for the residue (with degeneracy handling) and
/// accumulate into `n2sc[pos]` and the per-domain `dom_correction` total.
/// Equivalent to the inner null2 score loop inside C `rescore_isolated_domain`.
fn add_null2_correction(
    null2_arr: &[f32],
    abc: &Alphabet,
    dsq: &[Dsq],
    ienv: usize,
    jenv: usize,
    mut n2sc: Option<&mut [f32]>,
    dom_correction: &mut f32,
) {
    let mut log_null2 = [0.0_f32; 256];
    for (x, val) in null2_arr.iter().enumerate() {
        log_null2[x] = c_logf_to_f32(*val);
    }
    for pos in ienv..=jenv {
        let x = dsq[pos] as usize;
        let sc = null2_odds_for_code(null2_arr, &log_null2, abc, x);
        if let Some(ref mut n2sc) = n2sc {
            n2sc[pos] = sc;
        }
        *dom_correction += sc;
    }
}

/// Per-residue `log f'(x)/f(x)` lookup with degenerate-code handling.
/// Returns the pre-log odds from `log_null2` for canonical codes, 0 for
/// gaps/missing/nonresidue, and the log of the averaged odds across the
/// resolved canonical residues for ambiguity codes.
fn null2_odds_for_code(null2_arr: &[f32], log_null2: &[f32; 256], abc: &Alphabet, x: usize) -> f32 {
    if x < null2_arr.len() {
        return log_null2[x];
    }
    if x >= abc.kp || x == abc.k || x >= abc.kp.saturating_sub(2) {
        return 0.0;
    }
    let n = abc.ndegen[x];
    if n == 0 {
        return 0.0;
    }
    let mut odds = 0.0_f32;
    for (a, &null2) in null2_arr.iter().enumerate().take(abc.k) {
        if abc.degen[x][a] {
            odds += null2;
        }
    }
    c_logf_to_f32(odds / n as f32)
}

/// Per-residue null2 odds ratio (linear, not log) with degenerate-code
/// handling. Used in the multidomain region path where ratios are
/// accumulated across many stochastic traces before being log-summarized.
fn null2_ratio_for_code(null2_arr: &[f32], abc: &Alphabet, x: usize) -> f32 {
    if x < null2_arr.len() {
        return null2_arr[x];
    }
    if x >= abc.kp || x == abc.k || x >= abc.kp.saturating_sub(2) {
        return 1.0;
    }
    let n = abc.ndegen[x];
    if n == 0 {
        return 1.0;
    }
    let mut odds = 0.0_f32;
    for (a, &null2) in null2_arr.iter().enumerate().take(abc.k) {
        if abc.degen[x][a] {
            odds += null2;
        }
    }
    odds / n as f32
}

/// Score and align a single isolated domain envelope. Port of C
fn long_target_local_bg(
    dsq: &[Dsq],
    start: usize,
    end: usize,
    bg: &Bg,
    abc: &Alphabet,
    seq_len: usize,
) -> Bg {
    let k = abc.k;
    let mut counts = vec![0.0_f32; k];
    let mut total = 0.0_f32;

    for &x in dsq.iter().take(end + 1).skip(start) {
        let xu = x as usize;
        if xu < k {
            counts[xu] += 1.0;
            total += 1.0;
        } else if xu < abc.kp && abc.ndegen[xu] > 0 {
            let w = 1.0_f32 / abc.ndegen[xu] as f32;
            for (a, count) in counts.iter_mut().enumerate() {
                if abc.degen[xu][a] {
                    *count += w;
                }
            }
            total += 1.0;
        }
    }

    if total <= 0.0 {
        return bg.clone();
    }

    for c in &mut counts {
        *c /= total;
    }

    let clamped_n = seq_len.clamp(50, 100) as f32;
    let bg_smooth = 25.0_f32 / clamped_n;
    let mut local_bg = bg.clone();
    let mut sum = 0.0_f32;
    for (i, f) in local_bg.f.iter_mut().enumerate().take(k) {
        *f = bg_smooth * bg.f[i] + (1.0 - bg_smooth) * counts[i];
        sum += *f;
    }
    if sum > 0.0 {
        for f in local_bg.f.iter_mut().take(k) {
            *f /= sum;
        }
    }
    local_bg
}

/// `rescore_isolated_domain` from `p7_domaindef.c`.
///
/// Given the envelope coordinates `ienv..=jenv` of `dsq` and a unilocal
/// model, computes:
///   - Forward score over the envelope (per-domain bit score base);
///   - Backward and posterior decoding for PP annotation and null2 odds;
///   - Optimal-accuracy traceback for the alignment;
///   - Null2 bias correction (skipped when `null2_is_done`).
///
/// SIMD path (when available) reuses the caller-owned scratch buffers;
/// the non-SIMD fallback uses Gmx matrices. When `long_target` is set, the
/// nhmmer-specific reparameterization (envelope-local background, rest
/// length set to envelope length) is applied to the optimized profile,
/// mirroring C `reparameterize_model()`.
///
/// `seq_len` is the full sequence length (drives the null-model length
/// correction); `null_sc` is the per-sequence null log score from `p7_bg`.
#[allow(clippy::too_many_arguments)]
fn score_domain_envelope(
    dsq: &[Dsq],
    seq_len: usize,
    gm: &Profile,
    env_gm: &Profile,
    env_om: Option<&crate::simd::oprofile::OProfile>,
    hmm: &Hmm,
    ienv: usize,
    jenv: usize,
    null_sc: f32,
    bg: &Bg,
    mut n2sc: Option<&mut [f32]>,
    null2_is_done: bool,
    _match_odds: Option<&[f32]>,
    optacc_deltas: &OptAccTDelta,
    simd_scratch: &mut DomainSimdScratch,
    make_alignment: bool,
    make_alignment_display: bool,
    long_target: bool,
    do_null2: bool,
) -> Domain {
    debug_assert!(ienv >= 1 && jenv <= seq_len);
    debug_assert!(null_sc.is_finite());
    let env_len = jenv - ienv + 1;
    let l = seq_len; // full sequence length for length correction
    let null2_is_done = null2_is_done || !do_null2;
    let do_longtarget_length = long_target;
    let do_longtarget_reparam = long_target && do_null2;

    // Forward score for the envelope.
    let env_fwd_sc;
    let mut simd_null2_scratch: Option<&[f32]> = None;
    let mut env_pp_storage: Option<Gmx>;
    let mut env_pp_ptr: *mut Gmx = std::ptr::null_mut();
    #[cfg(striped)]
    let mut simd_oa_coords: Option<(usize, usize, usize, usize)> = None;
    #[cfg(striped)]
    let mut simd_oa_trace: Option<crate::trace::Trace> = None;
    #[cfg(striped)]
    let mut simd_oasc: Option<f32> = None;
    #[cfg(striped)]
    let mut pre_trim_local_bg_f: Option<Vec<f32>> = None;
    let use_simd = cfg!(striped);
    // The striped path reads the generic posterior matrix only to display alignments or when it
    // finds no optimal accuracy trace, so it is cleared only then.
    let lazy_pp = make_alignment && use_simd && !make_alignment_display;
    if make_alignment {
        if !lazy_pp {
            simd_scratch.pp_gmx.grow_to_zeroed(gm.m, env_len);
        }
        env_pp_ptr = &mut simd_scratch.pp_gmx;
    }
    let dsq_offset = ienv - 1;
    let mut sub_dsq = Vec::new();
    if !use_simd || make_alignment_display {
        sub_dsq.push(crate::alphabet::DSQ_SENTINEL);
        sub_dsq.extend_from_slice(&dsq[ienv..=jenv]);
        sub_dsq.push(crate::alphabet::DSQ_SENTINEL);
    }

    if use_simd {
        #[cfg(striped)]
        {
            use crate::simd::oprofile::OProfile;
            let env_om_storage;
            let long_target_om_storage;
            let env_om = if let Some(om) = env_om {
                om
            } else {
                env_om_storage = OProfile::convert(env_gm);
                &env_om_storage
            };
            // nhmmer (long_target) path: rescore the envelope with the profile's
            // rest length set to env_len AND the match emission log-odds
            // reparameterized against envelope-local residue frequency.
            // Matches C p7_domaindef.c:2108 `ReconfigRestLength(om, j-i+1)` and
            // p7_domaindef.c:2115 `reparameterize_model(bg, om, sq, i, j-i+1, ...)`.
            let (env_om, pre_trim_local_bg_f_inner) = if do_longtarget_length {
                let mut clone = env_om.clone();
                clone.reconfig_rest_length(env_len as i32);
                let local_bg_f = if do_longtarget_reparam {
                    let abc_lt = crate::alphabet::Alphabet::shared(hmm.abc_type);
                    let local_bg = long_target_local_bg(dsq, ienv, jenv, bg, &abc_lt, seq_len);
                    let fwd_emissions = env_om.get_fwd_emissions(bg, &abc_lt);
                    clone.update_fwd_emission_scores(&local_bg, &fwd_emissions, &abc_lt);
                    Some(local_bg.f.clone())
                } else {
                    None
                };
                long_target_om_storage = clone;
                (&long_target_om_storage, local_bg_f)
            } else {
                (env_om, None)
            };
            pre_trim_local_bg_f = pre_trim_local_bg_f_inner;
            if make_alignment || !null2_is_done {
                use crate::simd::oprofile::{P7O_C, P7O_J, P7O_LOOP, P7O_N};
                use crate::simd::probmx::{
                    p_decoding_to_gmx, p_null2_odds_from_omx_expectation_reuse,
                };

                {
                    let scratch = &mut *simd_scratch;
                    let canonical_env = is_canonical_window(dsq, dsq_offset, env_len, env_om.abc_k);
                    scratch.fwd_pmx.resize_full(gm.m, env_len);
                    env_fwd_sc = unsafe {
                        if canonical_env {
                            crate::simd::fwd_filter::forward_parser_pmx_offset_direct(
                                dsq,
                                dsq_offset,
                                env_len,
                                env_om,
                                &mut scratch.fwd_pmx,
                            )
                        } else {
                            crate::simd::fwd_filter::forward_parser_pmx_offset_with_scratch(
                                dsq,
                                dsq_offset,
                                env_len,
                                env_om,
                                &mut scratch.fwd_pmx,
                                &mut scratch.fwd_dp,
                            )
                        }
                    };
                    scratch.bck_pmx.resize_full(gm.m, env_len);
                    unsafe {
                        if canonical_env {
                            crate::simd::bck_filter::backward_parser_pmx_offset_direct(
                                dsq,
                                dsq_offset,
                                env_len,
                                env_om,
                                &mut scratch.bck_pmx,
                                Some(&scratch.fwd_pmx.row_scale),
                            );
                        } else {
                            crate::simd::bck_filter::backward_parser_pmx_offset_with_scratch(
                                dsq,
                                dsq_offset,
                                env_len,
                                env_om,
                                env_fwd_sc,
                                &mut scratch.bck_pmx,
                                Some(&scratch.fwd_pmx.row_scale),
                                &mut scratch.bck_prev,
                                &mut scratch.bck_cur,
                            );
                        }
                    };
                    let njc_loop = [
                        env_om.xf[P7O_N][P7O_LOOP],
                        env_om.xf[P7O_J][P7O_LOOP],
                        env_om.xf[P7O_C][P7O_LOOP],
                    ];
                    if make_alignment {
                        // Always populate SIMD pp matrix and run SIMD OA.
                        // Match C p7_OptimalAccuracy + p7_OATrace which are
                        // SIMD on the striped pp matrix — gives byte-identical
                        // trace boundaries. (Previously alignment_display path
                        // fell back to generic OA which drifted by 1 residue.)
                        unsafe {
                            simd_oasc = Some(crate::simd::optacc::decode_and_fill(
                                &scratch.fwd_pmx,
                                &scratch.bck_pmx,
                                env_om,
                                &mut scratch.pp_pmx,
                                &mut scratch.oa_pmx,
                                &mut scratch.pp_cells,
                                (!null2_is_done && !make_alignment_display)
                                    .then_some(&mut scratch.expected),
                            ));
                            simd_oa_coords = crate::simd::optacc::oa_trace_coords_pmx(
                                env_om,
                                &scratch.pp_pmx,
                                &scratch.oa_pmx,
                            );
                            // Full Trace for alignment_display_with_pp.
                            if make_alignment_display {
                                simd_oa_trace = Some(crate::simd::optacc::oa_trace_pmx(
                                    env_om,
                                    &scratch.pp_pmx,
                                    &scratch.oa_pmx,
                                ));
                            }
                        }
                        // Also populate generic env_pp for alignment_display
                        // PP-line annotations (still uses generic Gmx format).
                        if make_alignment_display {
                            p_decoding_to_gmx(
                                &scratch.fwd_pmx,
                                &scratch.bck_pmx,
                                gm.m,
                                njc_loop,
                                unsafe { &mut *env_pp_ptr },
                            );
                        }
                    }
                    if !null2_is_done {
                        if make_alignment && !make_alignment_display {
                            unsafe {
                                scratch.expected.null2(
                                    env_len,
                                    gm.abc_k,
                                    &env_om.rfv,
                                    &mut scratch.null2,
                                );
                            }
                        } else {
                            p_null2_odds_from_omx_expectation_reuse(
                                &scratch.fwd_pmx,
                                &scratch.bck_pmx,
                                gm.abc_k,
                                &env_om.rfv,
                                njc_loop,
                                &mut scratch.null2,
                                &mut scratch.exp_m,
                                &mut scratch.exp_i,
                            );
                        }
                        simd_null2_scratch = Some(&scratch.null2);
                    }
                }
            } else {
                {
                    let scratch = &mut *simd_scratch;
                    scratch.fwd_pmx.resize_full(gm.m, env_len);
                    env_fwd_sc = unsafe {
                        crate::simd::fwd_filter::forward_parser_pmx_offset_with_scratch(
                            dsq,
                            dsq_offset,
                            env_len,
                            env_om,
                            &mut scratch.fwd_pmx,
                            &mut scratch.fwd_dp,
                        )
                    };
                }
            }
        }
        #[cfg(not(striped))]
        unreachable!();
    } else {
        #[cfg(striped)]
        {
            let mut gx = Gmx::new(gm.m, env_len);
            env_fwd_sc = g_forward(&sub_dsq, env_len, env_gm, &mut gx);
        }
        #[cfg(not(striped))]
        {
            simd_scratch.env_fwd_gmx.grow_to(gm.m, env_len);
            env_fwd_sc = g_forward(&sub_dsq, env_len, env_gm, &mut simd_scratch.env_fwd_gmx);
        }
    }

    // Posterior decoding for PP annotation and null2. C HMMER fills n2sc[]
    // from these per-envelope posteriors for simple regions.
    if env_pp_ptr.is_null() && simd_null2_scratch.is_none() && !null2_is_done {
        env_pp_storage = Some(Gmx::new(gm.m, env_len));
        env_pp_ptr = env_pp_storage.as_mut().unwrap();
    }
    let needs_generic_pp = make_alignment && !use_simd
        || make_alignment_display
        || simd_null2_scratch.is_none() && !null2_is_done;
    if !env_pp_ptr.is_null() && unsafe { (*env_pp_ptr).l == 0 } && needs_generic_pp {
        #[cfg(striped)]
        {
            let mut gx_fwd = Gmx::new(gm.m, env_len);
            g_forward(&sub_dsq, env_len, env_gm, &mut gx_fwd);
            let mut gx_bck = Gmx::new(gm.m, env_len);
            g_backward(&sub_dsq, env_len, env_gm, &mut gx_bck);
            g_decoding(env_gm, &gx_fwd, &gx_bck, unsafe { &mut *env_pp_ptr });
        }
        #[cfg(not(striped))]
        {
            simd_scratch.env_fwd_gmx.grow_to(gm.m, env_len);
            g_forward(&sub_dsq, env_len, env_gm, &mut simd_scratch.env_fwd_gmx);
            simd_scratch.env_bck_gmx.grow_to(gm.m, env_len);
            g_backward(&sub_dsq, env_len, env_gm, &mut simd_scratch.env_bck_gmx);
            g_decoding(
                env_gm,
                &simd_scratch.env_fwd_gmx,
                &simd_scratch.env_bck_gmx,
                unsafe { &mut *env_pp_ptr },
            );
        }
    }
    let mut dom_correction = 0.0_f32;
    if null2_is_done {
        if let Some(ref n2sc) = n2sc {
            for pos in ienv..=jenv {
                dom_correction += n2sc[pos];
            }
        }
    } else {
        let abc = Alphabet::shared(hmm.abc_type);
        if let Some(null2_arr) = simd_null2_scratch {
            add_null2_correction(
                null2_arr,
                &abc,
                dsq,
                ienv,
                jenv,
                n2sc.as_deref_mut(),
                &mut dom_correction,
            );
        } else {
            let null2_arr =
                generic_null2::null2_by_expectation(env_gm, unsafe { &*env_pp_ptr }, 1, env_len);
            add_null2_correction(&null2_arr, &abc, dsq, ienv, jenv, n2sc, &mut dom_correction);
        }
    }

    let mut oasc = 0.0_f32;
    let ad = if make_alignment {
        let mut simd_ad: Option<AliDisplay> = None;
        let env_emission_odds =
            |k: usize, x: usize| env_om.and_then(|om| om.fwd_emission_odds(k, x));

        #[cfg(striped)]
        {
            if let Some(tr) = simd_oa_trace.take() {
                // When make_alignment_display=true: use SIMD trace with env_pp for
                // PP-line annotations. Matches C p7_alidisplay_Create with SIMD OA.
                oasc = simd_oasc.unwrap_or(0.0);
                let env_pp = unsafe { &*env_pp_ptr };
                let abc = Alphabet::shared(hmm.abc_type);
                simd_ad = crate::trace::alignment_display_with_pp_emission_odds(
                    &tr,
                    &sub_dsq,
                    hmm,
                    &abc,
                    Some(env_pp),
                    pre_trim_local_bg_f.as_deref(),
                    Some(&env_emission_odds),
                )
                .map(|mut ad| {
                    ad.sqfrom += ienv - 1;
                    ad.sqto += ienv - 1;
                    AliDisplay {
                        model: ad.model,
                        mline: ad.mline,
                        aseq: ad.aseq,
                        hmmfrom: ad.hmmfrom,
                        hmmto: ad.hmmto,
                        sqfrom: ad.sqfrom,
                        sqto: ad.sqto,
                        ppline: ad.ppline,
                        rfline: ad.rfline,
                    }
                });
            } else if let Some((hmmfrom, hmmto, mut sqfrom, mut sqto)) = simd_oa_coords.take() {
                oasc = simd_oasc.unwrap_or(0.0);
                if sqfrom > 0 {
                    sqfrom += ienv - 1;
                }
                if sqto > 0 {
                    sqto += ienv - 1;
                }
                simd_ad = Some(AliDisplay {
                    model: String::new(),
                    mline: String::new(),
                    aseq: String::new(),
                    hmmfrom,
                    hmmto,
                    sqfrom,
                    sqto,
                    ppline: String::new(),
                    rfline: String::new(),
                });
            }
        }

        if simd_ad.is_some() {
            simd_ad
        } else {
            if lazy_pp {
                simd_scratch.pp_gmx.grow_to_zeroed(gm.m, env_len);
            }
            let env_pp = unsafe { &*env_pp_ptr };
            {
                simd_scratch.oa_gmx.grow_to_zeroed(gm.m, env_len);
                oasc = g_optimal_accuracy_with_deltas(
                    env_gm,
                    env_pp,
                    &mut simd_scratch.oa_gmx,
                    optacc_deltas,
                );
                let tr = g_oa_trace(env_gm, env_pp, &simd_scratch.oa_gmx);
                if make_alignment_display {
                    let abc = Alphabet::shared(hmm.abc_type);
                    crate::trace::alignment_display_with_pp_emission_odds(
                        &tr,
                        &sub_dsq,
                        hmm,
                        &abc,
                        Some(env_pp),
                        None,
                        Some(&env_emission_odds),
                    )
                    .map(|mut ad| {
                        ad.sqfrom += ienv - 1;
                        ad.sqto += ienv - 1;
                        AliDisplay {
                            model: ad.model,
                            mline: ad.mline,
                            aseq: ad.aseq,
                            hmmfrom: ad.hmmfrom,
                            hmmto: ad.hmmto,
                            sqfrom: ad.sqfrom,
                            sqto: ad.sqto,
                            ppline: ad.ppline,
                            rfline: ad.rfline,
                        }
                    })
                } else {
                    crate::trace::alignment_coords(&tr).map(
                        |(hmmfrom, hmmto, mut sqfrom, mut sqto)| {
                            if sqfrom > 0 {
                                sqfrom += ienv - 1;
                            }
                            if sqto > 0 {
                                sqto += ienv - 1;
                            }
                            AliDisplay {
                                model: String::new(),
                                mline: String::new(),
                                aseq: String::new(),
                                hmmfrom,
                                hmmto,
                                sqfrom,
                                sqto,
                                ppline: String::new(),
                                rfline: String::new(),
                            }
                        },
                    )
                }
            }
        }
    } else {
        None
    };

    let (iali, jali) = if let Some(ref a) = ad {
        (a.sqfrom as i64, a.sqto as i64)
    } else {
        (ienv as i64, jenv as i64)
    };

    // long_target envelope trim + second Forward pass for bias correction
    // (C p7_domaindef.c:2155-2216).
    //
    // If the envelope extends more than max_env_extra (=20) residues beyond
    // the alignment bounds on either side, trim it to fit and re-run Forward
    // on the trimmed region with the profile reparameterized against that
    // shorter region. This matches C's "truncate away random-match tail"
    // behavior for nhmmer.
    //
    // Then run Forward with the DEFAULT (non-reparameterized) profile to get
    // domcorrection, used for bias. Final envsc = min(reparam, default);
    // domcorrection (nats) = default - envsc.
    #[cfg(striped)]
    let (env_fwd_sc, dom_correction, env_len, ienv, jenv, dsq_offset, trim_rebuilt_ad) =
        if do_longtarget_length {
            if let Some(base_om) = env_om {
                const MAX_ENV_EXTRA: i64 = 20;
                let mut new_ienv = ienv;
                let mut new_jenv = jenv;
                let mut new_env_len = env_len;
                let mut new_dsq_offset = dsq_offset;
                let mut env_fwd_sc_current = env_fwd_sc;
                let mut trim_rebuilt_ad: Option<(crate::trace::AlignmentDisplay, f32)> = None;

                let envelope_extends =
                    (iali - ienv as i64) > MAX_ENV_EXTRA || (jenv as i64 - jali) > MAX_ENV_EXTRA;
                if envelope_extends {
                    let trimmed_ienv = ((iali - MAX_ENV_EXTRA).max(1) as usize).max(ienv);
                    let trimmed_jenv = ((jali + MAX_ENV_EXTRA).max(1) as usize).min(jenv);
                    if trimmed_jenv >= trimmed_ienv {
                        new_ienv = trimmed_ienv;
                        new_jenv = trimmed_jenv;
                        new_env_len = new_jenv - new_ienv + 1;
                        new_dsq_offset = new_ienv - 1;

                        // Reconfigure env_om for the trimmed region and re-run
                        // Forward (mirrors p7_domaindef.c:2168-2176).
                        let mut trim_om = base_om.clone();
                        trim_om.reconfig_rest_length(new_env_len as i32);
                        let trim_local_bg_f = if do_longtarget_reparam {
                            let abc_lt = crate::alphabet::Alphabet::shared(hmm.abc_type);
                            let local_bg =
                                long_target_local_bg(dsq, new_ienv, new_jenv, bg, &abc_lt, seq_len);
                            let fwd_emissions = base_om.get_fwd_emissions(bg, &abc_lt);
                            trim_om.update_fwd_emission_scores(&local_bg, &fwd_emissions, &abc_lt);
                            Some(local_bg.f)
                        } else {
                            None
                        };

                        // Match C p7_domaindef.c:2180-2200: after trim Forward, also
                        // re-run Backward, Decoding, OA, OATrace and REBUILD the
                        // AliDisplay. Without this, Rust keeps the pre-trim ad with
                        // stale hmmto/sqto that diverge from C by 1 residue.
                        let canonical_trim =
                            is_canonical_window(dsq, new_dsq_offset, new_env_len, trim_om.abc_k);
                        if make_alignment {
                            let scratch = &mut *simd_scratch;
                            scratch.fwd_pmx.resize_full(gm.m, new_env_len);
                            env_fwd_sc_current = unsafe {
                                if canonical_trim {
                                    crate::simd::fwd_filter::forward_parser_pmx_offset_direct(
                                        dsq,
                                        new_dsq_offset,
                                        new_env_len,
                                        &trim_om,
                                        &mut scratch.fwd_pmx,
                                    )
                                } else {
                                    crate::simd::fwd_filter::forward_parser_pmx_offset_with_scratch(
                                        dsq,
                                        new_dsq_offset,
                                        new_env_len,
                                        &trim_om,
                                        &mut scratch.fwd_pmx,
                                        &mut scratch.fwd_dp,
                                    )
                                }
                            };
                            scratch.bck_pmx.resize_full(gm.m, new_env_len);
                            unsafe {
                                if canonical_trim {
                                    crate::simd::bck_filter::backward_parser_pmx_offset_direct(
                                        dsq,
                                        new_dsq_offset,
                                        new_env_len,
                                        &trim_om,
                                        &mut scratch.bck_pmx,
                                        Some(&scratch.fwd_pmx.row_scale),
                                    );
                                } else {
                                    crate::simd::bck_filter::backward_parser_pmx_offset_with_scratch(
                                        dsq,
                                        new_dsq_offset,
                                        new_env_len,
                                        &trim_om,
                                        env_fwd_sc_current,
                                        &mut scratch.bck_pmx,
                                        Some(&scratch.fwd_pmx.row_scale),
                                        &mut scratch.bck_prev,
                                        &mut scratch.bck_cur,
                                    );
                                }
                            }
                            let trim_oasc = unsafe {
                                crate::simd::optacc::decode_and_fill(
                                    &scratch.fwd_pmx,
                                    &scratch.bck_pmx,
                                    &trim_om,
                                    &mut scratch.pp_pmx,
                                    &mut scratch.oa_pmx,
                                    &mut scratch.pp_cells,
                                    None,
                                )
                            };
                            let trim_tr = unsafe {
                                crate::simd::optacc::oa_trace_pmx(
                                    &trim_om,
                                    &scratch.pp_pmx,
                                    &scratch.oa_pmx,
                                )
                            };
                            scratch.trim_pp_gmx.grow_to_zeroed(gm.m, new_env_len);
                            let njc_loop_trim = [
                                trim_om.xf[crate::simd::oprofile::P7O_N]
                                    [crate::simd::oprofile::P7O_LOOP],
                                trim_om.xf[crate::simd::oprofile::P7O_J]
                                    [crate::simd::oprofile::P7O_LOOP],
                                trim_om.xf[crate::simd::oprofile::P7O_C]
                                    [crate::simd::oprofile::P7O_LOOP],
                            ];
                            crate::simd::probmx::p_decoding_to_gmx(
                                &scratch.fwd_pmx,
                                &scratch.bck_pmx,
                                gm.m,
                                njc_loop_trim,
                                &mut scratch.trim_pp_gmx,
                            );
                            let trim_sub_start = new_dsq_offset;
                            let trim_sub_end = trim_sub_start + new_env_len;
                            scratch.trim_sub_dsq.clear();
                            scratch.trim_sub_dsq.push(crate::alphabet::DSQ_SENTINEL);
                            scratch
                                .trim_sub_dsq
                                .extend_from_slice(&dsq[trim_sub_start + 1..=trim_sub_end]);
                            scratch.trim_sub_dsq.push(crate::alphabet::DSQ_SENTINEL);
                            let abc = crate::alphabet::Alphabet::shared(hmm.abc_type);
                            let trim_emission_odds =
                                |k: usize, x: usize| trim_om.fwd_emission_odds(k, x);
                            if let Some(new_ad) =
                                crate::trace::alignment_display_with_pp_emission_odds(
                                    &trim_tr,
                                    &scratch.trim_sub_dsq,
                                    hmm,
                                    &abc,
                                    Some(&scratch.trim_pp_gmx),
                                    trim_local_bg_f.as_deref(),
                                    Some(&trim_emission_odds),
                                )
                            {
                                trim_rebuilt_ad = Some((new_ad, trim_oasc));
                            }
                        } else {
                            let scratch = &mut *simd_scratch;
                            scratch.fwd_pmx.resize_full(gm.m, new_env_len);
                            env_fwd_sc_current = unsafe {
                                if canonical_trim {
                                    crate::simd::fwd_filter::forward_parser_pmx_offset_direct(
                                        dsq,
                                        new_dsq_offset,
                                        new_env_len,
                                        &trim_om,
                                        &mut scratch.fwd_pmx,
                                    )
                                } else {
                                    crate::simd::fwd_filter::forward_parser_pmx_offset_with_scratch(
                                        dsq,
                                        new_dsq_offset,
                                        new_env_len,
                                        &trim_om,
                                        &mut scratch.fwd_pmx,
                                        &mut scratch.fwd_dp,
                                    )
                                }
                            };
                        }
                    }
                }

                let (envsc_min, lt_domcorrection, _) = if do_longtarget_reparam {
                    // Default-model Forward on the (possibly trimmed) region.
                    let mut default_om = base_om.clone();
                    default_om.reconfig_rest_length(new_env_len as i32);
                    let default_fwd_sc = {
                        let scratch = &mut *simd_scratch;
                        scratch.fwd_pmx.resize_full(gm.m, new_env_len);
                        let canonical_default =
                            is_canonical_window(dsq, new_dsq_offset, new_env_len, default_om.abc_k);
                        unsafe {
                            if canonical_default {
                                crate::simd::fwd_filter::forward_parser_pmx_offset_direct(
                                    dsq,
                                    new_dsq_offset,
                                    new_env_len,
                                    &default_om,
                                    &mut scratch.fwd_pmx,
                                )
                            } else {
                                crate::simd::fwd_filter::forward_parser_pmx_offset_with_scratch(
                                    dsq,
                                    new_dsq_offset,
                                    new_env_len,
                                    &default_om,
                                    &mut scratch.fwd_pmx,
                                    &mut scratch.fwd_dp,
                                )
                            }
                        }
                    };
                    let envsc_min = env_fwd_sc_current.min(default_fwd_sc);
                    (envsc_min, default_fwd_sc - envsc_min, default_fwd_sc)
                } else {
                    (env_fwd_sc_current, 0.0, env_fwd_sc_current)
                };
                (
                    envsc_min,
                    lt_domcorrection,
                    new_env_len,
                    new_ienv,
                    new_jenv,
                    new_dsq_offset,
                    trim_rebuilt_ad,
                )
            } else {
                (
                    env_fwd_sc,
                    dom_correction,
                    env_len,
                    ienv,
                    jenv,
                    dsq_offset,
                    None,
                )
            }
        } else {
            (
                env_fwd_sc,
                dom_correction,
                env_len,
                ienv,
                jenv,
                dsq_offset,
                None,
            )
        };
    // If trim rebuilt the ad, replace ad/iali/jali/oasc with post-trim values.
    #[cfg(striped)]
    let (ad, iali, jali, oasc) = if let Some((new_ad, new_oasc)) = trim_rebuilt_ad {
        // new_ad.sqfrom/sqto are 1-based in the trim sub_dsq. Convert to
        // absolute coords via the (post-trim) dsq_offset.
        let abs_sqfrom = new_ad.sqfrom + dsq_offset;
        let abs_sqto = new_ad.sqto + dsq_offset;
        let new_iali = abs_sqfrom as i64;
        let new_jali = abs_sqto as i64;
        let new_ad_abs = AliDisplay {
            model: new_ad.model,
            mline: new_ad.mline,
            aseq: new_ad.aseq,
            hmmfrom: new_ad.hmmfrom,
            hmmto: new_ad.hmmto,
            sqfrom: abs_sqfrom,
            sqto: abs_sqto,
            ppline: new_ad.ppline,
            rfline: new_ad.rfline,
        };
        (Some(new_ad_abs), new_iali, new_jali, new_oasc)
    } else {
        (ad, iali, jali, oasc)
    };
    #[cfg(not(striped))]
    let (env_fwd_sc, dom_correction) = (env_fwd_sc, dom_correction);

    // Omega-weighted null2 bias.
    let omega = bg.omega;
    let dom_bias = crate::logsum::p7_flogsum(0.0, c_log_f64(omega as f64) as f32 + dom_correction);

    // Domain bitscore:
    // * long_target: use C nhmmer's post-Viterbi long-target rescaling
    //   (hmmer/src/p7_pipeline.c:1390-1420). Shift entry/exit costs from env_len
    //   to max_length, adjust extension term, then bits = (bitscore - nullsc)/log(2).
    //   The dom_bias is NOT subtracted here; it's reported separately as dom.dombias.
    // * otherwise: hmmsearch-style formula (matches C rescore_isolated_domain).
    let dom_bitscore = if long_target {
        // Mirror C p7_pli_postViterbi_LongTarget (hmmer/src/p7_pipeline.c:1382-1416)
        // exactly. Keep the inner ratios in f32 where C uses `(float)` casts,
        // then promote to f64 for log and the running bitscore accumulator.
        let env_len_i = env_len as i32;
        // C uses the raw `dom->jali - dom->iali + 1` (p7_pipeline.c:1383), with
        // no clamp; its `if (ali_len < 8) continue` guard (p7_pipeline.c:1387)
        // ensures ali_len >= 8 by the time it reaches this rescaling.
        let ali_len_i = (jali - iali + 1) as i32;
        let max_length_i = if hmm.max_length > 0 {
            hmm.max_length
        } else {
            env_len_i
        };
        let mut bitscore = env_fwd_sc as f64;
        // (1) entrance/exit costs: env_len → max_length.
        // C: `2 * log(2. / (env_len+2))` — `2.` is double, (env_len+2) is int →
        // division yields double.
        bitscore -= 2.0 * c_log_f64(2.0_f64 / (env_len_i + 2) as f64);
        bitscore += 2.0 * c_log_f64(2.0_f64 / (max_length_i + 2) as f64);
        // (2) extension cost: replace env-bounds extension with max-bounds one.
        // C: `log((float)env_len / (env_len+2))` — ratio computed in f32, then
        // promoted to f64 for log. Multiplier `(env_len-ali_len)` is int.
        let env_ratio = (env_len_i as f32) / ((env_len_i + 2) as f32);
        bitscore -= (env_len_i - ali_len_i) as f64 * c_log_f64(env_ratio as f64);
        let max_len_or_env_i = max_length_i.max(env_len_i);
        // C: `log((float)om->max_length / (float)(om->max_length+2))`.
        let max_ratio = (max_length_i as f32) / ((max_length_i + 2) as f32);
        bitscore += (max_len_or_env_i - ali_len_i) as f64 * c_log_f64(max_ratio as f64);
        // Null score at max(max_length, env_len).
        let nsc_length = max_len_or_env_i as usize;
        let mut bg_local = bg.clone();
        bg_local.set_length(nsc_length);
        let nullsc = bg_local.null_one(nsc_length) as f64;
        ((bitscore - nullsc) / ESL_CONST_LOG2) as f32
    } else {
        // C p7_pipeline.c computes:
        //   bitscore = envsc + (sq->n-Ld) * log((float)sq->n / (float)(sq->n+3))
        // The ratio is rounded to float before log(), but log() itself is the
        // double-precision libc call. Preserve that ordering here.
        let ratio = (l as f32) / ((l + 3) as f32);
        let length_correction = (l - env_len) as f64 * c_log_f64(ratio as f64);
        let bitscore = (env_fwd_sc as f64 + length_correction) as f32;
        ((bitscore - (null_sc + dom_bias)) as f64 / ESL_CONST_LOG2) as f32
    };
    let tau = gm.evparam[crate::hmm::P7_FTAU] as f64;
    let lambda = gm.evparam[crate::hmm::P7_FLAMBDA] as f64;
    let dom_lnp = crate::stats::exponential::logsurv(dom_bitscore as f64, tau, lambda);

    // C p7_pipeline.c:1455 sets `hit->dcl[0].dombias = dom_bias` where
    // dom_bias = dom->domcorrection (raw nats). For the hmmsearch path the
    // existing convention is to store dombias in bits. Keep both paths
    // consistent by storing in bits.
    let domain_dombias = if long_target {
        dom_correction * (ESL_CONST_LOG2R as f32)
    } else {
        dom_bias * (ESL_CONST_LOG2R as f32)
    };
    let domain = Domain {
        iali,
        jali,
        ienv: ienv as i64,
        jenv: jenv as i64,
        bitscore: dom_bitscore,
        lnp: dom_lnp,
        // C prints dombias unclamped (p7_tophits.c:1680/1764), so keep the raw
        // (possibly slightly negative) value rather than flooring at 0.0.
        dombias: domain_dombias,
        oasc,
        envsc: env_fwd_sc,
        domcorrection: dom_correction,
        is_reported: false,
        is_included: false,
        ad,
    };

    domain
}

/// Define domains in a sequence using posterior probability heuristics.
/// Port of C `p7_domaindef_ByPosteriorHeuristics()` in `p7_domaindef.c`.
///
/// Pipeline:
///   1. Domain decoding produces per-position btot/etot/mocc arrays
///      (SIMD `p_domain_decoding_reuse` when available; generic
///      `domain_decoding` otherwise).
///   2. Region detection via `find_domain_regions` with `RT1`/`RT2`.
///   3. For each region, `is_multidomain_region` (threshold `RT3`)
///      decides between a direct single-envelope rescore and a
///      stochastic-traceback clustering pass (`NSAMPLES` samples,
///      `spensemble::cluster`).
///   4. Each envelope is scored and aligned via `score_domain_envelope`.
///
/// Caller may supply a pre-computed Forward parser matrix and score in
/// `fwd_pmx_input`/`fwd_sc_input`; otherwise they are recomputed. The
/// `long_target` flag enables the nhmmer-specific code paths inside
/// envelope rescoring. `seed` initializes the stochastic-trace RNG for
/// reproducibility.
///
/// Returns `(domains, nexpected, seq_bias_simple_nats,
/// seq_bias_pipeline_nats, stats)`; the two seq-bias values use the
/// C-equivalent sequential sum and Easel's Kahan-compensated sum
/// respectively.
#[allow(clippy::too_many_arguments)]
pub(crate) fn define_domains(
    dsq: &[Dsq],
    l: usize,
    gm: &Profile,
    om: Option<&crate::simd::oprofile::OProfile>,
    fwd_pmx_input: Option<&crate::simd::probmx::ProbMx>,
    fwd_sc_input: Option<f32>,
    hmm: &Hmm,
    bg: &Bg,
    null_sc: f32,
    seed: u32,
    #[cfg(striped)] simd_scratch: &mut DomainSimdScratch,
    make_alignment: bool,
    make_alignment_display: bool,
    long_target: bool,
    do_null2: bool,
) -> (Vec<Domain>, f32, f32, f32, DomainDefinitionStats) {
    crate::logsum::p7_flogsuminit();
    #[cfg(not(striped))]
    let mut simd_scratch_storage = DomainSimdScratch::new();
    #[cfg(not(striped))]
    let simd_scratch = &mut simd_scratch_storage;

    // Phase 1: Domain decoding (btot/etot/mocc) — SIMD when available
    let (mut btot, mut etot, mut mocc);

    let use_simd = cfg!(striped);
    if use_simd {
        #[cfg(striped)]
        {
            use crate::simd::oprofile::*;
            use crate::simd::probmx::p_domain_decoding_reuse;

            let om_storage;
            let om = if let Some(om) = om {
                om
            } else {
                om_storage = OProfile::convert(gm);
                &om_storage
            };
            let mut fwd_pmx_storage = None;
            let (fwd_pmx_ref, fwd_sc) =
                if let (Some(fwd_pmx), Some(fwd_sc)) = (fwd_pmx_input, fwd_sc_input) {
                    (fwd_pmx, fwd_sc)
                } else {
                    let mut fwd_pmx = crate::simd::probmx::ProbMx::new(l);
                    let fwd_sc = unsafe {
                        crate::simd::fwd_filter::forward_parser_pmx_offset_with_scratch(
                            dsq,
                            0,
                            l,
                            om,
                            &mut fwd_pmx,
                            &mut simd_scratch.fwd_dp,
                        )
                    };
                    let fwd_pmx_ref = fwd_pmx_storage.insert(fwd_pmx);
                    (&*fwd_pmx_ref, fwd_sc)
                };
            simd_scratch.bck_pmx.resize_parser(l);
            let bck_pmx = &mut simd_scratch.bck_pmx;
            unsafe {
                crate::simd::bck_filter::backward_parser_pmx_offset_with_scratch(
                    dsq,
                    0,
                    l,
                    om,
                    fwd_sc,
                    bck_pmx,
                    Some(&fwd_pmx_ref.row_scale),
                    &mut simd_scratch.bck_prev,
                    &mut simd_scratch.bck_cur,
                );
            };
            let njc_loop = [
                om.xf[P7O_N][P7O_LOOP],
                om.xf[P7O_J][P7O_LOOP],
                om.xf[P7O_C][P7O_LOOP],
            ];
            btot = std::mem::take(&mut simd_scratch.btot);
            etot = std::mem::take(&mut simd_scratch.etot);
            mocc = std::mem::take(&mut simd_scratch.mocc);
            p_domain_decoding_reuse(
                fwd_pmx_ref,
                bck_pmx,
                gm.m,
                l,
                njc_loop,
                &mut btot,
                &mut etot,
                &mut mocc,
            );
        }
        #[cfg(not(striped))]
        unreachable!();
    } else {
        #[cfg(striped)]
        {
            let mut gx_fwd = Gmx::new(gm.m, l);
            g_forward(dsq, l, gm, &mut gx_fwd);
            let mut gx_bck = Gmx::new(gm.m, l);
            g_backward(dsq, l, gm, &mut gx_bck);
            let r = domain_decoding(gm, &gx_fwd, &gx_bck);
            btot = r.0;
            etot = r.1;
            mocc = r.2;
        }
        #[cfg(not(striped))]
        {
            simd_scratch.fwd_gmx.grow_to(gm.m, l);
            g_forward(dsq, l, gm, &mut simd_scratch.fwd_gmx);
            simd_scratch.bck_gmx.grow_to(gm.m, l);
            g_backward(dsq, l, gm, &mut simd_scratch.bck_gmx);
            let r = domain_decoding(gm, &simd_scratch.fwd_gmx, &simd_scratch.bck_gmx);
            btot = r.0;
            etot = r.1;
            mocc = r.2;
        }
    }

    let nexpected = btot[l];

    // Region detection using C's state machine
    let regions = find_domain_regions(&btot, &etot, &mocc, l);
    let region_multi: Vec<bool> = regions
        .iter()
        .map(|&(ri, rj)| is_multidomain_region(&btot, &etot, ri, rj))
        .collect();
    #[cfg(striped)]
    if use_simd {
        simd_scratch.btot = btot;
        simd_scratch.etot = etot;
        simd_scratch.mocc = mocc;
    }
    let mut stats = DomainDefinitionStats {
        nregions: regions.len(),
        ..Default::default()
    };
    let mut n2sc = vec![0.0_f32; l + 1];
    let abc = Alphabet::shared(hmm.abc_type);
    let mut env_gm = gm.clone();
    reconfig_unihit(&mut env_gm, l as i32);
    let optacc_deltas = OptAccTDelta::from_profile(&env_gm);
    #[cfg(striped)]
    let simd_match_odds: Option<Vec<f32>> = None;
    #[cfg(not(striped))]
    let simd_match_odds: Option<Vec<f32>> = None;
    #[cfg(striped)]
    let env_om = if use_simd {
        if let Some(om) = om {
            let mut env_om = om.clone();
            env_om.reconfig_unihit(l as i32);
            Some(env_om)
        } else {
            Some(crate::simd::oprofile::OProfile::convert(&env_gm))
        }
    } else {
        None
    };
    #[cfg(not(striped))]
    let env_om: Option<crate::simd::oprofile::OProfile> = None;

    if regions.is_empty() {
        // No regions found — matches C: the region state machine never triggers,
        // so nregions/nenvelopes/ndom stay 0 and no domain is created.
        return (Vec::new(), nexpected, 0.0, 0.0, stats);
    }

    let mut domains = Vec::new();

    for (region_idx, (&(ri, rj), &is_multi)) in regions.iter().zip(region_multi.iter()).enumerate()
    {
        let _ = region_idx;
        if is_multi {
            stats.nclustered += 1;
            // Multi-domain region: resolve by stochastic traceback clustering
            // Run Forward on the region in multihit mode
            let region_len = rj - ri + 1;
            let mut region_gm = gm.clone();
            reconfig_multihit(&mut region_gm, l as i32);

            let mut sub_dsq_storage: Option<Vec<Dsq>> = None;
            let mut region_fwd_storage: Option<Gmx> = None;

            #[cfg(striped)]
            let region_trace_pmx = if use_simd {
                if let Some(om) = om {
                    let mut region_om = om.clone();
                    region_om.reconfig_multihit(l as i32);
                    let mut region_pmx = simd_scratch
                        .region_pmx
                        .take()
                        .unwrap_or_else(|| crate::simd::probmx::ProbMx::new_full(gm.m, region_len));
                    region_pmx.resize_full(gm.m, region_len);
                    unsafe {
                        crate::simd::fwd_filter::forward_parser_pmx_offset_with_scratch(
                            dsq,
                            ri - 1,
                            region_len,
                            &region_om,
                            &mut region_pmx,
                            &mut simd_scratch.fwd_dp,
                        );
                    }
                    Some((region_om, region_pmx))
                } else {
                    None
                }
            } else {
                None
            };

            let mut rng = MersenneTwister::new(seed);
            let mut segments = Vec::new();
            #[cfg(striped)]
            let mut trace_pmx_scratch = Trace::new();
            for value in n2sc.iter_mut().take(rj + 1).skip(ri) {
                *value = 0.0;
            }

            for trace_idx in 0..NSAMPLES {
                #[cfg(striped)]
                let tr_storage;
                #[cfg(striped)]
                let tr = if let Some((ref region_om, ref region_pmx)) = region_trace_pmx {
                    crate::dp::generic_stotrace::stochastic_trace_pmx_into(
                        &mut rng,
                        region_len,
                        region_om,
                        region_pmx,
                        &mut trace_pmx_scratch,
                    );
                    &trace_pmx_scratch
                } else {
                    if sub_dsq_storage.is_none() {
                        let mut sub_dsq = Vec::with_capacity(region_len + 2);
                        sub_dsq.push(crate::alphabet::DSQ_SENTINEL);
                        sub_dsq.extend_from_slice(&dsq[ri..=rj]);
                        sub_dsq.push(crate::alphabet::DSQ_SENTINEL);
                        sub_dsq_storage = Some(sub_dsq);
                    }
                    let sub_dsq = sub_dsq_storage.as_ref().unwrap();
                    if region_fwd_storage.is_none() {
                        let mut region_fwd = Gmx::new(gm.m, region_len);
                        g_forward(sub_dsq, region_len, &region_gm, &mut region_fwd);
                        region_fwd_storage = Some(region_fwd);
                    }
                    tr_storage = g_stochastic_trace(
                        &mut rng,
                        sub_dsq,
                        region_len,
                        &region_gm,
                        region_fwd_storage.as_ref().unwrap(),
                    );
                    &tr_storage
                };
                #[cfg(not(striped))]
                let tr_storage;
                #[cfg(not(striped))]
                let tr = {
                    if sub_dsq_storage.is_none() {
                        let mut sub_dsq = Vec::with_capacity(region_len + 2);
                        sub_dsq.push(crate::alphabet::DSQ_SENTINEL);
                        sub_dsq.extend_from_slice(&dsq[ri..=rj]);
                        sub_dsq.push(crate::alphabet::DSQ_SENTINEL);
                        sub_dsq_storage = Some(sub_dsq);
                    }
                    let sub_dsq = sub_dsq_storage.as_ref().unwrap();
                    if region_fwd_storage.is_none() {
                        let mut region_fwd = Gmx::new(gm.m, region_len);
                        g_forward(sub_dsq, region_len, &region_gm, &mut region_fwd);
                        region_fwd_storage = Some(region_fwd);
                    }
                    tr_storage = g_stochastic_trace(
                        &mut rng,
                        sub_dsq,
                        region_len,
                        &region_gm,
                        region_fwd_storage.as_ref().unwrap(),
                    );
                    &tr_storage
                };
                let mut in_domain = false;
                let mut seg_i = 0;
                let mut seg_k = 0;
                let mut seg_j = 0;
                let mut seg_m = 0;
                let mut zstart = 0usize;
                let mut trace_domains = Vec::new();

                for z in 0..tr.n {
                    match tr.st[z] {
                        State::B => {
                            in_domain = true;
                            seg_i = 0;
                            seg_k = 0;
                            zstart = z;
                        }
                        State::M if in_domain => {
                            if seg_i == 0 {
                                seg_i = tr.i[z];
                                seg_k = tr.k[z];
                            }
                            seg_j = tr.i[z];
                            seg_m = tr.k[z];
                        }
                        State::E if in_domain => {
                            if seg_i > 0 && seg_j > 0 {
                                let global_i = seg_i + ri - 1;
                                let global_j = seg_j + ri - 1;
                                segments.push(SegmentPair {
                                    i: global_i,
                                    j: global_j,
                                    k: seg_k,
                                    m: seg_m,
                                    trace_idx,
                                });
                                trace_domains.push((seg_i, seg_j, zstart, z));
                            }
                            in_domain = false;
                        }
                        _ => {}
                    }
                }

                let mut pos = 1usize;
                for (domain_idx, &(sqfrom, sqto, tfrom, tto)) in trace_domains.iter().enumerate() {
                    let _ = domain_idx;
                    #[cfg(striped)]
                    let null2 = if let Some((ref region_om, _)) = region_trace_pmx {
                        null2_by_trace_optimized(region_om, tr, tfrom, tto)
                    } else {
                        generic_null2::null2_by_trace(&region_gm, tr, tfrom, tto)
                    };
                    #[cfg(not(striped))]
                    let null2 = generic_null2::null2_by_trace(&region_gm, tr, tfrom, tto);
                    while pos <= sqfrom && pos <= region_len {
                        n2sc[ri + pos - 1] += 1.0;
                        pos += 1;
                    }
                    while pos <= sqto && pos <= region_len {
                        let x = dsq[ri + pos - 1] as usize;
                        n2sc[ri + pos - 1] += null2_ratio_for_code(&null2, &abc, x);
                        pos += 1;
                    }
                }
                while pos <= region_len {
                    n2sc[ri + pos - 1] += 1.0;
                    pos += 1;
                }
            }

            for value in n2sc.iter_mut().take(rj + 1).skip(ri) {
                *value = c_logf_to_f32(*value / NSAMPLES as f32);
            }

            if !segments.is_empty() {
                let params = ClusterParams::default();
                let envelopes = spensemble::cluster(&segments, NSAMPLES, &params);
                let mut last_jenv = 0usize;
                for (env_idx, env) in envelopes.iter().enumerate() {
                    let ienv = env.ienv.max(1).min(l);
                    let jenv = env.jenv.max(1).min(l);
                    if jenv >= ienv {
                        let _ = env_idx;
                        if ienv <= last_jenv {
                            stats.noverlaps += 1;
                        }
                        stats.nenvelopes += 1;
                        domains.push(score_domain_envelope(
                            dsq,
                            l,
                            gm,
                            &env_gm,
                            env_om.as_ref(),
                            hmm,
                            ienv,
                            jenv,
                            null_sc,
                            bg,
                            Some(&mut n2sc),
                            true,
                            simd_match_odds.as_deref(),
                            &optacc_deltas,
                            &mut *simd_scratch,
                            make_alignment,
                            make_alignment_display,
                            long_target,
                            do_null2,
                        ));
                        last_jenv = jenv;
                    }
                }
            } else {
                // Clustering failed — treat as single domain
                stats.nenvelopes += 1;
                domains.push(score_domain_envelope(
                    dsq,
                    l,
                    gm,
                    &env_gm,
                    env_om.as_ref(),
                    hmm,
                    ri,
                    rj,
                    null_sc,
                    bg,
                    Some(&mut n2sc),
                    false,
                    simd_match_odds.as_deref(),
                    &optacc_deltas,
                    &mut *simd_scratch,
                    make_alignment,
                    make_alignment_display,
                    long_target,
                    do_null2,
                ));
            }
            #[cfg(striped)]
            if let Some((_, region_pmx)) = region_trace_pmx {
                simd_scratch.region_pmx = Some(region_pmx);
            }
        } else {
            // Single-domain region: the region IS the envelope
            stats.nenvelopes += 1;
            domains.push(score_domain_envelope(
                dsq,
                l,
                gm,
                &env_gm,
                env_om.as_ref(),
                hmm,
                ri,
                rj,
                null_sc,
                bg,
                Some(&mut n2sc),
                false,
                simd_match_odds.as_deref(),
                &optacc_deltas,
                &mut *simd_scratch,
                make_alignment,
                make_alignment_display,
                long_target,
                do_null2,
            ));
        }
    }

    // Sort domains by envelope start position
    domains.sort_by_key(|d| d.ienv);

    let seq_bias = c_fsum(&n2sc);
    let seq_bias_fsum = esl_vec_fsum(&n2sc);
    (domains, nexpected, seq_bias, seq_bias_fsum, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::AlphabetType;

    #[test]
    fn long_target_local_bg_distributes_degenerates_and_normalizes() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let bg = Bg::new(&abc);
        let dsq = abc.digitize(b"NNNN");

        let local_bg = long_target_local_bg(&dsq, 1, 4, &bg, &abc, 4);
        let sum: f32 = local_bg.f.iter().take(abc.k).sum();

        assert!((sum - 1.0).abs() < 1e-6);
        for f in local_bg.f.iter().take(abc.k) {
            assert!(*f > 0.0);
        }
    }

    #[test]
    fn long_target_local_bg_falls_back_when_window_has_no_residue_evidence() {
        let abc = Alphabet::new(AlphabetType::Dna);
        let bg = Bg::new(&abc);
        let dsq = vec![crate::alphabet::DSQ_SENTINEL; 4];

        let local_bg = long_target_local_bg(&dsq, 1, 2, &bg, &abc, 2);

        for i in 0..abc.k {
            assert_eq!(local_bg.f[i].to_bits(), bg.f[i].to_bits());
        }
    }
}
