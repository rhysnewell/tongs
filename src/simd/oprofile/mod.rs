//! P7_OPROFILE - SSE2-optimized scoring profile.
//! Direct port of impl_sse/p7_oprofile.c.
#![allow(
    clippy::approx_constant,
    clippy::erasing_op,
    clippy::identity_op,
    clippy::needless_range_loop
)]

use crate::profile::*;
#[cfg(target_arch = "x86_64")]
use crate::util::cmath::c_log_to_f32;
use crate::util::cmath::{c_expf_to_f32, c_logf_to_f32, ESL_CONST_LOG2};

/// Striped segment length for byte-precision (MSV) vectors: ceil(M/16), min 2.
/// Mirrors C macro `p7O_NQB`.
pub fn nqb(m: usize) -> usize {
    2.max(((m.max(1) - 1) / 16) + 1)
}

/// Striped segment length for word-precision (Viterbi) vectors: ceil(M/8), min 2.
/// Mirrors C macro `p7O_NQW`.
pub fn nqw(m: usize) -> usize {
    2.max(((m.max(1) - 1) / 8) + 1)
}

/// Striped segment length for float-precision (Forward) vectors: ceil(M/4), min 2.
/// Mirrors C macro `p7O_NQF`.
pub fn nqf(m: usize) -> usize {
    2.max(((m.max(1) - 1) / 4) + 1)
}

// Viterbi transition order in twv
pub const P7O_BM: usize = 0;
pub const P7O_MM: usize = 1;
pub const P7O_IM: usize = 2;
pub const P7O_DM: usize = 3;
pub const P7O_MD: usize = 4;
pub const P7O_MI: usize = 5;
pub const P7O_II: usize = 6;
pub const P7O_DD: usize = 7;

// Special states
pub const P7O_E: usize = 0;
pub const P7O_N: usize = 1;
pub const P7O_J: usize = 2;
pub const P7O_C: usize = 3;
pub const P7O_NXSTATES: usize = 4;
pub const P7O_MOVE: usize = 0;
pub const P7O_LOOP: usize = 1;
pub const P7O_NXTRANS: usize = 2;

/// 16-byte aligned wrapper around `[f32; 4]` for use with SSE `_mm_load_ps`.
#[cfg(target_arch = "x86_64")]
#[repr(align(16))]
#[derive(Debug, Clone, Copy)]
pub struct AlignedF32x4(pub [f32; 4]);

#[cfg(target_arch = "x86_64")]
impl AlignedF32x4 {
    /// Wrap a `[f32; 4]` into an aligned lane container.
    #[inline]
    pub fn from_array(v: [f32; 4]) -> Self {
        Self(v)
    }

    /// Return a `*const f32` to the first lane (16-byte aligned).
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.0.as_ptr()
    }
}

/// Optimized profile for SSE2 SIMD operations.
#[derive(Debug, Clone)]
pub struct OProfile {
    // === MSV byte-precision fields ===
    /// MSV byte-precision match scores: `rbv[residue_code][q_vector][16 bytes]`
    pub rbv: Vec<Vec<[u8; 16]>>,
    /// SSV signed-byte match scores, with extra wrapped vectors for band code.
    pub sbv: Vec<Vec<[u8; 16]>>,
    pub ssv: Option<std::sync::Arc<crate::simd::ssv::SsvProfile>>,
    pub tbm_b: u8,
    pub tec_b: u8,
    pub tjb_b: u8,
    pub scale_b: f32,
    pub base_b: u8,
    pub bias_b: u8,

    // === Viterbi word-precision fields ===
    /// Word-precision match emission scores: `rwv[residue_code][q_vector][8 words]`
    pub rwv: Vec<Vec<[i16; 8]>>,
    /// Word-precision transition scores: twv[j][8 words]
    /// Layout: for each q, 7 transitions (BM,MM,IM,DM,MD,MI,II), then DD at end
    pub twv: Vec<[i16; 8]>,
    /// Special state word scores: `xw[state][transition]`
    pub xw: [[i16; P7O_NXTRANS]; P7O_NXSTATES],
    pub scale_w: f32,
    pub base_w: i16,
    pub ddbound_w: i16,
    pub ncj_roundoff: f32,

    // === Forward/Backward float-precision fields ===
    /// Float-precision emission scores (probability ratios): `rfv[residue][q_vector][4 floats]`
    pub rfv: Vec<Vec<[f32; 4]>>,
    /// Float-precision transition scores: tfv[j][4 floats]
    pub tfv: Vec<[f32; 4]>,
    #[cfg(target_arch = "x86_64")]
    pub rfv_a: Vec<Vec<AlignedF32x4>>,
    #[cfg(target_arch = "x86_64")]
    pub tfv_a: Vec<AlignedF32x4>,
    /// Special state float scores: `xf[state][transition]`
    pub xf: [[f32; P7O_NXTRANS]; P7O_NXSTATES],

    // === Common fields ===
    pub m: usize,
    pub l: i32,
    /// Upper bound on emitted sequence length (HMM `MAXL`); written as the
    /// `.h3f` `max_length` field. Mirrors C `om->max_length`, copied from
    /// `gm->max_length` in `p7_oprofile_Convert` (p7_oprofile.c:1097). `-1`
    /// when the model has no MAXL line.
    pub max_length: i32,
    pub nj: f32,
    pub mode: i32,
    pub evparam: [f32; 6],
    pub cutoff: [f32; 6],
    pub compo: [f32; 20],
    pub name: String,
    pub abc_k: usize,
    pub abc_kp: usize,
}

/// Convert a float log-odds score to a biased MSV byte (`u8`).
///
/// Matches C `biased_byteify`: `result = (uint8_t)(-round(scale_b * sc)) + bias_b`.
/// For positive `sc` the negated rounded value is negative; unsigned wrapping
/// then yields the correct offset byte. Saturates to 255.
mod convert;
use convert::*;

impl OProfile {
    /// Convert a standard `Profile` into the SSE-striped optimized profile.
    ///
    /// Ports C `p7_oprofile_Convert()`: runs the MSV (byte), Viterbi (word),
    /// and Forward/Backward (float) striping conversions in one pass.
    /// Requires `gm.m <= allocM` and matching alphabet. Sets `mode`, `L`, `M`,
    /// `nj`, `evparam`, `cutoff`, `compo`, and copies the model name.
    pub fn convert(gm: &Profile) -> Self {
        let m = gm.m;
        let kp = gm.abc_kp;
        let k = gm.abc_k;

        let mf = mf_conversion(gm);
        let vf = vf_conversion(gm);
        let fb = fb_conversion(gm);

        #[cfg(target_arch = "x86_64")]
        let rfv_a: Vec<Vec<AlignedF32x4>> = fb
            .rfv
            .iter()
            .map(|rows| rows.iter().copied().map(AlignedF32x4::from_array).collect())
            .collect();
        #[cfg(target_arch = "x86_64")]
        let tfv_a: Vec<AlignedF32x4> = fb
            .tfv
            .iter()
            .copied()
            .map(AlignedF32x4::from_array)
            .collect();

        let ssv = crate::simd::ssv::SsvProfile::new(&mf.sbv, m);
        let om = OProfile {
            rbv: mf.rbv,
            sbv: mf.sbv,
            ssv,
            tbm_b: mf.tbm_b,
            tec_b: mf.tec_b,
            tjb_b: mf.tjb_b,
            scale_b: mf.scale_b,
            base_b: mf.base_b,
            bias_b: mf.bias_b,
            rwv: vf.rwv,
            twv: vf.twv,
            xw: vf.xw,
            scale_w: vf.scale_w,
            base_w: vf.base_w,
            ddbound_w: vf.ddbound_w,
            ncj_roundoff: 0.0,
            rfv: fb.rfv,
            tfv: fb.tfv,
            #[cfg(target_arch = "x86_64")]
            rfv_a,
            #[cfg(target_arch = "x86_64")]
            tfv_a,
            xf: fb.xf,
            m,
            l: gm.l,
            max_length: gm.max_length,
            nj: gm.nj,
            mode: gm.mode,
            evparam: gm.evparam,
            cutoff: gm.cutoff,
            compo: gm.compo,
            name: gm.name.clone(),
            abc_k: k,
            abc_kp: kp,
        };

        om
    }

    /// Set the target sequence length of the MSVFilter part of the model.
    ///
    /// Ports `p7_oprofile_ReconfigMSVLength`: recomputes only `tjb_b` for a
    /// new mean target length `l`. The acceleration pipeline (and nhmmer's
    /// long-target SSV scan) uses this to defer reconfiguring the rest of the
    /// length model until after the MSV stage.
    pub fn reconfig_msv_length(&mut self, l: i32) {
        self.tjb_b = unbiased_byteify(self.scale_b, c_logf_to_f32(3.0_f32 / (l as f32 + 3.0)));
    }

    /// Set the target sequence length of the optimized model.
    ///
    /// Ports `p7_oprofile_ReconfigLength` (combines MSV + "rest" length
    /// reconfigs): updates `tjb_b`, N/C/J transition floats `xf` and words
    /// `xw` for a new mean target length `l`. Does not touch the null model
    /// (caller must also `p7_bg_SetLength`). Must be fast — sits in the hot
    /// path, called once per target sequence.
    pub fn reconfig_length(&mut self, l: i32) {
        self.reconfig_msv_length(l);
        self.reconfig_rest_length(l);
    }

    /// Set the target sequence length of the main profile, excluding MSVFilter.
    ///
    /// Ports `p7_oprofile_ReconfigRestLength`: updates N/C/J transition floats
    /// and words for Forward/Viterbi filters without touching `tjb_b`.
    pub fn reconfig_rest_length(&mut self, l: i32) {
        let pmove = (2.0 + self.nj) / (l as f32 + 2.0 + self.nj);
        let ploop = 1.0 - pmove;

        let log_pmove = c_logf_to_f32(pmove);
        self.xw[P7O_N][P7O_MOVE] = wordify(self.scale_w, log_pmove);
        self.xw[P7O_C][P7O_MOVE] = wordify(self.scale_w, log_pmove);
        self.xw[P7O_J][P7O_MOVE] = wordify(self.scale_w, log_pmove);

        self.xf[P7O_N][P7O_LOOP] = ploop;
        self.xf[P7O_N][P7O_MOVE] = pmove;
        self.xf[P7O_C][P7O_LOOP] = ploop;
        self.xf[P7O_C][P7O_MOVE] = pmove;
        self.xf[P7O_J][P7O_LOOP] = ploop;
        self.xf[P7O_J][P7O_MOVE] = pmove;

        self.l = l;
    }

    /// Reconfigure the optimized profile into multihit mode for target length `l`.
    ///
    /// Ports `p7_oprofile_ReconfigMultihit`: sets E->{move,loop}=0.5 and
    /// `nj=1`, mirroring the unihit/multihit flip done during domain
    /// definition. Calls `reconfig_length(l)` to refresh the length model.
    /// Avoids rebuilding through generic log-space scores so low-bit parity
    /// with the original is preserved.
    pub fn reconfig_multihit(&mut self, l: i32) {
        self.xf[P7O_E][P7O_MOVE] = 0.5;
        self.xf[P7O_E][P7O_LOOP] = 0.5;
        self.nj = 1.0;

        let e = -(ESL_CONST_LOG2 as f32);
        self.xw[P7O_E][P7O_MOVE] = wordify(self.scale_w, e);
        self.xw[P7O_E][P7O_LOOP] = wordify(self.scale_w, e);

        self.reconfig_length(l);
    }

    /// Reconfigure the optimized profile into unihit mode for target length `l`.
    ///
    /// Ports `p7_oprofile_ReconfigUnihit`: sets E->move=1, E->loop=0, `nj=0`,
    /// and refreshes the length model via `reconfig_length(l)`. Used by domain
    /// definition to score isolated single-domain envelopes.
    pub fn reconfig_unihit(&mut self, l: i32) {
        self.xf[P7O_E][P7O_MOVE] = 1.0;
        self.xf[P7O_E][P7O_LOOP] = 0.0;
        self.nj = 0.0;

        self.xw[P7O_E][P7O_MOVE] = 0;
        self.xw[P7O_E][P7O_LOOP] = i16::MIN;

        self.reconfig_length(l);
    }

    /// Retrieve Forward (float) residue emission values into a flat array.
    ///
    /// Ports `p7_oprofile_GetFwdEmissionArray`: extracts an implicit 2D
    /// `(M+1) * Kp` table from the striped/interleaved `rfv`, converting back
    /// to emission values weighted by background. Canonical residues:
    /// `arr[k * Kp + x] = rfv[x][q][z] * bg.f[x]`. Degenerate / ambiguity
    /// codes are filled via `esl_abc_FExpectScVec`-style expectation over
    /// canonical residues weighted by `bg.f`.
    pub fn get_fwd_emissions(
        &self,
        bg: &crate::bg::Bg,
        abc: &crate::alphabet::Alphabet,
    ) -> Vec<f32> {
        let m = self.m;
        let kp = self.abc_kp;
        let k = abc.k;
        let nq = nqf(m);
        let cell_cnt = (m + 1) * kp;
        let mut arr = vec![0.0_f32; cell_cnt];

        for x in 0..k {
            for q in 0..nq {
                let lanes = self.rfv[x][q];
                for z in 0..4 {
                    let node = q + 1 + z * nq;
                    let idx = kp * node + x;
                    if idx < cell_cnt {
                        arr[idx] = lanes[z] * bg.f[x];
                    }
                }
            }
        }

        // Degenerate residue expectations: for each model position, fill in the
        // ambiguity columns as the bg.f-weighted mean of the matching canonical
        // values. Mirrors esl_abc_FExpectScVec over raw probabilities.
        for pos in 0..=m {
            let row = &mut arr[pos * kp..(pos + 1) * kp];
            for x in (k + 1)..=kp.saturating_sub(3) {
                if !abc.is_residue(x as u8) {
                    continue;
                }
                let mut numer = 0.0_f32;
                let mut denom = 0.0_f32;
                for i in 0..k {
                    if abc.degen[x][i] {
                        numer += row[i] * bg.f[i];
                        denom += bg.f[i];
                    }
                }
                row[x] = if denom > 0.0 { numer / denom } else { 0.0 };
            }
        }

        arr
    }

    /// Return the striped Forward emission odds ratio for match node `k` and
    /// residue code `x`, matching C's `p7_oprofile_FGetEmission()`.
    pub fn fwd_emission_odds(&self, k: usize, x: usize) -> Option<f32> {
        if k == 0 || k > self.m || x >= self.rfv.len() {
            return None;
        }
        let nq = nqf(self.m);
        let q = (k - 1) % nq;
        let z = (k - 1) / nq;
        self.rfv
            .get(x)
            .and_then(|rows| rows.get(q))
            .and_then(|lanes| lanes.get(z))
            .copied()
    }

    /// Update the Forward/Backward match emissions for a new bg distribution.
    ///
    /// Ports `p7_oprofile_UpdateFwdEmissionScores`: rewrites the striped `rfv`
    /// (and its aligned mirror `rfv_a`) in place using precomputed raw
    /// Forward emissions from `get_fwd_emissions`. Reorders the loops over
    /// `(q, x, z)` to minimize working scratch. Used by nhmmer long_target
    /// per-envelope reparameterization.
    #[cfg(target_arch = "x86_64")]
    pub fn update_fwd_emission_scores(
        &mut self,
        bg: &crate::bg::Bg,
        fwd_emissions: &[f32],
        abc: &crate::alphabet::Alphabet,
    ) {
        let m = self.m;
        let kp = self.abc_kp;
        let k = abc.k;
        let nq = nqf(m);

        // Row in sc_arr: for each of the 4 lanes (z), Kp residue scores.
        let mut sc_arr = vec![0.0_f32; kp * 4];

        for q in 0..nq {
            // Canonical residues — compute log(prob / bg.f)
            for x in 0..k {
                for z in 0..4 {
                    let node = q + 1 + z * nq;
                    sc_arr[z * kp + x] = if node <= m {
                        c_log_to_f32((fwd_emissions[kp * node + x] as f64) / (bg.f[x] as f64))
                    } else {
                        f32::NEG_INFINITY
                    };
                }
                let tmp = [
                    sc_arr[0 * kp + x],
                    sc_arr[1 * kp + x],
                    sc_arr[2 * kp + x],
                    sc_arr[3 * kp + x],
                ];
                let packed = unsafe { esl_sse_expf4(tmp) };
                self.rfv[x][q] = packed;
                self.rfv_a[x][q] = AlignedF32x4::from_array(packed);
            }

            // Gap, nonresidue, missing codes — -infinity.
            for z in 0..4 {
                sc_arr[z * kp + k] = f32::NEG_INFINITY;
                sc_arr[z * kp + kp - 2] = f32::NEG_INFINITY;
                sc_arr[z * kp + kp - 1] = f32::NEG_INFINITY;
            }

            // Ambiguity codes: expectation over canonical codes.
            for z in 0..4 {
                let base = z * kp;
                for x in (k + 1)..=kp.saturating_sub(3) {
                    if !abc.is_residue(x as u8) {
                        continue;
                    }
                    let mut numer = 0.0_f32;
                    let mut denom = 0.0_f32;
                    for i in 0..k {
                        if abc.degen[x][i] {
                            numer += sc_arr[base + i] * bg.f[i];
                            denom += bg.f[i];
                        }
                    }
                    sc_arr[base + x] = if denom > 0.0 { numer / denom } else { 0.0 };
                }
            }

            for x in k..kp {
                let tmp = [
                    sc_arr[0 * kp + x],
                    sc_arr[1 * kp + x],
                    sc_arr[2 * kp + x],
                    sc_arr[3 * kp + x],
                ];
                let packed = unsafe { esl_sse_expf4(tmp) };
                self.rfv[x][q] = packed;
                self.rfv_a[x][q] = AlignedF32x4::from_array(packed);
            }
        }
    }

    /// Read an already-quantized Viterbi transition word from the SSE-striped
    /// `twv` layout. Used by wider SIMD restripers to preserve C/SSE word
    /// scores exactly, including `maxval` clamping for II and padding lanes.
    pub fn twv_word_at(&self, node: usize, tsc_type: usize) -> i16 {
        let nq = nqw(self.m);
        let q = (node) % nq;
        let z = (node) / nq;
        if z < 8 && q < self.twv.len() / 8 {
            // Map back: twv layout is [q*7+t] for t=0..6, then DD at 7*nq+q
            let idx = match tsc_type {
                P7P_BM => q * 7 + P7O_BM,
                P7P_MM => q * 7 + P7O_MM,
                P7P_IM => q * 7 + P7O_IM,
                P7P_DM => q * 7 + P7O_DM,
                P7P_MD => q * 7 + P7O_MD,
                P7P_MI => q * 7 + P7O_MI,
                P7P_II => q * 7 + P7O_II,
                P7P_DD => 7 * nq + q,
                _ => return -32768,
            };
            if idx < self.twv.len() && z < 8 {
                return self.twv[idx][z];
            }
        }
        -32768
    }
}

/// Public re-export of `wordify` for use by AVX2/NEON restriping.
/// Mirrors C `wordify`: rounds `scale_w * sc` and saturates to i16 range.
pub fn wordify_pub(scale_w: f32, sc: f32) -> i16 {
    let sc = (scale_w * sc).round();
    if sc >= 32767.0 {
        32767
    } else if sc <= -32768.0 {
        -32768
    } else {
        sc as i16
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/simd/oprofile_test.rs"]
mod tests;
