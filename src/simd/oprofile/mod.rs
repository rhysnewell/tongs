//! P7_OPROFILE - SSE2-optimized scoring profile.
//! Direct port of impl_sse/p7_oprofile.c.
#![allow(
    clippy::approx_constant,
    clippy::erasing_op,
    clippy::identity_op,
    clippy::needless_range_loop
)]

use std::sync::Arc;

use crate::profile::*;
#[cfg(striped)]
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
#[cfg(striped)]
#[repr(align(16))]
#[derive(Debug, Clone, Copy)]
pub struct AlignedF32x4(pub [f32; 4]);

#[cfg(striped)]
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
    pub rbv: Arc<Vec<Vec<[u8; 16]>>>,
    /// SSV signed-byte match scores, with extra wrapped vectors for band code.
    pub sbv: Arc<Vec<Vec<[u8; 16]>>>,
    pub ssv: Option<Arc<crate::simd::ssv::SsvProfile>>,
    #[cfg(target_arch = "x86_64")]
    pub vit: Option<Arc<crate::simd::vit_avx2::VitProfile>>,
    pub tbm_b: u8,
    pub tec_b: u8,
    pub tjb_b: u8,
    pub scale_b: f32,
    pub base_b: u8,
    pub bias_b: u8,

    // === Viterbi word-precision fields ===
    /// Word-precision match emission scores: `rwv[residue_code][q_vector][8 words]`
    pub rwv: Arc<Vec<Vec<[i16; 8]>>>,
    /// Word-precision transition scores: twv[j][8 words]
    /// Layout: for each q, 7 transitions (BM,MM,IM,DM,MD,MI,II), then DD at end
    pub twv: Arc<Vec<[i16; 8]>>,
    /// Special state word scores: `xw[state][transition]`
    pub xw: [[i16; P7O_NXTRANS]; P7O_NXSTATES],
    pub scale_w: f32,
    pub base_w: i16,
    pub ddbound_w: i16,
    pub ncj_roundoff: f32,

    // === Forward/Backward float-precision fields ===
    /// Float-precision emission scores (probability ratios): `rfv[residue][q_vector][4 floats]`
    pub rfv: Arc<Vec<Vec<[f32; 4]>>>,
    /// Float-precision transition scores: tfv[j][4 floats]
    pub tfv: Arc<Vec<[f32; 4]>>,
    #[cfg(striped)]
    pub rfv_a: Arc<Vec<Vec<AlignedF32x4>>>,
    #[cfg(striped)]
    pub tfv_a: Arc<Vec<AlignedF32x4>>,
    /// Special state float scores: `xf[state][transition]`
    pub xf: [[f32; P7O_NXTRANS]; P7O_NXSTATES],

    // === Common fields ===
    pub m: usize,
    pub l: i32,
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

mod convert;
use convert::*;

impl OProfile {
    pub fn convert(gm: &Profile) -> Self {
        let m = gm.m;
        let kp = gm.abc_kp;
        let k = gm.abc_k;

        let mf = mf_conversion(gm);
        let vf = vf_conversion(gm);
        let fb = fb_conversion(gm);

        #[cfg(striped)]
        let rfv_a: Vec<Vec<AlignedF32x4>> = fb
            .rfv
            .iter()
            .map(|rows| rows.iter().copied().map(AlignedF32x4::from_array).collect())
            .collect();
        #[cfg(striped)]
        let tfv_a: Vec<AlignedF32x4> = fb
            .tfv
            .iter()
            .copied()
            .map(AlignedF32x4::from_array)
            .collect();

        let ssv = crate::simd::ssv::SsvProfile::new(&mf.sbv, m);
        let om = OProfile {
            rbv: Arc::new(mf.rbv),
            sbv: Arc::new(mf.sbv),
            ssv,
            #[cfg(target_arch = "x86_64")]
            vit: None,
            tbm_b: mf.tbm_b,
            tec_b: mf.tec_b,
            tjb_b: mf.tjb_b,
            scale_b: mf.scale_b,
            base_b: mf.base_b,
            bias_b: mf.bias_b,
            rwv: Arc::new(vf.rwv),
            twv: Arc::new(vf.twv),
            xw: vf.xw,
            scale_w: vf.scale_w,
            base_w: vf.base_w,
            ddbound_w: vf.ddbound_w,
            ncj_roundoff: 0.0,
            rfv: Arc::new(fb.rfv),
            tfv: Arc::new(fb.tfv),
            #[cfg(striped)]
            rfv_a: Arc::new(rfv_a),
            #[cfg(striped)]
            tfv_a: Arc::new(tfv_a),
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

        #[cfg(target_arch = "x86_64")]
        let om = OProfile {
            vit: crate::simd::vit_avx2::VitProfile::new(&om),
            ..om
        };
        om
    }

    pub fn reconfig_msv_length(&mut self, l: i32) {
        self.tjb_b = unbiased_byteify(self.scale_b, c_logf_to_f32(3.0_f32 / (l as f32 + 3.0)));
    }

    pub fn reconfig_length(&mut self, l: i32) {
        self.reconfig_msv_length(l);
        self.reconfig_rest_length(l);
    }

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

    pub fn reconfig_multihit(&mut self, l: i32) {
        self.xf[P7O_E][P7O_MOVE] = 0.5;
        self.xf[P7O_E][P7O_LOOP] = 0.5;
        self.nj = 1.0;

        let e = -(ESL_CONST_LOG2 as f32);
        self.xw[P7O_E][P7O_MOVE] = wordify(self.scale_w, e);
        self.xw[P7O_E][P7O_LOOP] = wordify(self.scale_w, e);

        self.reconfig_length(l);
    }

    pub fn reconfig_unihit(&mut self, l: i32) {
        self.xf[P7O_E][P7O_MOVE] = 1.0;
        self.xf[P7O_E][P7O_LOOP] = 0.0;
        self.nj = 0.0;

        self.xw[P7O_E][P7O_MOVE] = 0;
        self.xw[P7O_E][P7O_LOOP] = i16::MIN;

        self.reconfig_length(l);
    }

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

    #[cfg(striped)]
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
                Arc::make_mut(&mut self.rfv)[x][q] = packed;
                Arc::make_mut(&mut self.rfv_a)[x][q] = AlignedF32x4::from_array(packed);
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
                Arc::make_mut(&mut self.rfv)[x][q] = packed;
                Arc::make_mut(&mut self.rfv_a)[x][q] = AlignedF32x4::from_array(packed);
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

/// Public re-export of `wordify` for use by AVX2 restriping.
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
