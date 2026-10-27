use super::*;

pub(super) fn biased_byteify(scale_b: f32, bias_b: u8, sc: f32) -> u8 {
    let negated = -(scale_b * sc).round();
    if negated > (255 - bias_b) as f32 {
        255
    } else {
        // Match C's unsigned wrap behavior: cast to i32 first, then wrapping add
        let ival = negated as i32;
        (ival as u8).wrapping_add(bias_b)
    }
}

/// Convert a float log-odds score to a Viterbi-precision signed word (`i16`).
/// Mirrors C `wordify`: rounds `scale_w * sc` and saturates to `[-32768, 32767]`.
pub(super) fn wordify(scale_w: f32, sc: f32) -> i16 {
    let sc = (scale_w * sc).round();
    if sc >= 32767.0 {
        32767
    } else if sc <= -32768.0 {
        -32768
    } else {
        sc as i16
    }
}

/// Convert a float log-odds score to an unbiased MSV byte cost.
/// Mirrors C `unbiased_byteify`: negates and rounds, saturating to 255.
pub(super) fn unbiased_byteify(scale_b: f32, sc: f32) -> u8 {
    let negated = -(scale_b * sc).round();
    if negated > 255.0 {
        255
    } else {
        negated as i32 as u8
    }
}

/// Vectorized 4-lane `expf` port of Easel's `esl_sse_expf`, used to lift log
/// scores into probability space when building the Forward/Backward profile.
/// Uses the Cephes minimax polynomial; falls back to `f32::INFINITY` / `0.0`
/// on over/underflow.
///
/// # Safety
/// Requires SSE2 to be available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub(super) unsafe fn esl_sse_expf4(x: [f32; 4]) -> [f32; 4] {
    use std::arch::x86_64::*;

    const CEPHES_P0: f32 = 1.987_569_150_0e-4;
    const CEPHES_P1: f32 = 1.398_199_950_7e-3;
    const CEPHES_P2: f32 = 8.333_451_907_3e-3;
    const CEPHES_P3: f32 = 4.166_579_589_4e-2;
    const CEPHES_P4: f32 = 1.666_666_545_9e-1;
    const CEPHES_P5: f32 = 5.000_000_120_1e-1;
    const C0: f32 = 0.693_359_375;
    const C1: f32 = -2.121_944_4e-4;
    const LOG2R: f32 = 1.442_695_040_888_963_4_f32;
    const MAXLOGF: f32 = 88.376_262_664_794_9;
    const MINLOGF: f32 = -88.376_262_664_794_9;

    let mut xv = _mm_loadu_ps(x.as_ptr());
    let maxmask = _mm_cmpgt_ps(xv, _mm_set1_ps(MAXLOGF));
    let minmask = _mm_cmple_ps(xv, _mm_set1_ps(MINLOGF));

    let mut fx = _mm_mul_ps(xv, _mm_set1_ps(LOG2R));
    fx = _mm_add_ps(fx, _mm_set1_ps(0.5));

    let mut k = _mm_cvttps_epi32(fx);
    let tmp = _mm_cvtepi32_ps(k);
    let mut mask = _mm_cmpgt_ps(tmp, fx);
    mask = _mm_and_ps(mask, _mm_set1_ps(1.0));
    fx = _mm_sub_ps(tmp, mask);
    k = _mm_cvttps_epi32(fx);

    let tmp = _mm_mul_ps(fx, _mm_set1_ps(C0));
    let zc = _mm_mul_ps(fx, _mm_set1_ps(C1));
    xv = _mm_sub_ps(xv, tmp);
    xv = _mm_sub_ps(xv, zc);
    let z = _mm_mul_ps(xv, xv);

    let mut y = _mm_set1_ps(CEPHES_P0);
    y = _mm_mul_ps(y, xv);
    y = _mm_add_ps(y, _mm_set1_ps(CEPHES_P1));
    y = _mm_mul_ps(y, xv);
    y = _mm_add_ps(y, _mm_set1_ps(CEPHES_P2));
    y = _mm_mul_ps(y, xv);
    y = _mm_add_ps(y, _mm_set1_ps(CEPHES_P3));
    y = _mm_mul_ps(y, xv);
    y = _mm_add_ps(y, _mm_set1_ps(CEPHES_P4));
    y = _mm_mul_ps(y, xv);
    y = _mm_add_ps(y, _mm_set1_ps(CEPHES_P5));
    y = _mm_mul_ps(y, z);
    y = _mm_add_ps(y, xv);
    y = _mm_add_ps(y, _mm_set1_ps(1.0));

    k = _mm_add_epi32(k, _mm_set1_epi32(127));
    k = _mm_slli_epi32(k, 23);
    fx = _mm_castsi128_ps(k);
    y = _mm_mul_ps(y, fx);

    let infv = _mm_set1_ps(f32::INFINITY);
    y = _mm_or_ps(_mm_and_ps(maxmask, infv), _mm_andnot_ps(maxmask, y));
    y = _mm_andnot_ps(minmask, y);

    let mut out = [0.0_f32; 4];
    _mm_storeu_ps(out.as_mut_ptr(), y);
    out
}

/// Scalar fallback for `esl_sse_expf4` on non-x86 builds.
#[cfg(not(target_arch = "x86_64"))]
pub(super) fn esl_sse_expf4(x: [f32; 4]) -> [f32; 4] {
    x.map(esl_sse_expf_lane)
}

/// Scalar fallback for one lane of Easel's `esl_sse_expf()`. Uses the same
/// Cephes minimax polynomial as the SSE version for bitwise parity.
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(super) fn esl_sse_expf_lane(mut x: f32) -> f32 {
    const CEPHES_P0: f32 = 1.9875691500e-4;
    const CEPHES_P1: f32 = 1.3981999507e-3;
    const CEPHES_P2: f32 = 8.3334519073e-3;
    const CEPHES_P3: f32 = 4.1665795894e-2;
    const CEPHES_P4: f32 = 1.6666665459e-1;
    const CEPHES_P5: f32 = 5.0000001201e-1;
    const C0: f32 = 0.693359375;
    const C1: f32 = -2.12194440e-4;
    const LOG2R: f32 = 1.44269504088896341_f32;
    const MAXLOGF: f32 = 88.3762626647949;
    const MINLOGF: f32 = -88.3762626647949;

    if x > MAXLOGF {
        return f32::INFINITY;
    }
    if x <= MINLOGF {
        return 0.0;
    }

    let mut fx = x * LOG2R;
    fx += 0.5;
    let mut k = fx as i32;
    let tmp = k as f32;
    if tmp > fx {
        fx = tmp - 1.0;
        k -= 1;
    } else {
        fx = tmp;
    }

    let tmp = fx * C0;
    let z = fx * C1;
    x -= tmp;
    x -= z;
    let z = x * x;

    let mut y = CEPHES_P0;
    y *= x;
    y += CEPHES_P1;
    y *= x;
    y += CEPHES_P2;
    y *= x;
    y += CEPHES_P3;
    y *= x;
    y += CEPHES_P4;
    y *= x;
    y += CEPHES_P5;
    y *= z;
    y += x;
    y += 1.0;

    let pow2 = f32::from_bits(((k + 127) as u32) << 23);
    y * pow2
}

pub(super) struct MfConversion {
    pub(super) rbv: Vec<Vec<[u8; 16]>>,
    pub(super) sbv: Vec<Vec<[u8; 16]>>,
    pub(super) tbm_b: u8,
    pub(super) tec_b: u8,
    pub(super) tjb_b: u8,
    pub(super) scale_b: f32,
    pub(super) base_b: u8,
    pub(super) bias_b: u8,
}

/// Port of C `mf_conversion()`: build the MSV/SSV byte-score profile.
pub(super) fn mf_conversion(gm: &Profile) -> MfConversion {
    let m = gm.m;
    let nq = nqb(m);
    let kp = gm.abc_kp;
    let k = gm.abc_k;
    let scale_b = (3.0 / ESL_CONST_LOG2) as f32;
    let base_b: u8 = 190;

    let mut max_sc = 0.0_f32;
    for x in 0..k {
        for node in 1..=m {
            let sc = gm.msc(node, x);
            if sc.is_finite() && sc > max_sc {
                max_sc = sc;
            }
        }
    }

    let bias_b = unbiased_byteify(scale_b, -max_sc);
    let mut rbv = vec![vec![[0u8; 16]; nq]; kp];
    for x in 0..kp {
        for q in 0..nq {
            let mut tmp = [0u8; 16];
            for z in 0..16 {
                let node = q + 1 + z * nq;
                tmp[z] = if node <= m {
                    biased_byteify(scale_b, bias_b, gm.msc(node, x))
                } else {
                    255
                };
            }
            rbv[x][q] = tmp;
        }
    }

    let extra_sb = 17;
    let mut sbv = vec![vec![[0u8; 16]; nq + extra_sb]; kp];
    for x in 0..kp {
        for q in 0..(nq + extra_sb) {
            let src = rbv[x][q % nq];
            let mut tmp = [0u8; 16];
            for z in 0..16 {
                tmp[z] = (bias_b.wrapping_add(127).saturating_sub(src[z])) ^ 127;
            }
            sbv[x][q] = tmp;
        }
    }

    let tbm_b = unbiased_byteify(
        scale_b,
        c_logf_to_f32(2.0_f32 / (m as f32 * (m as f32 + 1.0))),
    );
    let tec_b = unbiased_byteify(scale_b, c_logf_to_f32(0.5_f32));
    let tjb_b = unbiased_byteify(scale_b, c_logf_to_f32(3.0_f32 / (gm.l as f32 + 3.0)));

    MfConversion {
        rbv,
        sbv,
        tbm_b,
        tec_b,
        tjb_b,
        scale_b,
        base_b,
        bias_b,
    }
}

pub(super) struct VfConversion {
    pub(super) rwv: Vec<Vec<[i16; 8]>>,
    pub(super) twv: Vec<[i16; 8]>,
    pub(super) xw: [[i16; P7O_NXTRANS]; P7O_NXSTATES],
    pub(super) scale_w: f32,
    pub(super) base_w: i16,
    pub(super) ddbound_w: i16,
}

/// Port of C `vf_conversion()`: build the Viterbi word-score profile.
pub(super) fn vf_conversion(gm: &Profile) -> VfConversion {
    let m = gm.m;
    let kp = gm.abc_kp;
    let scale_w = (500.0 / ESL_CONST_LOG2) as f32;
    let base_w: i16 = 12000;
    let nqw = nqw(m);

    let mut rwv = vec![vec![[0i16; 8]; nqw]; kp];
    for x in 0..kp {
        for q in 0..nqw {
            let mut tmp = [0i16; 8];
            for z in 0..8 {
                let node = q + 1 + z * nqw;
                tmp[z] = if node <= m {
                    wordify(scale_w, gm.msc(node, x))
                } else {
                    -32768
                };
            }
            rwv[x][q] = tmp;
        }
    }

    let mut twv = vec![[0i16; 8]; 8 * nqw];
    let mut j = 0;
    for q in 0..nqw {
        let k = q + 1;
        let trans_specs: [(usize, usize, i16); 7] = [
            (P7P_BM, k.wrapping_sub(1), 0),
            (P7P_MM, k.wrapping_sub(1), 0),
            (P7P_IM, k.wrapping_sub(1), 0),
            (P7P_DM, k.wrapping_sub(1), 0),
            (P7P_MD, k, 0),
            (P7P_MI, k, 0),
            (P7P_II, k, -1),
        ];
        for &(tg, kb, maxval) in &trans_specs {
            let mut tmp = [0i16; 8];
            for z in 0..8 {
                let node = kb + z * nqw;
                let val = if node < m {
                    wordify(scale_w, gm.tsc(node, tg))
                } else {
                    -32768
                };
                tmp[z] = if val >= maxval { maxval } else { val };
            }
            twv[j] = tmp;
            j += 1;
        }
    }
    for q in 0..nqw {
        let k = q + 1;
        let mut tmp = [0i16; 8];
        for z in 0..8 {
            let node = k + z * nqw;
            tmp[z] = if node < m {
                wordify(scale_w, gm.tsc(node, P7P_DD))
            } else {
                -32768
            };
        }
        twv[j] = tmp;
        j += 1;
    }

    let mut xw = [[0i16; P7O_NXTRANS]; P7O_NXSTATES];
    xw[P7O_E][P7O_LOOP] = wordify(scale_w, gm.xsc[P7P_E][P7P_LOOP]);
    xw[P7O_E][P7O_MOVE] = wordify(scale_w, gm.xsc[P7P_E][P7P_MOVE]);
    xw[P7O_N][P7O_MOVE] = wordify(scale_w, gm.xsc[P7P_N][P7P_MOVE]);
    xw[P7O_N][P7O_LOOP] = 0;
    xw[P7O_C][P7O_MOVE] = wordify(scale_w, gm.xsc[P7P_C][P7P_MOVE]);
    xw[P7O_C][P7O_LOOP] = 0;
    xw[P7O_J][P7O_MOVE] = wordify(scale_w, gm.xsc[P7P_J][P7P_MOVE]);
    xw[P7O_J][P7O_LOOP] = 0;

    let mut ddbound_w: i16 = -32768;
    for node in 2..m.saturating_sub(1) {
        let dd = wordify(scale_w, gm.tsc(node, P7P_DD)) as i32;
        let dm = wordify(scale_w, gm.tsc(node + 1, P7P_DM)) as i32;
        let bm = wordify(scale_w, gm.tsc(node + 1, P7P_BM)) as i32;
        let ddtmp = dd + dm - bm;
        if ddtmp > ddbound_w as i32 {
            ddbound_w = ddtmp.min(32767) as i16;
        }
    }

    VfConversion {
        rwv,
        twv,
        xw,
        scale_w,
        base_w,
        ddbound_w,
    }
}

pub(super) struct FbConversion {
    pub(super) rfv: Vec<Vec<[f32; 4]>>,
    pub(super) tfv: Vec<[f32; 4]>,
    pub(super) xf: [[f32; P7O_NXTRANS]; P7O_NXSTATES],
}

/// Port of C `fb_conversion()`: build Forward/Backward probability vectors.
pub(super) fn fb_conversion(gm: &Profile) -> FbConversion {
    let m = gm.m;
    let kp = gm.abc_kp;
    let nqf = nqf(m);
    let mut rfv = vec![vec![[0.0f32; 4]; nqf]; kp];
    for x in 0..kp {
        for q in 0..nqf {
            let mut tmp = [0.0f32; 4];
            for z in 0..4 {
                let node = q + 1 + z * nqf;
                tmp[z] = if node <= m {
                    gm.msc(node, x)
                } else {
                    f32::NEG_INFINITY
                };
            }
            #[cfg(target_arch = "x86_64")]
            {
                rfv[x][q] = unsafe { esl_sse_expf4(tmp) };
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                rfv[x][q] = esl_sse_expf4(tmp);
            }
        }
    }

    let mut tfv = vec![[0.0f32; 4]; 8 * nqf];
    let mut j = 0;
    for q in 0..nqf {
        let k = q + 1;
        let trans_specs: [(usize, usize); 7] = [
            (P7P_BM, k.wrapping_sub(1)),
            (P7P_MM, k.wrapping_sub(1)),
            (P7P_IM, k.wrapping_sub(1)),
            (P7P_DM, k.wrapping_sub(1)),
            (P7P_MD, k),
            (P7P_MI, k),
            (P7P_II, k),
        ];
        for &(tg, kb) in &trans_specs {
            let mut tmp = [0.0f32; 4];
            for z in 0..4 {
                let node = kb + z * nqf;
                tmp[z] = if node < m {
                    gm.tsc(node, tg)
                } else {
                    f32::NEG_INFINITY
                };
            }
            #[cfg(target_arch = "x86_64")]
            {
                tfv[j] = unsafe { esl_sse_expf4(tmp) };
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                tfv[j] = esl_sse_expf4(tmp);
            }
            j += 1;
        }
    }
    for q in 0..nqf {
        let k = q + 1;
        let mut tmp = [0.0f32; 4];
        for z in 0..4 {
            let node = k + z * nqf;
            tmp[z] = if node < m {
                gm.tsc(node, P7P_DD)
            } else {
                f32::NEG_INFINITY
            };
        }
        #[cfg(target_arch = "x86_64")]
        {
            tfv[j] = unsafe { esl_sse_expf4(tmp) };
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            tfv[j] = esl_sse_expf4(tmp);
        }
        j += 1;
    }

    let mut xf = [[0.0f32; P7O_NXTRANS]; P7O_NXSTATES];
    xf[P7O_E][P7O_LOOP] = c_expf_to_f32(gm.xsc[P7P_E][P7P_LOOP]);
    xf[P7O_E][P7O_MOVE] = c_expf_to_f32(gm.xsc[P7P_E][P7P_MOVE]);
    xf[P7O_N][P7O_LOOP] = c_expf_to_f32(gm.xsc[P7P_N][P7P_LOOP]);
    xf[P7O_N][P7O_MOVE] = c_expf_to_f32(gm.xsc[P7P_N][P7P_MOVE]);
    xf[P7O_C][P7O_LOOP] = c_expf_to_f32(gm.xsc[P7P_C][P7P_LOOP]);
    xf[P7O_C][P7O_MOVE] = c_expf_to_f32(gm.xsc[P7P_C][P7P_MOVE]);
    xf[P7O_J][P7O_LOOP] = c_expf_to_f32(gm.xsc[P7P_J][P7P_LOOP]);
    xf[P7O_J][P7O_MOVE] = c_expf_to_f32(gm.xsc[P7P_J][P7P_MOVE]);

    FbConversion { rfv, tfv, xf }
}
