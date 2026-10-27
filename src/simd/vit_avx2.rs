use std::arch::x86_64::*;
use std::sync::Arc;

use crate::alphabet::Dsq;
use crate::simd::oprofile::*;
use crate::simd::vit_filter::VitResult;

const LANES: usize = 16;

// A 16-lane restriping of the 8-lane word scores. Viterbi takes maxima of saturating sums cell by
// cell, so every cell and the final score are the same whatever the lane count.
#[derive(Debug)]
pub struct VitProfile {
    q: usize,
    rwv: Vec<__m256i>,
    twv: Vec<__m256i>,
}

impl VitProfile {
    pub fn new(om: &OProfile) -> Option<Arc<Self>> {
        if !is_x86_feature_detected!("avx2") {
            return None;
        }
        let m = om.m;
        let q = 2.max((m.max(1) - 1) / LANES + 1);
        let nq = nqw(m);
        let word = |row: &[[i16; 8]], node: usize, offset: usize| match node < m + offset {
            true => row[(node - offset) % nq][(node - offset) / nq],
            false => i16::MIN,
        };
        let vector = |lane: &dyn Fn(usize) -> i16| unsafe {
            std::mem::transmute::<[i16; LANES], __m256i>(std::array::from_fn(lane))
        };
        let mut rwv = Vec::with_capacity(om.abc_kp * q);
        for row in om.rwv.iter() {
            for stripe in 0..q {
                rwv.push(vector(&|z| word(row, stripe + 1 + z * q, 1)));
            }
        }
        let mut twv = Vec::with_capacity(8 * q);
        for stripe in 0..q {
            for t in 0..7 {
                let offset = usize::from(t >= P7O_MD);
                twv.push(vector(&|z| {
                    transition(&om.twv, t, 7, stripe + offset + z * q, offset, m, nq)
                }));
            }
        }
        for stripe in 0..q {
            twv.push(vector(&|z| {
                transition(&om.twv, 7 * nq, 1, stripe + 1 + z * q, 1, m, nq)
            }));
        }
        Some(Arc::new(VitProfile { q, rwv, twv }))
    }
}

fn transition(
    twv: &[[i16; 8]],
    start: usize,
    stride: usize,
    node: usize,
    offset: usize,
    m: usize,
    nq: usize,
) -> i16 {
    match node < m {
        true => twv[start + (node - offset) % nq * stride][(node - offset) / nq],
        false => i16::MIN,
    }
}

#[target_feature(enable = "avx2")]
unsafe fn shifted(v: __m256i) -> __m256i {
    let low = _mm256_permute2x128_si256::<0x08>(v, v);
    _mm256_insert_epi16::<0>(_mm256_alignr_epi8::<14>(v, low), i16::MIN)
}

#[target_feature(enable = "avx2")]
unsafe fn hmax(v: __m256i) -> i16 {
    let mut m = _mm_max_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    m = _mm_max_epi16(m, _mm_srli_si128::<8>(m));
    m = _mm_max_epi16(m, _mm_srli_si128::<4>(m));
    m = _mm_max_epi16(m, _mm_srli_si128::<2>(m));
    _mm_extract_epi16::<0>(m) as i16
}

fn add(a: i16, b: i16) -> i16 {
    (a as i32 + b as i32) as i16
}

#[target_feature(enable = "avx2")]
pub unsafe fn viterbi_filter(
    dsq: &[Dsq],
    l: usize,
    om: &OProfile,
    vp: &VitProfile,
    dp: &mut Vec<__m256i>,
) -> VitResult {
    assert!(dsq.len() > l);
    let q_count = vp.q;
    let neg = _mm256_set1_epi16(i16::MIN);
    dp.clear();
    dp.resize(3 * q_count, neg);
    let cells = dp.as_mut_ptr();
    let (mmx, dmx, imx) = (cells, cells.add(1), cells.add(2));
    let twv = vp.twv.as_ptr();
    let dd = twv.add(7 * q_count);
    let xw = &om.xw;

    let mut xn = om.base_w;
    let mut xb = add(xn, xw[P7O_N][P7O_MOVE]);
    let mut xj = i16::MIN;
    let mut xc = i16::MIN;

    for i in 1..=l {
        let rsc = vp
            .rwv
            .as_ptr()
            .add(*dsq.get_unchecked(i) as usize * q_count);
        let mut dcv = neg;
        let mut xev = neg;
        let mut dmaxv = neg;
        let xbv = _mm256_set1_epi16(xb);
        let mut mpv = shifted(*mmx.add(3 * (q_count - 1)));
        let mut dpv = shifted(*dmx.add(3 * (q_count - 1)));
        let mut ipv = shifted(*imx.add(3 * (q_count - 1)));
        for q in 0..q_count {
            let t = twv.add(7 * q);
            let mut sv = _mm256_adds_epi16(xbv, *t);
            sv = _mm256_max_epi16(sv, _mm256_adds_epi16(mpv, *t.add(1)));
            sv = _mm256_max_epi16(sv, _mm256_adds_epi16(ipv, *t.add(2)));
            sv = _mm256_max_epi16(sv, _mm256_adds_epi16(dpv, *t.add(3)));
            sv = _mm256_adds_epi16(sv, *rsc.add(q));
            xev = _mm256_max_epi16(xev, sv);
            mpv = *mmx.add(3 * q);
            dpv = *dmx.add(3 * q);
            ipv = *imx.add(3 * q);
            *mmx.add(3 * q) = sv;
            *dmx.add(3 * q) = dcv;
            dcv = _mm256_adds_epi16(sv, *t.add(4));
            dmaxv = _mm256_max_epi16(dcv, dmaxv);
            *imx.add(3 * q) = _mm256_max_epi16(
                _mm256_adds_epi16(mpv, *t.add(5)),
                _mm256_adds_epi16(ipv, *t.add(6)),
            );
        }

        let xe = hmax(xev);
        if xe >= i16::MAX {
            return VitResult::Overflow;
        }
        xn = add(xn, xw[P7O_N][P7O_LOOP]);
        xc = add(xc, xw[P7O_C][P7O_LOOP]).max(add(xe, xw[P7O_E][P7O_MOVE]));
        xj = add(xj, xw[P7O_J][P7O_LOOP]).max(add(xe, xw[P7O_E][P7O_LOOP]));
        xb = add(xj, xw[P7O_J][P7O_MOVE]).max(add(xn, xw[P7O_N][P7O_MOVE]));

        if hmax(dmaxv) as i32 + om.ddbound_w as i32 > xb as i32 {
            dcv = shifted(dcv);
            for q in 0..q_count {
                let d = dmx.add(3 * q);
                *d = _mm256_max_epi16(dcv, *d);
                dcv = _mm256_adds_epi16(*d, *dd.add(q));
            }
            'passes: loop {
                dcv = shifted(dcv);
                for q in 0..q_count {
                    let d = dmx.add(3 * q);
                    if _mm256_movemask_epi8(_mm256_cmpgt_epi16(dcv, *d)) == 0 {
                        break 'passes;
                    }
                    *d = _mm256_max_epi16(dcv, *d);
                    dcv = _mm256_adds_epi16(*d, *dd.add(q));
                }
            }
        } else {
            *dmx = shifted(dcv);
        }
    }

    if xc > i16::MIN {
        let sc = (xc as f32 + xw[P7O_C][P7O_MOVE] as f32 - om.base_w as f32) / om.scale_w - 3.0;
        VitResult::Ok(sc)
    } else {
        VitResult::Ok(f32::NEG_INFINITY)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/simd/vit_avx2_test.rs"]
mod tests;
