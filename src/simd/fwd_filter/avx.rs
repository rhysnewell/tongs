use std::arch::x86_64::*;
use std::cell::Cell;

use crate::simd::oprofile::*;
use crate::simd::shuffle_mask;

use super::{advance, Forward};

thread_local! {
    static JOINT: Cell<Vec<__m256>> = const { Cell::new(Vec::new()) };
}

// Target a fills the low 128 bits of each register and target b the high 128, so each half runs the
// 4-lane arithmetic in its order and the pair costs one instruction where the lanes spend two.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn rows<'p>(
    om: &OProfile,
    a: &mut Forward<'p>,
    b: &mut Forward<'p>,
    common: usize,
) {
    let cells = nqf(om.m) * 3;
    let mut joint = JOINT.take();
    joint.clear();
    joint.resize(cells, _mm256_setzero_ps());
    let dp = joint.as_mut_ptr();
    for i in 1..=common {
        wide_row(om, a, b, dp, i);
        if i == common || a.rows.keeps_all() {
            unpack::<0>(dp, cells, a.rows.at(i));
        }
        if i == common || b.rows.keeps_all() {
            unpack::<1>(dp, cells, b.rows.at(i));
        }
    }
    JOINT.set(joint);
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack<const HALF: i32>(joint: *const __m256, cells: usize, row: *mut __m128) {
    for c in 0..cells {
        *row.add(c) = _mm256_extractf128_ps::<HALF>(*joint.add(c));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn shift(v: __m256) -> __m256 {
    _mm256_castsi256_ps(_mm256_slli_si256::<4>(_mm256_castps_si256(v)))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn both(om: &OProfile, at: usize) -> __m256 {
    _mm256_broadcast_ps(&*(*om.tfv.as_ptr().add(at)).as_ptr().cast::<__m128>())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn joined(low: *const AlignedF32x4, high: *const AlignedF32x4) -> __m256 {
    _mm256_set_m128(_mm_load_ps((*high).as_ptr()), _mm_load_ps((*low).as_ptr()))
}

#[target_feature(enable = "avx2")]
unsafe fn wide_row(om: &OProfile, a: &mut Forward, b: &mut Forward, dp: *mut __m256, i: usize) {
    let q_count = nqf(om.m);
    let zerov = _mm256_setzero_ps();
    let rfv = om.rfv.as_ptr();
    let rsc_a = (*rfv.add(*a.dsq.add(i) as usize)).as_ptr();
    let rsc_b = (*rfv.add(*b.dsq.add(i) as usize)).as_ptr();
    let xbv = _mm256_set_m128(_mm_set1_ps(b.xb), _mm_set1_ps(a.xb));
    let mut dcv = zerov;
    let mut xev = zerov;
    let mut mpv = shift(*dp.add((q_count - 1) * 3));
    let mut dpv = shift(*dp.add((q_count - 1) * 3 + 1));
    let mut ipv = shift(*dp.add((q_count - 1) * 3 + 2));

    for q in 0..q_count {
        let base = q * 7;
        let rsc = joined(rsc_a.add(q), rsc_b.add(q));
        let mut sv = _mm256_mul_ps(xbv, both(om, base));
        sv = _mm256_add_ps(sv, _mm256_mul_ps(mpv, both(om, base + 1)));
        sv = _mm256_add_ps(sv, _mm256_mul_ps(ipv, both(om, base + 2)));
        sv = _mm256_add_ps(sv, _mm256_mul_ps(dpv, both(om, base + 3)));
        sv = _mm256_mul_ps(sv, rsc);
        xev = _mm256_add_ps(xev, sv);
        let cell = dp.add(q * 3);
        mpv = *cell;
        dpv = *cell.add(1);
        ipv = *cell.add(2);
        *cell = sv;
        *cell.add(1) = dcv;
        dcv = _mm256_mul_ps(sv, both(om, base + 4));
        *cell.add(2) = _mm256_add_ps(
            _mm256_mul_ps(mpv, both(om, base + 5)),
            _mm256_mul_ps(ipv, both(om, base + 6)),
        );
    }

    let dd = 7 * q_count;
    dcv = shift(dcv);
    *dp.add(1) = zerov;
    for q in 0..q_count {
        let cell = dp.add(q * 3 + 1);
        *cell = _mm256_add_ps(dcv, *cell);
        dcv = _mm256_mul_ps(*cell, both(om, dd + q));
    }
    if om.m < 100 {
        for _ in 0..3 {
            dcv = shift(dcv);
            for q in 0..q_count {
                let cell = dp.add(q * 3 + 1);
                *cell = _mm256_add_ps(dcv, *cell);
                dcv = _mm256_mul_ps(dcv, both(om, dd + q));
            }
        }
    } else {
        // A stopped target's half carries zeros, and adding zero leaves its cells as SSE left them.
        let mut open = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
        for _ in 0..3 {
            dcv = _mm256_and_ps(shift(dcv), open);
            let mut cv = zerov;
            for q in 0..q_count {
                let cell = dp.add(q * 3 + 1);
                let sv = _mm256_add_ps(dcv, *cell);
                cv = _mm256_or_ps(cv, _mm256_cmp_ps::<_CMP_GT_OQ>(sv, *cell));
                *cell = sv;
                dcv = _mm256_mul_ps(dcv, both(om, dd + q));
            }
            let changed = _mm256_movemask_ps(cv);
            let (low, high) = (changed & 0x0f != 0, changed & 0xf0 != 0);
            let (low, high) = (-(low as i32), -(high as i32));
            open = _mm256_and_ps(
                open,
                _mm256_castsi256_ps(_mm256_set_epi32(high, high, high, high, low, low, low, low)),
            );
            if low == 0 && high == 0 {
                break;
            }
        }
    }

    for q in 0..q_count {
        xev = _mm256_add_ps(*dp.add(q * 3 + 1), xev);
    }
    xev = _mm256_add_ps(
        xev,
        _mm256_shuffle_ps::<{ shuffle_mask(0, 3, 2, 1) }>(xev, xev),
    );
    xev = _mm256_add_ps(
        xev,
        _mm256_shuffle_ps::<{ shuffle_mask(1, 0, 3, 2) }>(xev, xev),
    );
    _mm_store_ss(&mut a.xe, _mm256_castps256_ps128(xev));
    _mm_store_ss(&mut b.xe, _mm256_extractf128_ps::<1>(xev));

    let (scale_a, by_a) = advance(a);
    let (scale_b, by_b) = advance(b);
    if by_a.is_some() || by_b.is_some() {
        let scalev = _mm256_set_m128(
            _mm_set1_ps(by_b.unwrap_or(1.0)),
            _mm_set1_ps(by_a.unwrap_or(1.0)),
        );
        for cell in 0..q_count * 3 {
            let p = dp.add(cell);
            *p = _mm256_mul_ps(*p, scalev);
        }
    }
    a.store(i, scale_a);
    b.store(i, scale_b);
}
