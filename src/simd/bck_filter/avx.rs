use std::arch::x86_64::*;
use std::cell::Cell;

use crate::simd::oprofile::*;
use crate::simd::shuffle_mask;

use super::{advance, special_states, Backward};

thread_local! {
    static JOINT: Cell<Vec<__m256>> = const { Cell::new(Vec::new()) };
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn rows<'p>(
    om: &OProfile,
    a: &mut Backward<'p>,
    b: &mut Backward<'p>,
    common: usize,
) {
    if common < 2 {
        return;
    }
    let cells = nqf(om.m) * 3;
    let mut joint = JOINT.take();
    joint.clear();
    joint.resize(cells * 2, _mm256_setzero_ps());
    let (end_a, end_b) = (a.rows.at(a.l), b.rows.at(b.l));
    for (c, cell) in joint[..cells].iter_mut().enumerate() {
        *cell = _mm256_set_m128(*end_b.add(c), *end_a.add(c));
    }
    let mut prev = joint.as_mut_ptr();
    let mut cur = prev.add(cells);
    for k in 1..common {
        wide_row(om, a, b, prev, cur, k);
        let last = k + 1 == common;
        if last || a.rows.keeps_all() {
            unpack::<0>(cur, cells, a.rows.at(a.l - k));
        }
        if last || b.rows.keeps_all() {
            unpack::<1>(cur, cells, b.rows.at(b.l - k));
        }
        std::mem::swap(&mut prev, &mut cur);
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
unsafe fn shifted(v: __m256) -> __m256 {
    _mm256_castsi256_ps(_mm256_srli_si256::<4>(_mm256_castps_si256(v)))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn both(om: &OProfile, at: usize) -> __m256 {
    _mm256_broadcast_ps(&*(*om.tfv.as_ptr().add(at)).as_ptr().cast::<__m128>())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn emissions(om: &OProfile, low: usize, high: usize, qi: usize) -> __m256 {
    let rfv = om.rfv.as_ptr();
    let low = _mm_load_ps((*(*rfv.add(low)).as_ptr().add(qi)).as_ptr());
    let high = _mm_load_ps((*(*rfv.add(high)).as_ptr().add(qi)).as_ptr());
    _mm256_set_m128(high, low)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn sum_halves(mut v: __m256) -> (f32, f32) {
    v = _mm256_add_ps(v, _mm256_shuffle_ps::<{ shuffle_mask(0, 3, 2, 1) }>(v, v));
    v = _mm256_add_ps(v, _mm256_shuffle_ps::<{ shuffle_mask(1, 0, 3, 2) }>(v, v));
    (
        _mm256_cvtss_f32(v),
        _mm_cvtss_f32(_mm256_extractf128_ps::<1>(v)),
    )
}

#[target_feature(enable = "avx2")]
unsafe fn wide_row(
    om: &OProfile,
    a: &mut Backward,
    b: &mut Backward,
    dpp: *const __m256,
    dpc: *mut __m256,
    k: usize,
) {
    let q = nqf(om.m);
    let zerov = _mm256_setzero_ps();
    let (at_a, at_b) = (a.l - k, b.l - k);
    let next_a = *a.dsq.add(at_a + 1) as usize;
    let next_b = *b.dsq.add(at_b + 1) as usize;
    let mut mpv = shifted(_mm256_mul_ps(*dpp, emissions(om, next_a, next_b, 0)));
    let mut tmmv = shifted(both(om, 1));
    let mut timv = shifted(both(om, 2));
    let mut tdmv = shifted(both(om, 3));
    let mut x_bv = zerov;

    for qi in (0..q).rev() {
        let tii = both(om, qi * 7 + 6);
        let tmi = both(om, qi * 7 + 5);
        let tbm = both(om, qi * 7);
        let ipv = *dpp.add(qi * 3 + 2);
        *dpc.add(qi * 3 + 2) = _mm256_add_ps(_mm256_mul_ps(ipv, tii), _mm256_mul_ps(mpv, timv));
        *dpc.add(qi * 3 + 1) = _mm256_mul_ps(mpv, tdmv);
        *dpc.add(qi * 3) = _mm256_add_ps(_mm256_mul_ps(ipv, tmi), _mm256_mul_ps(mpv, tmmv));
        mpv = _mm256_mul_ps(*dpp.add(qi * 3), emissions(om, next_a, next_b, qi));
        x_bv = _mm256_add_ps(x_bv, _mm256_mul_ps(mpv, tbm));
        tdmv = both(om, qi * 7 + 3);
        timv = both(om, qi * 7 + 2);
        tmmv = both(om, qi * 7 + 1);
    }

    (a.x_b, b.x_b) = sum_halves(x_bv);
    special_states(a);
    special_states(b);
    let x_ev = _mm256_set_m128(_mm_set1_ps(b.x_e), _mm_set1_ps(a.x_e));

    let dd = 7 * q;
    let mut dpv = shifted(_mm256_add_ps(*dpc.add(1), x_ev));
    let mut dcv = zerov;
    for qi in (0..q).rev() {
        dcv = _mm256_mul_ps(dpv, both(om, dd + qi));
        let d_ptr = dpc.add(qi * 3 + 1);
        *d_ptr = _mm256_add_ps(*d_ptr, _mm256_add_ps(dcv, x_ev));
        dpv = *d_ptr;
        let m_ptr = dpc.add(qi * 3);
        *m_ptr = _mm256_add_ps(*m_ptr, x_ev);
    }
    for _ in 0..3 {
        dcv = shifted(dcv);
        for qi in (0..q).rev() {
            dcv = _mm256_mul_ps(dcv, both(om, dd + qi));
            let d_ptr = dpc.add(qi * 3 + 1);
            *d_ptr = _mm256_add_ps(*d_ptr, dcv);
        }
    }

    let mut mdv = shifted(*dpc.add(1));
    for qi in (0..q).rev() {
        let m_ptr = dpc.add(qi * 3);
        *m_ptr = _mm256_add_ps(*m_ptr, _mm256_mul_ps(mdv, both(om, qi * 7 + 4)));
        mdv = *dpc.add(qi * 3 + 1);
    }

    let (scale_a, by_a) = advance(a, at_a);
    let (scale_b, by_b) = advance(b, at_b);
    if by_a.is_some() || by_b.is_some() {
        let scalev = _mm256_set_m128(
            _mm_set1_ps(by_b.unwrap_or(1.0)),
            _mm_set1_ps(by_a.unwrap_or(1.0)),
        );
        for cell in 0..q * 3 {
            let p = dpc.add(cell);
            *p = _mm256_mul_ps(*p, scalev);
        }
    }
    a.store(at_a, scale_a);
    b.store(at_b, scale_b);
}
