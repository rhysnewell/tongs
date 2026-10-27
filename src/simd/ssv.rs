use std::sync::Arc;

use crate::simd::sse::*;

use crate::alphabet::Dsq;
use crate::simd::oprofile::{nqb, OProfile};

pub enum SsvResult {
    Ok(f32),
    Overflow,
    NoResult,
}

// Knudsen's band layout from HMMER's SSV filter. Fourteen bands leave two of the sixteen vector
// registers for the running maximum and the begin vector.
const BANDS: usize = 14;

#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
struct Chunk([u8; 32]);

#[derive(Debug)]
pub struct SsvProfile {
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    lanes: usize,
    q: usize,
    width: usize,
    scores: Vec<Chunk>,
}

impl SsvProfile {
    pub fn new(sbv: &[Vec<[u8; 16]>], m: usize) -> Option<Arc<Self>> {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            return Some(Arc::new(Self::striped(sbv, m, 32)));
        }
        Some(Arc::new(Self::striped(sbv, m, 16)))
    }

    fn striped(sbv: &[Vec<[u8; 16]>], m: usize, lanes: usize) -> Self {
        let q = 2.max((m.max(1) - 1) / lanes + 1);
        let q16 = nqb(m);
        let width = q + BANDS;
        let total = sbv.len() * width * lanes;
        let mut scores = vec![Chunk([0; 32]); total.div_ceil(32)];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(scores.as_mut_ptr().cast::<u8>(), total) };
        for (x, rows) in sbv.iter().enumerate() {
            for stripe in 0..width {
                for lane in 0..lanes {
                    let k = 1 + stripe % q + lane * q;
                    bytes[(x * width + stripe) * lanes + lane] = match k <= m {
                        true => rows[(k - 1) % q16][(k - 1) / q16],
                        false => 127,
                    };
                }
            }
        }
        SsvProfile {
            lanes,
            q,
            width,
            scores,
        }
    }

    pub fn score(&self, dsq: &[Dsq], l: usize, om: &OProfile) -> SsvResult {
        let (tjb, tbm, tec, bias) = (
            om.tjb_b as i32,
            om.tbm_b as i32,
            om.tec_b as i32,
            om.bias_b as i32,
        );
        if tjb + tbm + tec + bias >= 127 || l == 0 {
            return SsvResult::NoResult;
        }
        let xe = self.xe(dsq, l) as i32;
        let base = om.base_b as i32;
        if xe >= 255 - bias {
            return match base - tjb - tbm < 128 {
                true => SsvResult::NoResult,
                false => SsvResult::Overflow,
            };
        }
        let xe = (xe + base - tjb - tbm - 128) as u16 as i32;
        if xe >= 255 - bias {
            return SsvResult::Overflow;
        }
        let xj = (xe - tec) as u16 as i32;
        if xj > base {
            return SsvResult::NoResult;
        }
        SsvResult::Ok(((xj - tjb) as f32 - base as f32) / om.scale_b - 3.0)
    }

    fn xe(&self, dsq: &[Dsq], l: usize) -> u8 {
        assert!(dsq.len() > l);
        unsafe {
            #[cfg(target_arch = "x86_64")]
            if self.lanes == 32 {
                return avx2_xe(dsq.as_ptr().add(1), l, self);
            }
            sse2_xe(dsq.as_ptr().add(1), l, self)
        }
    }
}

macro_rules! ssv_step {
    (($subs:ident, $max:ident, $load:ident), $sv:ident, $xev:ident, $rsc:expr, [$($b:literal)*]) => {{
        let rsc = $rsc;
        $(
            $sv[$b] = $subs($sv[$b], $load(rsc.add($b)));
            $xev[$b & 3] = $max($xev[$b & 3], $sv[$b]);
        )*
    }};
}

macro_rules! ssv_turn {
    ($ops:tt, $wrap:ident, $sv:ident, $xev:ident, $beginv:ident, $row:ident, $i:ident, $q:expr,
     $limit:expr, $label:lifetime, $bs:tt, [$($j:literal)*]) => {
        $(
            if $i >= $limit {
                break $label;
            }
            ssv_step!($ops, $sv, $xev, $row($i).add($q - ($j + 1)), $bs);
            $sv[$j] = $wrap($sv[$j], $beginv);
            $i += 1;
        )*
    };
}

// Each band width gets its own function so its diagonals stay in registers, as in HMMER's
// fixed-width band functions.
macro_rules! ssv_bands {
    ($feature:literal, $vec:ty, $set1:ident, $wrap:ident, $hmax:ident, $xe:ident,
     ($subs:ident, $max:ident, $load:ident),
     $(($name:ident, $w:literal, $bs:tt, $rs:tt),)*) => {
        $(
            #[cfg_attr(target_arch = "x86_64", target_feature(enable = $feature))]
            #[allow(clippy::too_many_arguments, unused_assignments)]
            unsafe fn $name(
                dsq: *const Dsq,
                l: usize,
                scores: *const $vec,
                width: usize,
                big_q: usize,
                q: usize,
                beginv: $vec,
                xev: $vec,
            ) -> $vec {
                let mut sv = [beginv; $w];
                let mut acc = [xev; 4];
                let row = |i: usize| scores.add(*dsq.add(i) as usize * width);
                let mut i = 0;
                while i < l && i < big_q - q - $w {
                    ssv_step!(($subs, $max, $load), sv, acc, row(i).add(i + q), $bs);
                    i += 1;
                }
                i = big_q - q - $w;
                'first: {
                    ssv_turn!(($subs, $max, $load), $wrap, sv, acc, beginv, row, i, big_q, l, 'first, $bs, $rs);
                }
                let mut i2 = big_q - q;
                while i2 + big_q < l {
                    for ii in 0..big_q - $w {
                        ssv_step!(($subs, $max, $load), sv, acc, row(i2 + ii).add(ii), $bs);
                    }
                    i = i2 + big_q - $w;
                    'middle: {
                        ssv_turn!(($subs, $max, $load), $wrap, sv, acc, beginv, row, i, big_q, usize::MAX, 'middle, $bs, $rs);
                    }
                    i2 += big_q;
                }
                let mut ii = 0;
                while i2 + ii < l && ii < big_q - $w {
                    ssv_step!(($subs, $max, $load), sv, acc, row(i2 + ii).add(ii), $bs);
                    ii += 1;
                }
                i = i2 + ii;
                'last: {
                    ssv_turn!(($subs, $max, $load), $wrap, sv, acc, beginv, row, i, big_q, l, 'last, $bs, $rs);
                }
                $max($max(acc[0], acc[1]), $max(acc[2], acc[3]))
            }
        )*

        #[cfg_attr(target_arch = "x86_64", target_feature(enable = $feature))]
        unsafe fn $xe(dsq: *const Dsq, l: usize, profile: &SsvProfile) -> u8 {
            let beginv = $set1(-128);
            let mut xev = beginv;
            let scores = profile.scores.as_ptr().cast::<$vec>();
            let rounds = profile.q.div_ceil(BANDS);
            let mut last = 0;
            for round in 0..rounds {
                let q = profile.q * (round + 1) / rounds;
                xev = match q - last {
                    $($w => $name(dsq, l, scores, profile.width, profile.q, last, beginv, xev),)*
                    _ => unreachable!(),
                };
                last = q;
            }
            $hmax(xev)
        }
    };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn avx2_wrap(sv: __m256i, beginv: __m256i) -> __m256i {
    let low = _mm256_permute2x128_si256::<0x08>(sv, sv);
    _mm256_or_si256(_mm256_alignr_epi8::<15>(sv, low), beginv)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn avx2_hmax(v: __m256i) -> u8 {
    sse2_hmax(_mm_max_epu8(
        _mm256_castsi256_si128(v),
        _mm256_extracti128_si256::<1>(v),
    ))
}

#[inline]
unsafe fn sse2_wrap(sv: __m128i, beginv: __m128i) -> __m128i {
    _mm_or_si128(_mm_slli_si128::<1>(sv), beginv)
}

#[inline]
unsafe fn sse2_hmax(mut v: __m128i) -> u8 {
    v = _mm_max_epu8(v, _mm_srli_si128::<8>(v));
    v = _mm_max_epu8(v, _mm_srli_si128::<4>(v));
    v = _mm_max_epu8(v, _mm_srli_si128::<2>(v));
    v = _mm_max_epu8(v, _mm_srli_si128::<1>(v));
    _mm_cvtsi128_si32(v) as u8
}

#[cfg(target_arch = "x86_64")]
ssv_bands!(
    "avx2", __m256i, _mm256_set1_epi8, avx2_wrap, avx2_hmax, avx2_xe,
    (_mm256_subs_epi8, _mm256_max_epu8, _mm256_load_si256),
    (avx2_band1, 1, [0], [0]),
    (avx2_band2, 2, [0 1], [1 0]),
    (avx2_band3, 3, [0 1 2], [2 1 0]),
    (avx2_band4, 4, [0 1 2 3], [3 2 1 0]),
    (avx2_band5, 5, [0 1 2 3 4], [4 3 2 1 0]),
    (avx2_band6, 6, [0 1 2 3 4 5], [5 4 3 2 1 0]),
    (avx2_band7, 7, [0 1 2 3 4 5 6], [6 5 4 3 2 1 0]),
    (avx2_band8, 8, [0 1 2 3 4 5 6 7], [7 6 5 4 3 2 1 0]),
    (avx2_band9, 9, [0 1 2 3 4 5 6 7 8], [8 7 6 5 4 3 2 1 0]),
    (avx2_band10, 10, [0 1 2 3 4 5 6 7 8 9], [9 8 7 6 5 4 3 2 1 0]),
    (avx2_band11, 11, [0 1 2 3 4 5 6 7 8 9 10], [10 9 8 7 6 5 4 3 2 1 0]),
    (avx2_band12, 12, [0 1 2 3 4 5 6 7 8 9 10 11], [11 10 9 8 7 6 5 4 3 2 1 0]),
    (avx2_band13, 13, [0 1 2 3 4 5 6 7 8 9 10 11 12], [12 11 10 9 8 7 6 5 4 3 2 1 0]),
    (avx2_band14, 14, [0 1 2 3 4 5 6 7 8 9 10 11 12 13], [13 12 11 10 9 8 7 6 5 4 3 2 1 0]),
);

ssv_bands!(
    "sse2", __m128i, _mm_set1_epi8, sse2_wrap, sse2_hmax, sse2_xe,
    (_mm_subs_epi8, _mm_max_epu8, _mm_load_si128),
    (sse2_band1, 1, [0], [0]),
    (sse2_band2, 2, [0 1], [1 0]),
    (sse2_band3, 3, [0 1 2], [2 1 0]),
    (sse2_band4, 4, [0 1 2 3], [3 2 1 0]),
    (sse2_band5, 5, [0 1 2 3 4], [4 3 2 1 0]),
    (sse2_band6, 6, [0 1 2 3 4 5], [5 4 3 2 1 0]),
    (sse2_band7, 7, [0 1 2 3 4 5 6], [6 5 4 3 2 1 0]),
    (sse2_band8, 8, [0 1 2 3 4 5 6 7], [7 6 5 4 3 2 1 0]),
    (sse2_band9, 9, [0 1 2 3 4 5 6 7 8], [8 7 6 5 4 3 2 1 0]),
    (sse2_band10, 10, [0 1 2 3 4 5 6 7 8 9], [9 8 7 6 5 4 3 2 1 0]),
    (sse2_band11, 11, [0 1 2 3 4 5 6 7 8 9 10], [10 9 8 7 6 5 4 3 2 1 0]),
    (sse2_band12, 12, [0 1 2 3 4 5 6 7 8 9 10 11], [11 10 9 8 7 6 5 4 3 2 1 0]),
    (sse2_band13, 13, [0 1 2 3 4 5 6 7 8 9 10 11 12], [12 11 10 9 8 7 6 5 4 3 2 1 0]),
    (sse2_band14, 14, [0 1 2 3 4 5 6 7 8 9 10 11 12 13], [13 12 11 10 9 8 7 6 5 4 3 2 1 0]),
);

#[cfg(test)]
#[path = "../../tests/unit/simd/ssv_test.rs"]
mod tests;
