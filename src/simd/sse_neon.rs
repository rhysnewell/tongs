#![allow(non_camel_case_types, clippy::missing_safety_doc, unused_unsafe)]

// Each op keeps the SSE lane order and rounding (no fused multiply-add), so
// striped kernels give the same bits on aarch64 as on x86_64.

use std::arch::aarch64::*;
use std::mem::transmute;

pub type __m128 = float32x4_t;
pub type __m128i = uint8x16_t;

#[inline(always)]
fn u32s(a: __m128) -> uint32x4_t {
    unsafe { vreinterpretq_u32_f32(a) }
}

#[inline(always)]
fn f32s(a: uint32x4_t) -> __m128 {
    unsafe { vreinterpretq_f32_u32(a) }
}

#[inline(always)]
fn s16s(a: __m128i) -> int16x8_t {
    unsafe { vreinterpretq_s16_u8(a) }
}

#[inline(always)]
fn from_s16(a: int16x8_t) -> __m128i {
    unsafe { vreinterpretq_u8_s16(a) }
}

#[inline(always)]
fn s32s(a: __m128i) -> int32x4_t {
    unsafe { vreinterpretq_s32_u8(a) }
}

#[inline(always)]
fn from_s32(a: int32x4_t) -> __m128i {
    unsafe { vreinterpretq_u8_s32(a) }
}

#[inline(always)]
pub fn _mm_setzero_ps() -> __m128 {
    unsafe { vdupq_n_f32(0.0) }
}

#[inline(always)]
pub fn _mm_set1_ps(a: f32) -> __m128 {
    unsafe { vdupq_n_f32(a) }
}

#[inline(always)]
pub unsafe fn _mm_load_ps(p: *const f32) -> __m128 {
    unsafe { vld1q_f32(p) }
}

#[inline(always)]
pub unsafe fn _mm_loadu_ps(p: *const f32) -> __m128 {
    unsafe { vld1q_f32(p) }
}

#[inline(always)]
pub unsafe fn _mm_store_ps(p: *mut f32, a: __m128) {
    unsafe { vst1q_f32(p, a) }
}

#[inline(always)]
pub unsafe fn _mm_storeu_ps(p: *mut f32, a: __m128) {
    unsafe { vst1q_f32(p, a) }
}

#[inline(always)]
pub unsafe fn _mm_store_ss(p: *mut f32, a: __m128) {
    unsafe { *p = vgetq_lane_f32::<0>(a) }
}

#[inline(always)]
pub fn _mm_add_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { vaddq_f32(a, b) }
}

#[inline(always)]
pub fn _mm_sub_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { vsubq_f32(a, b) }
}

#[inline(always)]
pub fn _mm_mul_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { vmulq_f32(a, b) }
}

// MAXPS returns the second operand on a tie of signed zeros or a NaN.
#[inline(always)]
pub fn _mm_max_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { vbslq_f32(vcgtq_f32(a, b), a, b) }
}

#[inline(always)]
pub fn _mm_cmpgt_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { f32s(vcgtq_f32(a, b)) }
}

#[inline(always)]
pub fn _mm_cmple_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { f32s(vcleq_f32(a, b)) }
}

#[inline(always)]
pub fn _mm_and_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { f32s(vandq_u32(u32s(a), u32s(b))) }
}

#[inline(always)]
pub fn _mm_or_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { f32s(vorrq_u32(u32s(a), u32s(b))) }
}

#[inline(always)]
pub fn _mm_andnot_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { f32s(vbicq_u32(u32s(b), u32s(a))) }
}

#[inline(always)]
pub fn _mm_shuffle_ps<const MASK: i32>(a: __m128, b: __m128) -> __m128 {
    let a: [f32; 4] = unsafe { transmute(a) };
    let b: [f32; 4] = unsafe { transmute(b) };
    let m = MASK as usize;
    let r = [a[m & 3], a[(m >> 2) & 3], b[(m >> 4) & 3], b[(m >> 6) & 3]];
    unsafe { transmute(r) }
}

#[inline(always)]
pub fn _mm_movemask_ps(a: __m128) -> i32 {
    unsafe {
        let signs = vshrq_n_u32::<31>(u32s(a));
        let shifts = vld1q_s32([0, 1, 2, 3].as_ptr());
        vaddvq_u32(vshlq_u32(signs, shifts)) as i32
    }
}

#[inline(always)]
pub fn _mm_cvttps_epi32(a: __m128) -> __m128i {
    unsafe { from_s32(vcvtq_s32_f32(a)) }
}

#[inline(always)]
pub fn _mm_cvtepi32_ps(a: __m128i) -> __m128 {
    unsafe { vcvtq_f32_s32(s32s(a)) }
}

#[inline(always)]
pub fn _mm_cvtss_f32(a: __m128) -> f32 {
    unsafe { vgetq_lane_f32::<0>(a) }
}

#[inline(always)]
pub fn _mm_move_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { vsetq_lane_f32::<0>(vgetq_lane_f32::<0>(b), a) }
}

#[inline(always)]
pub fn _mm_castsi128_ps(a: __m128i) -> __m128 {
    unsafe { vreinterpretq_f32_u8(a) }
}

#[inline(always)]
pub fn _mm_castps_si128(a: __m128) -> __m128i {
    unsafe { vreinterpretq_u8_f32(a) }
}

#[inline(always)]
pub fn _mm_setzero_si128() -> __m128i {
    unsafe { vdupq_n_u8(0) }
}

#[inline(always)]
pub fn _mm_set1_epi8(a: i8) -> __m128i {
    unsafe { vdupq_n_u8(a as u8) }
}

#[inline(always)]
pub fn _mm_set1_epi16(a: i16) -> __m128i {
    unsafe { from_s16(vdupq_n_s16(a)) }
}

#[inline(always)]
pub fn _mm_set1_epi32(a: i32) -> __m128i {
    unsafe { from_s32(vdupq_n_s32(a)) }
}

#[inline(always)]
pub unsafe fn _mm_load_si128(p: *const __m128i) -> __m128i {
    unsafe { vld1q_u8(p as *const u8) }
}

#[inline(always)]
pub unsafe fn _mm_loadu_si128(p: *const __m128i) -> __m128i {
    unsafe { vld1q_u8(p as *const u8) }
}

#[inline(always)]
pub unsafe fn _mm_storeu_si128(p: *mut __m128i, a: __m128i) {
    unsafe { vst1q_u8(p as *mut u8, a) }
}

#[inline(always)]
pub fn _mm_max_epu8(a: __m128i, b: __m128i) -> __m128i {
    unsafe { vmaxq_u8(a, b) }
}

#[inline(always)]
pub fn _mm_max_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe { from_s16(vmaxq_s16(s16s(a), s16s(b))) }
}

#[inline(always)]
pub fn _mm_adds_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe { from_s16(vqaddq_s16(s16s(a), s16s(b))) }
}

#[inline(always)]
pub fn _mm_adds_epu8(a: __m128i, b: __m128i) -> __m128i {
    unsafe { vqaddq_u8(a, b) }
}

#[inline(always)]
pub fn _mm_subs_epu8(a: __m128i, b: __m128i) -> __m128i {
    unsafe { vqsubq_u8(a, b) }
}

#[inline(always)]
pub fn _mm_subs_epi8(a: __m128i, b: __m128i) -> __m128i {
    unsafe { vreinterpretq_u8_s8(vqsubq_s8(vreinterpretq_s8_u8(a), vreinterpretq_s8_u8(b))) }
}

#[inline(always)]
pub fn _mm_add_epi32(a: __m128i, b: __m128i) -> __m128i {
    unsafe { from_s32(vaddq_s32(s32s(a), s32s(b))) }
}

#[inline(always)]
pub fn _mm_or_si128(a: __m128i, b: __m128i) -> __m128i {
    unsafe { vorrq_u8(a, b) }
}

#[inline(always)]
pub fn _mm_cmpeq_epi8(a: __m128i, b: __m128i) -> __m128i {
    unsafe { vceqq_u8(a, b) }
}

#[inline(always)]
pub fn _mm_cmpgt_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe { vreinterpretq_u8_u16(vcgtq_s16(s16s(a), s16s(b))) }
}

#[inline(always)]
pub fn _mm_movemask_epi8(a: __m128i) -> i32 {
    unsafe {
        let high = vshrq_n_u8::<7>(a);
        let p16 = vreinterpretq_u32_u16(vsraq_n_u16::<7>(
            vreinterpretq_u16_u8(high),
            vreinterpretq_u16_u8(high),
        ));
        let p32 = vreinterpretq_u64_u32(vsraq_n_u32::<14>(p16, p16));
        let p64 = vreinterpretq_u8_u64(vsraq_n_u64::<28>(p32, p32));
        (vgetq_lane_u8::<0>(p64) as i32) | ((vgetq_lane_u8::<8>(p64) as i32) << 8)
    }
}

macro_rules! ext {
    ($n:expr, $a:expr, $b:expr; $($k:literal)*) => {
        match $n {
            $($k => vextq_u8::<$k>($a, $b),)*
            _ => unreachable!(),
        }
    };
}

#[inline(always)]
pub fn _mm_srli_si128<const IMM8: i32>(a: __m128i) -> __m128i {
    let z = _mm_setzero_si128();
    match IMM8 {
        0 => a,
        1..=15 => unsafe { ext!(IMM8, a, z; 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15) },
        _ => z,
    }
}

#[inline(always)]
pub fn _mm_slli_si128<const IMM8: i32>(a: __m128i) -> __m128i {
    let z = _mm_setzero_si128();
    match IMM8 {
        0 => a,
        1..=15 => unsafe { ext!(16 - IMM8, z, a; 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15) },
        _ => z,
    }
}

#[inline(always)]
pub fn _mm_srli_epi32<const IMM8: i32>(a: __m128i) -> __m128i {
    if !(0..32).contains(&IMM8) {
        return _mm_setzero_si128();
    }
    unsafe { vreinterpretq_u8_u32(vshlq_u32(vreinterpretq_u32_u8(a), vdupq_n_s32(-IMM8))) }
}

#[inline(always)]
pub fn _mm_slli_epi32<const IMM8: i32>(a: __m128i) -> __m128i {
    if !(0..32).contains(&IMM8) {
        return _mm_setzero_si128();
    }
    unsafe { vreinterpretq_u8_u32(vshlq_u32(vreinterpretq_u32_u8(a), vdupq_n_s32(IMM8))) }
}

#[inline(always)]
pub fn _mm_shuffle_epi32<const IMM8: i32>(a: __m128i) -> __m128i {
    let a: [u32; 4] = unsafe { transmute(a) };
    let m = IMM8 as usize;
    let r = [a[m & 3], a[(m >> 2) & 3], a[(m >> 4) & 3], a[(m >> 6) & 3]];
    unsafe { transmute(r) }
}

#[inline(always)]
pub fn _mm_shufflelo_epi16<const IMM8: i32>(a: __m128i) -> __m128i {
    let a: [u16; 8] = unsafe { transmute(a) };
    let m = IMM8 as usize;
    let r = [
        a[m & 3],
        a[(m >> 2) & 3],
        a[(m >> 4) & 3],
        a[(m >> 6) & 3],
        a[4],
        a[5],
        a[6],
        a[7],
    ];
    unsafe { transmute(r) }
}

#[inline(always)]
pub fn _mm_extract_epi16<const IMM8: i32>(a: __m128i) -> i32 {
    let a: [u16; 8] = unsafe { transmute(a) };
    a[(IMM8 & 7) as usize] as i32
}

#[inline(always)]
pub fn _mm_cvtsi128_si32(a: __m128i) -> i32 {
    unsafe { vgetq_lane_s32::<0>(s32s(a)) }
}
