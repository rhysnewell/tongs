pub mod oprofile;
pub mod optacc;

pub mod bck_filter;
pub mod fm_ssv;
pub mod fwd_filter;
pub mod msv_filter;
pub mod probmx;
pub mod ssv;
pub mod ssv_longtarget;
pub mod vit_filter;
pub mod vit_wide;

#[cfg(test)]
#[path = "../../tests/unit/simd/fixtures.rs"]
pub(crate) mod fixtures;

#[cfg(target_arch = "x86_64")]
pub use std::arch::x86_64 as sse;
#[cfg(target_arch = "aarch64")]
#[path = "sse_neon.rs"]
pub mod sse;

/// Build an 8-bit immediate shuffle mask for `_mm_shuffle_epi32` / `_mm_shufflelo_epi16`.
/// The four 2-bit fields select source lanes (z = high, w = low).
pub const fn shuffle_mask(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}
