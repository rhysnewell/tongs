//! SSE optimal-accuracy DP over striped posterior matrices.
//! Direct port of HMMER impl_sse/optacc.c p7_OptimalAccuracy() for the
//! coordinate-only domain path.

use std::arch::x86_64::*;

use crate::simd::oprofile::*;
use crate::simd::probmx::{ProbMx, PXB, PXC, PXE, PXJ, PXN};
use crate::trace::{State, Trace};

const NXCELLS: usize = 5;
const DP_CELLS_PER_K: usize = 3; // striped order: M, D, I
const OM_M: usize = 0;
const OM_D: usize = 1;
const OM_I: usize = 2;

/// Load the 4-lane stripe `(qi, state)` from a striped DP row.
#[inline(always)]
unsafe fn load_cell(row: *const f32, qi: usize, state: usize) -> __m128 {
    _mm_load_ps(row.add(qi * DP_CELLS_PER_K * 4 + state * 4))
}

/// Store the 4-lane stripe `(qi, state)` into a striped DP row.
#[inline(always)]
unsafe fn store_cell(row: *mut f32, qi: usize, state: usize, v: __m128) {
    _mm_store_ps(row.add(qi * DP_CELLS_PER_K * 4 + state * 4), v);
}

/// Striped right-shift by one lane (Easel `esl_sse_rightshift_ps`):
/// `[a,b,c,d]` becomes `[fill, a, b, c]`.
#[inline(always)]
unsafe fn rightshift_ps(v: __m128, fill: __m128) -> __m128 {
    let shifted = _mm_shuffle_ps::<{ crate::simd::shuffle_mask(2, 1, 0, 0) }>(v, v);
    _mm_move_ss(shifted, fill)
}

/// Horizontal max across the 4 lanes of `v` (Easel `esl_sse_hmax_ps`).
#[inline(always)]
unsafe fn hmax_ps(v: __m128) -> f32 {
    let h1 = _mm_max_ps(
        v,
        _mm_shuffle_ps::<{ crate::simd::shuffle_mask(0, 3, 2, 1) }>(v, v),
    );
    let h2 = _mm_max_ps(
        h1,
        _mm_shuffle_ps::<{ crate::simd::shuffle_mask(1, 0, 3, 2) }>(h1, h1),
    );
    let mut out = 0.0_f32;
    _mm_store_ss(&mut out, h2);
    out
}

mod decode;
mod fill;
mod trace;

pub use fill::decode_and_fill;
pub use trace::{oa_trace_coords_pmx, oa_trace_pmx};
