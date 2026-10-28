#![allow(clippy::too_many_arguments)]

use crate::dp::gmx::*;
use crate::simd::oprofile::{nqf, AlignedF32x4};
use crate::simd::sse::*;

pub const PXE: usize = 0;
pub const PXN: usize = 1;
pub const PXJ: usize = 2;
pub const PXB: usize = 3;
pub const PXC: usize = 4;
pub(crate) const NXCELLS: usize = 5;
const DP_CELLS_PER_K: usize = 3;

fn aligned_f32_offset(v: &[f32]) -> usize {
    let misalignment = (v.as_ptr() as usize) & 15;
    if misalignment == 0 {
        0
    } else {
        (16 - misalignment) / std::mem::size_of::<f32>()
    }
}

fn checked_probmx_shape(m: usize, l: usize) -> (usize, usize, usize) {
    let rows = l.checked_add(1).expect("ProbMx row count overflow");
    let q = nqf(m);
    let striped_row_width = q
        .checked_mul(DP_CELLS_PER_K * 4)
        .expect("ProbMx striped row width overflow");
    (rows, q, striped_row_width)
}

// A parser keeps two rows that the kernels roll through, so the pair and single kernels share one
// row walk with a full matrix.
#[derive(Default)]
pub struct ProbMx {
    pub m: usize,
    pub l: usize,
    pub xmx: Vec<f32>,
    pub row_scale: Vec<f32>,
    pub striped_dp: Vec<f32>,
    pub striped_dp_offset: usize,
    striped_row_width: usize,
    q: usize,
    pub has_dp: bool,
    pub has_own_scales: bool,
}

impl ProbMx {
    pub fn new_full(m: usize, l: usize) -> Self {
        let mut mx = ProbMx::default();
        mx.resize_full(m, l);
        mx
    }

    pub fn resize_parser(&mut self, m: usize, l: usize) {
        self.shape(m, l, 2, false);
    }

    pub fn resize_full(&mut self, m: usize, l: usize) {
        let rows = l.checked_add(1).expect("ProbMx row count overflow");
        self.shape(m, l, rows, true);
    }

    // The kernels overwrite every row they read, so storage only grows and only row 0 is cleared.
    fn shape(&mut self, m: usize, l: usize, dp_rows: usize, full: bool) {
        let (rows, q, striped_row_width) = checked_probmx_shape(m, l);
        let cells = dp_rows
            .checked_mul(striped_row_width)
            .and_then(|cells| cells.checked_add(4))
            .expect("ProbMx striped matrix size overflow");
        self.m = m;
        self.l = l;
        self.q = q;
        self.striped_row_width = striped_row_width;
        self.has_dp = full;
        self.has_own_scales = false;
        self.xmx.resize(rows * NXCELLS, 0.0);
        self.row_scale.resize(rows, 1.0);
        if self.striped_dp.len() < cells {
            self.striped_dp.resize(cells, 0.0);
        }
        self.striped_dp_offset = aligned_f32_offset(&self.striped_dp);
        let row0 = self.striped_dp_offset;
        self.striped_dp[row0..row0 + striped_row_width].fill(0.0);
    }

    #[inline]
    pub fn xmx(&self, i: usize, s: usize) -> f32 {
        self.xmx[i * NXCELLS + s]
    }

    #[inline]
    pub fn set_xmx(&mut self, i: usize, s: usize, val: f32) {
        self.xmx[i * NXCELLS + s] = val;
    }

    #[inline]
    pub fn striped_row_width(&self) -> usize {
        self.striped_row_width
    }

    #[inline]
    pub fn q_count(&self) -> usize {
        self.q
    }

    pub(crate) fn rows(&mut self) -> Rows {
        Rows {
            base: unsafe {
                self.striped_dp
                    .as_mut_ptr()
                    .add(self.striped_dp_offset)
                    .cast::<__m128>()
            },
            stride: self.q * DP_CELLS_PER_K,
            mask: if self.has_dp { usize::MAX } else { 1 },
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Rows {
    base: *mut __m128,
    stride: usize,
    mask: usize,
}

impl Rows {
    #[inline(always)]
    pub(crate) unsafe fn at(self, i: usize) -> *mut __m128 {
        self.base.add((i & self.mask) * self.stride)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub(crate) fn keeps_all(self) -> bool {
        self.mask == usize::MAX
    }
}

pub fn p_domain_decoding_reuse(
    fwd: &ProbMx,
    bck: &ProbMx,
    l: usize,
    njc_loop: [f32; 3],
    btot: &mut Vec<f32>,
    etot: &mut Vec<f32>,
    mocc: &mut Vec<f32>,
) {
    let [n_loop, j_loop, c_loop] = njc_loop;
    let bck_n0 = bck.xmx(0, PXN);
    let mut scaleproduct = if bck_n0 > 0.0 {
        (1.0_f64 / bck_n0 as f64) as f32
    } else {
        0.0
    };

    btot.resize(l + 1, 0.0);
    etot.resize(l + 1, 0.0);
    mocc.resize(l + 1, 0.0);
    btot[0] = 0.0;
    etot[0] = 0.0;
    mocc[0] = 0.0;

    for i in 1..=l {
        btot[i] = btot[i - 1];
        let mut b = fwd.xmx(i - 1, PXB);
        b *= bck.xmx(i - 1, PXB);
        b *= fwd.row_scale[i - 1];
        b *= scaleproduct;
        btot[i] += b;

        if bck.has_own_scales {
            scaleproduct *= fwd.row_scale[i - 1] / bck.row_scale[i - 1];
        }

        etot[i] = etot[i - 1];
        let mut e = fwd.xmx(i, PXE);
        e *= bck.xmx(i, PXE);
        e *= fwd.row_scale[i];
        e *= scaleproduct;
        etot[i] += e;

        let mut njcp = fwd.xmx(i - 1, PXN);
        njcp *= bck.xmx(i, PXN);
        njcp *= n_loop;
        njcp *= scaleproduct;

        let mut term = fwd.xmx(i - 1, PXJ);
        term *= bck.xmx(i, PXJ);
        term *= j_loop;
        term *= scaleproduct;
        njcp += term;

        term = fwd.xmx(i - 1, PXC);
        term *= bck.xmx(i, PXC);
        term *= c_loop;
        term *= scaleproduct;
        njcp += term;
        mocc[i] = (1.0_f64 - njcp as f64) as f32;
    }
}

pub fn p_decoding_to_gmx(fwd: &ProbMx, bck: &ProbMx, m: usize, njc_loop: [f32; 3], pp: &mut Gmx) {
    let l = fwd.l;
    let fdp = fwd.striped_dp.as_ptr().wrapping_add(fwd.striped_dp_offset);
    let bdp = bck.striped_dp.as_ptr().wrapping_add(bck.striped_dp_offset);
    let fx = fwd.xmx.as_ptr();
    let bx = bck.xmx.as_ptr();
    let ppdp = pp.dp_mem.as_mut_ptr();
    let ppx = pp.xmx.as_mut_ptr();

    let bck_n0 = unsafe { *bx.add(PXN) };
    let inv_total = if bck_n0 > 0.0 { 1.0 / bck_n0 } else { 0.0 };
    let q = fwd.q;
    let pp_stride = pp.row_width() * P7G_NSCELLS;

    pp.m = m;
    pp.l = l;

    unsafe {
        for node in 0..=m {
            let idx = node * P7G_NSCELLS;
            *ppdp.add(idx + P7G_M) = 0.0;
            *ppdp.add(idx + P7G_I) = 0.0;
            *ppdp.add(idx + P7G_D) = 0.0;
        }
        for s in [P7G_E, P7G_N, P7G_J, P7G_B, P7G_C] {
            *ppx.add(s) = 0.0;
        }

        let mut scaleproduct = inv_total;
        for i in 1..=l {
            let dp_scale = scaleproduct * fwd.row_scale[i];
            let frow = i * fwd.striped_row_width;
            let brow = i * bck.striped_row_width;
            let prow = i * pp_stride;
            let xrow = i * P7G_NXCELLS;

            *ppdp.add(prow + P7G_M) = 0.0;
            *ppdp.add(prow + P7G_I) = 0.0;
            *ppdp.add(prow + P7G_D) = 0.0;

            for qi in 0..q {
                let fbase = frow + qi * DP_CELLS_PER_K * 4;
                let bbase = brow + qi * DP_CELLS_PER_K * 4;
                for lane in 0..4 {
                    let node = 1 + qi + lane * q;
                    if node > m {
                        continue;
                    }
                    let pidx = prow + node * P7G_NSCELLS;
                    *ppdp.add(pidx + P7G_M) =
                        *fdp.add(fbase + lane) * *bdp.add(bbase + lane) * dp_scale;
                    *ppdp.add(pidx + P7G_D) = 0.0;
                    *ppdp.add(pidx + P7G_I) = if node < m {
                        *fdp.add(fbase + 8 + lane) * *bdp.add(bbase + 8 + lane) * dp_scale
                    } else {
                        0.0
                    };
                }
            }

            let special = |s: usize, t: usize| {
                *fx.add((i - 1) * NXCELLS + s)
                    * *bx.add(i * NXCELLS + s)
                    * njc_loop[t]
                    * scaleproduct
            };
            *ppx.add(xrow + P7G_E) = 0.0;
            *ppx.add(xrow + P7G_N) = special(PXN, 0);
            *ppx.add(xrow + P7G_J) = special(PXJ, 1);
            *ppx.add(xrow + P7G_B) = 0.0;
            *ppx.add(xrow + P7G_C) = special(PXC, 2);

            if bck.has_own_scales {
                scaleproduct *= fwd.row_scale[i] / bck.row_scale[i];
            }
        }
    }
}

pub fn p_null2_odds_from_omx_expectation_reuse(
    fwd: &ProbMx,
    bck: &ProbMx,
    k: usize,
    rfv: &[Vec<AlignedF32x4>],
    njc_loop: [f32; 3],
    null2: &mut Vec<f32>,
    exp_m: &mut Vec<f32>,
    exp_i: &mut Vec<f32>,
) {
    if !fwd.has_dp || !bck.has_dp {
        return;
    }

    let ld = fwd.l;
    let q = fwd.q;
    let exp_len = q * 4;
    exp_m.clear();
    exp_m.resize(exp_len, 0.0);
    exp_i.clear();
    exp_i.resize(exp_len, 0.0);

    let fdp = fwd.striped_dp.as_ptr().wrapping_add(fwd.striped_dp_offset);
    let bdp = bck.striped_dp.as_ptr().wrapping_add(bck.striped_dp_offset);
    let fx = fwd.xmx.as_ptr();
    let bx = bck.xmx.as_ptr();
    let exp_m_ptr = exp_m.as_mut_ptr();
    let exp_i_ptr = exp_i.as_mut_ptr();

    let mut exp_n = 0.0_f32;
    let mut exp_c = 0.0_f32;
    let mut exp_j = 0.0_f32;
    let bck_n0 = bck.xmx(0, PXN);
    let mut scaleproduct = if bck_n0 > 0.0 { 1.0 / bck_n0 } else { 0.0 };

    unsafe {
        for i in 1..=ld {
            let scalev = _mm_set1_ps(scaleproduct * fwd.row_scale[i]);
            let frow = i * fwd.striped_row_width;
            let brow = i * bck.striped_row_width;

            for qi in 0..q {
                let fbase = frow + qi * DP_CELLS_PER_K * 4;
                let bbase = brow + qi * DP_CELLS_PER_K * 4;
                let ebase = qi * 4;

                let row_m = _mm_mul_ps(
                    _mm_mul_ps(_mm_load_ps(fdp.add(fbase)), _mm_load_ps(bdp.add(bbase))),
                    scalev,
                );
                let old_m = _mm_loadu_ps(exp_m_ptr.add(ebase));
                _mm_storeu_ps(exp_m_ptr.add(ebase), _mm_add_ps(row_m, old_m));

                let row_i = _mm_mul_ps(
                    _mm_mul_ps(
                        _mm_load_ps(fdp.add(fbase + 8)),
                        _mm_load_ps(bdp.add(bbase + 8)),
                    ),
                    scalev,
                );
                let old_i = _mm_loadu_ps(exp_i_ptr.add(ebase));
                _mm_storeu_ps(exp_i_ptr.add(ebase), _mm_add_ps(row_i, old_i));
            }

            let special = |s: usize, t: usize| {
                *fx.add((i - 1) * NXCELLS + s)
                    * *bx.add(i * NXCELLS + s)
                    * njc_loop[t]
                    * scaleproduct
            };
            exp_n += special(PXN, 0);
            exp_j += special(PXJ, 1);
            exp_c += special(PXC, 2);

            if bck.has_own_scales {
                scaleproduct *= fwd.row_scale[i] / bck.row_scale[i];
            }
        }

        let norm = 1.0 / ld as f32;
        let normv = _mm_set1_ps(norm);
        for qi in 0..q {
            let ebase = qi * 4;
            let m = _mm_mul_ps(_mm_loadu_ps(exp_m_ptr.add(ebase)), normv);
            let ins = _mm_mul_ps(_mm_loadu_ps(exp_i_ptr.add(ebase)), normv);
            _mm_storeu_ps(exp_m_ptr.add(ebase), m);
            _mm_storeu_ps(exp_i_ptr.add(ebase), ins);
        }
        exp_n *= norm;
        exp_c *= norm;
        exp_j *= norm;
        let xfactor = exp_n + exp_c + exp_j;

        null2.resize(k, 1.0);
        for (odds, slot) in rfv.iter().zip(null2.iter_mut()) {
            let mut sv = _mm_setzero_ps();
            for (qi, stripe) in odds.iter().enumerate().take(q) {
                let ebase = qi * 4;
                let odds = _mm_loadu_ps(stripe.as_ptr());
                sv = _mm_add_ps(sv, _mm_mul_ps(_mm_loadu_ps(exp_m_ptr.add(ebase)), odds));
                sv = _mm_add_ps(sv, _mm_loadu_ps(exp_i_ptr.add(ebase)));
            }
            sv = _mm_add_ps(sv, _mm_shuffle_ps::<0b00_11_10_01>(sv, sv));
            sv = _mm_add_ps(sv, _mm_shuffle_ps::<0b01_00_11_10>(sv, sv));
            *slot = _mm_cvtss_f32(sv) + xfactor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_matrix_shape_uses_minimum_two_simd_stripes() {
        let mx = ProbMx::new_full(1, 1);
        assert_eq!(mx.striped_row_width(), 2 * DP_CELLS_PER_K * 4);
        assert_eq!(mx.q_count(), 2);

        let mut reused = ProbMx::default();
        reused.resize_full(1, 2);
        assert_eq!(reused.striped_row_width(), 2 * DP_CELLS_PER_K * 4);
        assert_eq!(reused.q_count(), 2);
    }

    #[test]
    #[should_panic(expected = "ProbMx")]
    fn checked_shape_rejects_size_overflow() {
        let mut mx = ProbMx::default();
        mx.resize_full(usize::MAX / 2, usize::MAX / 2);
    }
}
