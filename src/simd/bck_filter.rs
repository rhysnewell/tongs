#![allow(clippy::missing_safety_doc)]

use crate::alphabet::Dsq;
use crate::simd::oprofile::*;
use crate::simd::probmx::{ProbMx, Rows, NXCELLS, PXB, PXC, PXE, PXJ, PXN};
use crate::simd::sse::*;
use crate::util::cmath::c_log_f64;

type Xf = [[f32; P7O_NXTRANS]; P7O_NXSTATES];

// Backward's D->D wing chains a multiply and two adds per stripe, so two targets share each pass
// as Forward's do, each lane scaled by its own Forward rows.
pub struct Backward<'a> {
    dsq: *const Dsq,
    l: usize,
    q: usize,
    xf: Xf,
    fwd_row_scales: *const f32,
    rows: Rows,
    xmx: *mut f32,
    row_scale: *mut f32,
    own_scales: &'a mut bool,
    x_e: f32,
    x_n: f32,
    x_j: f32,
    x_b: f32,
    x_c: f32,
    totscale: f64,
    has_own_scales: bool,
}

impl<'a> Backward<'a> {
    // Without Forward's row scales the pass picks its own, and only then is its score meaningful.
    pub fn new(
        dsq: &'a [Dsq],
        offset: usize,
        l: usize,
        xf: Xf,
        fwd_row_scales: Option<&'a [f32]>,
        pmx: &'a mut ProbMx,
    ) -> Self {
        assert!(l > 0 && pmx.l >= l && offset + l < dsq.len());
        assert!(fwd_row_scales.is_none_or(|scales| scales.len() > l));
        let rows = pmx.rows();
        Backward {
            dsq: unsafe { dsq.as_ptr().add(offset) },
            l,
            q: pmx.q_count(),
            xf,
            fwd_row_scales: fwd_row_scales.map_or(std::ptr::null(), |scales| scales.as_ptr()),
            rows,
            xmx: pmx.xmx.as_mut_ptr(),
            row_scale: pmx.row_scale.as_mut_ptr(),
            own_scales: &mut pmx.has_own_scales,
            x_e: 0.0,
            x_n: 0.0,
            x_j: 0.0,
            x_b: 0.0,
            x_c: 0.0,
            totscale: 0.0,
            has_own_scales: fwd_row_scales.is_none(),
        }
    }

    fn tracks_scales(&self) -> bool {
        self.fwd_row_scales.is_null()
    }

    #[inline(always)]
    unsafe fn store(&mut self, i: usize, row_scale: f32) {
        let x = self.xmx.add(i * NXCELLS);
        *x.add(PXE) = self.x_e;
        *x.add(PXN) = self.x_n;
        *x.add(PXJ) = self.x_j;
        *x.add(PXB) = self.x_b;
        *x.add(PXC) = self.x_c;
        *self.row_scale.add(i) = row_scale;
    }
}

pub unsafe fn backward(om: &OProfile, a: &mut Backward) -> f32 {
    last_row(om, a);
    for k in 1..a.l {
        row::<1>(om, [&mut *a], k);
    }
    first_row(om, a)
}

pub unsafe fn backward_pair<'p>(om: &OProfile, a: &mut Backward<'p>, b: &mut Backward<'p>) {
    last_row(om, a);
    last_row(om, b);
    let common = a.l.min(b.l);
    for k in 1..common {
        row::<2>(om, [&mut *a, &mut *b], k);
    }
    let longer = if a.l > b.l { &mut *a } else { &mut *b };
    for k in common.max(1)..longer.l {
        row::<1>(om, [&mut *longer], k);
    }
    first_row(om, a);
    first_row(om, b);
}

pub unsafe fn backward_parser_pmx(
    dsq: &[Dsq],
    l: usize,
    om: &OProfile,
    fwd_row_scales: Option<&[f32]>,
    pmx: &mut ProbMx,
) -> f32 {
    backward_parser_pmx_offset(dsq, 0, l, om, fwd_row_scales, pmx)
}

pub unsafe fn backward_parser_pmx_offset(
    dsq: &[Dsq],
    offset: usize,
    l: usize,
    om: &OProfile,
    fwd_row_scales: Option<&[f32]>,
    pmx: &mut ProbMx,
) -> f32 {
    backward(
        om,
        &mut Backward::new(dsq, offset, l, om.xf, fwd_row_scales, pmx),
    )
}

#[inline(always)]
unsafe fn shifted(v: __m128) -> __m128 {
    _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(v)))
}

#[inline(always)]
unsafe fn tfv(om: &OProfile, qi: usize, tidx: usize) -> __m128 {
    _mm_load_ps((*om.tfv.as_ptr().add(qi * 7 + tidx)).as_ptr())
}

#[inline(always)]
unsafe fn tdd(om: &OProfile, q: usize, qi: usize) -> __m128 {
    _mm_load_ps((*om.tfv.as_ptr().add(7 * q + qi)).as_ptr())
}

#[inline(always)]
unsafe fn rfv(om: &OProfile, x: usize, qi: usize) -> __m128 {
    _mm_load_ps((*(*om.rfv.as_ptr().add(x)).as_ptr().add(qi)).as_ptr())
}

#[inline(always)]
unsafe fn hsum(v: __m128) -> f32 {
    let v = _mm_add_ps(
        v,
        _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(v, v),
    );
    let v = _mm_add_ps(
        v,
        _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(v, v),
    );
    _mm_cvtss_f32(v)
}

unsafe fn last_row(om: &OProfile, lane: &mut Backward) {
    let q = nqf(om.m);
    assert_eq!(lane.q, q);
    let zerov = _mm_setzero_ps();
    let l = lane.l;
    lane.x_c = lane.xf[P7O_C][P7O_MOVE];
    lane.x_e = lane.x_c * lane.xf[P7O_E][P7O_MOVE];
    let dp = lane.rows.at(l);
    let x_ev = _mm_set1_ps(lane.x_e);
    for qi in 0..q {
        *dp.add(qi * 3) = x_ev;
        *dp.add(qi * 3 + 1) = x_ev;
        *dp.add(qi * 3 + 2) = zerov;
    }
    let mut dpv = shifted(*dp.add((q - 1) * 3 + 1));
    let mut dcv = zerov;
    for qi in (0..q).rev() {
        dcv = _mm_mul_ps(dpv, tdd(om, q, qi));
        let d_ptr = dp.add(qi * 3 + 1);
        *d_ptr = _mm_add_ps(*d_ptr, dcv);
        dpv = *d_ptr;
    }
    for _ in 0..3 {
        dcv = shifted(dcv);
        for qi in (0..q).rev() {
            dcv = _mm_mul_ps(dcv, tdd(om, q, qi));
            let d_ptr = dp.add(qi * 3 + 1);
            *d_ptr = _mm_add_ps(*d_ptr, dcv);
        }
    }
    let mut dcv = shifted(*dp.add(1));
    for qi in (0..q).rev() {
        let m_ptr = dp.add(qi * 3);
        *m_ptr = _mm_add_ps(*m_ptr, _mm_mul_ps(dcv, tfv(om, qi, 4)));
        dcv = *dp.add(qi * 3 + 1);
    }

    let row_scale = if lane.tracks_scales() {
        1.0
    } else {
        *lane.fwd_row_scales.add(l)
    };
    if row_scale > 1.0 {
        let scalev = _mm_set1_ps(1.0 / row_scale);
        lane.x_e /= row_scale;
        lane.x_n /= row_scale;
        lane.x_j /= row_scale;
        lane.x_b /= row_scale;
        lane.x_c /= row_scale;
        for cell in 0..q * 3 {
            *dp.add(cell) = _mm_mul_ps(*dp.add(cell), scalev);
        }
    }
    lane.store(l, row_scale);
}

// Step k fills row l - k of every lane, so lanes of different lengths stay in step from their ends.
#[inline(always)]
unsafe fn row<const N: usize>(om: &OProfile, lanes: [&mut Backward; N], k: usize) {
    let q = nqf(om.m);
    let zerov = _mm_setzero_ps();
    let mut at = [0usize; N];
    let mut next = [0usize; N];
    let mut dpp = [std::ptr::null::<__m128>(); N];
    let mut dpc = [std::ptr::null_mut::<__m128>(); N];
    let mut mpv = [zerov; N];
    let mut x_bv = [zerov; N];
    for t in 0..N {
        at[t] = lanes[t].l - k;
        next[t] = *lanes[t].dsq.add(at[t] + 1) as usize;
        dpp[t] = lanes[t].rows.at(at[t] + 1);
        dpc[t] = lanes[t].rows.at(at[t]);
        mpv[t] = shifted(_mm_mul_ps(*dpp[t], rfv(om, next[t], 0)));
    }
    let mut tmmv = shifted(tfv(om, 0, 1));
    let mut timv = shifted(tfv(om, 0, 2));
    let mut tdmv = shifted(tfv(om, 0, 3));

    for qi in (0..q).rev() {
        let tii = tfv(om, qi, 6);
        let tmi = tfv(om, qi, 5);
        let tbm = tfv(om, qi, 0);
        for t in 0..N {
            let ipv = *dpp[t].add(qi * 3 + 2);
            *dpc[t].add(qi * 3 + 2) = _mm_add_ps(_mm_mul_ps(ipv, tii), _mm_mul_ps(mpv[t], timv));
            *dpc[t].add(qi * 3 + 1) = _mm_mul_ps(mpv[t], tdmv);
            *dpc[t].add(qi * 3) = _mm_add_ps(_mm_mul_ps(ipv, tmi), _mm_mul_ps(mpv[t], tmmv));
            mpv[t] = _mm_mul_ps(*dpp[t].add(qi * 3), rfv(om, next[t], qi));
            x_bv[t] = _mm_add_ps(x_bv[t], _mm_mul_ps(mpv[t], tbm));
        }
        tdmv = tfv(om, qi, 3);
        timv = tfv(om, qi, 2);
        tmmv = tfv(om, qi, 1);
    }

    let mut x_ev = [zerov; N];
    for t in 0..N {
        let lane = &mut *lanes[t];
        lane.x_b = hsum(x_bv[t]);
        let xf = &lane.xf;
        lane.x_c *= xf[P7O_C][P7O_LOOP];
        lane.x_j = lane.x_b * xf[P7O_J][P7O_MOVE] + lane.x_j * xf[P7O_J][P7O_LOOP];
        lane.x_n = lane.x_b * xf[P7O_N][P7O_MOVE] + lane.x_n * xf[P7O_N][P7O_LOOP];
        lane.x_e = lane.x_c * xf[P7O_E][P7O_MOVE] + lane.x_j * xf[P7O_E][P7O_LOOP];
        x_ev[t] = _mm_set1_ps(lane.x_e);
    }

    let mut dpv = [zerov; N];
    let mut dcv = [zerov; N];
    for t in 0..N {
        dpv[t] = shifted(_mm_add_ps(*dpc[t].add(1), x_ev[t]));
    }
    for qi in (0..q).rev() {
        let tddv = tdd(om, q, qi);
        for t in 0..N {
            dcv[t] = _mm_mul_ps(dpv[t], tddv);
            let d_ptr = dpc[t].add(qi * 3 + 1);
            *d_ptr = _mm_add_ps(*d_ptr, _mm_add_ps(dcv[t], x_ev[t]));
            dpv[t] = *d_ptr;
            let m_ptr = dpc[t].add(qi * 3);
            *m_ptr = _mm_add_ps(*m_ptr, x_ev[t]);
        }
    }
    for _ in 0..3 {
        for v in dcv.iter_mut() {
            *v = shifted(*v);
        }
        for qi in (0..q).rev() {
            let tddv = tdd(om, q, qi);
            for t in 0..N {
                dcv[t] = _mm_mul_ps(dcv[t], tddv);
                let d_ptr = dpc[t].add(qi * 3 + 1);
                *d_ptr = _mm_add_ps(*d_ptr, dcv[t]);
            }
        }
    }

    let mut mdv = [zerov; N];
    for t in 0..N {
        mdv[t] = shifted(*dpc[t].add(1));
    }
    for qi in (0..q).rev() {
        let tmd = tfv(om, qi, 4);
        for t in 0..N {
            let m_ptr = dpc[t].add(qi * 3);
            *m_ptr = _mm_add_ps(*m_ptr, _mm_mul_ps(mdv[t], tmd));
            mdv[t] = *dpc[t].add(qi * 3 + 1);
        }
    }

    for t in 0..N {
        rescale(&mut *lanes[t], dpc[t], q, at[t]);
    }
}

#[inline(always)]
unsafe fn rescale(lane: &mut Backward, dpc: *mut __m128, q: usize, i: usize) {
    if lane.x_b > 1.0e16 {
        lane.has_own_scales = true;
    }
    let row_scale = if lane.has_own_scales {
        if lane.x_b > 1.0e4 {
            lane.x_b
        } else {
            1.0
        }
    } else {
        *lane.fwd_row_scales.add(i)
    };
    if row_scale > 1.0 {
        let scalev = _mm_set1_ps(1.0 / row_scale);
        lane.x_e /= row_scale;
        lane.x_n /= row_scale;
        lane.x_j /= row_scale;
        lane.x_c /= row_scale;
        for cell in 0..q * 3 {
            *dpc.add(cell) = _mm_mul_ps(*dpc.add(cell), scalev);
        }
        if lane.tracks_scales() {
            lane.totscale += c_log_f64(row_scale as f64);
        }
        if lane.has_own_scales {
            lane.x_b = 1.0;
        } else {
            lane.x_b /= row_scale;
        }
    }
    lane.store(i, row_scale);
}

unsafe fn first_row(om: &OProfile, lane: &mut Backward) -> f32 {
    let q = nqf(om.m);
    let dp = lane.rows.at(1);
    let x1 = *lane.dsq.add(1) as usize;
    let mut x_bv = _mm_setzero_ps();
    for qi in 0..q {
        let mpv = _mm_mul_ps(*dp.add(qi * 3), rfv(om, x1, qi));
        x_bv = _mm_add_ps(x_bv, _mm_mul_ps(mpv, tfv(om, qi, 0)));
    }
    lane.x_b = hsum(x_bv);
    lane.x_n = lane.x_b * lane.xf[P7O_N][P7O_MOVE] + lane.x_n * lane.xf[P7O_N][P7O_LOOP];
    lane.x_e = 0.0;
    lane.x_j = 0.0;
    lane.x_c = 0.0;
    lane.store(0, 1.0);
    *lane.own_scales = lane.has_own_scales;
    (lane.totscale + c_log_f64(lane.x_n as f64)) as f32
}
