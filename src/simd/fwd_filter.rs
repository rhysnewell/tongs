#![allow(clippy::missing_safety_doc)]

use crate::alphabet::Dsq;
use crate::simd::oprofile::*;
use crate::simd::probmx::{ProbMx, Rows, NXCELLS, PXB, PXC, PXE, PXJ, PXN};
use crate::simd::sse::*;
use crate::util::cmath::c_log_f64;

type Xf = [[f32; P7O_NXTRANS]; P7O_NXSTATES];

// One target's Forward state. Two lanes share each pass against one profile so the D->D add and
// multiply chain of one fills the latency of the other, each lane in the single target's order.
pub struct Forward<'a> {
    dsq: *const Dsq,
    l: usize,
    q: usize,
    xf: Xf,
    rows: Rows,
    xmx: *mut f32,
    row_scale: *mut f32,
    _pmx: std::marker::PhantomData<&'a mut ProbMx>,
    xe: f32,
    xn: f32,
    xj: f32,
    xb: f32,
    xc: f32,
    totscale: f32,
}

impl<'a> Forward<'a> {
    // `xf` is the target's own length model, since a shared profile holds whichever was set last.
    pub fn new(dsq: &'a [Dsq], offset: usize, l: usize, xf: Xf, pmx: &'a mut ProbMx) -> Self {
        assert!(pmx.l >= l && offset + l < dsq.len());
        let xb = xf[P7O_N][P7O_MOVE];
        Forward {
            dsq: unsafe { dsq.as_ptr().add(offset) },
            l,
            q: pmx.q_count(),
            xf,
            rows: pmx.rows(),
            xmx: pmx.xmx.as_mut_ptr(),
            row_scale: pmx.row_scale.as_mut_ptr(),
            _pmx: std::marker::PhantomData,
            xe: 0.0,
            xn: 1.0,
            xj: 0.0,
            xb,
            xc: 0.0,
            totscale: 0.0,
        }
    }

    unsafe fn start(&mut self, q: usize) {
        assert_eq!(self.q, q);
        let row = self.rows.at(0);
        for cell in 0..q * 3 {
            *row.add(cell) = _mm_setzero_ps();
        }
        self.store(0, 1.0);
    }

    #[inline(always)]
    unsafe fn store(&mut self, i: usize, row_scale: f32) {
        let x = self.xmx.add(i * NXCELLS);
        *x.add(PXE) = self.xe;
        *x.add(PXN) = self.xn;
        *x.add(PXJ) = self.xj;
        *x.add(PXB) = self.xb;
        *x.add(PXC) = self.xc;
        *self.row_scale.add(i) = row_scale;
    }

    fn score(&self) -> f32 {
        let xc = self.xc;
        if xc.is_nan() || (self.l > 0 && xc == 0.0) || xc.is_infinite() {
            f32::NEG_INFINITY
        } else {
            (self.totscale as f64 + c_log_f64((xc * self.xf[P7O_C][P7O_MOVE]) as f64)) as f32
        }
    }
}

pub unsafe fn forward(om: &OProfile, a: &mut Forward) -> f32 {
    let q = nqf(om.m);
    a.start(q);
    for i in 1..=a.l {
        row::<1>(om, [&mut *a], i);
    }
    a.score()
}

pub unsafe fn forward_pair<'p>(
    om: &OProfile,
    a: &mut Forward<'p>,
    b: &mut Forward<'p>,
) -> (f32, f32) {
    let q = nqf(om.m);
    a.start(q);
    b.start(q);
    let common = a.l.min(b.l);
    for i in 1..=common {
        row::<2>(om, [&mut *a, &mut *b], i);
    }
    let longer = if a.l > b.l { &mut *a } else { &mut *b };
    for i in common + 1..=longer.l {
        row::<1>(om, [&mut *longer], i);
    }
    (a.score(), b.score())
}

pub unsafe fn forward_parser(dsq: &[Dsq], l: usize, om: &OProfile) -> f32 {
    let mut pmx = ProbMx::default();
    pmx.resize_parser(om.m, l);
    forward_parser_pmx(dsq, l, om, &mut pmx)
}

pub unsafe fn forward_parser_pmx(dsq: &[Dsq], l: usize, om: &OProfile, pmx: &mut ProbMx) -> f32 {
    forward_parser_pmx_offset(dsq, 0, l, om, pmx)
}

pub unsafe fn forward_parser_pmx_offset(
    dsq: &[Dsq],
    offset: usize,
    l: usize,
    om: &OProfile,
    pmx: &mut ProbMx,
) -> f32 {
    forward(om, &mut Forward::new(dsq, offset, l, om.xf, pmx))
}

#[inline(always)]
unsafe fn row<const N: usize>(om: &OProfile, lanes: [&mut Forward; N], i: usize) {
    let q_count = nqf(om.m);
    let zerov = _mm_setzero_ps();
    let rfv_ptr = om.rfv.as_ptr();
    let tfv_ptr = om.tfv.as_ptr();
    let mut prev = [std::ptr::null::<__m128>(); N];
    let mut cur = [std::ptr::null_mut::<__m128>(); N];
    let mut rsc = [std::ptr::null::<AlignedF32x4>(); N];
    let mut xbv = [zerov; N];
    let mut dcv = [zerov; N];
    let mut xev = [zerov; N];
    let mut mpv = [zerov; N];
    let mut dpv = [zerov; N];
    let mut ipv = [zerov; N];
    for t in 0..N {
        prev[t] = lanes[t].rows.at(i - 1);
        cur[t] = lanes[t].rows.at(i);
        rsc[t] = (*rfv_ptr.add(*lanes[t].dsq.add(i) as usize)).as_ptr();
        xbv[t] = _mm_set1_ps(lanes[t].xb);
        mpv[t] = rightshift_float(*prev[t].add((q_count - 1) * 3));
        dpv[t] = rightshift_float(*prev[t].add((q_count - 1) * 3 + 1));
        ipv[t] = rightshift_float(*prev[t].add((q_count - 1) * 3 + 2));
    }

    for q in 0..q_count {
        let base = q * 7;
        let tbm = _mm_load_ps((*tfv_ptr.add(base)).as_ptr());
        let tmm = _mm_load_ps((*tfv_ptr.add(base + 1)).as_ptr());
        let tim = _mm_load_ps((*tfv_ptr.add(base + 2)).as_ptr());
        let tdm = _mm_load_ps((*tfv_ptr.add(base + 3)).as_ptr());
        let tmd = _mm_load_ps((*tfv_ptr.add(base + 4)).as_ptr());
        let tmi = _mm_load_ps((*tfv_ptr.add(base + 5)).as_ptr());
        let tii = _mm_load_ps((*tfv_ptr.add(base + 6)).as_ptr());
        for t in 0..N {
            let rsc_v = _mm_load_ps((*rsc[t].add(q)).as_ptr());
            let mut sv = _mm_mul_ps(xbv[t], tbm);
            sv = _mm_add_ps(sv, _mm_mul_ps(mpv[t], tmm));
            sv = _mm_add_ps(sv, _mm_mul_ps(ipv[t], tim));
            sv = _mm_add_ps(sv, _mm_mul_ps(dpv[t], tdm));
            sv = _mm_mul_ps(sv, rsc_v);
            xev[t] = _mm_add_ps(xev[t], sv);
            let old = prev[t].add(q * 3);
            mpv[t] = *old;
            dpv[t] = *old.add(1);
            ipv[t] = *old.add(2);
            let cell = cur[t].add(q * 3);
            *cell = sv;
            *cell.add(1) = dcv[t];
            dcv[t] = _mm_mul_ps(sv, tmd);
            *cell.add(2) = _mm_add_ps(_mm_mul_ps(mpv[t], tmi), _mm_mul_ps(ipv[t], tii));
        }
    }

    let dd = tfv_ptr.add(7 * q_count);
    for t in 0..N {
        dcv[t] = rightshift_float(dcv[t]);
        *cur[t].add(1) = zerov;
    }
    for q in 0..q_count {
        let tdd = _mm_load_ps((*dd.add(q)).as_ptr());
        for t in 0..N {
            let cell = cur[t].add(q * 3 + 1);
            *cell = _mm_add_ps(dcv[t], *cell);
            dcv[t] = _mm_mul_ps(*cell, tdd);
        }
    }
    if om.m < 100 {
        for _ in 0..3 {
            for v in dcv.iter_mut() {
                *v = rightshift_float(*v);
            }
            for q in 0..q_count {
                let tdd = _mm_load_ps((*dd.add(q)).as_ptr());
                for t in 0..N {
                    let cell = cur[t].add(q * 3 + 1);
                    *cell = _mm_add_ps(dcv[t], *cell);
                    dcv[t] = _mm_mul_ps(dcv[t], tdd);
                }
            }
        }
    } else {
        let mut open = [true; N];
        for _ in 0..3 {
            let mut cv = [zerov; N];
            for v in dcv.iter_mut() {
                *v = rightshift_float(*v);
            }
            for q in 0..q_count {
                let tdd = _mm_load_ps((*dd.add(q)).as_ptr());
                for t in 0..N {
                    if open[t] {
                        let cell = cur[t].add(q * 3 + 1);
                        let sv = _mm_add_ps(dcv[t], *cell);
                        cv[t] = _mm_or_ps(cv[t], _mm_cmpgt_ps(sv, *cell));
                        *cell = sv;
                        dcv[t] = _mm_mul_ps(dcv[t], tdd);
                    }
                }
            }
            for t in 0..N {
                open[t] = open[t] && _mm_movemask_ps(cv[t]) != 0;
            }
            if open.iter().all(|o| !o) {
                break;
            }
        }
    }

    for t in 0..N {
        let lane = &mut *lanes[t];
        let mut xe_sum = xev[t];
        for q in 0..q_count {
            xe_sum = _mm_add_ps(*cur[t].add(q * 3 + 1), xe_sum);
        }
        xe_sum = _mm_add_ps(
            xe_sum,
            _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(xe_sum, xe_sum),
        );
        xe_sum = _mm_add_ps(
            xe_sum,
            _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(xe_sum, xe_sum),
        );
        _mm_store_ss(&mut lane.xe, xe_sum);
        specials(lane, cur[t], q_count, i);
    }
}

#[inline(always)]
unsafe fn specials(lane: &mut Forward, cur: *mut __m128, q_count: usize, i: usize) {
    let xf = &lane.xf;
    lane.xn *= xf[P7O_N][P7O_LOOP];
    lane.xc = lane.xc * xf[P7O_C][P7O_LOOP] + lane.xe * xf[P7O_E][P7O_MOVE];
    lane.xj = lane.xj * xf[P7O_J][P7O_LOOP] + lane.xe * xf[P7O_E][P7O_LOOP];
    lane.xb = lane.xj * xf[P7O_J][P7O_MOVE] + lane.xn * xf[P7O_N][P7O_MOVE];
    let row_scale = if lane.xe > 1.0e4 {
        let row_scale = lane.xe;
        let scalev = _mm_set1_ps(1.0 / lane.xe);
        lane.xn /= row_scale;
        lane.xc /= row_scale;
        lane.xj /= row_scale;
        lane.xb /= row_scale;
        for cell in 0..q_count * 3 {
            let p = cur.add(cell);
            *p = _mm_mul_ps(*p, scalev);
        }
        lane.totscale = (lane.totscale as f64 + c_log_f64(lane.xe as f64)) as f32;
        lane.xe = 1.0;
        row_scale
    } else {
        1.0
    };
    lane.store(i, row_scale);
}

#[inline(always)]
pub(crate) unsafe fn rightshift_float(v: __m128) -> __m128 {
    _mm_castsi128_ps(_mm_slli_si128::<4>(_mm_castps_si128(v)))
}
