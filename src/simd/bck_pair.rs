use crate::alphabet::Dsq;
use crate::simd::oprofile::*;
use crate::simd::probmx::{ProbMx, PXB, PXC, PXE, PXJ, PXN};
use crate::simd::sse::*;

// Backward's D->D wing chains a multiply and two adds per stripe, so two targets share each pass
// as Forward's pair does. Each lane runs the parser's arithmetic in its order, scaled by its own
// Forward rows.
pub struct BackParse<'a> {
    dsq: &'a [Dsq],
    l: usize,
    xf: [[f32; P7O_NXTRANS]; P7O_NXSTATES],
    fwd_row_scales: &'a [f32],
    pmx: &'a mut ProbMx,
    dpp: &'a mut Vec<__m128>,
    dpc: &'a mut Vec<__m128>,
    x_e: f32,
    x_n: f32,
    x_j: f32,
    x_b: f32,
    x_c: f32,
    has_own_scales: bool,
}

impl<'a> BackParse<'a> {
    pub fn new(
        dsq: &'a [Dsq],
        l: usize,
        xf: [[f32; P7O_NXTRANS]; P7O_NXSTATES],
        fwd_row_scales: &'a [f32],
        pmx: &'a mut ProbMx,
        dpp: &'a mut Vec<__m128>,
        dpc: &'a mut Vec<__m128>,
    ) -> Self {
        pmx.resize_parser(l);
        BackParse {
            dsq,
            l,
            xf,
            fwd_row_scales,
            pmx,
            dpp,
            dpc,
            x_e: 0.0,
            x_n: 0.0,
            x_j: 0.0,
            x_b: 0.0,
            x_c: 0.0,
            has_own_scales: false,
        }
    }
}

pub unsafe fn backward_parser_pair<'p>(
    om: &OProfile,
    a: &mut BackParse<'p>,
    b: &mut BackParse<'p>,
) {
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

#[inline(always)]
unsafe fn shifted(v: __m128) -> __m128 {
    _mm_castsi128_ps(_mm_srli_si128::<4>(_mm_castps_si128(v)))
}

#[inline(always)]
unsafe fn tfv(om: &OProfile, qi: usize, tidx: usize) -> __m128 {
    _mm_load_ps((*om.tfv_a.as_ptr().add(qi * 7 + tidx)).as_ptr())
}

#[inline(always)]
unsafe fn tdd(om: &OProfile, q: usize, qi: usize) -> __m128 {
    _mm_load_ps((*om.tfv_a.as_ptr().add(7 * q + qi)).as_ptr())
}

#[inline(always)]
unsafe fn rfv(om: &OProfile, x: usize, qi: usize) -> __m128 {
    _mm_load_ps((*(*om.rfv_a.as_ptr().add(x)).as_ptr().add(qi)).as_ptr())
}

unsafe fn last_row(om: &OProfile, lane: &mut BackParse) {
    let q = nqf(om.m);
    let zerov = _mm_setzero_ps();
    let l = lane.l;
    lane.dpp.clear();
    lane.dpp.resize(q * 3, zerov);
    lane.dpc.clear();
    lane.dpc.resize(q * 3, zerov);
    lane.x_c = lane.xf[P7O_C][P7O_MOVE];
    lane.x_e = lane.x_c * lane.xf[P7O_E][P7O_MOVE];
    let dp = lane.dpp.as_mut_ptr();
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

    let row_scale = lane.fwd_row_scales[l];
    if row_scale > 1.0 {
        let scalev = _mm_set1_ps(1.0 / row_scale);
        lane.x_e /= row_scale;
        lane.x_n /= row_scale;
        lane.x_j /= row_scale;
        lane.x_b /= row_scale;
        lane.x_c /= row_scale;
        for v in lane.dpp.iter_mut() {
            *v = _mm_mul_ps(*v, scalev);
        }
    }
    let pmx = &mut *lane.pmx;
    pmx.set_xmx(l, PXE, lane.x_e);
    pmx.set_xmx(l, PXN, 0.0);
    pmx.set_xmx(l, PXJ, 0.0);
    pmx.set_xmx(l, PXB, 0.0);
    pmx.set_xmx(l, PXC, lane.x_c);
    pmx.scale[l] = 0.0;
    pmx.row_scale[l] = row_scale;
}

// Step k fills row l - k of every lane, so lanes of different lengths stay in step from their ends.
#[inline(always)]
unsafe fn row<const N: usize>(om: &OProfile, lanes: [&mut BackParse; N], k: usize) {
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
        next[t] = lanes[t].dsq[at[t] + 1] as usize;
        dpp[t] = lanes[t].dpp.as_ptr();
        dpc[t] = lanes[t].dpc.as_mut_ptr();
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
            let mcv = _mm_add_ps(_mm_mul_ps(ipv, tmi), _mm_mul_ps(mpv[t], tmmv));
            let next_mpv = _mm_mul_ps(*dpp[t].add(qi * 3), rfv(om, next[t], qi));
            *dpc[t].add(qi * 3) = mcv;
            mpv[t] = next_mpv;
            x_bv[t] = _mm_add_ps(x_bv[t], _mm_mul_ps(mpv[t], tbm));
        }
        tdmv = tfv(om, qi, 3);
        timv = tfv(om, qi, 2);
        tmmv = tfv(om, qi, 1);
    }

    let mut x_ev = [zerov; N];
    for t in 0..N {
        let lane = &mut *lanes[t];
        let mut sum = x_bv[t];
        sum = _mm_add_ps(
            sum,
            _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(sum, sum),
        );
        sum = _mm_add_ps(
            sum,
            _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(sum, sum),
        );
        _mm_store_ss(&mut lane.x_b, sum);
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
        rescale(&mut *lanes[t], q, at[t]);
    }
}

#[inline(always)]
unsafe fn rescale(lane: &mut BackParse, q: usize, i: usize) {
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
        lane.fwd_row_scales[i]
    };
    if row_scale > 1.0 {
        let scalev = _mm_set1_ps(1.0 / row_scale);
        lane.x_e /= row_scale;
        lane.x_n /= row_scale;
        lane.x_j /= row_scale;
        lane.x_c /= row_scale;
        let dpc = lane.dpc.as_mut_ptr();
        for cell in 0..q * 3 {
            let p = dpc.add(cell);
            *p = _mm_mul_ps(*p, scalev);
        }
        if lane.has_own_scales {
            lane.x_b = 1.0;
        } else {
            lane.x_b /= row_scale;
        }
    }
    let pmx = &mut *lane.pmx;
    pmx.set_xmx(i, PXE, lane.x_e);
    pmx.set_xmx(i, PXN, lane.x_n);
    pmx.set_xmx(i, PXJ, lane.x_j);
    pmx.set_xmx(i, PXB, lane.x_b);
    pmx.set_xmx(i, PXC, lane.x_c);
    pmx.scale[i] = 0.0;
    pmx.row_scale[i] = row_scale;
    std::mem::swap(lane.dpp, lane.dpc);
}

unsafe fn first_row(om: &OProfile, lane: &mut BackParse) {
    let q = nqf(om.m);
    let dp = lane.dpp.as_ptr();
    let x1 = lane.dsq[1] as usize;
    let mut x_bv = _mm_setzero_ps();
    for qi in 0..q {
        let mpv = _mm_mul_ps(*dp.add(qi * 3), rfv(om, x1, qi));
        x_bv = _mm_add_ps(x_bv, _mm_mul_ps(mpv, tfv(om, qi, 0)));
    }
    x_bv = _mm_add_ps(
        x_bv,
        _mm_shuffle_ps::<{ super::shuffle_mask(0, 3, 2, 1) }>(x_bv, x_bv),
    );
    x_bv = _mm_add_ps(
        x_bv,
        _mm_shuffle_ps::<{ super::shuffle_mask(1, 0, 3, 2) }>(x_bv, x_bv),
    );
    _mm_store_ss(&mut lane.x_b, x_bv);
    lane.x_n = lane.x_b * lane.xf[P7O_N][P7O_MOVE] + lane.x_n * lane.xf[P7O_N][P7O_LOOP];
    let pmx = &mut *lane.pmx;
    pmx.set_xmx(0, PXE, 0.0);
    pmx.set_xmx(0, PXN, lane.x_n);
    pmx.set_xmx(0, PXJ, 0.0);
    pmx.set_xmx(0, PXB, lane.x_b);
    pmx.set_xmx(0, PXC, 0.0);
    pmx.scale[0] = 0.0;
    pmx.row_scale[0] = 1.0;
    pmx.has_own_scales = lane.has_own_scales;
}
