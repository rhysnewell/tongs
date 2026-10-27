use super::decode::Decoding;
use super::*;

pub(super) struct Fill<'a> {
    om: &'a OProfile,
    l: usize,
    q: usize,
    row_width: usize,
    ppdp: *const f32,
    pdp_x: *const f32,
    oxdp: *mut f32,
    oxx: *mut f32,
    tfv: *const AlignedF32x4,
    dd_tfv: *const AlignedF32x4,
}

impl<'a> Fill<'a> {
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn start(om: &'a OProfile, pp: &ProbMx, ox: &mut ProbMx) -> Self {
        let m = om.m;
        let l = pp.l;
        ox.resize_full(m, l);

        let q = pp.q_count();
        let row_width = pp.striped_row_width();
        let ppdp = pp.striped_dp.as_ptr().add(pp.striped_dp_offset);
        let pdp_x = pp.xmx.as_ptr();
        let oxdp = ox.striped_dp.as_mut_ptr().add(ox.striped_dp_offset);
        let oxx = ox.xmx.as_mut_ptr();
        let tfv = om.tfv_a.as_ptr();
        let dd_tfv = tfv.add(7 * q);
        let infv = _mm_set1_ps(f32::NEG_INFINITY);

        for qi in 0..q {
            store_cell(oxdp, qi, OM_M, infv);
            store_cell(oxdp, qi, OM_D, infv);
            store_cell(oxdp, qi, OM_I, infv);
        }
        *oxx.add(PXE) = f32::NEG_INFINITY;
        *oxx.add(PXN) = 0.0;
        *oxx.add(PXJ) = f32::NEG_INFINITY;
        *oxx.add(PXB) = 0.0;
        *oxx.add(PXC) = f32::NEG_INFINITY;
        Fill {
            om,
            l,
            q,
            row_width,
            ppdp,
            pdp_x,
            oxdp,
            oxx,
            tfv,
            dd_tfv,
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn row(&self, i: usize) {
        let Fill {
            om,
            q,
            row_width,
            ppdp,
            pdp_x,
            oxdp,
            oxx,
            tfv,
            dd_tfv,
            ..
        } = *self;
        let zerov = _mm_setzero_ps();
        let infv = _mm_set1_ps(f32::NEG_INFINITY);
        let dpp = oxdp.add((i - 1) * row_width);
        let dpc = oxdp.add(i * row_width);
        let ppp = ppdp.add(i * row_width);
        let mut dcv = infv;
        let mut xev = infv;
        let xbv = _mm_set1_ps(*oxx.add((i - 1) * NXCELLS + PXB));

        let mut mpv = rightshift_ps(load_cell(dpp, q - 1, OM_M), infv);
        let mut dpv = rightshift_ps(load_cell(dpp, q - 1, OM_D), infv);
        let mut ipv = rightshift_ps(load_cell(dpp, q - 1, OM_I), infv);

        for qi in 0..q {
            let tp = tfv.add(qi * 7);
            let tbm = _mm_load_ps((*tp.add(P7O_BM)).as_ptr());
            let tmm = _mm_load_ps((*tp.add(P7O_MM)).as_ptr());
            let tim = _mm_load_ps((*tp.add(P7O_IM)).as_ptr());
            let tdm = _mm_load_ps((*tp.add(P7O_DM)).as_ptr());
            let tmd = _mm_load_ps((*tp.add(P7O_MD)).as_ptr());
            let tmi = _mm_load_ps((*tp.add(P7O_MI)).as_ptr());
            let tii = _mm_load_ps((*tp.add(P7O_II)).as_ptr());
            let pcell = ppp.add(qi * DP_CELLS_PER_K * 4);
            let dpp_cell = dpp.add(qi * DP_CELLS_PER_K * 4);
            let dpc_cell = dpc.add(qi * DP_CELLS_PER_K * 4);

            let mut sv = _mm_and_ps(_mm_cmpgt_ps(tbm, zerov), xbv);
            sv = _mm_max_ps(sv, _mm_and_ps(_mm_cmpgt_ps(tmm, zerov), mpv));
            sv = _mm_max_ps(sv, _mm_and_ps(_mm_cmpgt_ps(tim, zerov), ipv));
            sv = _mm_max_ps(sv, _mm_and_ps(_mm_cmpgt_ps(tdm, zerov), dpv));
            sv = _mm_add_ps(sv, _mm_load_ps(pcell.add(OM_M * 4)));
            xev = _mm_max_ps(xev, sv);

            mpv = _mm_load_ps(dpp_cell.add(OM_M * 4));
            dpv = _mm_load_ps(dpp_cell.add(OM_D * 4));
            ipv = _mm_load_ps(dpp_cell.add(OM_I * 4));

            _mm_store_ps(dpc_cell.add(OM_M * 4), sv);
            _mm_store_ps(dpc_cell.add(OM_D * 4), dcv);

            dcv = _mm_and_ps(_mm_cmpgt_ps(tmd, zerov), sv);

            sv = _mm_and_ps(_mm_cmpgt_ps(tmi, zerov), mpv);
            sv = _mm_max_ps(sv, _mm_and_ps(_mm_cmpgt_ps(tii, zerov), ipv));
            _mm_store_ps(
                dpc_cell.add(OM_I * 4),
                _mm_add_ps(sv, _mm_load_ps(pcell.add(OM_I * 4))),
            );
        }

        dcv = rightshift_ps(dcv, infv);
        for qi in 0..q {
            let dptr = dpc.add(qi * DP_CELLS_PER_K * 4 + OM_D * 4);
            let d = _mm_max_ps(dcv, _mm_load_ps(dptr));
            _mm_store_ps(dptr, d);
            let tdd = _mm_load_ps((*dd_tfv.add(qi)).as_ptr());
            dcv = _mm_and_ps(_mm_cmpgt_ps(tdd, zerov), d);
        }

        for _ in 1..4 {
            dcv = rightshift_ps(dcv, infv);
            for qi in 0..q {
                let dptr = dpc.add(qi * DP_CELLS_PER_K * 4 + OM_D * 4);
                let d = _mm_max_ps(dcv, _mm_load_ps(dptr));
                _mm_store_ps(dptr, d);
                let tdd = _mm_load_ps((*dd_tfv.add(qi)).as_ptr());
                dcv = _mm_and_ps(_mm_cmpgt_ps(tdd, zerov), dcv);
            }
        }

        for qi in 0..q {
            xev = _mm_max_ps(xev, load_cell(dpc, qi, OM_D));
        }

        let xrow = i * NXCELLS;
        let xprev = xrow - NXCELLS;
        *oxx.add(xrow + PXE) = hmax_ps(xev);

        let t1 = if om.xf[P7O_J][P7O_LOOP] == 0.0 {
            0.0
        } else {
            *oxx.add(xprev + PXJ) + *pdp_x.add(xrow + PXJ)
        };
        let t2 = if om.xf[P7O_E][P7O_LOOP] == 0.0 {
            0.0
        } else {
            *oxx.add(xrow + PXE)
        };
        *oxx.add(xrow + PXJ) = if t1 > t2 { t1 } else { t2 };

        let t1 = if om.xf[P7O_C][P7O_LOOP] == 0.0 {
            0.0
        } else {
            *oxx.add(xprev + PXC) + *pdp_x.add(xrow + PXC)
        };
        let t2 = if om.xf[P7O_E][P7O_MOVE] == 0.0 {
            0.0
        } else {
            *oxx.add(xrow + PXE)
        };
        *oxx.add(xrow + PXC) = if t1 > t2 { t1 } else { t2 };

        *oxx.add(xrow + PXN) = if om.xf[P7O_N][P7O_LOOP] == 0.0 {
            0.0
        } else {
            *oxx.add(xprev + PXN) + *pdp_x.add(xrow + PXN)
        };

        let t1 = if om.xf[P7O_N][P7O_MOVE] == 0.0 {
            0.0
        } else {
            *oxx.add(xrow + PXN)
        };
        let t2 = if om.xf[P7O_J][P7O_MOVE] == 0.0 {
            0.0
        } else {
            *oxx.add(xrow + PXJ)
        };
        *oxx.add(xrow + PXB) = if t1 > t2 { t1 } else { t2 };
    }

    unsafe fn finish(&self) -> f32 {
        *self.oxx.add(self.l * NXCELLS + PXC)
    }
}

// Decoding a row and filling it straight away keeps the posterior row in cache for the fill.
#[target_feature(enable = "sse2")]
pub unsafe fn decode_and_fill(
    fwd: &ProbMx,
    bck: &ProbMx,
    om: &OProfile,
    pp: &mut ProbMx,
    ox: &mut ProbMx,
) -> f32 {
    let mut decoding = Decoding::start(fwd, bck, om, pp);
    let fill = Fill::start(om, pp, ox);
    for i in 1..=fwd.l {
        decoding.row(i);
        fill.row(i);
    }
    fill.finish()
}
