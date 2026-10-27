use super::*;

pub(super) struct Decoding<'a> {
    fwd: &'a ProbMx,
    bck: &'a ProbMx,
    om: &'a OProfile,
    q: usize,
    row_width: usize,
    m: usize,
    fdp: *const f32,
    bdp: *const f32,
    pdp: *mut f32,
    fx: *const f32,
    bx: *const f32,
    px: *mut f32,
    scaleproduct: f32,
}

impl<'a> Decoding<'a> {
    pub(super) unsafe fn start(
        fwd: &'a ProbMx,
        bck: &'a ProbMx,
        om: &'a OProfile,
        pp: &mut ProbMx,
        cells: &mut Vec<__m128>,
    ) -> Self {
        let l = fwd.l;
        let m = om.m;
        pp.resize_parser(m, l);

        let q = fwd.q_count();
        let row_width = fwd.striped_row_width();
        let fdp = fwd.striped_dp.as_ptr().add(fwd.striped_dp_offset);
        let bdp = bck.striped_dp.as_ptr().add(bck.striped_dp_offset);
        cells.clear();
        cells.resize(3 * q, _mm_setzero_ps());
        let pdp = cells.as_mut_ptr().cast::<f32>();
        let fx = fwd.xmx.as_ptr();
        let bx = bck.xmx.as_ptr();
        let px = pp.xmx.as_mut_ptr();

        let inv_total = if *bx.add(PXN) > 0.0 {
            1.0 / *bx.add(PXN)
        } else {
            0.0
        };
        pp.m = m;
        pp.l = l;

        for s in 0..NXCELLS {
            *px.add(s) = 0.0;
        }

        let scaleproduct = inv_total;
        Decoding {
            fwd,
            bck,
            om,
            q,
            row_width,
            m,
            fdp,
            bdp,
            pdp,
            fx,
            bx,
            px,
            scaleproduct,
        }
    }

    pub(super) unsafe fn row(&mut self, i: usize) {
        let Decoding {
            fwd,
            bck,
            om,
            q,
            row_width,
            m,
            fdp,
            bdp,
            pdp,
            fx,
            bx,
            px,
            mut scaleproduct,
        } = *self;
        let dp_scale = scaleproduct * *fwd.row_scale.as_ptr().add(i);
        let scalev = _mm_set1_ps(dp_scale);
        let frow = fdp.add(i * row_width);
        let brow = bdp.add(i * row_width);
        let prow = pdp;

        for qi in 0..q {
            let fm = load_cell(frow, qi, OM_M);
            let bm = load_cell(brow, qi, OM_M);
            let pm = _mm_mul_ps(_mm_mul_ps(fm, bm), scalev);
            store_cell(prow, qi, OM_M, pm);

            // C's p7_Decoding stores D posteriors as zero; D is not a residue assignment.
            store_cell(prow, qi, OM_D, _mm_setzero_ps());

            let fi = load_cell(frow, qi, OM_I);
            let bi = load_cell(brow, qi, OM_I);
            let pi = _mm_mul_ps(_mm_mul_ps(fi, bi), scalev);
            store_cell(prow, qi, OM_I, pi);
        }

        // Node M has no insert state. The padded lane values are irrelevant to C
        // traceback but keeping I_M zero matches the generic matrix semantics.
        let m_q = (m - 1) % q;
        let m_lane = (m - 1) / q;
        let mut lanes = [0.0_f32; 4];
        _mm_storeu_ps(lanes.as_mut_ptr(), load_cell(prow, m_q, OM_I));
        lanes[m_lane] = 0.0;
        store_cell(prow, m_q, OM_I, _mm_loadu_ps(lanes.as_ptr()));

        let xrow = i * NXCELLS;
        *px.add(xrow + PXE) = 0.0;
        *px.add(xrow + PXN) = *fx.add((i - 1) * NXCELLS + PXN)
            * *bx.add(i * NXCELLS + PXN)
            * om.xf[P7O_N][P7O_LOOP]
            * scaleproduct;
        *px.add(xrow + PXJ) = *fx.add((i - 1) * NXCELLS + PXJ)
            * *bx.add(i * NXCELLS + PXJ)
            * om.xf[P7O_J][P7O_LOOP]
            * scaleproduct;
        *px.add(xrow + PXB) = 0.0;
        *px.add(xrow + PXC) = *fx.add((i - 1) * NXCELLS + PXC)
            * *bx.add(i * NXCELLS + PXC)
            * om.xf[P7O_C][P7O_LOOP]
            * scaleproduct;

        if bck.has_own_scales {
            scaleproduct *= *fwd.row_scale.as_ptr().add(i) / *bck.row_scale.as_ptr().add(i);
        }
        self.scaleproduct = scaleproduct;
    }
}
