use super::*;

pub(super) type AlignedTrace = (tongs::trace::Trace, Vec<f32>);

// The matrices are reused across sequences so only each trace and its O(L) posteriors are kept.
pub(super) fn align_traces(
    hmm: &tongs::hmm::Hmm,
    abc: &Alphabet,
    bg: &Bg,
    sequences: &[Sequence],
) -> Result<Vec<AlignedTrace>, Unalignable> {
    let mut gm = Profile::new(hmm.m, abc);
    profile::profile_config(hmm, bg, &mut gm, 100, P7_UNILOCAL);
    let mut om = OProfile::convert(&gm);
    let mut fwd = ProbMx::new_full(hmm.m, 0);
    let mut bck = ProbMx::new_full(hmm.m, 0);
    let mut pp = ProbMx::new_full(hmm.m, 0);
    let mut oa = ProbMx::new_full(hmm.m, 0);
    let mut pp_gmx = Gmx::new(hmm.m, 0);
    let mut cells = Vec::new();
    let mut generic = None;

    let mut out = Vec::with_capacity(sequences.len());
    for sq in sequences {
        if sq.n == 0 {
            out.push(empty_trace());
            continue;
        }
        om.reconfig_length(sq.n as i32);
        fwd.resize_full(hmm.m, sq.n);
        bck.resize_full(hmm.m, sq.n);
        oa.resize_full(hmm.m, sq.n);
        pp_gmx.grow_to(hmm.m, sq.n);

        let score =
            unsafe { tongs::simd::fwd_filter::forward_parser_pmx(&sq.dsq, sq.n, &om, &mut fwd) };
        if !score.is_finite() {
            return Err(Unalignable {
                name: sq.name.clone(),
            });
        }
        unsafe {
            tongs::simd::bck_filter::backward_parser_pmx(
                &sq.dsq,
                sq.n,
                &om,
                Some(&fwd.row_scale),
                &mut bck,
            );
        }
        if decoding_overflows(&fwd, &bck) {
            out.push(generic.get_or_insert_with(|| Generic::new(&gm)).trace(sq));
            continue;
        }
        unsafe {
            tongs::simd::optacc::decode_and_fill(
                &fwd, &bck, &om, &mut pp, &mut oa, &mut cells, None,
            );
        }
        let tr = unsafe { tongs::simd::optacc::oa_trace_pmx(&om, &pp, &oa) };
        let njc_loop = [
            om.xf[P7O_N][P7O_LOOP],
            om.xf[P7O_J][P7O_LOOP],
            om.xf[P7O_C][P7O_LOOP],
        ];
        p_decoding_to_gmx(&fwd, &bck, hmm.m, njc_loop, &mut pp_gmx);
        let pp_z = trace_posteriors(&tr, &pp_gmx);
        out.push((tr, pp_z));
    }
    Ok(out)
}

struct Generic {
    gm: Profile,
    fwd: Gmx,
    bck: Gmx,
    pp: Gmx,
    oa: Gmx,
}

impl Generic {
    fn new(gm: &Profile) -> Self {
        Generic {
            gm: gm.clone(),
            fwd: Gmx::new(gm.m, 0),
            bck: Gmx::new(gm.m, 0),
            pp: Gmx::new(gm.m, 0),
            oa: Gmx::new(gm.m, 0),
        }
    }

    fn trace(&mut self, sq: &Sequence) -> AlignedTrace {
        let (m, n) = (self.gm.m, sq.n);
        profile::reconfig_length(&mut self.gm, n as i32);
        self.fwd.grow_to(m, n);
        self.bck.grow_to(m, n);
        self.pp.grow_to(m, n);
        self.oa.grow_to_zeroed(m, n);
        g_forward(&sq.dsq, n, &self.gm, &mut self.fwd);
        g_backward(&sq.dsq, n, &self.gm, &mut self.bck);
        g_decoding(&self.gm, &self.fwd, &self.bck, &mut self.pp);
        let deltas = OptAccTDelta::from_profile(&self.gm);
        g_optimal_accuracy_with_deltas(&self.gm, &self.pp, &mut self.oa, &deltas);
        let tr = g_oa_trace(&self.gm, &self.pp, &self.oa);
        let pp_z = trace_posteriors(&tr, &self.pp);
        (tr, pp_z)
    }
}

// An empty sequence emits nothing, so C hmmalign gives it a row of gaps.
fn empty_trace() -> AlignedTrace {
    let mut tr = tongs::trace::Trace::new();
    for state in [State::S, State::N, State::B, State::E, State::C, State::T] {
        tr.append(state, 0, 0);
    }
    let pp_z = vec![0.0; tr.n];
    (tr, pp_z)
}

/// Extract the per-residue posterior probability at each trace position, while
/// the posterior matrix is still alive, so the O(M*L) matrix can be dropped.
fn trace_posteriors(tr: &tongs::trace::Trace, pp: &Gmx) -> Vec<f32> {
    let mut v = vec![0.0_f32; tr.n];
    for (z, value) in v.iter_mut().enumerate().take(tr.n) {
        *value = match tr.st[z] {
            State::M => pp.mmx(tr.i[z], tr.k[z]),
            State::I => pp.imx(tr.i[z], tr.k[z]),
            State::N | State::C => pp.xmx(tr.i[z], state_pp_index(tr.st[z])),
            _ => 0.0,
        };
    }
    v
}
