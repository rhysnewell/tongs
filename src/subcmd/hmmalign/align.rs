use super::*;

/// Align every input sequence to `hmm` (Forward/Backward + posterior decoding +
/// optimal accuracy traceback) and stitch the per-sequence traces into a single
/// text MSA, computing per-column PP_cons. Counterpart to the alignment block of
/// `main()` in hmmalign.c when no `--mapali` is supplied.
/// A per-sequence alignment result: the optimal-accuracy traceback and a small
/// per-trace-position posterior vector (`pp_z[z]` = the posterior for the residue
/// emitted at trace position `z`; 0 for non-emitting states).
pub(super) type AlignedTrace = (tongs::trace::Trace, Vec<f32>);

/// Forward/Backward + posterior decoding + optimal-accuracy traceback for every
/// sequence. Mirrors the per-sequence loop in C `p7_tracealign_Seqs`: the DP
/// matrices (and the profile) are allocated ONCE and reused (grown) across
/// sequences via `p7_gmx_GrowTo`-style reuse, so only the trace and a small
/// O(L) posterior vector are retained per sequence — never the O(M*L) matrices.
pub(super) fn align_traces(
    hmm: &tongs::hmm::Hmm,
    abc: &Alphabet,
    bg: &Bg,
    sequences: &[Sequence],
) -> Vec<AlignedTrace> {
    {
        align_traces_optimized(hmm, abc, bg, sequences)
    }
}

fn align_traces_optimized(
    hmm: &tongs::hmm::Hmm,
    abc: &Alphabet,
    bg: &Bg,
    sequences: &[Sequence],
) -> Vec<AlignedTrace> {
    let mut gm = Profile::new(hmm.m, abc);
    profile::profile_config(hmm, bg, &mut gm, 100, P7_UNILOCAL);
    let mut om = OProfile::convert(&gm);
    let mut fwd = ProbMx::new_full(hmm.m, 0);
    let mut bck = ProbMx::new_full(hmm.m, 0);
    let mut pp = ProbMx::new_full(hmm.m, 0);
    let mut oa = ProbMx::new_full(hmm.m, 0);
    let mut pp_gmx = Gmx::new(hmm.m, 0);
    let mut cells = Vec::new();

    let mut out = Vec::with_capacity(sequences.len());
    for sq in sequences {
        om.reconfig_length(sq.n as i32);
        fwd.resize_full(hmm.m, sq.n);
        bck.resize_full(hmm.m, sq.n);
        oa.resize_full(hmm.m, sq.n);
        pp_gmx.grow_to(hmm.m, sq.n);

        unsafe {
            tongs::simd::fwd_filter::forward_parser_pmx(&sq.dsq, sq.n, &om, &mut fwd);
            tongs::simd::bck_filter::backward_parser_pmx(
                &sq.dsq,
                sq.n,
                &om,
                Some(&fwd.row_scale),
                &mut bck,
            );
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
    out
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
