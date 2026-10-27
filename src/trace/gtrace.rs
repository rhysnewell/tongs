use super::*;

/// Float near-equality test. Faithful port of `esl_FCompare_old`
/// (hmmer/easel/easel.c:2400). Used by the Viterbi traceback to compare a DP
/// cell value against `prev + tsc + emission` candidate predecessors.
#[inline]
fn esl_fcompare_old(a: f32, b: f32, tol: f32) -> bool {
    if a.is_infinite() && b.is_infinite() {
        return true;
    }
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    if a == b {
        return true;
    }
    if a.abs() == 0.0 && b.abs() <= tol {
        return true;
    }
    if b.abs() == 0.0 && a.abs() <= tol {
        return true;
    }
    if 2.0 * (a - b).abs() / (a + b).abs() <= tol {
        return true;
    }
    false
}

/// Viterbi traceback: walk a filled generic Viterbi DP matrix `gx` backwards
/// from T to S to recover the optimal state path. Faithful port of
/// `p7_GTrace()` in `hmmer/src/generic_vtrace.c`. Reconstructs an M+I+D path
/// plus the surrounding S/N/B/E/C/J/T scaffold and returns it forward-ordered.
///
/// This is a reconstruction traceback: predecessor scores are compared against
/// the full `prev + tsc + emission` sum in exactly the order Viterbi computed
/// them (B,M,I,D for M-states), per the J1/121 note in the C source, so the
/// near-equality tie-break matches C cell-for-cell.
pub fn g_trace(dsq: &[Dsq], l: usize, gm: &Profile, gx: &Gmx) -> Trace {
    let m = gm.m;
    let mut tr = Trace::new();
    let tol = 1e-5_f32;

    let mut i = l; // position in seq (1..L)
    let mut k = 0usize; // position in model (1..M)

    tr.append(State::T, k, i);
    tr.append(State::C, k, i);
    let mut sprv = State::C;

    while sprv != State::S {
        // rsc[dsq[i]] available only for i>0; emission scores indexed with x=dsq[i].
        let scur: State = match sprv {
            State::C => {
                // C(i) comes from C(i-1) or E(i)
                if esl_fcompare_old(
                    gx.xmx(i, P7G_C),
                    gx.xmx(i - 1, P7G_C) + gm.xsc[P7P_C][P7P_LOOP],
                    tol,
                ) {
                    State::C
                } else if esl_fcompare_old(
                    gx.xmx(i, P7G_C),
                    gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_MOVE],
                    tol,
                ) {
                    State::E
                } else {
                    // C couldn't be traced (impossible in a valid matrix).
                    break;
                }
            }
            State::E => {
                // E connects from any M state; k set here.
                if gm.is_local() {
                    // Can't come from D in a local Viterbi trace.
                    let mut found = 0usize;
                    let mut kk = m;
                    while kk >= 1 {
                        if esl_fcompare_old(gx.xmx(i, P7G_E), gx.mmx(i, kk), tol) {
                            found = kk;
                            break;
                        }
                        kk -= 1;
                    }
                    if found == 0 {
                        break; // E couldn't be traced
                    }
                    k = found;
                    State::M
                } else {
                    // glocal mode: we come from either M_M or D_M
                    if esl_fcompare_old(gx.xmx(i, P7G_E), gx.mmx(i, m), tol) {
                        k = m;
                        State::M
                    } else if esl_fcompare_old(gx.xmx(i, P7G_E), gx.dmx(i, m), tol) {
                        k = m;
                        State::D
                    } else {
                        break; // E couldn't be traced
                    }
                }
            }
            State::M => {
                // M connects from i-1,k-1, or B. Test B-entry FIRST, then MM, IM, DM,
                // comparing against the full prev+tsc+emission sum (J1/121).
                let msc = gm.msc(k, dsq[i] as usize);
                let s = if esl_fcompare_old(
                    gx.mmx(i, k),
                    gx.xmx(i - 1, P7G_B) + gm.tsc(k - 1, P7P_BM) + msc,
                    tol,
                ) {
                    State::B
                } else if esl_fcompare_old(
                    gx.mmx(i, k),
                    gx.mmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_MM) + msc,
                    tol,
                ) {
                    State::M
                } else if esl_fcompare_old(
                    gx.mmx(i, k),
                    gx.imx(i - 1, k - 1) + gm.tsc(k - 1, P7P_IM) + msc,
                    tol,
                ) {
                    State::I
                } else if esl_fcompare_old(
                    gx.mmx(i, k),
                    gx.dmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_DM) + msc,
                    tol,
                ) {
                    State::D
                } else {
                    break; // M couldn't be traced
                };
                k -= 1;
                i -= 1;
                s
            }
            State::D => {
                // D connects from M,D at i,k-1
                let s = if esl_fcompare_old(
                    gx.dmx(i, k),
                    gx.mmx(i, k - 1) + gm.tsc(k - 1, P7P_MD),
                    tol,
                ) {
                    State::M
                } else if esl_fcompare_old(
                    gx.dmx(i, k),
                    gx.dmx(i, k - 1) + gm.tsc(k - 1, P7P_DD),
                    tol,
                ) {
                    State::D
                } else {
                    break; // D couldn't be traced
                };
                k -= 1;
                s
            }
            State::I => {
                // I connects from M,I at i-1,k
                let isc = gm.isc(k, dsq[i] as usize);
                let s = if esl_fcompare_old(
                    gx.imx(i, k),
                    gx.mmx(i - 1, k) + gm.tsc(k, P7P_MI) + isc,
                    tol,
                ) {
                    State::M
                } else if esl_fcompare_old(
                    gx.imx(i, k),
                    gx.imx(i - 1, k) + gm.tsc(k, P7P_II) + isc,
                    tol,
                ) {
                    State::I
                } else {
                    break; // I couldn't be traced
                };
                i -= 1;
                s
            }
            State::N => {
                // N connects from S, N
                if i == 0 {
                    State::S
                } else {
                    State::N
                }
            }
            State::B => {
                // B connects from N, J
                if esl_fcompare_old(
                    gx.xmx(i, P7G_B),
                    gx.xmx(i, P7G_N) + gm.xsc[P7P_N][P7P_MOVE],
                    tol,
                ) {
                    State::N
                } else if esl_fcompare_old(
                    gx.xmx(i, P7G_B),
                    gx.xmx(i, P7G_J) + gm.xsc[P7P_J][P7P_MOVE],
                    tol,
                ) {
                    State::J
                } else {
                    break; // B couldn't be traced
                }
            }
            State::J => {
                // J connects from E(i) or J(i-1)
                if esl_fcompare_old(
                    gx.xmx(i, P7G_J),
                    gx.xmx(i - 1, P7G_J) + gm.xsc[P7P_J][P7P_LOOP],
                    tol,
                ) {
                    State::J
                } else if esl_fcompare_old(
                    gx.xmx(i, P7G_J),
                    gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_LOOP],
                    tol,
                ) {
                    State::E
                } else {
                    break; // J couldn't be traced
                }
            }
            _ => break, // bogus state
        };

        // Append this state and the current i,k to be explained.
        tr.append(scur, k, i);

        // For NCJ, we had to defer the i decrement.
        if (scur == State::N || scur == State::J || scur == State::C) && scur == sprv {
            i -= 1;
        }

        sprv = scur;
    }

    tr.m = gm.m;
    tr.l = l;

    // Faithful port of p7_trace_Reverse (p7_trace.c:1108-1151).
    // For emit-on-transition N/C/J runs built backwards, pull residues back by
    // one (C-,Cx,Cx,Cx -> Cx,Cx,Cx,C-) before the in-place reversal.
    if tr.n > 0 {
        for z in 0..(tr.n - 1) {
            let run = (tr.st[z] == State::N && tr.st[z + 1] == State::N)
                || (tr.st[z] == State::C && tr.st[z + 1] == State::C)
                || (tr.st[z] == State::J && tr.st[z + 1] == State::J);
            if run && tr.i[z] == 0 && tr.i[z + 1] > 0 {
                tr.i[z] = tr.i[z + 1];
                tr.i[z + 1] = 0;
                if let Some(pp) = &mut tr.pp {
                    pp[z] = pp[z + 1];
                    pp[z + 1] = 0.0;
                }
            }
        }
    }

    tr.st.reverse();
    tr.k.reverse();
    tr.i.reverse();
    if let Some(pp) = &mut tr.pp {
        pp.reverse();
    }

    tr
}
