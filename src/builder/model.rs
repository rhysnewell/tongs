use super::*;

/// Compute HMMER's beta-derived maximum likely emitted sequence length.
///
/// This is the Rust counterpart of C `p7_Builder_MaxLength()`: it accumulates
/// the emitted-length distribution for the core model until the remaining tail
/// mass is below `emit_thresh`, bounded by `min(20*M, 100000)` and at least `M`.
pub fn max_length_from_beta(hmm: &Hmm, emit_thresh: f64) -> i32 {
    let model_len = hmm.m;
    if model_len <= 1 {
        return 1;
    }

    let length_bound = model_len.max((20 * model_len).min(100_000));
    let mut max_length = length_bound as i32;

    let mut i_dp = vec![[0.0_f64; 2]; model_len + 1];
    let mut m_dp = vec![[0.0_f64; 2]; model_len + 1];
    let mut d_dp = vec![[0.0_f64; 2]; model_len + 1];

    m_dp[1][0] = 1.0;
    i_dp[1][0] = 0.0;
    d_dp[1][0] = 0.0;
    m_dp[2][0] = 0.0;
    i_dp[2][0] = 0.0;
    d_dp[2][0] = hmm.t[1][MD] as f64;
    for k in 3..=model_len {
        m_dp[k][0] = 0.0;
        i_dp[k][0] = 0.0;
        d_dp[k][0] = hmm.t[k - 1][DD] as f64 * d_dp[k - 1][0];
    }

    m_dp[1][1] = 0.0;
    d_dp[1][1] = 0.0;
    d_dp[2][1] = 0.0;
    i_dp[2][1] = 0.0;
    i_dp[1][1] = hmm.t[1][MI] as f64 * m_dp[1][0];
    m_dp[2][1] = hmm.t[1][MM] as f64 * m_dp[1][0];
    for k in 3..=model_len {
        m_dp[k][1] = hmm.t[k - 1][DM] as f64 * d_dp[k - 1][0];
        i_dp[k][1] = 0.0;
        d_dp[k][1] =
            hmm.t[k - 1][MD] as f64 * m_dp[k - 1][1] + hmm.t[k - 1][DD] as f64 * d_dp[k - 1][1];
    }

    let mut p_sum =
        m_dp[model_len][0] + m_dp[model_len][1] + d_dp[model_len][0] + d_dp[model_len][1];

    let mut col_ptr = 0usize;
    for col in 3..=length_bound {
        let prev_col_ptr = 1 - col_ptr;
        let mut surv = 0.0_f64;

        m_dp[1][col_ptr] = 0.0;
        d_dp[1][col_ptr] = 0.0;
        i_dp[1][col_ptr] = hmm.t[1][II] as f64 * i_dp[1][prev_col_ptr];
        surv += i_dp[1][col_ptr];

        for k in 2..=model_len {
            m_dp[k][col_ptr] = hmm.t[k - 1][MM] as f64 * m_dp[k - 1][prev_col_ptr]
                + hmm.t[k - 1][DM] as f64 * d_dp[k - 1][prev_col_ptr]
                + hmm.t[k - 1][IM] as f64 * i_dp[k - 1][prev_col_ptr];
            i_dp[k][col_ptr] = hmm.t[k][MI] as f64 * m_dp[k][prev_col_ptr]
                + hmm.t[k][II] as f64 * i_dp[k][prev_col_ptr];
            d_dp[k][col_ptr] = hmm.t[k - 1][MD] as f64 * m_dp[k - 1][col_ptr]
                + hmm.t[k - 1][DD] as f64 * d_dp[k - 1][col_ptr];

            surv += i_dp[k][col_ptr]
                + m_dp[k][col_ptr] * (1.0 - hmm.t[k][MD] as f64)
                + d_dp[k][col_ptr] * (1.0 - hmm.t[k][DD] as f64);
        }
        surv += m_dp[model_len][col_ptr] * hmm.t[model_len][MD] as f64
            + d_dp[model_len][col_ptr] * hmm.t[model_len][DD] as f64
            - i_dp[model_len][col_ptr];

        p_sum += m_dp[model_len][col_ptr] + d_dp[model_len][col_ptr];
        let denom = surv + p_sum;
        if denom > 0.0 {
            surv /= denom;
        }

        if surv < emit_thresh {
            max_length = col as i32;
            break;
        }

        col_ptr = 1 - col_ptr;
    }

    max_length
}

pub fn set_max_length_from_beta(hmm: &mut Hmm, emit_thresh: f64) {
    hmm.max_length = max_length_from_beta(hmm, emit_thresh);
}

pub(super) fn model_mask_from_digitized(
    msa: &Msa,
    abc: &Alphabet,
    ax: &[Vec<u8>],
    weights: &[f64],
    symfrac: f32,
    hand_arch: bool,
) -> Vec<u8> {
    let gap = abc.gap_code();
    let missing = abc.missing_code();
    let mut mask = vec![b'.'; msa.alen];

    for (col, mask_col) in mask.iter_mut().enumerate().take(msa.alen) {
        let mut residue_wt = 0.0_f32;
        let mut total_wt = 0.0_f32;
        for seq in 0..msa.nseq {
            let pos = col + 1;
            if pos < ax[seq].len() - 1 {
                let residue = ax[seq][pos];
                if abc.is_residue(residue) {
                    residue_wt = (residue_wt as f64 + weights[seq]) as f32;
                    total_wt = (total_wt as f64 + weights[seq]) as f32;
                } else if residue == gap {
                    total_wt = (total_wt as f64 + weights[seq]) as f32;
                } else if residue == missing {
                    continue;
                }
            }
        }
        if residue_wt > 0.0 && total_wt > 0.0 && residue_wt / total_wt >= symfrac {
            *mask_col = b'x';
        }
    }

    // Only `--hand` architecture uses input RF; fast architecture ignores it.
    if hand_arch {
        if let Some(ref rf) = msa.rf {
            for col in 0..msa.alen.min(rf.len()) {
                mask[col] = if rf[col] != b'.' && rf[col] != b'-' && rf[col] != b' ' {
                    b'x'
                } else {
                    b'.'
                };
            }
        }
    }

    mask
}

/// Compute the average residue composition implied by the HMM's match and
/// insert emissions, weighted by per-node occupancy, and store in `hmm.compo`.
/// Counterpart to C's `p7_hmm_SetComposition()`.
pub(super) fn set_hmm_composition(hmm: &mut Hmm) {
    let mut mocc = vec![0.0_f32; hmm.m + 1];
    let mut iocc = vec![0.0_f32; hmm.m + 1];

    mocc[0] = 0.0;
    if hmm.m >= 1 {
        mocc[1] = hmm.t[0][MI] + hmm.t[0][MM];
    }
    for k in 2..=hmm.m {
        mocc[k] = mocc[k - 1] * (hmm.t[k - 1][MM] + hmm.t[k - 1][MI])
            + (1.0 - mocc[k - 1]) * hmm.t[k - 1][DM];
    }

    if hmm.t[0][IM] > 0.0 {
        iocc[0] = hmm.t[0][MI] / hmm.t[0][IM];
    }
    for k in 1..=hmm.m {
        if hmm.t[k][IM] > 0.0 {
            iocc[k] = mocc[k] * hmm.t[k][MI] / hmm.t[k][IM];
        }
    }

    hmm.compo.fill(0.0);
    for x in 0..hmm.abc_k {
        hmm.compo[x] += hmm.ins[0][x] * iocc[0];
    }
    for k in 1..=hmm.m {
        for x in 0..hmm.abc_k {
            hmm.compo[x] += hmm.mat[k][x] * mocc[k];
            hmm.compo[x] += hmm.ins[k][x] * iocc[k];
        }
    }

    let sum: f32 = hmm.compo[..hmm.abc_k].iter().sum();
    if sum > 0.0 {
        for x in 0..hmm.abc_k {
            hmm.compo[x] /= sum;
        }
    }
    hmm.flags |= P7H_COMPO;
}

/// Derive a consensus residue string from the most-probable match emission
/// per node. Uppercase if its probability >= threshold (0.9 for nucleic,
/// 0.5 for amino), otherwise lowercase. Counterpart to C's `p7_hmm_SetConsensus()`.
pub(super) fn set_hmm_consensus(hmm: &mut Hmm, abc: &Alphabet) {
    let threshold = match hmm.abc_type {
        crate::alphabet::AlphabetType::Dna | crate::alphabet::AlphabetType::Rna => 0.9,
        _ => 0.5,
    };
    let mut cons = vec![b' '; hmm.m + 2];
    for (node, cons_slot) in cons.iter_mut().enumerate().take(hmm.m + 1).skip(1) {
        let mut best_x = 0usize;
        let mut best_p = f32::NEG_INFINITY;
        for x in 0..abc.k {
            if hmm.mat[node][x] > best_p {
                best_p = hmm.mat[node][x];
                best_x = x;
            }
        }
        let symbol = abc.sym[best_x];
        *cons_slot = if best_p >= threshold {
            symbol.to_ascii_uppercase()
        } else {
            symbol.to_ascii_lowercase()
        };
    }
    hmm.consensus = Some(cons);
    hmm.flags |= P7H_CONS;
}

/// Generate one faux traceback per MSA row consistent with the `matassign[]`
/// architecture, then run trace doctoring to remove D->I / I->D conflicts.
/// Counterpart to C's `p7_trace_FauxFromMSA()` + `p7_trace_Doctor()`.
pub(super) fn faux_trace_from_msa(
    ax: &[Vec<u8>],
    matassign: &[bool],
    abc: &Alphabet,
) -> Vec<Trace> {
    let mut traces = Vec::with_capacity(ax.len());
    for row in ax {
        let mut tr = Trace::new();
        tr.append(TraceState::B, 0, 0);

        let mut k = 0usize;
        for apos in 1..=matassign.len() {
            let sym = row[apos];
            if matassign[apos - 1] {
                k += 1;
                if abc.is_residue(sym) || sym == abc.nonresidue_code() {
                    tr.append(TraceState::M, k, apos);
                } else if abc.is_gap(sym) {
                    tr.append(TraceState::D, k, 0);
                } else if abc.is_missing(sym) && tr.st.last().copied() != Some(TraceState::X) {
                    tr.append(TraceState::X, k, 0);
                }
            } else if abc.is_residue(sym) || sym == abc.nonresidue_code() {
                tr.append(TraceState::I, k, apos);
            } else if abc.is_missing(sym) && tr.st.last().copied() != Some(TraceState::X) {
                tr.append(TraceState::X, k, 0);
            }
        }
        tr.append(TraceState::E, 0, 0);
        tr.m = k;
        tr.l = matassign.len();
        doctor_trace(&mut tr);
        traces.push(tr);
    }
    traces
}

/// Mark short fragment sequences' leading/trailing gaps as missing-data
/// symbols so they don't contribute spurious counts at flanking positions.
/// Counterpart to Easel's `esl_msa_MarkFragments_old()`.
pub(super) fn mark_fragments_old(ax: &mut [Vec<u8>], abc: &Alphabet, alen: usize, fragthresh: f32) {
    let missing = abc.missing_code();
    for row in ax {
        let rlen = row
            .iter()
            .skip(1)
            .take(alen)
            .filter(|&&sym| abc.is_residue(sym))
            .count();
        if (rlen as f32) <= fragthresh * alen as f32 {
            for sym in row.iter_mut().take(alen + 1).skip(1) {
                if abc.is_residue(*sym) {
                    break;
                }
                *sym = missing;
            }
            for sym in row[1..=alen].iter_mut().rev() {
                if abc.is_residue(*sym) {
                    break;
                }
                *sym = missing;
            }
        }
    }
}

/// Collapse adjacent D-I / I-D pairs in a faux trace into single M states,
/// matching the HMMER architecture's disallowed D<->I transitions.
/// Counterpart to C's `p7_trace_Doctor()`.
fn doctor_trace(tr: &mut Trace) {
    let mut new_st = Vec::with_capacity(tr.n);
    let mut new_k = Vec::with_capacity(tr.n);
    let mut new_i = Vec::with_capacity(tr.n);
    let mut opos = 0usize;

    while opos < tr.n {
        if opos + 1 < tr.n && tr.st[opos] == TraceState::D && tr.st[opos + 1] == TraceState::I {
            new_st.push(TraceState::M);
            new_k.push(tr.k[opos]);
            new_i.push(tr.i[opos + 1]);
            opos += 2;
        } else if opos + 1 < tr.n
            && tr.st[opos] == TraceState::I
            && tr.st[opos + 1] == TraceState::D
        {
            new_st.push(TraceState::M);
            new_k.push(tr.k[opos + 1]);
            new_i.push(tr.i[opos]);
            opos += 2;
        } else {
            new_st.push(tr.st[opos]);
            new_k.push(tr.k[opos]);
            new_i.push(tr.i[opos]);
            opos += 1;
        }
    }

    tr.st = new_st;
    tr.k = new_k;
    tr.i = new_i;
    tr.n = tr.st.len();
}

/// Accumulate weight `wt` for symbol `sym` into count vector `ct[0..K]`.
/// Distributes degenerate residues evenly over their canonical members.
/// Counterpart to Easel's `esl_abc_FCount()`.
fn fcount(abc: &Alphabet, ct: &mut [f32], sym: u8, wt: f32) {
    if abc.is_canonical(sym) || abc.is_gap(sym) {
        if let Some(slot) = ct.get_mut(sym as usize) {
            *slot += wt;
        }
    } else if abc.is_missing(sym) || sym == abc.nonresidue_code() {
    } else if abc.is_degenerate(sym) {
        let denom = abc.ndegen[sym as usize] as f32;
        if denom > 0.0 {
            for (y, count) in ct.iter_mut().enumerate().take(abc.k) {
                if abc.degen[sym as usize][y] {
                    *count += wt / denom;
                }
            }
        }
    }
}

/// Accumulate weighted emission and transition counts from one trace into
/// `hmm`. Skips X (missing data) regions at the trace ends. Counterpart to
/// C's `p7_trace_Count()`.
pub(super) fn count_trace(hmm: &mut Hmm, dsq: &[u8], wt: f32, tr: &Trace, abc: &Alphabet) {
    let mut z1 = 0usize;
    let mut z2 = tr.n.saturating_sub(1);

    if tr.n >= 2 && tr.st[0] == TraceState::B && tr.st[1] == TraceState::X {
        for z in 2..tr.n.saturating_sub(1) {
            if tr.st[z] == TraceState::M {
                z1 = z;
                break;
            }
        }
    }
    if tr.n >= 2 && tr.st[tr.n - 1] == TraceState::E && tr.st[tr.n - 2] == TraceState::X {
        for z in (1..=tr.n.saturating_sub(3)).rev() {
            if tr.st[z] == TraceState::M {
                z2 = z;
                break;
            }
        }
    }

    for z in z1..z2 {
        if tr.st[z] == TraceState::X {
            continue;
        }

        let st = tr.st[z];
        let st2 = tr.st[z + 1];
        let k = tr.k[z];
        let k2 = tr.k[z + 1];
        let i = tr.i[z];

        if st == TraceState::M {
            fcount(abc, &mut hmm.mat[k], dsq[i], wt);
        } else if st == TraceState::I {
            fcount(abc, &mut hmm.ins[k], dsq[i], wt);
        }

        if st2 == TraceState::X {
            continue;
        }

        if st == TraceState::B {
            if st2 == TraceState::M && k2 > 1 {
                hmm.t[0][MD] += wt;
                for ktmp in 1..k2.saturating_sub(1) {
                    hmm.t[ktmp][DD] += wt;
                }
                hmm.t[k2 - 1][DM] += wt;
            } else {
                match st2 {
                    TraceState::M => hmm.t[0][MM] += wt,
                    TraceState::I => hmm.t[0][MI] += wt,
                    TraceState::D => hmm.t[0][MD] += wt,
                    _ => {}
                }
            }
        } else if st == TraceState::M {
            match st2 {
                TraceState::M => hmm.t[k][MM] += wt,
                TraceState::I => hmm.t[k][MI] += wt,
                TraceState::D => hmm.t[k][MD] += wt,
                TraceState::E => hmm.t[k][MM] += wt,
                _ => {}
            }
        } else if st == TraceState::I {
            match st2 {
                TraceState::M => hmm.t[k][IM] += wt,
                TraceState::I => hmm.t[k][II] += wt,
                TraceState::E => hmm.t[k][IM] += wt,
                _ => {}
            }
        } else if st == TraceState::D {
            match st2 {
                TraceState::M => hmm.t[k][DM] += wt,
                TraceState::D => hmm.t[k][DD] += wt,
                TraceState::E => hmm.t[k][DM] += wt,
                _ => {}
            }
        }
    }
}
