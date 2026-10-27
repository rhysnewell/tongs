use super::*;

pub unsafe fn oa_trace_coords_pmx(
    om: &OProfile,
    pp: &ProbMx,
    ox: &ProbMx,
) -> Option<(usize, usize, usize, usize)> {
    let mut i = ox.l;
    let mut k = 0usize;
    let mut sprv = State::C;
    let mut hmmfrom = usize::MAX;
    let mut hmmto = 0usize;
    let mut sqfrom = usize::MAX;
    let mut sqto = 0usize;

    while sprv != State::S {
        let scur = match sprv {
            State::M => {
                let s = select_m(om, ox, i, k);
                k -= 1;
                i -= 1;
                s
            }
            State::D => {
                let s = select_d(om, ox, i, k);
                k -= 1;
                s
            }
            State::I => {
                let s = select_i(om, ox, i, k);
                i -= 1;
                s
            }
            State::N => {
                if i == 0 {
                    State::S
                } else {
                    State::N
                }
            }
            State::C => select_c(om, pp, ox, i),
            State::J => select_j(om, pp, ox, i),
            State::E => select_e(om, ox, i, &mut k),
            State::B => select_b(om, ox, i),
            _ => State::S,
        };

        match scur {
            State::M => {
                if k > 0 {
                    hmmfrom = hmmfrom.min(k);
                    hmmto = hmmto.max(k);
                }
                if i > 0 {
                    sqfrom = sqfrom.min(i);
                    sqto = sqto.max(i);
                }
            }
            State::D => {
                if k > 0 {
                    hmmfrom = hmmfrom.min(k);
                    hmmto = hmmto.max(k);
                }
            }
            State::I if i > 0 => {
                sqfrom = sqfrom.min(i);
                sqto = sqto.max(i);
            }
            _ => {}
        }

        if matches!(scur, State::N | State::J | State::C) && scur == sprv {
            i -= 1;
        }
        sprv = scur;
    }

    if hmmto > 0 || sqto > 0 {
        Some((
            if hmmfrom == usize::MAX { 0 } else { hmmfrom },
            hmmto,
            if sqfrom == usize::MAX { 0 } else { sqfrom },
            sqto,
        ))
    } else {
        None
    }
}

pub unsafe fn oa_trace_pmx(om: &OProfile, pp: &ProbMx, ox: &ProbMx) -> Trace {
    let mut tr = Trace::new();
    let mut i = ox.l;
    let mut k = 0usize;
    let mut sprv = State::C;

    tr.append(State::T, 0, i);
    tr.append(State::C, 0, i);

    while sprv != State::S {
        let scur = match sprv {
            State::M => {
                let s = select_m(om, ox, i, k);
                k -= 1;
                i -= 1;
                s
            }
            State::D => {
                let s = select_d(om, ox, i, k);
                k -= 1;
                s
            }
            State::I => {
                let s = select_i(om, ox, i, k);
                i -= 1;
                s
            }
            State::N => {
                if i == 0 {
                    State::S
                } else {
                    State::N
                }
            }
            State::C => select_c(om, pp, ox, i),
            State::J => select_j(om, pp, ox, i),
            State::E => select_e(om, ox, i, &mut k),
            State::B => select_b(om, ox, i),
            _ => State::S,
        };

        tr.append(scur, k, i);
        if matches!(scur, State::N | State::J | State::C) && scur == sprv {
            i -= 1;
        }
        sprv = scur;
    }

    tr.st.reverse();
    tr.k.reverse();
    tr.i.reverse();
    tr
}

/// Read a single lane of the striped OA cell `(i, q, state)`.
#[inline(always)]
unsafe fn cell_lane(ox: &ProbMx, i: usize, q: usize, state: usize, lane: usize) -> f32 {
    let row = ox
        .striped_dp
        .as_ptr()
        .add(ox.striped_dp_offset + i * ox.striped_row_width());
    let mut lanes = [0.0_f32; 4];
    _mm_storeu_ps(lanes.as_mut_ptr(), load_cell(row, q, state));
    lanes[lane]
}

/// Read a single lane of transition `tidx` at stripe `q` from `om.tfv`.
#[inline(always)]
unsafe fn tfv_lane(om: &OProfile, q: usize, tidx: usize, lane: usize) -> f32 {
    om.tfv[q * 7 + tidx][lane]
}

/// Read a single lane of the DD transition stripe at stripe `q` (stored after
/// the 7 standard transitions in `om.tfv`).
#[inline(always)]
unsafe fn tdd_lane(om: &OProfile, q: usize, lane: usize) -> f32 {
    let q_count = crate::simd::oprofile::nqf(om.m);
    om.tfv[7 * q_count + q][lane]
}

/// Pick the predecessor state of an M_k at row `i`: one of M, I, D, or B,
/// whichever scored the OA cell. Mirrors C `select_m` in optacc.c.
unsafe fn select_m(om: &OProfile, ox: &ProbMx, i: usize, k: usize) -> State {
    let q_count = ox.q_count();
    let q = (k - 1) % q_count;
    let lane = (k - 1) / q_count;
    let x_b = ox.xmx[(i - 1) * NXCELLS + PXB];

    let (mpv, dpv, ipv) = if q > 0 {
        (
            cell_lane(ox, i - 1, q - 1, OM_M, lane),
            cell_lane(ox, i - 1, q - 1, OM_D, lane),
            cell_lane(ox, i - 1, q - 1, OM_I, lane),
        )
    } else if lane == 0 {
        (0.0, 0.0, 0.0)
    } else {
        (
            cell_lane(ox, i - 1, q_count - 1, OM_M, lane - 1),
            cell_lane(ox, i - 1, q_count - 1, OM_D, lane - 1),
            cell_lane(ox, i - 1, q_count - 1, OM_I, lane - 1),
        )
    };

    let paths = [
        if tfv_lane(om, q, P7O_MM, lane) == 0.0 {
            f32::NEG_INFINITY
        } else {
            mpv
        },
        if tfv_lane(om, q, P7O_IM, lane) == 0.0 {
            f32::NEG_INFINITY
        } else {
            ipv
        },
        if tfv_lane(om, q, P7O_DM, lane) == 0.0 {
            f32::NEG_INFINITY
        } else {
            dpv
        },
        if tfv_lane(om, q, P7O_BM, lane) == 0.0 {
            f32::NEG_INFINITY
        } else {
            x_b
        },
    ];
    let states = [State::M, State::I, State::D, State::B];
    states[argmax_first(&paths)]
}

/// Pick the predecessor of a D_k cell: M (M->D) or D (D->D). Mirrors C `select_d`.
unsafe fn select_d(om: &OProfile, ox: &ProbMx, i: usize, k: usize) -> State {
    let q_count = ox.q_count();
    let q = (k - 1) % q_count;
    let lane = (k - 1) / q_count;
    let (mpv, dpv, tmd, tdd) = if q > 0 {
        (
            cell_lane(ox, i, q - 1, OM_M, lane),
            cell_lane(ox, i, q - 1, OM_D, lane),
            tfv_lane(om, q - 1, P7O_MD, lane),
            tdd_lane(om, q - 1, lane),
        )
    } else if lane == 0 {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        (
            cell_lane(ox, i, q_count - 1, OM_M, lane - 1),
            cell_lane(ox, i, q_count - 1, OM_D, lane - 1),
            tfv_lane(om, q_count - 1, P7O_MD, lane - 1),
            tdd_lane(om, q_count - 1, lane - 1),
        )
    };
    let md = if tmd == 0.0 { f32::NEG_INFINITY } else { mpv };
    let dd = if tdd == 0.0 { f32::NEG_INFINITY } else { dpv };
    if md >= dd {
        State::M
    } else {
        State::D
    }
}

/// Pick the predecessor of an I_k cell: M (M->I) or I (I->I). Mirrors C `select_i`.
unsafe fn select_i(om: &OProfile, ox: &ProbMx, i: usize, k: usize) -> State {
    let q_count = ox.q_count();
    let q = (k - 1) % q_count;
    let lane = (k - 1) / q_count;
    let mpv = cell_lane(ox, i - 1, q, OM_M, lane);
    let ipv = cell_lane(ox, i - 1, q, OM_I, lane);
    let mi = if tfv_lane(om, q, P7O_MI, lane) == 0.0 {
        f32::NEG_INFINITY
    } else {
        mpv
    };
    let ii = if tfv_lane(om, q, P7O_II, lane) == 0.0 {
        f32::NEG_INFINITY
    } else {
        ipv
    };
    if mi >= ii {
        State::M
    } else {
        State::I
    }
}

/// Pick predecessor of C at row `i`: C (C->C loop) or E (E->C move). Mirrors C `select_c`.
#[inline]
fn select_c(om: &OProfile, pp: &ProbMx, ox: &ProbMx, i: usize) -> State {
    let c_loop = if om.xf[P7O_C][P7O_LOOP] == 0.0 {
        f32::NEG_INFINITY
    } else {
        ox.xmx[(i - 1) * NXCELLS + PXC] + pp.xmx[i * NXCELLS + PXC]
    };
    let e_move = if om.xf[P7O_E][P7O_MOVE] == 0.0 {
        f32::NEG_INFINITY
    } else {
        ox.xmx[i * NXCELLS + PXE]
    };
    if c_loop > e_move {
        State::C
    } else {
        State::E
    }
}

/// Pick predecessor of J at row `i`: J (J->J loop) or E (E->J loop). Mirrors C `select_j`.
#[inline]
fn select_j(om: &OProfile, pp: &ProbMx, ox: &ProbMx, i: usize) -> State {
    let j_loop = if om.xf[P7O_J][P7O_LOOP] == 0.0 {
        f32::NEG_INFINITY
    } else {
        ox.xmx[(i - 1) * NXCELLS + PXJ] + pp.xmx[i * NXCELLS + PXJ]
    };
    let e_loop = if om.xf[P7O_E][P7O_LOOP] == 0.0 {
        f32::NEG_INFINITY
    } else {
        ox.xmx[i * NXCELLS + PXE]
    };
    if j_loop > e_loop {
        State::J
    } else {
        State::E
    }
}

/// Pick the (k, state) at which E was entered on row `i`: max over all M_k
/// (M ties beat D via `>=`) then all D_k (D wins only on strict `>`).
/// Mirrors C `select_e` iteration order exactly so tie-break picks match.
unsafe fn select_e(om: &OProfile, ox: &ProbMx, i: usize, ret_k: &mut usize) -> State {
    let _ = om;
    // Mirror C hmmer/src/impl_sse/optacc.c:select_e iteration order: for each
    // q, check all 4 M lanes first (M ties beat D via `>=`), then all 4 D
    // lanes (D only wins on strict `>`). Interleaving M/D per lane changes
    // tie-break picks and can shift the chosen k.
    let q_count = ox.q_count();
    let mut max = f32::NEG_INFINITY;
    let mut smax = State::M;
    let mut kmax = 1usize;
    for q in 0..q_count {
        for lane in 0..4 {
            let k = lane * q_count + q + 1;
            let m = cell_lane(ox, i, q, OM_M, lane);
            if m >= max {
                max = m;
                smax = State::M;
                kmax = k;
            }
        }
        for lane in 0..4 {
            let k = lane * q_count + q + 1;
            let d = cell_lane(ox, i, q, OM_D, lane);
            if d > max {
                max = d;
                smax = State::D;
                kmax = k;
            }
        }
    }
    *ret_k = kmax;
    smax
}

/// Pick predecessor of B at row `i`: N (N->B move) or J (J->B move). Mirrors C `select_b`.
#[inline]
fn select_b(om: &OProfile, ox: &ProbMx, i: usize) -> State {
    let n_move = if om.xf[P7O_N][P7O_MOVE] == 0.0 {
        f32::NEG_INFINITY
    } else {
        ox.xmx[i * NXCELLS + PXN]
    };
    let j_move = if om.xf[P7O_J][P7O_MOVE] == 0.0 {
        f32::NEG_INFINITY
    } else {
        ox.xmx[i * NXCELLS + PXJ]
    };
    if n_move > j_move {
        State::N
    } else {
        State::J
    }
}

/// Index of the maximum value in `values`; ties go to the first occurrence
/// (the input order encodes the C tie-break priority: M, I, D, B).
#[inline]
fn argmax_first(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = values[0];
    for (idx, &val) in values.iter().enumerate().skip(1) {
        if val > best_val {
            best = idx;
            best_val = val;
        }
    }
    best
}
