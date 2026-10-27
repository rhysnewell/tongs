use super::*;

pub(super) fn simple_consensus(hmm: &Hmm, abc: &Alphabet) -> Vec<u8> {
    let mut seq = Vec::with_capacity(hmm.m);
    for node in 1..=hmm.m {
        if masked_node(hmm, node) {
            seq.push(abc.sym[abc.unknown_code() as usize]);
        } else {
            seq.push(abc.sym[argmax(&hmm.mat[node][..abc.k])]);
        }
    }
    seq
}

pub(super) fn fancy_consensus(
    hmm: &Hmm,
    abc: &Alphabet,
    min_lower: f32,
    min_upper: f32,
) -> Vec<u8> {
    let unknown = (abc.sym[abc.unknown_code() as usize] as char).to_ascii_lowercase() as u8;
    let mut seq = Vec::with_capacity(hmm.m);
    for node in 1..=hmm.m {
        if masked_node(hmm, node) {
            seq.push(unknown);
        } else {
            let x = argmax(&hmm.mat[node][..abc.k]);
            let p = hmm.mat[node][x];
            let c = abc.sym[x] as char;
            if p < min_lower {
                seq.push(unknown);
            } else if p >= min_upper {
                seq.push(c.to_ascii_uppercase() as u8);
            } else {
                seq.push(c.to_ascii_lowercase() as u8);
            }
        }
    }
    seq
}

pub(super) fn core_emit(hmm: &Hmm, abc: &Alphabet, rng: &mut MersenneTwister) -> (Sequence, Trace) {
    let mut sq = Sequence::new();
    let mut tr = Trace::new();
    let mut k = 0usize;
    let mut i = 0usize;
    let mut state = State::B;

    tr.append(state, k, i);
    while state != State::E {
        state = match state {
            State::B | State::M => match rng.sample_discrete(&hmm.t[k][MM..=MD]) {
                0 => State::M,
                1 => State::I,
                _ => State::D,
            },
            State::I => match rng.sample_discrete(&hmm.t[k][IM..=II]) {
                0 => State::M,
                _ => State::I,
            },
            State::D => match rng.sample_discrete(&hmm.t[k][DM..=DD]) {
                0 => State::M,
                _ => State::D,
            },
            _ => panic!("invalid core emit state"),
        };

        if matches!(state, State::M | State::D) {
            k += 1;
        }
        if matches!(state, State::M | State::I) {
            i += 1;
        }
        if k == hmm.m + 1 {
            if state == State::M {
                state = State::E;
                k = 0;
            } else {
                panic!("core emitter failed to reach E from terminal match state");
            }
        }

        let residue: Option<Dsq> = match state {
            State::M => Some(rng.sample_residue(&hmm.mat[k][..abc.k])),
            State::I => Some(rng.sample_residue(&hmm.ins[k][..abc.k])),
            _ => None,
        };
        tr.append(state, k, i);
        if let Some(x) = residue {
            sq.dsq.push(x);
        }
    }
    sq.dsq.push(DSQ_SENTINEL);
    sq.n = sq.dsq.len().saturating_sub(2);
    sq.l = sq.n;
    tr.m = hmm.m;
    tr.l = i;
    (sq, tr)
}

pub(super) fn profile_emit(
    hmm: &Hmm,
    gm: &Profile,
    bg: &Bg,
    rng: &mut MersenneTwister,
) -> (Sequence, Trace) {
    let mut sq = Sequence::new();
    let mut tr = Trace::new();
    let mut k = 0usize;
    let mut i = 0usize;
    let mut kend = hmm.m;
    let mut state = State::N;

    tr.append(State::S, k, i);
    tr.append(State::N, k, i);
    while state != State::T {
        let previous = state;
        state = match state {
            State::B => {
                if gm.is_local() {
                    let (kstart, end) = sample_endpoints(rng, gm);
                    k = kstart;
                    kend = end;
                    State::M
                } else {
                    match rng.sample_discrete(&hmm.t[0][MM..=MD]) {
                        0 => {
                            k = 1;
                            State::M
                        }
                        1 => {
                            k = 0;
                            State::I
                        }
                        _ => {
                            k = 1;
                            State::D
                        }
                    }
                }
            }
            State::M => {
                if k == kend {
                    State::E
                } else {
                    match rng.sample_discrete(&hmm.t[k][MM..=MD]) {
                        0 => State::M,
                        1 => State::I,
                        _ => State::D,
                    }
                }
            }
            State::D => {
                if k == kend {
                    State::E
                } else if rng.sample_discrete(&hmm.t[k][DM..=DD]) == 0 {
                    State::M
                } else {
                    State::D
                }
            }
            State::I => {
                if rng.sample_discrete(&hmm.t[k][IM..=II]) == 0 {
                    State::M
                } else {
                    State::I
                }
            }
            State::N => {
                if sample_special(rng, gm, P7P_N) == P7P_MOVE {
                    State::B
                } else {
                    State::N
                }
            }
            State::E => {
                if sample_special(rng, gm, P7P_E) == P7P_MOVE {
                    State::C
                } else {
                    State::J
                }
            }
            State::C => {
                if sample_special(rng, gm, P7P_C) == P7P_MOVE {
                    State::T
                } else {
                    State::C
                }
            }
            State::J => {
                if sample_special(rng, gm, P7P_J) == P7P_MOVE {
                    State::B
                } else {
                    State::J
                }
            }
            _ => panic!("invalid profile emit state"),
        };

        if state == State::E {
            k = 0;
        } else if state == State::M && previous != State::B {
            k += 1;
        } else if state == State::D {
            k += 1;
        }

        let residue = if state == State::M {
            Some(rng.sample_residue(&hmm.mat[k][..gm.abc_k]))
        } else if state == State::I {
            Some(rng.sample_residue(&hmm.ins[k][..gm.abc_k]))
        } else if matches!(state, State::N | State::C | State::J) && previous == state {
            Some(rng.sample_residue(&bg.f[..gm.abc_k]))
        } else {
            None
        };

        if let Some(x) = residue {
            i += 1;
            sq.dsq.push(x);
        }
        tr.append(state, k, i);
    }

    sq.dsq.push(DSQ_SENTINEL);
    sq.n = sq.dsq.len().saturating_sub(2);
    sq.l = sq.n;
    tr.m = hmm.m;
    tr.l = i;
    (sq, tr)
}

fn sample_special(rng: &mut MersenneTwister, gm: &Profile, state: usize) -> usize {
    let probs = [
        c_exp_to_f32(gm.xsc[state][P7P_LOOP] as f64),
        c_exp_to_f32(gm.xsc[state][P7P_MOVE] as f64),
    ];
    rng.sample_discrete(&probs)
}

fn sample_endpoints(rng: &mut MersenneTwister, gm: &Profile) -> (usize, usize) {
    let mut pstart = vec![0.0_f32; gm.m + 1];
    for (k, pk) in pstart.iter_mut().enumerate().take(gm.m + 1).skip(1) {
        *pk = c_exp_to_f32(gm.tsc(k - 1, P7P_BM) as f64) * (gm.m - k + 1) as f32;
    }
    let kstart = rng.sample_discrete(&pstart);
    let kend = kstart + rng.roll(gm.m - kstart + 1);
    (kstart, kend)
}
