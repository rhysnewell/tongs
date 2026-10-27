//! Generic stochastic traceback — sample alignment from Forward distribution.
//! Port of generic_stotrace.c p7_GStochasticTrace().

#![allow(clippy::needless_range_loop)]

use crate::alphabet::Dsq;
use crate::dp::gmx::*;
use crate::profile::*;
use crate::trace::{State, Trace};
use crate::util::random::MersenneTwister;
use crate::util::vectorops::{f_lognorm, f_norm};

use crate::simd::oprofile::*;
use crate::simd::probmx::{ProbMx, PXB, PXC, PXE, PXJ, PXN};

/// Sample a stochastic traceback directly from a SIMD probability-space Forward matrix.
///
/// Convenience wrapper that allocates a `Trace` and forwards to
/// `stochastic_trace_pmx_into`. Works on the striped `ProbMx` produced by the
/// SIMD Forward parser.
pub fn stochastic_trace_pmx(
    rng: &mut MersenneTwister,
    l: usize,
    om: &OProfile,
    ox: &ProbMx,
) -> Trace {
    let mut tr = Trace::new();
    stochastic_trace_pmx_into(rng, l, om, ox, &mut tr);
    tr
}

/// In-place stochastic traceback from a SIMD probability-space Forward matrix.
///
/// Samples one path from the posterior alignment distribution and appends it
/// to `tr` (which is cleared first). Reverses the appended states so the
/// trace reads `S..T` on return.
pub fn stochastic_trace_pmx_into(
    rng: &mut MersenneTwister,
    l: usize,
    om: &OProfile,
    ox: &ProbMx,
    tr: &mut Trace,
) {
    tr.clear();
    let mut i = l;
    let mut k = 0usize;
    tr.append(State::T, k, i);
    tr.append(State::C, k, i);
    let mut s0 = State::C;
    let sx = Striped::new(om, ox);

    while s0 != State::S {
        let s1 = match s0 {
            State::M => {
                let next = select_m_pmx(rng, &sx, i, k);
                k -= 1;
                i -= 1;
                next
            }
            State::D => {
                let next = select_d_pmx(rng, &sx, i, k);
                k -= 1;
                next
            }
            State::I => {
                let next = select_i_pmx(rng, &sx, i, k);
                i -= 1;
                next
            }
            State::N => {
                if i == 0 {
                    State::S
                } else {
                    State::N
                }
            }
            State::C => select_c_pmx(rng, &sx, i),
            State::J => select_j_pmx(rng, &sx, i),
            State::E => select_e_pmx(rng, &sx, i, &mut k),
            State::B => select_b_pmx(rng, &sx, i),
            _ => State::S,
        };
        tr.append(s1, k, i);
        if matches!(s1, State::N | State::J | State::C) && s1 == s0 {
            i -= 1;
        }
        s0 = s1;
    }

    tr.st.reverse();
    tr.k.reverse();
    tr.i.reverse();
}

struct Striped<'a> {
    q: usize,
    width: usize,
    cells: &'a [f32],
    xmx: &'a [f32],
    row_scale: &'a [f32],
    tfv: &'a [[f32; 4]],
    xf: &'a [[f32; P7O_NXTRANS]; P7O_NXSTATES],
    m: usize,
}

impl<'a> Striped<'a> {
    fn new(om: &'a OProfile, ox: &'a ProbMx) -> Self {
        Striped {
            q: ox.q_count(),
            width: ox.striped_row_width(),
            cells: &ox.striped_dp[ox.striped_dp_offset..],
            xmx: &ox.xmx,
            row_scale: &ox.row_scale,
            tfv: &om.tfv,
            xf: &om.xf,
            m: om.m,
        }
    }

    fn slot(&self, k: usize) -> (usize, usize) {
        ((k - 1) % self.q, (k - 1) / self.q)
    }

    fn mdi(&self, i: usize, k: usize) -> (f32, f32, f32) {
        let (stripe, lane) = self.slot(k);
        let base = i * self.width + stripe * 12 + lane;
        (self.cells[base], self.cells[base + 4], self.cells[base + 8])
    }

    fn t(&self, k: usize, t: usize) -> f32 {
        let (stripe, lane) = self.slot(k);
        self.tfv[7 * stripe + t][lane]
    }

    fn dd(&self, k: usize) -> f32 {
        let (stripe, lane) = self.slot(k);
        self.tfv[7 * self.q + stripe][lane]
    }

    fn x(&self, i: usize, s: usize) -> f32 {
        self.xmx[i * 5 + s]
    }
}

fn select_m_pmx(rng: &mut MersenneTwister, sx: &Striped, i: usize, k: usize) -> State {
    let prev_k = k - 1;
    let (m, d, ins) = match prev_k > 0 {
        true => sx.mdi(i - 1, prev_k),
        false => (0.0, 0.0, 0.0),
    };
    let bm = sx.x(i - 1, PXB) * sx.t(k, P7O_BM);
    let mm = m * sx.t(k, P7O_MM);
    let im = ins * sx.t(k, P7O_IM);
    let dm = d * sx.t(k, P7O_DM);
    match choose_probs(rng, [bm, mm, im, dm]) {
        0 => State::B,
        1 => State::M,
        2 => State::I,
        _ => State::D,
    }
}

fn select_d_pmx(rng: &mut MersenneTwister, sx: &Striped, i: usize, k: usize) -> State {
    let prev_k = k - 1;
    let (m, d, _) = sx.mdi(i, prev_k);
    let md = m * sx.t(prev_k, P7O_MD);
    let dd = d * sx.dd(prev_k);
    match choose_probs(rng, [md, dd]) {
        0 => State::M,
        _ => State::D,
    }
}

fn select_i_pmx(rng: &mut MersenneTwister, sx: &Striped, i: usize, k: usize) -> State {
    let (m, _, ins) = sx.mdi(i - 1, k);
    let mi = m * sx.t(k, P7O_MI);
    let ii = ins * sx.t(k, P7O_II);
    match choose_probs(rng, [mi, ii]) {
        0 => State::M,
        _ => State::I,
    }
}

fn select_c_pmx(rng: &mut MersenneTwister, sx: &Striped, i: usize) -> State {
    let c = sx.x(i - 1, PXC) * sx.xf[P7O_C][P7O_LOOP];
    let e = sx.x(i, PXE) * sx.xf[P7O_E][P7O_MOVE] * sx.row_scale[i];
    match choose_probs(rng, [c, e]) {
        0 => State::C,
        _ => State::E,
    }
}

fn select_j_pmx(rng: &mut MersenneTwister, sx: &Striped, i: usize) -> State {
    let j = sx.x(i - 1, PXJ) * sx.xf[P7O_J][P7O_LOOP];
    let e = sx.x(i, PXE) * sx.xf[P7O_E][P7O_LOOP] * sx.row_scale[i];
    match choose_probs(rng, [j, e]) {
        0 => State::J,
        _ => State::E,
    }
}

fn select_b_pmx(rng: &mut MersenneTwister, sx: &Striped, i: usize) -> State {
    let n = sx.x(i, PXN) * sx.xf[P7O_N][P7O_MOVE];
    let j = sx.x(i, PXJ) * sx.xf[P7O_J][P7O_MOVE];
    match choose_probs(rng, [n, j]) {
        0 => State::N,
        _ => State::J,
    }
}

fn select_e_pmx(rng: &mut MersenneTwister, sx: &Striped, i: usize, ret_k: &mut usize) -> State {
    let roll = rng.next_f64();
    let norm = (1.0 / sx.x(i, PXE) as f64) as f32;
    let row = &sx.cells[i * sx.width..];
    let mut sum = 0.0_f64;
    for stripe in 0..sx.q {
        let cell = &row[stripe * 12..stripe * 12 + 8];
        for lane in 0..4 {
            sum += (cell[lane] * norm) as f64;
            if roll < sum {
                *ret_k = lane * sx.q + stripe + 1;
                return State::M;
            }
        }
        for lane in 0..4 {
            sum += (cell[4 + lane] * norm) as f64;
            if roll < sum {
                *ret_k = lane * sx.q + stripe + 1;
                return State::D;
            }
        }
    }
    *ret_k = sx.m;
    State::M
}

fn choose_probs<const N: usize>(rng: &mut MersenneTwister, mut probs: [f32; N]) -> usize {
    f_norm(&mut probs);
    rng.sample_discrete(&probs)
}

/// Stochastic traceback of a Forward matrix.
///
/// Samples an alignment of digital sequence `dsq` (length `L`) to profile `gm`
/// from the filled Forward DP matrix `gx`. Returns the sampled `Trace`,
/// reversed to read `S..T`. Counterpart of `p7_GStochasticTrace`.
pub fn g_stochastic_trace(
    rng: &mut MersenneTwister,
    dsq: &[Dsq],
    l: usize,
    gm: &Profile,
    gx: &Gmx,
) -> Trace {
    let m = gm.m;
    let mut tr = Trace::new();

    // Start from T->C at position L
    tr.append(State::T, 0, 0);
    tr.append(State::C, 0, l);
    let mut i = l;
    let mut cur_state = State::C;

    loop {
        match cur_state {
            State::C => {
                if i == 0 {
                    break;
                }
                // C from C(i-1) or E(i)
                let c_sc = gx.xmx(i - 1, P7G_C) + gm.xsc[P7P_C][P7P_LOOP];
                let e_sc = gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_MOVE];
                if sample_two(rng, c_sc, e_sc) == 0 {
                    tr.append(State::C, 0, i);
                    i -= 1;
                } else {
                    cur_state = State::E;
                    tr.append(State::E, 0, i);
                }
            }
            State::E => {
                // Local E connects from any M_k or any D_k for k=2..M.
                let mut scores = vec![f32::NEG_INFINITY; 2 * m + 1];
                for k in 1..=m {
                    scores[k] = gx.mmx(i, k);
                }
                for k in 2..=m {
                    scores[k + m] = gx.dmx(i, k);
                }
                let choice = sample_from(rng, &scores);
                if choice <= m {
                    let k = choice;
                    cur_state = State::M;
                    tr.append(State::M, k, i);
                } else {
                    let k = choice - m;
                    cur_state = State::D;
                    tr.append(State::D, k, i);
                }
            }
            State::M => {
                let k = *tr.k.last().unwrap();
                // Predecessor scores weighted by transition probabilities
                // (emission score is implicit in the Forward matrix values)
                debug_assert!(i <= dsq.len() - 2, "sequence position out of bounds");
                let bm = gx.xmx(i - 1, P7G_B) + gm.tsc(k - 1, P7P_BM);
                let mm = gx.mmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_MM);
                let im = gx.imx(i - 1, k - 1) + gm.tsc(k - 1, P7P_IM);
                let dm = gx.dmx(i - 1, k - 1) + gm.tsc(k - 1, P7P_DM);
                let choice = sample_from(rng, &[bm, mm, im, dm]);
                i -= 1;
                match choice {
                    0 => {
                        cur_state = State::B;
                        tr.append(State::B, 0, i);
                    }
                    1 => {
                        cur_state = State::M;
                        tr.append(State::M, k - 1, i);
                    }
                    2 => {
                        cur_state = State::I;
                        tr.append(State::I, k - 1, i);
                    }
                    _ => {
                        cur_state = State::D;
                        tr.append(State::D, k - 1, i);
                    }
                }
            }
            State::D => {
                // D connects from M,D at i,k-1 (generic_stotrace.c:120-129).
                // No k<=1 short-circuit in C: D_1 is reachable as an intermediate;
                // its predecessor scores are -inf so it normalizes away, but C still
                // consumes the RNG draw and appends the chosen state at k-1.
                let k = *tr.k.last().unwrap();
                let md = gx.mmx(i, k - 1) + gm.tsc(k - 1, P7P_MD);
                let dd = gx.dmx(i, k - 1) + gm.tsc(k - 1, P7P_DD);
                if sample_two(rng, md, dd) == 0 {
                    cur_state = State::M;
                    tr.append(State::M, k - 1, i);
                } else {
                    cur_state = State::D;
                    tr.append(State::D, k - 1, i);
                }
            }
            State::I => {
                let k = *tr.k.last().unwrap();
                if i == 0 {
                    break;
                }
                i -= 1;
                let mi = gx.mmx(i, k) + gm.tsc(k, P7P_MI);
                let ii = gx.imx(i, k) + gm.tsc(k, P7P_II);
                if sample_two(rng, mi, ii) == 0 {
                    cur_state = State::M;
                    tr.append(State::M, k, i);
                } else {
                    cur_state = State::I;
                    tr.append(State::I, k, i);
                }
            }
            State::B => {
                let bn = gx.xmx(i, P7G_N) + gm.xsc[P7P_N][P7P_MOVE];
                let bj = gx.xmx(i, P7G_J) + gm.xsc[P7P_J][P7P_MOVE];
                if sample_two(rng, bn, bj) == 0 {
                    cur_state = State::N;
                    tr.append(State::N, 0, i);
                } else {
                    cur_state = State::J;
                    tr.append(State::J, 0, i);
                }
            }
            State::N => {
                if i == 0 {
                    tr.append(State::S, 0, 0);
                    break;
                }
                tr.append(State::N, 0, i);
                i -= 1;
            }
            State::J => {
                if i == 0 {
                    cur_state = State::E;
                    tr.append(State::E, 0, 0);
                    continue;
                }
                let jj = gx.xmx(i - 1, P7G_J) + gm.xsc[P7P_J][P7P_LOOP];
                let je = gx.xmx(i, P7G_E) + gm.xsc[P7P_E][P7P_LOOP];
                if sample_two(rng, jj, je) == 0 {
                    tr.append(State::J, 0, i);
                    i -= 1;
                } else {
                    cur_state = State::E;
                    tr.append(State::E, 0, i);
                }
            }
            _ => break,
        }

        if tr.n > l + m + 100 {
            break;
        }
    }

    tr.st.reverse();
    tr.k.reverse();
    tr.i.reverse();
    tr
}

/// Normalize two log-space scores with `esl_vec_FLogNorm` and draw 0/1 with
/// `esl_rnd_FChoose` (= `sample_discrete`). Mirrors C's
/// `esl_vec_FLogNorm(sc,2); esl_rnd_FChoose(r, sc, 2)`.
fn sample_two(rng: &mut MersenneTwister, a: f32, b: f32) -> usize {
    let mut sc = [a, b];
    f_lognorm(&mut sc);
    rng.sample_discrete(&sc)
}

/// Normalize a log-space score vector with `esl_vec_FLogNorm` and draw an index
/// with `esl_rnd_FChoose` (= `sample_discrete`).
fn sample_from(rng: &mut MersenneTwister, scores: &[f32]) -> usize {
    let mut sc = scores.to_vec();
    f_lognorm(&mut sc);
    rng.sample_discrete(&sc)
}
