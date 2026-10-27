//! Stochastic traceback through a striped Forward matrix, a port of impl_sse/stotrace.c.

use crate::trace::{State, Trace};
use crate::util::random::MersenneTwister;
use crate::util::vectorops::f_norm;

use crate::simd::oprofile::*;
use crate::simd::probmx::{ProbMx, PXB, PXC, PXE, PXJ, PXN};

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
        for (state, lanes) in [(State::M, &cell[..4]), (State::D, &cell[4..])] {
            for (lane, &p) in lanes.iter().enumerate() {
                sum += (p * norm) as f64;
                if roll < sum {
                    *ret_k = lane * sx.q + stripe + 1;
                    return state;
                }
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
