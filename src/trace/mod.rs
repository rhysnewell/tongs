//! P7_TRACE — Viterbi traceback path through the HMM.
//! Port of p7_trace.c and generic_vtrace.c.

use crate::alphabet::{Alphabet, Dsq};
use crate::dp::gmx::*;
use crate::profile::*;
use crate::tophits::AliDisplay;

mod gtrace;
pub use gtrace::*;
mod display;
pub use display::*;

/// State types in a trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum State {
    S = 4,  // Start
    N = 5,  // N-terminal
    B = 6,  // Begin
    M = 1,  // Match
    D = 2,  // Delete
    I = 3,  // Insert
    E = 7,  // End
    C = 8,  // C-terminal
    T = 9,  // Terminal
    J = 10, // Join (multi-domain)
    X = 11, // Missing data / fragment marker
}

/// A traceback path through the HMM.
#[derive(Debug, Clone)]
pub struct Trace {
    /// State type at each position
    pub st: Vec<State>,
    /// Model node index (1..M for M/D/I, 0 otherwise)
    pub k: Vec<usize>,
    /// Sequence position (1..L for emitting states, 0 otherwise)
    pub i: Vec<usize>,
    /// Trace length
    pub n: usize,
    /// Model length for traces constructed relative to a model/MSA
    pub m: usize,
    /// Sequence/alignment coordinate length
    pub l: usize,
    /// Optional posterior probability annotation, parallel to state path.
    pub pp: Option<Vec<f32>>,
}

impl Trace {
    /// Allocate a new empty (growable, reusable) traceback.
    /// Port of `p7_trace_Create()`; PP storage is allocated lazily on the first
    /// `append_with_pp()` call (matching `p7_trace_CreateWithPP()`).
    pub fn new() -> Self {
        Trace {
            st: Vec::new(),
            k: Vec::new(),
            i: Vec::new(),
            n: 0,
            m: 0,
            l: 0,
            pp: None,
        }
    }

    /// Reinitialize the trace, retaining allocations for reuse.
    /// Port of `p7_trace_Reuse()`.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.st.clear();
        self.k.clear();
        self.i.clear();
        if let Some(pp) = &mut self.pp {
            pp.clear();
        }
        self.n = 0;
    }

    /// Append one state (no PP) to a left-to-right-growing trace.
    /// Port of `p7_trace_Append()`. For emit-on-transition states (N/C/J), the
    /// first state in a run is recorded with i=0; subsequent ones carry `i`.
    #[inline(always)]
    pub fn append(&mut self, state: State, k: usize, i: usize) {
        self.append_internal(state, k, i, None);
    }

    /// Append one state with an associated posterior probability for emitted
    /// residues. Port of `p7_trace_AppendWithPP()`; lazily allocates `self.pp`.
    /// `pp` is silently ignored for nonemitting states (recorded as 0.0).
    #[inline(always)]
    pub fn append_with_pp(&mut self, state: State, k: usize, i: usize, pp: f32) {
        self.append_internal(state, k, i, Some(pp));
    }

    /// Shared implementation backing both `append` and `append_with_pp`.
    /// Encodes the state-specific rules for which of (k, i, pp) get stored.
    #[inline(always)]
    fn append_internal(&mut self, state: State, k: usize, i: usize, pp: Option<f32>) {
        self.st.push(state);
        // Per-state stored pp (C zeroes pp for all nonemitting states, and for
        // the first N/C/J in a run; see p7_trace_AppendWithPP, p7_trace.c:1056-1085).
        let mut stored_pp = pp.unwrap_or(0.0);
        match state {
            State::N | State::C | State::J => {
                let emitted = self.st.len() > 1 && self.st[self.st.len() - 2] == state;
                self.k.push(0);
                if emitted {
                    self.i.push(i);
                } else {
                    // First N/C/J in a run is non-emitting: i=0, pp=0.0.
                    self.i.push(0);
                    stored_pp = 0.0;
                }
            }
            State::S | State::B | State::E | State::T => {
                self.k.push(0);
                self.i.push(0);
                stored_pp = 0.0;
            }
            State::X => {
                self.k.push(k);
                self.i.push(i);
                stored_pp = 0.0;
            }
            State::D => {
                self.k.push(k);
                self.i.push(0);
                stored_pp = 0.0;
            }
            State::M | State::I => {
                self.k.push(k);
                self.i.push(i);
            }
        }
        if pp.is_some() {
            if self.pp.is_none() {
                self.pp = Some(vec![0.0; self.n]);
            }
            self.pp.as_mut().unwrap().push(stored_pp);
        } else if let Some(pp_values) = &mut self.pp {
            pp_values.push(0.0);
        }
        self.n += 1;
    }

    /// Return the (hmmfrom, hmmto, sqfrom, sqto) bounds of the first domain
    /// (states between the first B and the next E). Port of
    /// `p7_trace_GetDomainCoords()` restricted to the first domain.
    pub fn domain_coords(&self) -> Option<(usize, usize, usize, usize)> {
        let mut hmmfrom = 0;
        let mut hmmto = 0;
        let mut sqfrom = 0;
        let mut sqto = 0;
        let mut in_domain = false;

        for z in 0..self.n {
            if self.st[z] == State::B {
                in_domain = true;
            }
            if in_domain && self.st[z] == State::M {
                if hmmfrom == 0 {
                    hmmfrom = self.k[z];
                    sqfrom = self.i[z];
                }
                hmmto = self.k[z];
                sqto = self.i[z];
            }
            if self.st[z] == State::E {
                break;
            }
        }

        if hmmfrom > 0 {
            Some((hmmfrom, hmmto, sqfrom, sqto))
        } else {
            None
        }
    }
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/trace_test.rs"]
mod tests;
