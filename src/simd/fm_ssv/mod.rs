//! Exact SSV-over-FM diagonal kernel for nhmmer's FM-index long-target path.
#![allow(clippy::needless_range_loop)]
//!
//! This is an additive, faithful port of the core of C's
//! `p7_SSVFM_longlarget` pipeline (hmmer/src/fm_ssv.c): the single FM trie
//! traversal (`FM_getSeeds` -> `FM_Recurse`) that scores SSV diagonals
//! *directly* over FM-index intervals using the per-position match log-odds
//! score table (C's `ssvdata->ssv_scores_f[k*Kp + x]`), carrying the full
//! `FM_DP_PAIR` DP state and pruning exactly as C does, followed by
//! `FM_mergeSeeds` and `FM_extendSeed`.
//!
//! SCOPE / FAITHFULNESS:
//!
//! * Both of C's sweeps are ported. The `fm_direction == fm_forward` sweep uses
//!   `FmIndex::prepend_interval` (C `fm_updateIntervalReverse(fmf, ...)`) on the
//!   reversed-text index; the `fm_direction == fm_backward` sweep uses
//!   `FmIndex::update_interval_forward` (C `fm_updateIntervalForward(fmb, ...)`)
//!   on the forward-text index, carrying the second (fmf-locatable) interval.
//!   The full `FM_DP_PAIR` state and pruning clauses are reproduced, including
//!   C's `opt_ext_fwd/rev` look-ahead prune — see `fm_recurse`.
//!
//! * The kernel is unit-agnostic in its score inputs: `SsvScores`, `sc_thresh_fm`,
//!   and the score-valued `FmSsvConfig` fields (`drop_lim`, `score_density_req`)
//!   must all share a unit. The nhmmer caller supplies them in NATS (`ln(p/q)`;
//!   bit thresholds * ln2) to mirror C's `ssv_scores_f`/`fm_cfg`. The kernel
//!   only emits seed *coordinates*, so the unit stays isolated from the
//!   bit-based downstream window construction.
//!
//! Wired into the nhmmer FM-index path via `fm_ssv_augment_windows`
//! (`src/subcmd/nhmmer.rs`), where it augments the seed-then-rescore candidate
//! windows to recover weak diagonals with no exact model k-mer match.
//! `FmIndex` carries C `FM_DATA`'s `occCnts_sb`/`occCnts_b` sampled rank, so
//! `occ` is O(`FREQ_CNT_B`) and this exhaustive trie traversal is tractable on
//! genome-scale blocks.

use crate::fm_index::{FmIndex, FmInterval};

mod seeds;
pub use seeds::*;
mod extend;
pub use extend::*;

/// Forward / backward orientation of the diagonal over the *model*.
/// Port of C's `fm_forward` / `fm_backward` for `model_direction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelDirection {
    Forward,
    Backward,
}

/// Strand of the target. Port of C's `p7_NOCOMPLEMENT` / `p7_COMPLEMENT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Complementarity {
    NoComplement,
    Complement,
}

/// Which FM index / interval discipline a `FM_Recurse` sweep uses. Port of C's
/// `fm_forward` / `fm_backward` for `fm_direction` (distinct from the
/// per-diagonal `model_direction`).
///
/// * `Forward`: standard backward-search step on `fmf` (the BWT of the reversed
///   block text; Rust `kind=0`), via [`FmIndex::prepend_interval`]. The single
///   interval is itself locatable on `fmf`.
/// * `Backward`: bi-directional forward-search step on `fmb` (the BWT of the
///   forward block text; Rust `kind=1`), via [`FmIndex::update_interval_forward`].
///   It carries a second interval that is locatable on `fmf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FmDirection {
    Forward,
    Backward,
}

/// Tunable thresholds and pruning parameters.
/// Mirrors the fields of C's `FM_CFG` / `FM_METADATA` that `FM_Recurse`,
/// `FM_mergeSeeds`, and `FM_extendSeed` read.
#[derive(Debug, Clone, Copy)]
pub struct FmSsvConfig {
    /// `fm_cfg->max_depth`: trie pruning length.
    pub max_depth: usize,
    /// `fm_cfg->drop_max_len`: max consecutive low-scoring run past the peak.
    pub drop_max_len: usize,
    /// `fm_cfg->drop_lim`: how close to the peak still counts as a new peak.
    pub drop_lim: f32,
    /// `fm_cfg->consec_pos_req`: required run of positive-scoring matches.
    pub consec_pos_req: usize,
    /// `fm_cfg->consensus_match_req`: consecutive consensus matches that force a seed.
    pub consensus_match_req: usize,
    /// `fm_cfg->score_density_req`: minimum bits/position.
    pub score_density_req: f32,
    /// `fm_cfg->ssv_length`: window length around a seed for `FM_mergeSeeds`/`FM_extendSeed`.
    pub ssv_length: usize,
}

impl Default for FmSsvConfig {
    fn default() -> Self {
        // Defaults mirror nhmmer.rs NhmmerFmSeedConfig::default / C fm defaults.
        Self {
            max_depth: 15,
            drop_max_len: 4,
            drop_lim: 0.3,
            consec_pos_req: 5,
            consensus_match_req: 11,
            score_density_req: 0.75,
            ssv_length: 100,
        }
    }
}

/// Compact per-diagonal DP state. Faithful port of C `FM_DP_PAIR`
/// (hmmer/src/hmmer.h). `pos` is the model position `k` reached on this path.
#[derive(Debug, Clone, Copy)]
pub struct FmDpPair {
    pub pos: i32,
    pub score: f32,
    pub max_score: f32,
    pub score_peak_len: i32,
    pub consec_pos: i16,
    pub max_consec_pos: i16,
    pub consec_consensus: i16,
    pub complementarity: Complementarity,
    pub model_direction: ModelDirection,
}

/// A diagonal seed. Faithful subset of C `FM_DIAG` (the fields the kernel and
/// `FM_mergeSeeds`/`FM_extendSeed` actually use).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmDiag {
    /// Target start position (FM coordinate), C `diag->n`.
    pub n: i64,
    /// Model position, C `diag->k`.
    pub k: i32,
    /// Diagonal length, C `diag->length`.
    pub length: i32,
    pub score: f32,
    pub complementarity: Complementarity,
}

impl FmDiag {
    /// C `sortkey`: groups by strand, then diagonal (n - k), then a fractional
    /// model-position tiebreak. Used by `FM_mergeSeeds`'s qsort.
    fn sortkey(&self, fm_n: i64, m: i32) -> f64 {
        let strand_base = if self.complementarity == Complementarity::Complement {
            (fm_n + 1) as f64
        } else {
            0.0
        };
        strand_base + (self.n - self.k as i64) as f64 + (self.k as f64) / ((m + 1) as f64)
    }
}

/// Per-position match score table. Index `[k * kp + x]` gives the LOD score in
/// bits for emitting alphabet symbol `x` at model node `k` (1..=M). This is the
/// Rust analog of C's `ssvdata->ssv_scores_f`. `kp` is the alphabet stride.
#[derive(Debug, Clone)]
pub struct SsvScores {
    pub scores: Vec<f32>,
    pub m: i32,
    pub kp: usize,
}

impl SsvScores {
    pub fn new(m: i32, kp: usize) -> Self {
        let len = ((m as usize) + 1) * kp;
        Self {
            scores: vec![0.0; len],
            m,
            kp,
        }
    }

    #[inline]
    pub fn get(&self, k: i32, x: usize) -> f32 {
        self.scores[(k as usize) * self.kp + x]
    }

    #[inline]
    pub fn set(&mut self, k: i32, x: usize, v: f32) {
        self.scores[(k as usize) * self.kp + x] = v;
    }
}

/// Optimal-extension look-ahead tables (C `P7_SCOREDATA.opt_ext_fwd/rev`).
/// `fwd[k][j]` = the best possible score from extending model position `k`
/// forward by `j+1` positions (sum of the per-node max match scores); `rev[r][j]`
/// likewise extending backward from `r`. Used by `FM_Recurse` to prune diagonals
/// that cannot reach threshold even with a perfect extension near the depth limit.
#[derive(Debug)]
struct OptExt {
    fwd: Vec<[f32; 10]>,
    rev: Vec<[f32; 10]>,
}

impl OptExt {
    /// Faithful port of `scoredata_GetSSVScoreArrays`'s opt_ext loop
    /// (`p7_scoredata.c`).
    fn build(scores: &SsvScores) -> Self {
        let m = scores.m as usize;
        let mut max_scores = vec![0f32; m + 1];
        for k in 1..=m {
            let mut best = 0f32;
            for x in 0..scores.kp {
                let v = scores.get(k as i32, x);
                if v > best {
                    best = v;
                }
            }
            max_scores[k] = best;
        }
        let mut fwd = vec![[0f32; 10]; m + 1];
        let mut rev = vec![[0f32; 10]; m + 1];
        for i in 1..m {
            let mut sc_fwd = 0f32;
            let mut sc_rev = 0f32;
            let mut j = 0usize;
            while j < 10 && i + j < m {
                sc_fwd += max_scores[i + j + 1];
                fwd[i][j] = sc_fwd;
                sc_rev += max_scores[m - i - j];
                rev[m - i][j] = sc_rev;
                j += 1;
            }
            while j < 10 {
                fwd[i][j] = fwd[i][j - 1];
                rev[m - i][j] = rev[m - i][j - 1];
                j += 1;
            }
        }
        OptExt { fwd, rev }
    }
}

/// Metadata describing the (DNA) alphabet for the FM trie traversal.
/// Mirrors the bits of C's `FM_METADATA` the kernel reads: alphabet size and
/// complement map.
#[derive(Debug, Clone)]
pub struct FmAlphabet {
    /// Number of "real" characters to enumerate over the trie (C `alph_size`,
    /// e.g. 4 for ACGT).
    pub alph_size: usize,
    /// Complement code map, `compl_alph[c]` (e.g. A<->T, C<->G).
    pub compl_alph: Vec<usize>,
}

impl FmAlphabet {
    /// Standard DNA: codes 0=A,1=C,2=G,3=T with Watson-Crick complement.
    pub fn dna() -> Self {
        Self {
            alph_size: 4,
            // A<->T (0<->3), C<->G (1<->2)
            compl_alph: vec![3, 2, 1, 0],
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/simd/fm_ssv_test.rs"]
mod tests;
