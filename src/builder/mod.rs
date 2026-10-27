//! HMM builder — construct profile HMMs from multiple sequence alignments.
//! Simplified port of p7_builder.c and build.c.

use crate::alphabet::{Alphabet, AlphabetType};
use crate::bg::Bg;
use crate::calibrate::CalibrationConfig;
use crate::hmm::*;
use crate::msa::{self, Msa};
use crate::prior::PriorStrategy;
use crate::trace::{State as TraceState, Trace};
use crate::util::random::{esl_rand64_deal, Rand64};
use crate::util::scorematrix::{ScoreMatrix, ScoreMatrixError};

mod weights;
pub use weights::*;
mod consensus;
use consensus::*;
mod tracealign;
pub use tracealign::*;
mod build;
pub use build::*;
mod model;
pub use model::*;
mod single;
pub use single::*;

pub const DEFAULT_WINDOW_BETA: f64 = 1e-7;

/// Default PB-weighting config values, mirroring `ESL_MSAWEIGHT_CFG`
/// (esl_msaweight.h:30-37). `ignore_rf` is supplied per-call because the
/// HMMER builder flips it with `--hand` (p7_builder.c:814); the rest are the
/// fixed Easel defaults.
const PB_FRAGTHRESH: f32 = 0.5;
const PB_SYMFRAC: f32 = 0.5;
/// `eslMSAWEIGHT_ALLOW_SAMP` (TRUE): permit the deep-alignment subsampling
/// consensus path. (esl_msaweight.h:33)
const PB_ALLOW_SAMP: bool = true;
/// `eslMSAWEIGHT_SAMPTHRESH` (50000): if `nseq > sampthresh`, determine
/// consensus from a subsample rather than all sequences. (esl_msaweight.h:34)
const PB_SAMPTHRESH: usize = 50000;
/// `eslMSAWEIGHT_NSAMP` (10000): number of sequences in the subsample.
/// (esl_msaweight.h:35)
const PB_NSAMP: usize = 10000;
/// `eslMSAWEIGHT_MAXFRAG` (5000): if the subsample contains more than this
/// many fragments, reject sampling and fall back to all sequences.
/// (esl_msaweight.h:36)
const PB_MAXFRAG: usize = 5000;
/// `eslMSAWEIGHT_RNGSEED` (42): fixed RNG seed for reproducible subsampling.
/// (esl_msaweight.h:37)
const PB_RNGSEED: u64 = 42;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EffectiveSeqNumber {
    Entropy {
        target_re: Option<f64>,
        target_sigma: Option<f64>,
    },
    EntropyExp {
        target_re: Option<f64>,
        target_sigma: Option<f64>,
    },
    Cluster {
        identity_cutoff: f64,
    },
    None,
    Set(f32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RelativeWeighting {
    PositionBased,
    Gsc,
    Blosum { identity_cutoff: f64 },
    Given,
    None,
}

#[cfg(test)]
#[path = "../../tests/unit/builder_test.rs"]
mod tests;
