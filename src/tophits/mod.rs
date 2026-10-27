//! P7_TOPHITS - Ranked hit list for search results.
//! Simplified port focused on what hmmsearch needs.

#![allow(clippy::doc_lazy_continuation)]

use crate::alphabet::Alphabet;
use crate::msa::Msa;
use crate::sequence::Sequence;
use crate::trace::{State, Trace};
use crate::util::cmath::c_exp_f64;

mod display;
pub use display::*;
mod alignment;
pub use alignment::*;

/// Alignment display data for one domain.
#[derive(Debug, Clone, Default)]
pub struct AliDisplay {
    pub model: String,  // consensus model line
    pub mline: String,  // match/identity line
    pub aseq: String,   // aligned target sequence
    pub ppline: String, // posterior probability annotation
    pub rfline: String, // RF annotation (empty if HMM has no RF)
    pub hmmfrom: usize,
    pub hmmto: usize,
    pub sqfrom: usize,
    pub sqto: usize,
}

/// A single domain within a hit.
#[derive(Debug, Clone)]
pub struct Domain {
    pub iali: i64, // alignment start in seq (1-based)
    pub jali: i64, // alignment end in seq
    pub ienv: i64, // envelope start
    pub jenv: i64, // envelope end
    pub bitscore: f32,
    pub lnp: f64,     // log P-value
    pub dombias: f32, // bias correction
    pub oasc: f32,    // optimal accuracy score
    pub envsc: f32,   // envelope score
    pub domcorrection: f32,
    pub is_reported: bool,
    pub is_included: bool,
    pub ad: Option<AliDisplay>, // alignment display data
}

/// A sequence-level hit.
#[derive(Debug, Clone)]
pub struct Hit {
    pub name: String,
    pub acc: String,
    pub desc: String,
    pub n: usize,       // target sequence length
    pub sortkey: f64,   // primary sort key (lnP, or negative score for score-threshold ranking)
    pub score: f32,     // overall bit score
    pub bias: f32,      // bias correction in bits
    pub pre_score: f32, // pre-bias-correction score
    pub sum_score: f32, // sum score
    pub lnp: f64,       // log P-value
    pub pre_lnp: f64,
    pub sum_lnp: f64,
    pub nexpected: f32, // expected number of domains
    pub nregions: usize,
    pub nclustered: usize,
    pub noverlaps: usize,
    pub nenvelopes: usize,
    pub ndom: usize, // actual number of domains
    pub nreported: usize,
    pub nincluded: usize,
    pub dcl: Vec<Domain>,
    pub flags: u32,
    pub seqidx: i64,
    pub subseq_start: i64,
}

pub const P7_IS_REPORTED: u32 = 1 << 0;
pub const P7_IS_INCLUDED: u32 = 1 << 1;
pub const P7_IS_NEW: u32 = 1 << 2;
pub const P7_IS_DROPPED: u32 = 1 << 3;
pub const P7_IS_DUPLICATE: u32 = 1 << 4;

/// Collection of ranked search results.
#[derive(Debug)]
pub struct TopHits {
    pub hits: Vec<Hit>,
    pub nreported: usize,
    pub nincluded: usize,
    pub is_sorted: bool,
}

impl TopHits {
    /// Allocate a new (empty) ranked hit list.
    /// Port of `p7_tophits_Create()`. Hits are added with `create_next_hit()`.
    pub fn new() -> Self {
        TopHits {
            hits: Vec::new(),
            nreported: 0,
            nincluded: 0,
            is_sorted: false,
        }
    }

    /// Append a fresh zero-initialized `Hit` and return a mutable reference for
    /// the caller to fill in. Port of `p7_tophits_CreateNextHit()`.
    pub fn create_next_hit(&mut self) -> &mut Hit {
        self.hits.push(Hit {
            name: String::new(),
            acc: String::new(),
            desc: String::new(),
            n: 0,
            sortkey: 0.0,
            score: 0.0,
            bias: 0.0,
            pre_score: 0.0,
            sum_score: 0.0,
            lnp: 0.0,
            pre_lnp: 0.0,
            sum_lnp: 0.0,
            nexpected: 0.0,
            nregions: 0,
            nclustered: 0,
            noverlaps: 0,
            nenvelopes: 0,
            ndom: 0,
            nreported: 0,
            nincluded: 0,
            dcl: Vec::new(),
            flags: P7_IS_NEW,
            seqidx: -1,
            subseq_start: 0,
        });
        self.is_sorted = false;
        self.hits.last_mut().unwrap()
    }

    /// Remove duplicate hits produced by overlapping long-target windows
    /// finding the same alignment. Port of `p7_tophits_RemoveDuplicates`
    /// (hmmer/src/p7_tophits.c:823).
    ///
    /// Scans consecutive hits (caller must have sorted the list by
    /// sequence/position). Two hits are duplicates when they target the
    /// same model name and same source sequence on the same strand AND
    /// overlap on the HMM AND at least one of:
    ///   - ali start within 3 positions
    ///   - ali end within 3 positions
    ///   - overlap covers ≥95% of the shorter alignment
    ///   The worse-E hit is marked `P7_IS_DUPLICATE` (and stripped of the
    ///   REPORTED/INCLUDED flags).
    pub fn remove_duplicates(&mut self) {
        if self.hits.len() < 2 {
            return;
        }
        let n = self.hits.len();
        let mut prev = 0usize;
        for i in 1..n {
            // Extract comparison fields without holding a borrow.
            let (p_j, s_j_raw, e_j_raw, hmm_from_j, hmm_to_j) = {
                let h = &self.hits[prev];
                let dom = h.dcl.first();
                let (sj, ej) = dom.map(|d| (d.iali, d.jali)).unwrap_or((0, 0));
                let (hf, ht) = dom
                    .and_then(|d| d.ad.as_ref().map(|a| (a.hmmfrom as i64, a.hmmto as i64)))
                    .unwrap_or((0, 0));
                (h.lnp, sj, ej, hf, ht)
            };
            let (p_i, s_i_raw, e_i_raw, hmm_from_i, hmm_to_i) = {
                let h = &self.hits[i];
                let dom = h.dcl.first();
                let (si, ei) = dom.map(|d| (d.iali, d.jali)).unwrap_or((0, 0));
                let (hf, ht) = dom
                    .and_then(|d| d.ad.as_ref().map(|a| (a.hmmfrom as i64, a.hmmto as i64)))
                    .unwrap_or((0, 0));
                (h.lnp, si, ei, hf, ht)
            };
            // C (p7_tophits.c:861-862) compares the name/seqidx of hit i against
            // the *immediately adjacent* prior hit (i-1), even if i-1 was itself
            // flagged duplicate — distinct from `j` (the last *kept* hit) used for
            // the coordinate/p-value fields above.
            let name_i = self.hits[i].name.clone();
            let seqidx_i = self.hits[i].seqidx;
            let name_j = self.hits[i - 1].name.clone();
            let seqidx_j = self.hits[i - 1].seqidx;
            let dir_j = if s_j_raw < e_j_raw { 1 } else { -1 };
            let dir_i = if s_i_raw < e_i_raw { 1 } else { -1 };
            let (s_j, e_j) = if dir_j == -1 {
                (e_j_raw, s_j_raw)
            } else {
                (s_j_raw, e_j_raw)
            };
            let (s_i, e_i) = if dir_i == -1 {
                (e_i_raw, s_i_raw)
            } else {
                (s_i_raw, e_i_raw)
            };
            let len_j = e_j - s_j + 1;
            let len_i = e_i - s_i + 1;
            let intersect_alistart = s_i.max(s_j);
            let intersect_aliend = e_i.min(e_j);
            let intersect_alilen = intersect_aliend - intersect_alistart + 1;
            let intersect_hmmstart = hmm_from_i.max(hmm_from_j);
            let intersect_hmmend = hmm_to_i.min(hmm_to_j);
            let intersect_hmmlen = intersect_hmmend - intersect_hmmstart + 1;

            let flush_start = (s_i - s_j).abs() <= 3;
            let flush_end = (e_i - e_j).abs() <= 3;
            let mostly_i = len_i > 0 && (intersect_alilen as f64) >= (len_i as f64 * 0.95);
            let mostly_j = len_j > 0 && (intersect_alilen as f64) >= (len_j as f64 * 0.95);

            let is_dup = name_i == name_j
                && seqidx_i == seqidx_j
                && dir_i == dir_j
                && intersect_hmmlen > 0
                && (flush_start || flush_end || mostly_i || mostly_j);

            if is_dup {
                // Keep the one with lower (better) lnp.
                let remove = if p_i < p_j { prev } else { i };
                self.hits[remove].flags |= P7_IS_DUPLICATE;
                self.hits[remove].flags &= !P7_IS_REPORTED;
                self.hits[remove].flags &= !P7_IS_INCLUDED;
                // Prev becomes whichever one we kept.
                if remove == prev {
                    prev = i;
                }
            } else {
                prev = i;
            }
        }
        // Recount reported/included.
        self.nreported = self
            .hits
            .iter()
            .filter(|h| h.flags & P7_IS_REPORTED != 0)
            .count();
        self.nincluded = self
            .hits
            .iter()
            .filter(|h| h.flags & P7_IS_INCLUDED != 0)
            .count();
    }

    /// Sort hits by (seqidx, alignment start position) for duplicate
    /// detection. Mirrors C `p7_tophits_SortBySeqidxAndAlipos` which is
    /// invoked immediately before `p7_tophits_RemoveDuplicates` in
    /// `nhmmer.c`. After sorting here, duplicate hits at the same
    /// position are adjacent.
    pub fn sort_by_seqidx_and_alipos(&mut self) {
        self.hits.sort_by(|a, b| {
            a.seqidx
                .cmp(&b.seqidx)
                .then_with(|| compare_hit_alipos(a, b))
        });
    }

    /// Sort hits by (model name, alignment position) for duplicate detection in
    /// nhmmscan. Mirrors C `p7_tophits_SortByModelnameAndAlipos`.
    pub fn sort_by_modelname_and_alipos(&mut self) {
        self.hits
            .sort_by(|a, b| a.name.cmp(&b.name).then_with(|| compare_hit_alipos(a, b)));
    }

    /// Sort hits by sort key (E-value / score) with C HMMER's tiebreakers
    /// (p7_tophits.c:hit_sorter_by_sortkey):
    ///   1. sortkey ascending (lnP ascending = most-significant first).
    ///   2. name.
    ///   3. strand (positive first).
    ///   4. `dcl[0].iali` ascending.
    ///
    /// Stability note (audit 09 finding, intentional): Rust `sort_by` is a
    /// stable sort, whereas C uses `qsort`, whose ordering on a *full* tie
    /// (every comparator key equal) is unspecified. C's tie order therefore
    /// cannot be reproduced faithfully; a stable sort that preserves input
    /// order is the deterministic, safe choice. This only affects output line
    /// order among exact ties, which is effectively unobservable. The same
    /// reasoning applies to `sort_by_seqidx_and_alipos` and
    /// `sort_by_modelname_and_alipos`.
    pub fn sort_by_sortkey(&mut self) {
        self.hits.sort_by(|a, b| {
            a.sortkey
                .partial_cmp(&b.sortkey)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.name.cmp(&b.name))
                .then_with(|| {
                    let a_dom = a.dcl.first();
                    let b_dom = b.dcl.first();
                    let a_dir = a_dom
                        .map(|d| if d.iali < d.jali { 1 } else { -1 })
                        .unwrap_or(1);
                    let b_dir = b_dom
                        .map(|d| if d.iali < d.jali { 1 } else { -1 })
                        .unwrap_or(1);
                    if a_dir != b_dir {
                        // + strand (dir=1) goes before - strand (dir=-1).
                        b_dir.cmp(&a_dir)
                    } else {
                        let a_i = a_dom.map(|d| d.iali).unwrap_or(0);
                        let b_i = b_dom.map(|d| d.iali).unwrap_or(0);
                        a_i.cmp(&b_i)
                    }
                })
        });
        self.is_sorted = true;
    }

    /// Apply reporting and inclusion thresholds.
    /// Uses E-value thresholds: inc_e for inclusion, report_e for reporting.
    pub fn threshold(&mut self, pli: &super::pipeline::Pipeline, z: f64, domz: f64) {
        self.nreported = 0;
        self.nincluded = 0;

        if pli.use_bit_cutoffs != super::pipeline::BitCutoff::None {
            for hit in &mut self.hits {
                hit.sortkey = pli.hit_sortkey(hit.score, hit.lnp);
                hit.nreported = hit.dcl.iter().filter(|dom| dom.is_reported).count();
                hit.nincluded = hit.dcl.iter().filter(|dom| dom.is_included).count();
                if hit.flags & P7_IS_REPORTED != 0 {
                    self.nreported += 1;
                }
                if hit.flags & P7_IS_INCLUDED != 0 {
                    self.nincluded += 1;
                }
            }
            self.workaround_bug_h74();
            if pli.score_sort_active() {
                self.sort_by_sortkey();
            }
            return;
        }

        for hit in &mut self.hits {
            hit.flags &= !(P7_IS_REPORTED | P7_IS_INCLUDED | P7_IS_DROPPED);
            hit.nreported = 0;
            hit.nincluded = 0;
            hit.sortkey = pli.hit_sortkey(hit.score, hit.lnp);
            for dom in &mut hit.dcl {
                dom.is_reported = false;
                dom.is_included = false;
            }

            // Skip hits already marked duplicate by remove_duplicates.
            if hit.flags & P7_IS_DUPLICATE != 0 {
                continue;
            }

            let evalue = z * c_exp_f64(hit.lnp);

            let reported = if pli.by_e {
                evalue <= pli.e_value_threshold
            } else if let Some(t) = pli.t {
                hit.score as f64 >= t
            } else {
                evalue <= pli.e_value_threshold
            };

            if reported {
                hit.flags |= P7_IS_REPORTED;
                self.nreported += 1;
            } else {
                hit.flags |= P7_IS_DROPPED;
            }

            let included = if pli.inc_by_e {
                evalue <= pli.inc_e
            } else if let Some(t) = pli.inc_t {
                hit.score as f64 >= t
            } else {
                evalue <= pli.inc_e
            };
            if reported && included {
                hit.flags |= P7_IS_INCLUDED;
                self.nincluded += 1;
            }
        }

        for hit in &mut self.hits {
            if hit.flags & P7_IS_REPORTED == 0 {
                continue;
            }

            if pli.long_target {
                if let Some(dom) = hit.dcl.first_mut() {
                    dom.is_reported = true;
                    hit.nreported = 1;
                    if hit.flags & P7_IS_INCLUDED != 0 {
                        dom.is_included = true;
                        hit.nincluded = 1;
                    }
                }
                continue;
            }

            let target_included = hit.flags & P7_IS_INCLUDED != 0;
            for dom in &mut hit.dcl {
                let dom_evalue = domz * c_exp_f64(dom.lnp);

                let dom_reported = if pli.dom_by_e {
                    dom_evalue <= pli.dom_e_value_threshold
                } else if let Some(t) = pli.dom_t {
                    dom.bitscore as f64 >= t
                } else {
                    dom_evalue <= pli.dom_e_value_threshold
                };

                if dom_reported {
                    dom.is_reported = true;
                    hit.nreported += 1;
                }

                let dom_included = target_included
                    && if pli.incdom_by_e {
                        dom_evalue <= pli.inc_dome
                    } else if let Some(t) = pli.inc_dom_t {
                        dom.bitscore as f64 >= t
                    } else {
                        dom_evalue <= pli.inc_dome
                    };
                if dom_included {
                    dom.is_included = true;
                    hit.nincluded += 1;
                }
            }
        }

        self.workaround_bug_h74();

        if pli.score_sort_active() {
            self.sort_by_sortkey();
        }
    }

    /// Workaround for HMMER bug #h74: two distinct envelopes can produce
    /// identical optimal-accuracy alignments (same `iali`/`jali`) because H3's
    /// Forward/Backward cannot be limited to a profile-coordinate range. When
    /// this happens we un-report / un-include the lower-scoring duplicate domain
    /// so only one copy is shown. Faithful port of `workaround_bug_h74()` in
    /// `hmmer/src/p7_tophits.c:756`, called at the end of `p7_tophits_Threshold`.
    ///
    /// Only hits flagged with overlapping envelopes (`noverlaps != 0`) are
    /// examined; this adjusts per-domain `is_reported`/`is_included` flags and
    /// the per-hit `nreported`/`nincluded` domain counts (not the target-level
    /// counts), matching C exactly.
    fn workaround_bug_h74(&mut self) {
        for hit in &mut self.hits {
            if hit.noverlaps == 0 {
                continue;
            }
            let ndom = hit.dcl.len();
            for d1 in 0..ndom {
                for d2 in (d1 + 1)..ndom {
                    if hit.dcl[d1].iali == hit.dcl[d2].iali && hit.dcl[d1].jali == hit.dcl[d2].jali
                    {
                        let dremoved = if hit.dcl[d1].bitscore >= hit.dcl[d2].bitscore {
                            d2
                        } else {
                            d1
                        };
                        if hit.dcl[dremoved].is_reported {
                            hit.dcl[dremoved].is_reported = false;
                            hit.nreported = hit.nreported.saturating_sub(1);
                        }
                        if hit.dcl[dremoved].is_included {
                            hit.dcl[dremoved].is_included = false;
                            hit.nincluded = hit.nincluded.saturating_sub(1);
                        }
                    }
                }
            }
        }
    }
}

impl Default for TopHits {
    fn default() -> Self {
        Self::new()
    }
}

fn compare_hit_alipos(a: &Hit, b: &Hit) -> std::cmp::Ordering {
    let (a_start, a_end, a_dir) = normalized_domain_span(a.dcl.first());
    let (b_start, b_end, b_dir) = normalized_domain_span(b.dcl.first());
    if a_dir != b_dir {
        return b_dir.cmp(&a_dir);
    }
    a_start.cmp(&b_start).then_with(|| b_end.cmp(&a_end))
}

fn normalized_domain_span(dom: Option<&Domain>) -> (i64, i64, i32) {
    let (mut start, mut end) = dom.map(|d| (d.iali, d.jali)).unwrap_or((0, 0));
    let dir = if start < end { 1 } else { -1 };
    if dir == -1 {
        std::mem::swap(&mut start, &mut end);
    }
    (start, end, dir)
}

#[cfg(test)]
#[path = "../../tests/unit/tophits_test.rs"]
mod tests;
