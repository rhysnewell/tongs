use std::cmp::Reverse;

use rayon::prelude::*;

use crate::bg::Bg;
use crate::hmm::Hmm;
use crate::pipeline::Pipeline;
use crate::profile::Profile;
use crate::sequence::Sequence;
use crate::simd::oprofile::OProfile;
use crate::tophits::{Hit, TopHits};
use crate::util::simd_env::FlushToZero;

pub struct Query<'a> {
    pub hmm: &'a Hmm,
    pub gm: Profile,
    pub om: OProfile,
    pub bg: Bg,
    pub pli: Pipeline,
}

#[derive(Default)]
pub struct Scan {
    pub hits: Vec<Hit>,
    pub n_past_msv: u64,
    pub n_past_bias: u64,
    pub n_past_vit: u64,
    pub n_past_fwd: u64,
}

impl Scan {
    pub fn absorb(&mut self, other: Scan) {
        self.hits.extend(other.hits);
        self.n_past_msv += other.n_past_msv;
        self.n_past_bias += other.n_past_bias;
        self.n_past_vit += other.n_past_vit;
        self.n_past_fwd += other.n_past_fwd;
    }
}

impl Query<'_> {
    pub fn scan(&self, targets: &[Sequence], first: u64) -> Scan {
        let _mode = FlushToZero::enter();
        let mut pli = self.pli.fork();
        let mut gm = self.gm.clone();
        let mut om = self.om.clone();
        let mut bg = self.bg.clone();
        let mut hits = Vec::new();
        let mut found = TopHits::new();
        for (index, sq) in (first..).zip(targets) {
            pli.n_targets = index;
            bg.set_length(sq.n);
            if pli.run(&mut gm, &mut om, &bg, self.hmm, sq, &mut found) {
                hits.append(&mut found.hits);
            }
            found.hits.clear();
        }
        Scan {
            hits,
            n_past_msv: pli.n_past_msv,
            n_past_bias: pli.n_past_bias,
            n_past_vit: pli.n_past_vit,
            n_past_fwd: pli.n_past_fwd,
        }
    }
}

// Each query scans its own slices of the block, so a large set of models parallelises over models
// and a single model over target slices. Hits merge back in target order, as a serial scan finds them.
pub fn scan_all(queries: &[Query], targets: &[Sequence], first: u64) -> Vec<Scan> {
    if queries.is_empty() || targets.is_empty() {
        return queries.iter().map(|_| Scan::default()).collect();
    }
    let slices = (4 * rayon::current_num_threads())
        .div_ceil(queries.len())
        .min(targets.len());
    let size = targets.len().div_ceil(slices);
    let mut units = (0..queries.len())
        .flat_map(|query| (0..slices).map(move |slice| (query, slice)))
        .collect::<Vec<_>>();
    units.sort_by_key(|&(query, _)| Reverse(queries[query].gm.m));
    let mut found = units
        .par_iter()
        .with_max_len(1)
        .map(|&(query, slice)| {
            let start = (slice * size).min(targets.len());
            let end = (start + size).min(targets.len());
            let scan = queries[query].scan(&targets[start..end], first + start as u64);
            (query * slices + slice, scan)
        })
        .collect::<Vec<_>>();
    found.sort_unstable_by_key(|(unit, _)| *unit);
    let mut scans = queries.iter().map(|_| Scan::default()).collect::<Vec<_>>();
    for (unit, scan) in found {
        scans[unit / slices].absorb(scan);
    }
    scans
}
