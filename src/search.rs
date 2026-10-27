use std::cmp::Reverse;

use rayon::prelude::*;

use crate::alphabet::Alphabet;
use crate::bg::Bg;
use crate::hmm::Hmm;
use crate::pipeline::Pipeline;
use crate::profile::{profile_config, reconfig_length, Profile, P7P_NXSTATES, P7P_NXTRANS, P7_LOCAL};
use crate::sequence::Sequence;
use crate::simd::oprofile::{OProfile, P7O_NXSTATES, P7O_NXTRANS};
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

// Every length-dependent term a target sets, kept so a later target of the same length skips the
// logarithms behind them.
#[derive(Clone, Copy)]
struct Lengthed {
    l: usize,
    xsc: [[f32; P7P_NXTRANS]; P7P_NXSTATES],
    tjb_b: u8,
    xw: [[i16; P7O_NXTRANS]; P7O_NXSTATES],
    xf: [[f32; P7O_NXTRANS]; P7O_NXSTATES],
    p1: f32,
    fhmm: [f32; 3],
    null_sc: f32,
}

impl Lengthed {
    fn set(l: usize, gm: &mut Profile, om: &mut OProfile, bg: &mut Bg) -> Self {
        bg.set_length(l);
        reconfig_length(gm, l as i32);
        om.reconfig_length(l as i32);
        Lengthed {
            l,
            xsc: gm.xsc,
            tjb_b: om.tjb_b,
            xw: om.xw,
            xf: om.xf,
            p1: bg.p1,
            fhmm: bg.fhmm_t[0],
            null_sc: bg.null_one(l),
        }
    }

    fn restore(&self, gm: &mut Profile, om: &mut OProfile, bg: &mut Bg) {
        gm.xsc = self.xsc;
        gm.l = self.l as i32;
        om.tjb_b = self.tjb_b;
        om.xw = self.xw;
        om.xf = self.xf;
        om.l = self.l as i32;
        bg.p1 = self.p1;
        bg.fhmm_t[0] = self.fhmm;
    }
}

const LENGTH_SLOTS: usize = 1024;

impl<'a> Query<'a> {
    // Configured under the float mode the scan runs in, as hmmsearch configures on its workers.
    pub fn new(hmm: &'a Hmm) -> Self {
        let _mode = FlushToZero::enter();
        let abc = Alphabet::new(hmm.abc_type);
        let mut bg = Bg::new(&abc);
        let mut gm = Profile::new(hmm.m, &abc);
        profile_config(hmm, &bg, &mut gm, 100, P7_LOCAL);
        bg.set_filter(hmm.m, &hmm.compo);
        let om = OProfile::convert(&gm);
        let mut pli = Pipeline::new();
        pli.new_model(&gm);
        Query {
            hmm,
            gm,
            om,
            bg,
            pli,
        }
    }

    pub fn scan(&self, targets: &[Sequence], first: u64) -> Scan {
        let _mode = FlushToZero::enter();
        let mut pli = self.pli.fork();
        let mut gm = self.gm.clone();
        let mut om = self.om.clone();
        let mut bg = self.bg.clone();
        let mut hits = Vec::new();
        let mut found = TopHits::new();
        let mut lengths: Vec<Option<Lengthed>> = vec![None; LENGTH_SLOTS];
        let mut held: Option<Held> = None;
        for (index, sq) in (first..).zip(targets) {
            pli.n_targets = index;
            if sq.n == 0 {
                continue;
            }
            let slot = &mut lengths[sq.n % LENGTH_SLOTS];
            let kept = match slot {
                Some(kept) if kept.l == sq.n => {
                    kept.restore(&mut gm, &mut om, &mut bg);
                    *kept
                }
                _ => *slot.insert(Lengthed::set(sq.n, &mut gm, &mut om, &mut bg)),
            };
            #[cfg(not(striped))]
            pli.run_configured(&gm, &om, &bg, self.hmm, sq, kept.null_sc, &mut found);
            #[cfg(striped)]
            {
                let Some(filtersc) = pli.filters(&gm, &om, &bg, sq, kept.null_sc) else {
                    continue;
                };
                let this = Held {
                    sq,
                    kept,
                    filtersc,
                    counted: pli.n_targets,
                };
                let Some(partner) = held.take() else {
                    held = Some(this);
                    continue;
                };
                let (partner_sc, this_sc) =
                    pli.forward_pair(&om, (partner.sq, partner.kept.xf), (sq, kept.xf));
                if pli.passes_forward(partner_sc, partner.filtersc)
                    && pli.passes_forward(this_sc, filtersc)
                {
                    pli.backward_pair(&om, (partner.sq, partner.kept.xf), (sq, kept.xf));
                }
                self.finish(&mut pli, &mut gm, &mut om, &mut bg, &partner, partner_sc, &mut found);
                pli.take_partner();
                self.finish(&mut pli, &mut gm, &mut om, &mut bg, &this, this_sc, &mut found);
            }
            hits.append(&mut found.hits);
        }
        if let Some(last) = held {
            last.kept.restore(&mut gm, &mut om, &mut bg);
            let sc = pli.forward(&gm, &om, last.sq);
            self.finish(&mut pli, &mut gm, &mut om, &mut bg, &last, sc, &mut found);
            hits.append(&mut found.hits);
        }
        Scan {
            hits,
            n_past_msv: pli.n_past_msv,
            n_past_bias: pli.n_past_bias,
            n_past_vit: pli.n_past_vit,
            n_past_fwd: pli.n_past_fwd,
        }
    }

    // A held target is finished under its own length model and target count, as if it ran alone.
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        pli: &mut Pipeline,
        gm: &mut Profile,
        om: &mut OProfile,
        bg: &mut Bg,
        held: &Held,
        fwd_sc: f32,
        found: &mut TopHits,
    ) {
        let counted = pli.n_targets;
        pli.n_targets = held.counted;
        held.kept.restore(gm, om, bg);
        pli.after_forward(gm, om, bg, self.hmm, held.sq, held.kept.null_sc, held.filtersc, fwd_sc, found);
        pli.n_targets = counted.max(held.counted);
    }
}

struct Held<'t> {
    sq: &'t Sequence,
    kept: Lengthed,
    filtersc: f32,
    counted: u64,
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
