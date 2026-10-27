use std::cmp::Reverse;

use rayon::prelude::*;

use crate::alphabet::Alphabet;
use crate::bg::Bg;
use crate::hmm::Hmm;
use crate::pipeline::Pipeline;
use crate::profile::{
    profile_config, reconfig_length, Profile, P7P_NXSTATES, P7P_NXTRANS, P7_LOCAL,
};
use crate::sequence::Sequence;
use crate::simd::oprofile::{OProfile, P7O_NXSTATES, P7O_NXTRANS};
use crate::tophits::{Hit, TopHits};
use crate::util::simd_env::FlushToZero;
use crate::{
    domaindef::{Definition, Model, Target},
    simd::bck_filter::{backward, backward_pair, Backward},
    simd::fwd_filter::{forward, forward_pair, Forward},
    simd::probmx::ProbMx,
};

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
        let mut scanner = Scanner {
            query: self,
            pli: self.pli.fork(),
            gm: self.gm.clone(),
            om: self.om.clone(),
            bg: self.bg.clone(),
            found: TopHits::new(),
            hits: Vec::new(),
            pool: Vec::new(),
            forward_held: None,
            backward_held: None,
        };
        let mut lengths: Vec<Option<Lengthed>> = vec![None; LENGTH_SLOTS];
        for (index, sq) in (first..).zip(targets) {
            scanner.pli.n_targets = index;
            if sq.n == 0 {
                continue;
            }
            let slot = &mut lengths[sq.n % LENGTH_SLOTS];
            let kept = match slot {
                Some(kept) if kept.l == sq.n => {
                    kept.restore(&mut scanner.gm, &mut scanner.om, &mut scanner.bg);
                    *kept
                }
                _ => *slot.insert(Lengthed::set(
                    sq.n,
                    &mut scanner.gm,
                    &mut scanner.om,
                    &mut scanner.bg,
                )),
            };
            scanner.add(sq, kept);
        }
        scanner.finish()
    }
}

#[derive(Default)]
struct Parse {
    fwd: ProbMx,
    bck: ProbMx,
}

struct Pending<'t> {
    sq: &'t Sequence,
    kept: Lengthed,
    filtersc: f32,
    counted: u64,
    fwd_sc: f32,
    parse: Parse,
}

impl Pending<'_> {
    fn target(&self) -> Target<'_> {
        Target {
            dsq: &self.sq.dsq,
            l: self.sq.n,
            null_sc: self.kept.null_sc,
            xf: self.kept.xf,
            fwd: &self.parse.fwd,
            bck: &self.parse.bck,
        }
    }
}

// Filter survivors pair up for Forward, and Forward survivors pair up in file order for Backward
// and domain definition, so a survivor whose partner failed waits for the next one.
struct Scanner<'q, 't> {
    query: &'q Query<'q>,
    pli: Pipeline,
    gm: Profile,
    om: OProfile,
    bg: Bg,
    found: TopHits,
    hits: Vec<Hit>,
    pool: Vec<Parse>,
    forward_held: Option<Pending<'t>>,
    backward_held: Option<Pending<'t>>,
}

impl<'t> Scanner<'_, 't> {
    fn add(&mut self, sq: &'t Sequence, kept: Lengthed) {
        let Some(filtersc) = self.pli.filters(&self.om, &self.bg, sq, kept.null_sc) else {
            return;
        };
        let this = Pending {
            sq,
            kept,
            filtersc,
            counted: self.pli.n_targets,
            fwd_sc: 0.0,
            parse: self.pool.pop().unwrap_or_default(),
        };
        let Some(mut partner) = self.forward_held.take() else {
            self.forward_held = Some(this);
            return;
        };
        let mut this = this;
        let m = self.om.m;
        partner.parse.fwd.resize_parser(m, partner.sq.n);
        this.parse.fwd.resize_parser(m, this.sq.n);
        let mut a = Forward::new(
            &partner.sq.dsq,
            0,
            partner.sq.n,
            partner.kept.xf,
            &mut partner.parse.fwd,
        );
        let mut b = Forward::new(
            &this.sq.dsq,
            0,
            this.sq.n,
            this.kept.xf,
            &mut this.parse.fwd,
        );
        (partner.fwd_sc, this.fwd_sc) = unsafe { forward_pair(&self.om, &mut a, &mut b) };
        self.queue(partner);
        self.queue(this);
    }

    fn queue(&mut self, p: Pending<'t>) {
        if !self.pli.survives_forward(p.fwd_sc, p.filtersc) {
            self.pool.push(p.parse);
            return;
        }
        let Some(mut first) = self.backward_held.take() else {
            self.backward_held = Some(p);
            return;
        };
        let mut second = p;
        let m = self.om.m;
        first.parse.bck.resize_parser(m, first.sq.n);
        second.parse.bck.resize_parser(m, second.sq.n);
        unsafe {
            backward_pair(
                &self.om,
                &mut backward_lane(&mut first),
                &mut backward_lane(&mut second),
            )
        };
        let model = Model {
            gm: &self.gm,
            om: &self.om,
            hmm: self.query.hmm,
            bg: &self.bg,
        };
        let definitions = self.pli.define(&model, [&first.target(), &second.target()]);
        for (p, definition) in [first, second].into_iter().zip(definitions) {
            self.report(p, definition);
        }
    }

    fn report(&mut self, p: Pending<'t>, definition: Definition) {
        let counted = self.pli.n_targets;
        self.pli.n_targets = p.counted;
        let (sq, null_sc) = (p.sq, p.kept.null_sc);
        self.pli.report(
            &self.gm,
            &self.bg,
            sq,
            null_sc,
            p.fwd_sc,
            definition,
            &mut self.found,
        );
        self.pli.n_targets = counted.max(p.counted);
        self.hits.append(&mut self.found.hits);
        self.pool.push(p.parse);
    }

    fn finish(mut self) -> Scan {
        if let Some(mut last) = self.forward_held.take() {
            last.parse.fwd.resize_parser(self.om.m, last.sq.n);
            let mut lane = Forward::new(
                &last.sq.dsq,
                0,
                last.sq.n,
                last.kept.xf,
                &mut last.parse.fwd,
            );
            last.fwd_sc = unsafe { forward(&self.om, &mut lane) };
            self.queue(last);
        }
        if let Some(mut last) = self.backward_held.take() {
            last.parse.bck.resize_parser(self.om.m, last.sq.n);
            unsafe { backward(&self.om, &mut backward_lane(&mut last)) };
            let model = Model {
                gm: &self.gm,
                om: &self.om,
                hmm: self.query.hmm,
                bg: &self.bg,
            };
            let [definition] = self.pli.define(&model, [&last.target()]);
            self.report(last, definition);
        }
        self.into_scan()
    }

    fn into_scan(self) -> Scan {
        Scan {
            hits: self.hits,
            n_past_msv: self.pli.n_past_msv,
            n_past_bias: self.pli.n_past_bias,
            n_past_vit: self.pli.n_past_vit,
            n_past_fwd: self.pli.n_past_fwd,
        }
    }
}

fn backward_lane<'p>(p: &'p mut Pending) -> Backward<'p> {
    Backward::new(
        &p.sq.dsq,
        0,
        p.sq.n,
        p.kept.xf,
        Some(&p.parse.fwd.row_scale),
        &mut p.parse.bck,
    )
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
