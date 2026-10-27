mod envelope;
mod long_target;
mod null2;
mod regions;
mod score;

use crate::alphabet::{Alphabet, Dsq};
use crate::bg::Bg;
use crate::hmm::Hmm;
use crate::profile::Profile;
use crate::tophits::Domain;
use crate::trace::Trace;

use regions::{cluster_region, find_domain_regions, is_multidomain_region, Job};

#[derive(Debug, Clone, Copy, Default)]
pub struct DomainDefinitionStats {
    pub nregions: usize,
    pub nclustered: usize,
    pub noverlaps: usize,
    pub nenvelopes: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct Settings {
    pub seed: u32,
    pub make_alignment: bool,
    pub make_alignment_display: bool,
    pub long_target: bool,
    pub do_null2: bool,
}

pub(crate) struct Definition {
    pub domains: Vec<Domain>,
    pub nexpected: f32,
    pub seq_bias: f32,
    pub stats: DomainDefinitionStats,
}

pub(crate) struct Model<'a> {
    pub gm: &'a Profile,
    pub om: &'a crate::simd::oprofile::OProfile,
    pub hmm: &'a Hmm,
    pub bg: &'a Bg,
}

pub(crate) struct Target<'a> {
    pub dsq: &'a [Dsq],
    pub l: usize,
    pub null_sc: f32,
    pub xf: [[f32; crate::simd::oprofile::P7O_NXTRANS]; crate::simd::oprofile::P7O_NXSTATES],
    pub fwd: &'a crate::simd::probmx::ProbMx,
    pub bck: &'a crate::simd::probmx::ProbMx,
}

#[derive(Default)]
pub(crate) struct Scratch {
    btot: Vec<f32>,
    etot: Vec<f32>,
    mocc: Vec<f32>,
    trace: Trace,
    region_pmx: crate::simd::probmx::ProbMx,
    slots: [envelope::Slot; 2],
    shared: envelope::Shared,
}

struct Plan {
    jobs: Vec<Job>,
    n2sc: Vec<f32>,
    nexpected: f32,
    stats: DomainDefinitionStats,
    domains: Vec<Domain>,
}

fn null2_done(job: Job, settings: Settings) -> bool {
    job.null2_is_done || !settings.do_null2
}

// Every target's regions are clustered before any envelope is rescored, so the envelopes of a
// target pair can share their Forward and Backward passes.
pub(crate) fn define<const N: usize>(
    model: &Model,
    targets: [&Target; N],
    settings: Settings,
    scratch: &mut Scratch,
) -> [Definition; N] {
    crate::logsum::p7_flogsuminit();
    let mut plans = targets.map(|t| plan(model, t, settings, scratch));
    rescore(model, &targets, &mut plans, settings, scratch);
    plans.map(|mut plan| {
        plan.domains.sort_by_key(|d| d.ienv);
        Definition {
            domains: plan.domains,
            nexpected: plan.nexpected,
            seq_bias: crate::util::vectorops::f_sum(&plan.n2sc),
            stats: plan.stats,
        }
    })
}

fn plan(model: &Model, t: &Target, settings: Settings, scratch: &mut Scratch) -> Plan {
    let l = t.l;
    decode(model, t, scratch);
    let nexpected = scratch.btot[l];
    let regions = find_domain_regions(&scratch.btot, &scratch.etot, &scratch.mocc, l);
    let multi: Vec<bool> = regions
        .iter()
        .map(|&(ri, rj)| is_multidomain_region(&scratch.btot, &scratch.etot, ri, rj))
        .collect();
    let mut stats = DomainDefinitionStats {
        nregions: regions.len(),
        ..Default::default()
    };
    let mut n2sc = vec![0.0_f32; l + 1];
    let mut jobs = Vec::new();
    let abc = Alphabet::shared(model.hmm.abc_type);
    let mut region_model = None;
    for (&region, multi) in regions.iter().zip(multi) {
        if !multi {
            stats.nenvelopes += 1;
            jobs.push(Job {
                ienv: region.0,
                jenv: region.1,
                null2_is_done: false,
            });
            continue;
        }
        stats.nclustered += 1;
        let region_model = region_model.get_or_insert_with(|| multihit(model, l));
        let mut sampler = StripedSampler::new(t, region, region_model, &mut scratch.region_pmx);
        cluster_region(
            t.dsq,
            l,
            region,
            abc,
            settings.seed,
            &mut sampler,
            &mut scratch.trace,
            &mut n2sc,
            &mut stats,
            &mut jobs,
        );
    }
    Plan {
        jobs,
        n2sc,
        nexpected,
        stats,
        domains: Vec::new(),
    }
}

fn decode(_model: &Model, t: &Target, scratch: &mut Scratch) {
    use crate::simd::oprofile::{P7O_C, P7O_J, P7O_LOOP, P7O_N};
    let njc_loop = [
        t.xf[P7O_N][P7O_LOOP],
        t.xf[P7O_J][P7O_LOOP],
        t.xf[P7O_C][P7O_LOOP],
    ];
    crate::simd::probmx::p_domain_decoding_reuse(
        t.fwd,
        t.bck,
        t.l,
        njc_loop,
        &mut scratch.btot,
        &mut scratch.etot,
        &mut scratch.mocc,
    );
}

fn multihit(model: &Model, l: usize) -> crate::simd::oprofile::OProfile {
    let mut om = model.om.clone();
    om.reconfig_multihit(l as i32);
    om
}

struct StripedSampler<'a> {
    om: &'a crate::simd::oprofile::OProfile,
    pmx: &'a crate::simd::probmx::ProbMx,
    len: usize,
}

impl<'a> StripedSampler<'a> {
    fn new(
        t: &Target,
        (ri, rj): (usize, usize),
        om: &'a crate::simd::oprofile::OProfile,
        pmx: &'a mut crate::simd::probmx::ProbMx,
    ) -> Self {
        let len = rj - ri + 1;
        pmx.resize_full(om.m, len);
        unsafe { crate::simd::fwd_filter::forward_parser_pmx_offset(t.dsq, ri - 1, len, om, pmx) };
        StripedSampler { om, pmx, len }
    }
}

impl regions::Sampler for StripedSampler<'_> {
    fn sample(&mut self, rng: &mut crate::util::random::MersenneTwister, tr: &mut Trace) {
        crate::dp::generic_stotrace::stochastic_trace_pmx_into(
            rng, self.len, self.om, self.pmx, tr,
        );
    }

    fn null2(&self, tr: &Trace, zstart: usize, zend: usize) -> Vec<f32> {
        null2::null2_by_trace_striped(self.om, tr, zstart, zend)
    }
}

fn rescore<const N: usize>(
    model: &Model,
    targets: &[&Target; N],
    plans: &mut [Plan; N],
    settings: Settings,
    scratch: &mut Scratch,
) {
    use envelope::{fill, needs_backward, Lane};
    let env_oms = targets.map(|t| {
        let mut om = model.om.clone();
        om.reconfig_unihit(t.l as i32);
        om
    });
    let queue: Vec<(usize, Job)> = plans
        .iter()
        .enumerate()
        .flat_map(|(t, plan)| plan.jobs.iter().map(move |&job| (t, job)))
        .collect();
    if settings.long_target {
        for (t, job) in queue {
            let plan = &mut plans[t];
            let domain = long_target::score(
                targets[t],
                model,
                settings,
                job,
                &env_oms[t],
                &mut scratch.slots,
                &mut plan.n2sc,
                &mut scratch.shared,
            );
            plan.domains.push(domain);
        }
        return;
    }
    let lane = |&(t, job): &(usize, Job)| Lane {
        dsq: targets[t].dsq,
        xf: env_oms[t].xf,
        job,
        backward: needs_backward(job, settings),
    };
    for chunk in queue.chunks(2) {
        match chunk {
            [a] => fill(model.om, &[lane(a)], &mut scratch.slots),
            [a, b] => fill(model.om, &[lane(a), lane(b)], &mut scratch.slots),
            _ => unreachable!(),
        }
        for (slot, &(t, job)) in scratch.slots.iter().zip(chunk) {
            let plan = &mut plans[t];
            let r = envelope::rescore(
                targets[t],
                model,
                settings,
                job,
                &env_oms[t],
                &env_oms[t],
                None,
                slot,
                &mut plan.n2sc,
                &mut scratch.shared,
            );
            let domain = score::hmmsearch_domain(targets[t], model, job, slot.fwd_sc, r);
            plan.domains.push(domain);
        }
    }
}
