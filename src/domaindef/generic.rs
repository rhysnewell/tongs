use crate::alphabet::{Alphabet, Dsq, DSQ_SENTINEL};
use crate::dp::generic_backward::g_backward;
use crate::dp::generic_decoding::{domain_decoding, g_decoding};
use crate::dp::generic_fwdback::g_forward;
use crate::dp::generic_null2::{null2_by_expectation, null2_by_trace};
use crate::dp::generic_optacc::{g_oa_trace, g_optimal_accuracy_with_deltas, OptAccTDelta};
use crate::dp::generic_stotrace::g_stochastic_trace;
use crate::dp::gmx::Gmx;
use crate::profile::{reconfig_unihit, Profile};
use crate::tophits::Domain;
use crate::trace::{alignment_coords, alignment_display_with_pp_emission_odds, Trace};
use crate::util::cmath::ESL_CONST_LOG2R;
use crate::util::random::MersenneTwister;

use super::null2::add_null2_correction;
use super::regions::Job;
use super::score::{self, from_coords, shifted, Rescored};
use super::{null2_done, Model, Settings, Target};

pub(super) struct Scratch {
    fwd: Gmx,
    bck: Gmx,
    pp: Gmx,
    oa: Gmx,
}

impl Default for Scratch {
    fn default() -> Self {
        Scratch {
            fwd: Gmx::new(0, 0),
            bck: Gmx::new(0, 0),
            pp: Gmx::new(0, 0),
            oa: Gmx::new(0, 0),
        }
    }
}

fn window(dsq: &[Dsq], from: usize, to: usize) -> Vec<Dsq> {
    let mut sub = Vec::with_capacity(to - from + 3);
    sub.push(DSQ_SENTINEL);
    sub.extend_from_slice(&dsq[from..=to]);
    sub.push(DSQ_SENTINEL);
    sub
}

impl Scratch {
    pub fn decode(&mut self, gm: &Profile, t: &Target) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        self.fwd.grow_to(gm.m, t.l);
        g_forward(t.dsq, t.l, gm, &mut self.fwd);
        self.bck.grow_to(gm.m, t.l);
        g_backward(t.dsq, t.l, gm, &mut self.bck);
        domain_decoding(gm, &self.fwd, &self.bck)
    }

    pub fn score(
        &mut self,
        model: &Model,
        t: &Target,
        settings: Settings,
        job: Job,
        n2sc: &mut [f32],
    ) -> Domain {
        let Job { ienv, jenv, .. } = job;
        let env_len = jenv - ienv + 1;
        let m = model.gm.m;
        let null2_done = null2_done(job, settings);
        let abc = Alphabet::shared(model.hmm.abc_type);
        let mut env_gm = model.gm.clone();
        reconfig_unihit(&mut env_gm, t.l as i32);
        let sub_dsq = window(t.dsq, ienv, jenv);

        self.fwd.grow_to(m, env_len);
        let env_fwd_sc = g_forward(&sub_dsq, env_len, &env_gm, &mut self.fwd);
        let wants_pp = settings.make_alignment || !null2_done;
        if wants_pp {
            self.pp.grow_to_zeroed(m, env_len);
            self.bck.grow_to(m, env_len);
            g_backward(&sub_dsq, env_len, &env_gm, &mut self.bck);
            g_decoding(&env_gm, &self.fwd, &self.bck, &mut self.pp);
        }

        let dom_correction = if null2_done {
            n2sc[ienv..=jenv].iter().fold(0.0_f32, |sum, &sc| sum + sc)
        } else {
            let null2 = null2_by_expectation(&env_gm, &self.pp, 1, env_len);
            add_null2_correction(&null2, abc, t.dsq, ienv, jenv, n2sc)
        };

        let mut oasc = 0.0_f32;
        let ad = if settings.make_alignment {
            let deltas = OptAccTDelta::from_profile(&env_gm);
            self.oa.grow_to_zeroed(m, env_len);
            oasc = g_optimal_accuracy_with_deltas(&env_gm, &self.pp, &mut self.oa, &deltas);
            let tr = g_oa_trace(&env_gm, &self.pp, &self.oa);
            if settings.make_alignment_display {
                let no_odds = |_: usize, _: usize| None;
                alignment_display_with_pp_emission_odds(
                    &tr,
                    &sub_dsq,
                    model.hmm,
                    abc,
                    Some(&self.pp),
                    None,
                    Some(&no_odds),
                )
                .map(|ad| shifted(ad, ienv - 1))
            } else {
                alignment_coords(&tr).map(|coords| from_coords(coords, ienv - 1))
            }
        } else {
            None
        };

        let r = Rescored {
            ad,
            oasc,
            dom_correction,
        };
        if !settings.long_target {
            return score::hmmsearch_domain(t, model, job, env_fwd_sc, r);
        }
        let (iali, jali) = r.ali_bounds(job);
        let bitscore = score::window_bitscore(model, job, (iali, jali), env_fwd_sc);
        score::build(
            model,
            (iali, jali, ienv, jenv),
            bitscore,
            dom_correction * ESL_CONST_LOG2R as f32,
            env_fwd_sc,
            r,
        )
    }
}

pub(super) struct Sampler<'a> {
    gm: &'a Profile,
    sub_dsq: Vec<Dsq>,
    fwd: Gmx,
    len: usize,
}

impl<'a> Sampler<'a> {
    pub fn new(t: &Target, (ri, rj): (usize, usize), gm: &'a Profile) -> Self {
        let len = rj - ri + 1;
        let sub_dsq = window(t.dsq, ri, rj);
        let mut fwd = Gmx::new(gm.m, len);
        g_forward(&sub_dsq, len, gm, &mut fwd);
        Sampler {
            gm,
            sub_dsq,
            fwd,
            len,
        }
    }
}

impl super::regions::Sampler for Sampler<'_> {
    fn sample(&mut self, rng: &mut MersenneTwister, tr: &mut Trace) {
        *tr = g_stochastic_trace(rng, &self.sub_dsq, self.len, self.gm, &self.fwd);
    }

    fn null2(&self, tr: &Trace, zstart: usize, zend: usize) -> Vec<f32> {
        null2_by_trace(self.gm, tr, zstart, zend)
    }
}
