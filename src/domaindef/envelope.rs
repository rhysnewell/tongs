use crate::alphabet::{Alphabet, Dsq, DSQ_SENTINEL};
use crate::dp::generic_optacc::{g_oa_trace, g_optimal_accuracy_with_deltas, OptAccTDelta};
use crate::dp::gmx::Gmx;
use crate::profile::reconfig_unihit;
use crate::simd::bck_filter::{backward, backward_pair, Backward};
use crate::simd::fwd_filter::{forward, forward_pair, Forward};
use crate::simd::oprofile::{OProfile, P7O_C, P7O_J, P7O_LOOP, P7O_N, P7O_NXSTATES, P7O_NXTRANS};
use crate::simd::optacc::{decode_and_fill, oa_trace_coords_pmx, oa_trace_pmx, Expectation};
use crate::simd::probmx::{p_decoding_to_gmx, p_null2_odds_from_omx_expectation_reuse, ProbMx};
use crate::simd::sse::__m128;
use crate::trace::{alignment_coords, alignment_display_with_pp_emission_odds};

use super::null2::add_null2_correction;
use super::regions::Job;
use super::score::{from_coords, shifted, Rescored};
use super::{null2_done, Model, Settings, Target};

#[derive(Default)]
pub(super) struct Slot {
    pub fwd: ProbMx,
    pub bck: ProbMx,
    pub fwd_sc: f32,
}

pub(super) struct Shared {
    pub pp_pmx: ProbMx,
    pub oa_pmx: ProbMx,
    pub pp_cells: Vec<__m128>,
    expected: Expectation,
    null2: Vec<f32>,
    exp_m: Vec<f32>,
    exp_i: Vec<f32>,
    pub pp_gmx: Gmx,
    oa_gmx: Gmx,
    pub sub_dsq: Vec<Dsq>,
}

impl Default for Shared {
    fn default() -> Self {
        Shared {
            pp_pmx: ProbMx::default(),
            oa_pmx: ProbMx::default(),
            pp_cells: Vec::new(),
            expected: Expectation::default(),
            null2: Vec::new(),
            exp_m: Vec::new(),
            exp_i: Vec::new(),
            pp_gmx: Gmx::new(0, 0),
            oa_gmx: Gmx::new(0, 0),
            sub_dsq: Vec::new(),
        }
    }
}

pub(super) struct Lane<'a> {
    pub dsq: &'a [Dsq],
    pub xf: [[f32; P7O_NXTRANS]; P7O_NXSTATES],
    pub job: Job,
    pub backward: bool,
}

pub(super) fn needs_backward(job: Job, settings: Settings) -> bool {
    settings.make_alignment || !null2_done(job, settings)
}

// `om` lends only its tables. Each lane brings its own length model.
pub(super) fn fill(om: &OProfile, lanes: &[Lane], slots: &mut [Slot; 2]) {
    let [s0, s1] = slots;
    for (lane, slot) in lanes.iter().zip([&mut *s0, &mut *s1]) {
        let len = lane.job.jenv - lane.job.ienv + 1;
        if lane.backward {
            slot.fwd.resize_full(om.m, len);
            slot.bck.resize_full(om.m, len);
        } else {
            slot.fwd.resize_parser(om.m, len);
        }
    }
    let forward_lane = |lane: &Lane<'_>, fwd: &mut ProbMx| {
        let job = lane.job;
        let len = job.jenv - job.ienv + 1;
        unsafe {
            forward(
                om,
                &mut Forward::new(lane.dsq, job.ienv - 1, len, lane.xf, fwd),
            )
        }
    };
    match lanes {
        [a] => {
            s0.fwd_sc = forward_lane(a, &mut s0.fwd);
            if a.backward {
                unsafe { backward(om, &mut backward_lane(a, s0)) };
            }
        }
        [a, b] => {
            let (a_len, b_len) = (a.job.jenv - a.job.ienv + 1, b.job.jenv - b.job.ienv + 1);
            let mut fa = Forward::new(a.dsq, a.job.ienv - 1, a_len, a.xf, &mut s0.fwd);
            let mut fb = Forward::new(b.dsq, b.job.ienv - 1, b_len, b.xf, &mut s1.fwd);
            (s0.fwd_sc, s1.fwd_sc) = unsafe { forward_pair(om, &mut fa, &mut fb) };
            match (a.backward, b.backward) {
                (true, true) => unsafe {
                    backward_pair(om, &mut backward_lane(a, s0), &mut backward_lane(b, s1))
                },
                (true, false) => unsafe {
                    backward(om, &mut backward_lane(a, s0));
                },
                (false, true) => unsafe {
                    backward(om, &mut backward_lane(b, s1));
                },
                (false, false) => {}
            }
        }
        _ => unreachable!("an envelope pass holds one or two lanes"),
    }
}

fn backward_lane<'s>(lane: &Lane<'s>, slot: &'s mut Slot) -> Backward<'s> {
    let len = lane.job.jenv - lane.job.ienv + 1;
    Backward::new(
        lane.dsq,
        lane.job.ienv - 1,
        len,
        lane.xf,
        Some(&slot.fwd.row_scale),
        &mut slot.bck,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rescore(
    t: &Target,
    model: &Model,
    settings: Settings,
    job: Job,
    om: &OProfile,
    odds_om: &OProfile,
    display_bg: Option<&[f32]>,
    slot: &Slot,
    n2sc: &mut [f32],
    shared: &mut Shared,
) -> Rescored {
    let Job { ienv, jenv, .. } = job;
    let env_len = jenv - ienv + 1;
    let m = model.gm.m;
    let display = settings.make_alignment_display;
    let null2_done = null2_done(job, settings);
    let abc = Alphabet::shared(model.hmm.abc_type);
    let njc_loop = [
        om.xf[P7O_N][P7O_LOOP],
        om.xf[P7O_J][P7O_LOOP],
        om.xf[P7O_C][P7O_LOOP],
    ];
    let odds = |k: usize, x: usize| odds_om.fwd_emission_odds(k, x);
    if display {
        shared.sub_dsq.clear();
        shared.sub_dsq.push(DSQ_SENTINEL);
        shared.sub_dsq.extend_from_slice(&t.dsq[ienv..=jenv]);
        shared.sub_dsq.push(DSQ_SENTINEL);
    }

    let mut ad = None;
    let mut oasc = 0.0_f32;
    if settings.make_alignment {
        if display {
            shared.pp_gmx.grow_to_zeroed(m, env_len);
        }
        let expected = (!null2_done && !display).then_some(&mut shared.expected);
        let sc = unsafe {
            decode_and_fill(
                &slot.fwd,
                &slot.bck,
                om,
                &mut shared.pp_pmx,
                &mut shared.oa_pmx,
                &mut shared.pp_cells,
                expected,
            )
        };
        if display {
            let tr = unsafe { oa_trace_pmx(om, &shared.pp_pmx, &shared.oa_pmx) };
            p_decoding_to_gmx(&slot.fwd, &slot.bck, m, njc_loop, &mut shared.pp_gmx);
            oasc = sc;
            ad = alignment_display_with_pp_emission_odds(
                &tr,
                &shared.sub_dsq,
                model.hmm,
                abc,
                Some(&shared.pp_gmx),
                display_bg,
                Some(&odds),
            )
            .map(|ad| shifted(ad, ienv - 1));
        } else if let Some(coords) =
            unsafe { oa_trace_coords_pmx(om, &shared.pp_pmx, &shared.oa_pmx) }
        {
            oasc = sc;
            ad = Some(from_coords(coords, ienv - 1));
        }
    }

    let dom_correction = if null2_done {
        n2sc[ienv..=jenv].iter().fold(0.0_f32, |sum, &sc| sum + sc)
    } else {
        if settings.make_alignment && !display {
            unsafe {
                shared
                    .expected
                    .null2(env_len, model.gm.abc_k, &om.rfv, &mut shared.null2)
            };
        } else {
            p_null2_odds_from_omx_expectation_reuse(
                &slot.fwd,
                &slot.bck,
                model.gm.abc_k,
                &om.rfv,
                njc_loop,
                &mut shared.null2,
                &mut shared.exp_m,
                &mut shared.exp_i,
            );
        }
        add_null2_correction(&shared.null2, abc, t.dsq, ienv, jenv, n2sc)
    };

    if settings.make_alignment && ad.is_none() {
        if !display {
            shared.pp_gmx.grow_to_zeroed(m, env_len);
        }
        let mut env_gm = model.gm.clone();
        reconfig_unihit(&mut env_gm, t.l as i32);
        let deltas = OptAccTDelta::from_profile(&env_gm);
        shared.oa_gmx.grow_to_zeroed(m, env_len);
        oasc = g_optimal_accuracy_with_deltas(&env_gm, &shared.pp_gmx, &mut shared.oa_gmx, &deltas);
        let tr = g_oa_trace(&env_gm, &shared.pp_gmx, &shared.oa_gmx);
        ad = if display {
            alignment_display_with_pp_emission_odds(
                &tr,
                &shared.sub_dsq,
                model.hmm,
                abc,
                Some(&shared.pp_gmx),
                None,
                Some(&odds),
            )
            .map(|ad| shifted(ad, ienv - 1))
        } else {
            alignment_coords(&tr).map(|coords| from_coords(coords, ienv - 1))
        };
    }

    Rescored {
        ad,
        oasc,
        dom_correction,
    }
}
