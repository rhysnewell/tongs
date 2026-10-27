use crate::alphabet::{Alphabet, Dsq, DSQ_SENTINEL};
use crate::bg::Bg;
use crate::simd::bck_filter::backward_parser_pmx_offset;
use crate::simd::fwd_filter::forward_parser_pmx_offset;
use crate::simd::oprofile::{OProfile, P7O_C, P7O_J, P7O_LOOP, P7O_N};
use crate::simd::optacc::{decode_and_fill, oa_trace_pmx};
use crate::simd::probmx::p_decoding_to_gmx;
use crate::tophits::{AliDisplay, Domain};
use crate::trace::alignment_display_with_pp_emission_odds;
use crate::util::cmath::ESL_CONST_LOG2R;

use super::envelope::{self, Lane, Shared, Slot};
use super::regions::Job;
use super::score::{self, shifted};
use super::{Model, Settings, Target};

const MAX_ENV_EXTRA: i64 = 20;

pub(super) fn local_bg(
    dsq: &[Dsq],
    start: usize,
    end: usize,
    bg: &Bg,
    abc: &Alphabet,
    seq_len: usize,
) -> Bg {
    let k = abc.k;
    let mut counts = vec![0.0_f32; k];
    let mut total = 0.0_f32;

    for &x in dsq.iter().take(end + 1).skip(start) {
        let xu = x as usize;
        if xu < k {
            counts[xu] += 1.0;
            total += 1.0;
        } else if xu < abc.kp && abc.ndegen[xu] > 0 {
            let w = 1.0_f32 / abc.ndegen[xu] as f32;
            for (a, count) in counts.iter_mut().enumerate() {
                if abc.degen[xu][a] {
                    *count += w;
                }
            }
            total += 1.0;
        }
    }

    if total <= 0.0 {
        return bg.clone();
    }
    for c in &mut counts {
        *c /= total;
    }

    let bg_smooth = 25.0_f32 / seq_len.clamp(50, 100) as f32;
    let mut local = bg.clone();
    let mut sum = 0.0_f32;
    for (i, f) in local.f.iter_mut().enumerate().take(k) {
        *f = bg_smooth * bg.f[i] + (1.0 - bg_smooth) * counts[i];
        sum += *f;
    }
    if sum > 0.0 {
        for f in local.f.iter_mut().take(k) {
            *f /= sum;
        }
    }
    local
}

fn window_profile(
    t: &Target,
    model: &Model,
    base: &OProfile,
    (start, end): (usize, usize),
    reparam: bool,
) -> (OProfile, Option<Vec<f32>>) {
    let mut om = base.clone();
    om.reconfig_rest_length((end - start + 1) as i32);
    if !reparam {
        return (om, None);
    }
    let abc = Alphabet::shared(model.hmm.abc_type);
    let local = local_bg(t.dsq, start, end, model.bg, abc, t.l);
    let emissions = base.get_fwd_emissions(model.bg, abc);
    om.update_fwd_emission_scores(&local, &emissions, abc);
    (om, Some(local.f))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn score(
    t: &Target,
    model: &Model,
    settings: Settings,
    job: Job,
    base: &OProfile,
    slots: &mut [Slot; 2],
    n2sc: &mut [f32],
    shared: &mut Shared,
) -> Domain {
    let reparam = settings.do_null2;
    let (om, display_bg) = window_profile(t, model, base, (job.ienv, job.jenv), reparam);
    let lane = Lane {
        dsq: t.dsq,
        xf: om.xf,
        job,
        backward: envelope::needs_backward(job, settings),
    };
    envelope::fill(&om, &[lane], slots);
    let slot = &mut slots[0];
    let mut r = envelope::rescore(
        t,
        model,
        settings,
        job,
        &om,
        base,
        display_bg.as_deref(),
        slot,
        n2sc,
        shared,
    );
    let (mut iali, mut jali) = r.ali_bounds(job);
    let mut window = job;
    let mut env_fwd_sc = slot.fwd_sc;
    let extends = iali - job.ienv as i64 > MAX_ENV_EXTRA || job.jenv as i64 - jali > MAX_ENV_EXTRA;
    let trimmed_ienv = ((iali - MAX_ENV_EXTRA).max(1) as usize).max(job.ienv);
    let trimmed_jenv = ((jali + MAX_ENV_EXTRA).max(1) as usize).min(job.jenv);
    if extends && trimmed_jenv >= trimmed_ienv {
        window.ienv = trimmed_ienv;
        window.jenv = trimmed_jenv;
        let (trim_om, trim_bg) =
            window_profile(t, model, base, (trimmed_ienv, trimmed_jenv), reparam);
        env_fwd_sc = window_forward(t, model, &trim_om, window, slot, settings.make_alignment);
        if settings.make_alignment {
            if let Some((ad, oasc)) =
                realign(t, model, &trim_om, window, trim_bg.as_deref(), slot, shared)
            {
                (iali, jali) = (ad.sqfrom as i64, ad.sqto as i64);
                r.ad = Some(ad);
                r.oasc = oasc;
            }
        }
    }

    let (env_fwd_sc, dom_correction) = if reparam {
        let mut default_om = base.clone();
        default_om.reconfig_rest_length((window.jenv - window.ienv + 1) as i32);
        let default_sc = window_forward(t, model, &default_om, window, slot, false);
        let envsc = env_fwd_sc.min(default_sc);
        (envsc, default_sc - envsc)
    } else {
        (env_fwd_sc, 0.0)
    };
    r.dom_correction = dom_correction;

    let bitscore = score::window_bitscore(model, window, (iali, jali), env_fwd_sc);
    score::build(
        model,
        (iali, jali, window.ienv, window.jenv),
        bitscore,
        dom_correction * ESL_CONST_LOG2R as f32,
        env_fwd_sc,
        r,
    )
}

fn window_forward(
    t: &Target,
    model: &Model,
    om: &OProfile,
    window: Job,
    slot: &mut Slot,
    with_backward: bool,
) -> f32 {
    let len = window.jenv - window.ienv + 1;
    let offset = window.ienv - 1;
    slot.fwd.resize_full(model.gm.m, len);
    let sc = unsafe { forward_parser_pmx_offset(t.dsq, offset, len, om, &mut slot.fwd) };
    if with_backward {
        slot.bck.resize_full(model.gm.m, len);
        unsafe {
            backward_parser_pmx_offset(
                t.dsq,
                offset,
                len,
                om,
                Some(&slot.fwd.row_scale),
                &mut slot.bck,
            )
        };
    }
    sc
}

// C rebuilds the alignment display of a trimmed envelope, or its bounds drift by a residue.
fn realign(
    t: &Target,
    model: &Model,
    om: &OProfile,
    window: Job,
    display_bg: Option<&[f32]>,
    slot: &Slot,
    shared: &mut Shared,
) -> Option<(AliDisplay, f32)> {
    let len = window.jenv - window.ienv + 1;
    let oasc = unsafe {
        decode_and_fill(
            &slot.fwd,
            &slot.bck,
            om,
            &mut shared.pp_pmx,
            &mut shared.oa_pmx,
            &mut shared.pp_cells,
            None,
        )
    };
    let tr = unsafe { oa_trace_pmx(om, &shared.pp_pmx, &shared.oa_pmx) };
    shared.pp_gmx.grow_to_zeroed(model.gm.m, len);
    let njc_loop = [
        om.xf[P7O_N][P7O_LOOP],
        om.xf[P7O_J][P7O_LOOP],
        om.xf[P7O_C][P7O_LOOP],
    ];
    p_decoding_to_gmx(
        &slot.fwd,
        &slot.bck,
        model.gm.m,
        njc_loop,
        &mut shared.pp_gmx,
    );
    shared.sub_dsq.clear();
    shared.sub_dsq.push(DSQ_SENTINEL);
    shared
        .sub_dsq
        .extend_from_slice(&t.dsq[window.ienv..=window.jenv]);
    shared.sub_dsq.push(DSQ_SENTINEL);
    let odds = |k: usize, x: usize| om.fwd_emission_odds(k, x);
    alignment_display_with_pp_emission_odds(
        &tr,
        &shared.sub_dsq,
        model.hmm,
        Alphabet::shared(model.hmm.abc_type),
        Some(&shared.pp_gmx),
        display_bg,
        Some(&odds),
    )
    .map(|ad| (shifted(ad, window.ienv - 1), oasc))
}

#[cfg(test)]
#[path = "../../tests/unit/domaindef/long_target_test.rs"]
mod tests;
