use crate::tophits::{AliDisplay, Domain};
use crate::util::cmath::{c_log_f64, ESL_CONST_LOG2, ESL_CONST_LOG2R};

use super::regions::Job;
use super::{Model, Target};

pub(super) struct Rescored {
    pub ad: Option<AliDisplay>,
    pub oasc: f32,
    pub dom_correction: f32,
}

impl Rescored {
    pub fn ali_bounds(&self, job: Job) -> (i64, i64) {
        self.ad
            .as_ref()
            .map_or((job.ienv as i64, job.jenv as i64), |ad| {
                (ad.sqfrom as i64, ad.sqto as i64)
            })
    }
}

pub(super) fn shifted(mut ad: AliDisplay, offset: usize) -> AliDisplay {
    ad.sqfrom += offset;
    ad.sqto += offset;
    ad
}

pub(super) fn from_coords(
    (hmmfrom, hmmto, sqfrom, sqto): (usize, usize, usize, usize),
    offset: usize,
) -> AliDisplay {
    let shift = |sq: usize| if sq > 0 { sq + offset } else { sq };
    AliDisplay {
        hmmfrom,
        hmmto,
        sqfrom: shift(sqfrom),
        sqto: shift(sqto),
        ..Default::default()
    }
}

pub(super) fn hmmsearch_domain(
    t: &Target,
    model: &Model,
    job: Job,
    env_fwd_sc: f32,
    r: Rescored,
) -> Domain {
    let Job { ienv, jenv, .. } = job;
    let env_len = jenv - ienv + 1;
    let l = t.l;
    let (iali, jali) = r.ali_bounds(job);
    let dom_bias = crate::logsum::p7_flogsum(
        0.0,
        c_log_f64(model.bg.omega as f64) as f32 + r.dom_correction,
    );
    let ratio = (l as f32) / ((l + 3) as f32);
    let length_correction = (l - env_len) as f64 * c_log_f64(ratio as f64);
    let bitscore = (env_fwd_sc as f64 + length_correction) as f32;
    let bitscore = ((bitscore - (t.null_sc + dom_bias)) as f64 / ESL_CONST_LOG2) as f32;
    build(
        model,
        (iali, jali, ienv, jenv),
        bitscore,
        dom_bias * ESL_CONST_LOG2R as f32,
        env_fwd_sc,
        r,
    )
}

pub(super) fn build(
    model: &Model,
    (iali, jali, ienv, jenv): (i64, i64, usize, usize),
    bitscore: f32,
    dombias: f32,
    envsc: f32,
    r: Rescored,
) -> Domain {
    let tau = model.gm.evparam[crate::hmm::P7_FTAU] as f64;
    let lambda = model.gm.evparam[crate::hmm::P7_FLAMBDA] as f64;
    Domain {
        iali,
        jali,
        ienv: ienv as i64,
        jenv: jenv as i64,
        bitscore,
        lnp: crate::stats::exponential::logsurv(bitscore as f64, tau, lambda),
        dombias,
        oasc: r.oasc,
        envsc,
        domcorrection: r.dom_correction,
        is_reported: false,
        is_included: false,
        ad: r.ad,
    }
}

// C keeps these ratios in single precision where it casts to float.
pub(super) fn window_bitscore(
    model: &Model,
    window: Job,
    (iali, jali): (i64, i64),
    env_fwd_sc: f32,
) -> f32 {
    let env_len = (window.jenv - window.ienv + 1) as i32;
    let ali_len = (jali - iali + 1) as i32;
    let max_length = if model.hmm.max_length > 0 {
        model.hmm.max_length
    } else {
        env_len
    };
    let mut bitscore = env_fwd_sc as f64;
    bitscore -= 2.0 * c_log_f64(2.0_f64 / (env_len + 2) as f64);
    bitscore += 2.0 * c_log_f64(2.0_f64 / (max_length + 2) as f64);
    let env_ratio = (env_len as f32) / ((env_len + 2) as f32);
    bitscore -= (env_len - ali_len) as f64 * c_log_f64(env_ratio as f64);
    let max_len_or_env = max_length.max(env_len);
    let max_ratio = (max_length as f32) / ((max_length + 2) as f32);
    bitscore += (max_len_or_env - ali_len) as f64 * c_log_f64(max_ratio as f64);
    let nsc_length = max_len_or_env as usize;
    let mut bg = model.bg.clone();
    bg.set_length(nsc_length);
    let nullsc = bg.null_one(nsc_length) as f64;
    ((bitscore - nullsc) / ESL_CONST_LOG2) as f32
}
