use super::*;

#[derive(Clone, Copy)]
pub(super) struct ModelParams {
    pub(super) mu: f64,
    pub(super) lambda: f64,
}

pub(super) fn calibrated_model_params(
    score_mode: ScoreMode,
    hmm: &Hmm,
    abc: &Alphabet,
    bg: &Bg,
    args: &Args,
    rng: &mut Mt19937,
) -> ModelParams {
    let lambda = p7_lambda(hmm, bg) as f64;
    match score_mode {
        ScoreMode::Msv => ModelParams {
            mu: calibrate_msv_mu(hmm, abc, bg, args.l, args.em_l, args.em_n, lambda, rng),
            lambda,
        },
        ScoreMode::Viterbi => ModelParams {
            mu: calibrate_viterbi_mu(hmm, abc, bg, args.l, args.ev_l, args.ev_n, lambda, rng),
            lambda,
        },
        ScoreMode::Forward => ModelParams {
            mu: calibrate_forward_tau(
                hmm, abc, bg, args.l, args.ef_l, args.ef_n, args.eft, lambda, rng,
            ),
            lambda,
        },
        ScoreMode::Hybrid => ModelParams { mu: 0.0, lambda },
    }
}

fn calibrate_msv_mu(
    hmm: &Hmm,
    abc: &Alphabet,
    bg: &Bg,
    sim_l: usize,
    l: usize,
    n: usize,
    lambda: f64,
    rng: &mut Mt19937,
) -> f64 {
    let scores = simulate_optimized_calibration_scores(
        hmm,
        abc,
        bg,
        sim_l,
        l,
        n,
        rng,
        OptimizedCalibrationMode::Msv,
    );
    gumbel::fit_complete_loc(&scores, lambda).unwrap_or_else(|_| fallback_gumbel(&scores).0)
}

fn calibrate_viterbi_mu(
    hmm: &Hmm,
    abc: &Alphabet,
    bg: &Bg,
    sim_l: usize,
    l: usize,
    n: usize,
    lambda: f64,
    rng: &mut Mt19937,
) -> f64 {
    let scores = simulate_optimized_calibration_scores(
        hmm,
        abc,
        bg,
        sim_l,
        l,
        n,
        rng,
        OptimizedCalibrationMode::Viterbi,
    );
    gumbel::fit_complete_loc(&scores, lambda).unwrap_or_else(|_| fallback_gumbel(&scores).0)
}

fn calibrate_forward_tau(
    hmm: &Hmm,
    abc: &Alphabet,
    bg: &Bg,
    sim_l: usize,
    l: usize,
    n: usize,
    tailp: f64,
    lambda: f64,
    rng: &mut Mt19937,
) -> f64 {
    let scores = simulate_optimized_calibration_scores(
        hmm,
        abc,
        bg,
        sim_l,
        l,
        n,
        rng,
        OptimizedCalibrationMode::Forward,
    );
    let (gmu, glam) = gumbel::fit_complete(&scores).unwrap_or_else(|_| fallback_gumbel(&scores));
    let tau = gumbel::invcdf(1.0 - tailp, gmu, glam) + c_log_f64(tailp) / lambda;
    if tau.is_finite() {
        tau
    } else {
        fallback_exponential(&scores).0
    }
}

pub(super) enum OptimizedCalibrationMode {
    Msv,
    Viterbi,
    Forward,
}

fn simulate_optimized_calibration_scores(
    hmm: &Hmm,
    abc: &Alphabet,
    bg: &Bg,
    sim_l: usize,
    l: usize,
    n: usize,
    rng: &mut Mt19937,
    mode: OptimizedCalibrationMode,
) -> Vec<f64> {
    let mut bg = bg.clone();
    bg.set_length(sim_l);
    let mut gm = Profile::new(hmm.m, abc);
    profile::profile_config(hmm, &bg, &mut gm, sim_l as i32, P7_LOCAL);
    let mut om = OProfile::convert(&gm);
    om.reconfig_length(l as i32);
    bg.set_length(l);
    let msv_maxsc = (255.0 - om.base_b as f32) / om.scale_b;
    let vit_maxsc = (32767.0 - om.base_w as f32) / om.scale_w;
    let mut scores = Vec::with_capacity(n);

    for _ in 0..n {
        let mut dsq = Vec::with_capacity(l + 2);
        dsq.push(tongs::alphabet::DSQ_SENTINEL);
        for _ in 0..l {
            dsq.push(rng.sample_residue(&bg.f));
        }
        dsq.push(tongs::alphabet::DSQ_SENTINEL);

        let sc = match mode {
            OptimizedCalibrationMode::Msv => optimized_msv_filter_score(&dsq, l, &om, msv_maxsc),
            OptimizedCalibrationMode::Viterbi => {
                optimized_viterbi_filter_score(&dsq, l, &om, vit_maxsc)
            }
            OptimizedCalibrationMode::Forward => optimized_forward_parser_score(&dsq, l, &om),
        };
        let null_sc = bg.null_one(l);
        let bits = (sc as f64 - null_sc as f64) / ESL_CONST_LOG2;
        if bits.is_finite() {
            scores.push(bits);
        }
    }

    if scores.is_empty() {
        scores.push(0.0);
    }
    scores
}

fn optimized_msv_filter_score(dsq: &[u8], l: usize, om: &OProfile, maxsc: f32) -> f32 {
    match unsafe { tongs::simd::msv_filter::p7_msv_filter(dsq, l, om) } {
        tongs::simd::msv_filter::MsvResult::Ok(s) => s,
        tongs::simd::msv_filter::MsvResult::Overflow => maxsc,
    }
}

fn optimized_viterbi_filter_score(dsq: &[u8], l: usize, om: &OProfile, maxsc: f32) -> f32 {
    match unsafe { tongs::simd::vit_filter::p7_viterbi_filter(dsq, l, om) } {
        tongs::simd::vit_filter::VitResult::Ok(s) => s,
        tongs::simd::vit_filter::VitResult::Overflow => maxsc,
    }
}

fn optimized_forward_parser_score(dsq: &[u8], l: usize, om: &OProfile) -> f32 {
    unsafe { tongs::simd::fwd_filter::forward_parser(dsq, l, om) }
}
