use super::*;

#[derive(Clone, Copy)]
pub(super) struct ScoreFit {
    pub(super) kind: FitKind,
    pub(super) model_mu: f64,
    pub(super) model_lambda: f64,
}

#[derive(Clone, Copy)]
pub(super) enum FitKind {
    Gumbel,
    Exponential,
}

impl ScoreFit {
    pub(super) fn fit(score_mode: ScoreMode, scores: &[f64], model_params: ModelParams) -> Self {
        let kind = match score_mode {
            ScoreMode::Forward | ScoreMode::Hybrid => FitKind::Exponential,
            ScoreMode::Viterbi | ScoreMode::Msv => FitKind::Gumbel,
        };
        let (model_mu, model_lambda) = (model_params.mu, model_params.lambda);
        let (fit_mu, fit_lambda) = match kind {
            FitKind::Gumbel => {
                gumbel::fit_complete(scores).unwrap_or_else(|_| fallback_gumbel(scores))
            }
            FitKind::Exponential => fit_exponential_tail(scores, 0.02),
        };
        let (model_mu, model_lambda) =
            sanitize_model_params(model_mu, model_lambda, fit_mu, fit_lambda);

        Self {
            kind,
            model_mu,
            model_lambda,
        }
    }

    pub(super) fn model_survival(&self, score: f64) -> f64 {
        match self.kind {
            FitKind::Gumbel => gumbel::surv(score, self.model_mu, self.model_lambda),
            FitKind::Exponential => exponential::surv(score, self.model_mu, self.model_lambda),
        }
    }
}

fn fit_exponential_tail(scores: &[f64], tail_mass: f64) -> (f64, f64) {
    let mut ranked = ranked_scores(scores);
    let tail_n = ((ranked.len() as f64 * tail_mass).ceil() as usize)
        .max(2)
        .min(ranked.len());
    ranked.truncate(tail_n);
    let (mu, lambda) = exponential::fit_complete(&ranked);
    if lambda.is_finite() && lambda > 0.0 {
        (mu, lambda)
    } else {
        fallback_exponential(scores)
    }
}

pub(super) fn fallback_gumbel(scores: &[f64]) -> (f64, f64) {
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let var = sample_variance(scores, mean).max(1e-6);
    let lambda = std::f64::consts::PI / c_sqrt_f64(6.0 * var);
    let euler_gamma = 0.5772156649015329_f64;
    (mean - euler_gamma / lambda, lambda)
}

pub(super) fn fallback_exponential(scores: &[f64]) -> (f64, f64) {
    let mu = scores.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let lambda = 1.0 / (mean - mu).max(1e-6);
    (mu, lambda)
}

fn sample_variance(scores: &[f64], mean: f64) -> f64 {
    if scores.len() < 2 {
        return 0.0;
    }
    scores
        .iter()
        .map(|score| {
            let delta = score - mean;
            delta * delta
        })
        .sum::<f64>()
        / (scores.len() - 1) as f64
}

fn sanitize_model_params(mu: f64, lambda: f64, fit_mu: f64, fit_lambda: f64) -> (f64, f64) {
    if mu.is_finite() && lambda.is_finite() && lambda > 0.0 {
        (mu, lambda)
    } else {
        (fit_mu, fit_lambda)
    }
}

fn ranked_scores(scores: &[f64]) -> Vec<f64> {
    let mut ranked = scores.to_vec();
    ranked.sort_by(|a, b| b.total_cmp(a));
    ranked
}

pub(super) fn elide_length_model(gm: &mut Profile, bg: &mut Bg) {
    bg.p1 = 350.0 / 351.0;
    let loop_sc = c_logf_to_f32(bg.p1);
    let move_sc = c_logf_to_f32(1.0 - bg.p1);
    gm.xsc[P7P_N][P7P_LOOP] = loop_sc;
    gm.xsc[P7P_C][P7P_LOOP] = loop_sc;
    gm.xsc[P7P_J][P7P_LOOP] = loop_sc;
    gm.xsc[P7P_N][P7P_MOVE] = move_sc;
    gm.xsc[P7P_C][P7P_MOVE] = move_sc;
    gm.xsc[P7P_J][P7P_MOVE] = move_sc;
}
