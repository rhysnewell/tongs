use super::*;

pub(super) fn write_optional_outputs(
    args: &Args,
    hmm_name: &str,
    model_params: ModelParams,
    score_mode: ScoreMode,
    scores: &[f64],
    alilens: &[usize],
) -> std::io::Result<()> {
    let fit = ScoreFit::fit(score_mode, scores, model_params);
    let histogram = ScoreHistogram::from_scores(scores, -50.0, 50.0, 0.2);

    if let Some(path) = &args.xfile {
        let mut file = std::fs::File::create(path)?;
        for &score in scores {
            file.write_all(&score.to_ne_bytes())?;
        }
    }

    if let Some(path) = &args.afile {
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "# {}", hmm_name)?;
        writeln!(file, "# alilen bitscore")?;
        for (&alilen, &score) in alilens.iter().zip(scores.iter()) {
            writeln!(file, "{} {}", alilen, fmt_fixed4(score))?;
        }
    }

    if let Some(path) = &args.ffile {
        let mut file = std::fs::File::create(path)?;
        if !matches!(score_mode, ScoreMode::Forward) {
            let mut npass = 0usize;
            for &score in scores {
                if fit.model_survival(score) <= args.pthresh {
                    npass += 1;
                }
            }
            writeln!(
                file,
                "{}\t{}\t{}",
                hmm_name,
                npass,
                fmt_fixed4(npass as f64 / scores.len() as f64)
            )?;
        }
    }

    if let Some(path) = &args.efile {
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "# {}", hmm_name)?;
        for rank in 1..=scores.len().min(1000) {
            let score = histogram.rank(rank);
            writeln!(
                file,
                "{} {}",
                rank,
                fmt_g(scores.len() as f64 * fit.model_survival(score))
            )?;
        }
        writeln!(file, "&")?;
    }

    if let Some(path) = &args.pfile {
        let mut file = std::fs::File::create(path)?;
        histogram.write_survival_plot(&mut file)?;
        match score_mode {
            ScoreMode::Forward => {
                let fits = forward_tail_fits(args, &histogram);
                let (mu, lambda) = fits
                    .last()
                    .map(|fit| (fit.mu, fit.lambda))
                    .unwrap_or_else(|| fallback_exponential(scores));
                write_exponential_plot(&mut file, mu, lambda, mu, histogram.xmax + 5.0, 0.1)?;
                write_exponential_plot(&mut file, mu, 0.693147, mu, histogram.xmax + 5.0, 0.1)?;
            }
            ScoreMode::Viterbi | ScoreMode::Hybrid | ScoreMode::Msv => {
                let (mu, lambda) =
                    gumbel::fit_complete(scores).unwrap_or_else(|_| fallback_gumbel(scores));
                let mufix = gumbel::fit_complete_loc(scores, 0.693147).unwrap_or(mu);
                write_gumbel_plot(
                    &mut file,
                    mu,
                    lambda,
                    histogram.xmin - 5.0,
                    histogram.xmax + 5.0,
                    0.1,
                )?;
                write_gumbel_plot(
                    &mut file,
                    mufix,
                    0.693147,
                    histogram.xmin - 5.0,
                    histogram.xmax + 5.0,
                    0.1,
                )?;
            }
        }
    }

    Ok(())
}

#[derive(Clone)]
pub(super) struct ScoreHistogram {
    pub(super) obs: Vec<u64>,
    pub(super) w: f64,
    pub(super) bmin: f64,
    pub(super) bmax: f64,
    pub(super) imin: usize,
    pub(super) imax: usize,
    pub(super) xmin: f64,
    pub(super) xmax: f64,
    pub(super) sorted: Vec<f64>,
}

impl ScoreHistogram {
    pub(super) fn from_scores(scores: &[f64], bmin: f64, bmax: f64, w: f64) -> Self {
        let nb = ((bmax - bmin) / w) as usize;
        let mut histogram = Self {
            obs: vec![0; nb],
            w,
            bmin,
            bmax,
            imin: nb,
            imax: 0,
            xmin: f64::INFINITY,
            xmax: f64::NEG_INFINITY,
            sorted: Vec::with_capacity(scores.len()),
        };
        for &score in scores {
            if score.is_finite() {
                histogram.add(score);
            }
        }
        histogram.sorted.sort_by(|a, b| a.total_cmp(b));
        histogram
    }

    fn add(&mut self, score: f64) {
        let mut bin = (((score - self.bmin) / self.w) - 1.0).ceil() as isize;
        if bin < 0 {
            let nnew = (-bin * 2) as usize;
            let mut obs = vec![0; nnew + self.obs.len()];
            obs[nnew..].copy_from_slice(&self.obs);
            self.obs = obs;
            self.bmin -= nnew as f64 * self.w;
            if !self.sorted.is_empty() {
                self.imin += nnew;
                self.imax += nnew;
            }
            bin += nnew as isize;
        } else if bin as usize >= self.obs.len() {
            let nnew = (bin as usize - self.obs.len() + 1) * 2;
            self.obs.resize(self.obs.len() + nnew, 0);
            self.bmax += nnew as f64 * self.w;
        }

        let bin = bin as usize;
        self.obs[bin] += 1;
        if self.sorted.is_empty() {
            self.imin = bin;
            self.imax = bin;
        } else {
            self.imin = self.imin.min(bin);
            self.imax = self.imax.max(bin);
        }
        self.xmin = self.xmin.min(score);
        self.xmax = self.xmax.max(score);
        self.sorted.push(score);
    }

    fn rank(&self, rank: usize) -> f64 {
        self.sorted[self.sorted.len() - rank]
    }

    fn tail_by_mass(&self, mass: f64) -> &[f64] {
        // Easel esl_histogram_GetTailByMass() truncates `n * pmass`, so the
        // returned tail mass is <= the requested mass.
        let n = ((self.sorted.len() as f64) * mass.min(1.0)) as usize;
        let n = n.min(self.sorted.len());
        &self.sorted[self.sorted.len() - n..]
    }

    fn bin_lbound(&self, bin: usize) -> f64 {
        self.w * bin as f64 + self.bmin
    }

    fn write_survival_plot(&self, out: &mut dyn Write) -> std::io::Result<()> {
        let mut cumulative = 0_u64;
        let n = self.sorted.len() as f64;
        if self.obs[self.imax] > 1 {
            writeln!(out, "{}\t{}", fmt_fixed6(self.xmax), fmt_g(1.0 / n))?;
        }
        for bin in (self.imin..=self.imax).rev() {
            if self.obs[bin] > 0 {
                cumulative += self.obs[bin];
                writeln!(
                    out,
                    "{}\t{}",
                    fmt_fixed6(self.bin_lbound(bin)),
                    fmt_g(cumulative as f64 / n)
                )?;
            }
        }
        writeln!(out, "&")
    }
}

#[derive(Clone, Copy)]
pub(super) struct ForwardTailFit {
    pub(super) tailp: f64,
    pub(super) mu: f64,
    pub(super) lambda: f64,
}

fn forward_tail_fits(args: &Args, histogram: &ScoreHistogram) -> Vec<ForwardTailFit> {
    let mut fits = Vec::new();
    let mut tailp = args.tmin;
    loop {
        if tailp > 1.0 {
            tailp = 1.0;
        }
        let tail = histogram.tail_by_mass(tailp);
        let fit = if tail.is_empty() {
            fallback_exponential(&histogram.sorted)
        } else {
            let (mu, lambda) = exponential::fit_complete(tail);
            if lambda.is_finite() && lambda > 0.0 {
                (mu, lambda)
            } else {
                fallback_exponential(&histogram.sorted)
            }
        };
        fits.push(ForwardTailFit {
            tailp,
            mu: fit.0,
            lambda: fit.1,
        });

        if args.tpoints == 1 {
            break;
        }
        // C (hmmsim.c:815) holds tpoints as a `double`, so `tpoints-1` is float
        // arithmetic; with --tpoints 0 this is -1.0, not an integer underflow.
        // Mirror that to match C and avoid a usize subtraction panic.
        let tpoints = args.tpoints as f64;
        let prev = tailp;
        if args.tlinear {
            tailp += (args.tmax - args.tmin) / (tpoints - 1.0);
        } else {
            tailp *= (c_log_f64(args.tmax / args.tmin) / (tpoints - 1.0)).exp();
        }
        // The sweep only makes sense for tpoints >= 2 with tmax > tmin, where the
        // step strictly advances tailp toward tmax. For degenerate inputs C accepts
        // but does not loop (e.g. --tpoints 0 with tmin==tmax leaves the step at a
        // no-op; C aborts on the empty fit). Guard against non-advancement so we
        // terminate instead of spinning forever, and cap iterations at tpoints.
        if !(tailp > prev) || tailp > args.tmax + 1e-7 || fits.len() >= args.tpoints {
            break;
        }
    }
    fits
}

/// Format a value the way C's `%8.4f` does: render the digits with C's
/// `%.4f` (via `fmt_fixed4`, which calls the C library), then right-justify
/// to a minimum field width of 8 with spaces. C printf never truncates when
/// the content exceeds the width, matching Rust's `{:>8}` on the string.
fn fmt_w8_4(val: f64) -> String {
    format!("{:>8}", fmt_fixed4(val))
}

pub(super) fn write_forward_tail_summary(
    args: &Args,
    out: &mut dyn Write,
    hmm_name: &str,
    model_params: ModelParams,
    histogram: &ScoreHistogram,
) -> std::io::Result<()> {
    let rank = histogram.sorted.len().min(10).max(1);
    let x10 = histogram.rank(rank);
    let fits = forward_tail_fits(args, histogram);
    for fit in fits {
        let e10 =
            histogram.sorted.len() as f64 * fit.tailp * exponential::surv(x10, fit.mu, fit.lambda);
        let e10fix =
            histogram.sorted.len() as f64 * fit.tailp * exponential::surv(x10, fit.mu, 0.693147);
        let e10p = histogram.sorted.len() as f64
            * exponential::surv(x10, model_params.mu, model_params.lambda);
        // C: fprintf("%-20s  %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n",
        //            name, tailp, mu, lambda, E10, mufix(=mu), E10fix, pmu, plambda, E10p)
        writeln!(
            out,
            "{:<20}  {} {} {} {} {} {} {} {} {}",
            hmm_name,
            fmt_w8_4(fit.tailp),
            fmt_w8_4(fit.mu),
            fmt_w8_4(fit.lambda),
            fmt_w8_4(e10),
            fmt_w8_4(fit.mu),
            fmt_w8_4(e10fix),
            fmt_w8_4(model_params.mu),
            fmt_w8_4(model_params.lambda),
            fmt_w8_4(e10p)
        )?;
    }
    Ok(())
}

pub(super) fn write_gumbel_summary(
    out: &mut dyn Write,
    hmm_name: &str,
    model_params: ModelParams,
    histogram: &ScoreHistogram,
    alilens: Option<&[usize]>,
) -> std::io::Result<()> {
    let rank = histogram.sorted.len().min(10).max(1);
    let x10 = histogram.rank(rank);

    // tailp field is always 1.0 for the Gumbel-fit modes.
    let tailp = 1.0;

    let (mu, lambda) = gumbel::fit_complete(&histogram.sorted)
        .unwrap_or_else(|_| fallback_gumbel(&histogram.sorted));
    let e10 = histogram.sorted.len() as f64 * gumbel::surv(x10, mu, lambda);

    let mufix = gumbel::fit_complete_loc(&histogram.sorted, 0.693147).unwrap_or(mu);
    let e10fix = histogram.sorted.len() as f64 * gumbel::surv(x10, mufix, 0.693147);

    let mufix2 =
        gumbel::fit_complete_loc(&histogram.sorted, model_params.lambda).unwrap_or(model_params.mu);
    let e10fix2 = histogram.sorted.len() as f64 * gumbel::surv(x10, mufix2, model_params.lambda);

    let e10p =
        histogram.sorted.len() as f64 * gumbel::surv(x10, model_params.mu, model_params.lambda);

    // C: fprintf("%-20s  %8.4f x11", name, tailp, mu, lambda, E10, mufix, E10fix,
    //            mufix2, E10fix2, pmu, plambda, E10p)  -- no trailing newline here;
    // the newline (or the optional -a columns) is appended afterward.
    write!(
        out,
        "{:<20}  {} {} {} {} {} {} {} {} {} {} {}",
        hmm_name,
        fmt_w8_4(tailp),
        fmt_w8_4(mu),
        fmt_w8_4(lambda),
        fmt_w8_4(e10),
        fmt_w8_4(mufix),
        fmt_w8_4(e10fix),
        fmt_w8_4(mufix2),
        fmt_w8_4(e10fix2),
        fmt_w8_4(model_params.mu),
        fmt_w8_4(model_params.lambda),
        fmt_w8_4(e10p)
    )?;

    if let Some(alilens) = alilens {
        // C: esl_stats_IMean over the alignment lengths, then
        //    fprintf(" %8.4f %8.4f\n", almean, sqrt(alvar))
        let (almean, alvar) = i_mean(alilens);
        writeln!(out, " {} {}", fmt_w8_4(almean), fmt_w8_4(c_sqrt_f64(alvar)))?;
    } else {
        writeln!(out)?;
    }
    Ok(())
}

/// Reproduce Easel's `esl_stats_IMean`: sample mean and (n>1) sample variance
/// of an integer vector, computed in f64. The variance uses `fabs(...)` to
/// avoid a tiny negative result for zero variance, matching the C code.
fn i_mean(x: &[usize]) -> (f64, f64) {
    let n = x.len();
    let mut sum = 0.0_f64;
    let mut sqsum = 0.0_f64;
    for &xi in x {
        let xi = xi as f64;
        sum += xi;
        sqsum += xi * xi;
    }
    let mean = sum / n as f64;
    let var = if n > 1 {
        ((sqsum - sum * sum / n as f64) / (n as f64 - 1.0)).abs()
    } else {
        0.0
    };
    (mean, var)
}

fn write_gumbel_plot(
    out: &mut dyn Write,
    mu: f64,
    lambda: f64,
    xmin: f64,
    xmax: f64,
    xstep: f64,
) -> std::io::Result<()> {
    let mut x = xmin;
    while x <= xmax {
        writeln!(
            out,
            "{}\t{}",
            fmt_fixed6(x),
            fmt_g(gumbel::surv(x, mu, lambda))
        )?;
        x += xstep;
    }
    writeln!(out, "&")
}

fn write_exponential_plot(
    out: &mut dyn Write,
    mu: f64,
    lambda: f64,
    xmin: f64,
    xmax: f64,
    xstep: f64,
) -> std::io::Result<()> {
    let mut x = xmin;
    while x <= xmax {
        writeln!(
            out,
            "{}\t{}",
            fmt_fixed6(x),
            fmt_g(exponential::surv(x, mu, lambda))
        )?;
        x += xstep;
    }
    writeln!(out, "&")
}
