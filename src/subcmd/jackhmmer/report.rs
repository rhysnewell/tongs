use super::*;

#[derive(Default)]
pub(super) struct PipelineStats {
    pub(super) n_past_msv: u64,
    pub(super) n_past_bias: u64,
    pub(super) n_past_vit: u64,
    pub(super) n_past_fwd: u64,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_pipeline_stats<W: Write>(
    out: &mut W,
    model_len: usize,
    n_targets: u64,
    total_residues: u64,
    _nreported: u64,
    z: f64,
    domz: f64,
    z_from_option: bool,
    domz_from_option: bool,
    stats: &PipelineStats,
    f1: f64,
    f2: f64,
    f3: f64,
) {
    let expected_msv = (f1 * n_targets as f64).max(0.0);
    let expected_vit = (f2 * n_targets as f64).max(0.0);
    let expected_fwd = (f3 * n_targets as f64).max(0.0);
    let frac = |n: u64| {
        if n_targets > 0 {
            n as f64 / n_targets as f64
        } else {
            0.0
        }
    };

    writeln!(out, "Internal pipeline statistics summary:").unwrap();
    writeln!(out, "-------------------------------------").unwrap();
    writeln!(
        out,
        "Query model(s):                  {:>11}  ({} nodes)",
        1, model_len
    )
    .unwrap();
    writeln!(
        out,
        "Target sequences:                {:>11}  ({} residues searched)",
        n_targets, total_residues
    )
    .unwrap();
    writeln!(
        out,
        "Passed MSV filter:               {:>11}  ({}); expected {} ({})",
        stats.n_past_msv,
        tongs::output::fmt_g(frac(stats.n_past_msv)),
        tongs::output::fmt_fixed1(expected_msv),
        tongs::output::fmt_g(f1)
    )
    .unwrap();
    writeln!(
        out,
        "Passed bias filter:              {:>11}  ({}); expected {} ({})",
        stats.n_past_bias,
        tongs::output::fmt_g(frac(stats.n_past_bias)),
        tongs::output::fmt_fixed1(expected_msv),
        tongs::output::fmt_g(f1)
    )
    .unwrap();
    writeln!(
        out,
        "Passed Vit filter:               {:>11}  ({}); expected {} ({})",
        stats.n_past_vit,
        tongs::output::fmt_g(frac(stats.n_past_vit)),
        tongs::output::fmt_fixed1(expected_vit),
        tongs::output::fmt_g(f2)
    )
    .unwrap();
    writeln!(
        out,
        "Passed Fwd filter:               {:>11}  ({}); expected {} ({})",
        stats.n_past_fwd,
        tongs::output::fmt_g(frac(stats.n_past_fwd)),
        tongs::output::fmt_fixed1(expected_fwd),
        tongs::output::fmt_g(f3)
    )
    .unwrap();
    writeln!(
        out,
        "Initial search space (Z):        {}  {}",
        tongs::output::fmt_width11_0(z),
        if z_from_option {
            "[as set by --Z on cmdline]"
        } else {
            "[actual number of targets]"
        }
    )
    .unwrap();
    writeln!(
        out,
        "Domain search space  (domZ):     {}  {}",
        tongs::output::fmt_width11_0(domz),
        if domz_from_option {
            "[as set by --domZ on cmdline]"
        } else {
            "[number of targets reported over threshold]"
        }
    )
    .unwrap();
}
