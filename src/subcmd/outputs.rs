use std::io::Write;
use std::path::Path;

use tongs::bg::Bg;
use tongs::output::{fmt_fixed1, fmt_g, fmt_width11_0};
use tongs::tophits::Hit;
use tongs::util::cmath::{c_log_f64, ESL_CONST_LOG2R};

pub(crate) fn create_output_file_or_exit(
    path: &std::path::Path,
    message_template: &str,
) -> std::fs::File {
    std::fs::File::create(path).unwrap_or_else(|_| {
        eprintln!(
            "{}",
            message_template.replace("{path}", &path.display().to_string())
        );
        std::process::exit(1);
    })
}
pub(crate) fn create_file_or_exit(path: &Path, what: &str) -> std::fs::File {
    std::fs::File::create(path).unwrap_or_else(|e| {
        eprintln!("Error creating {what}: {e}");
        std::process::exit(1);
    })
}

#[derive(Default)]
pub(crate) struct PipelineStats {
    pub(crate) n_past_msv: u64,
    pub(crate) n_past_bias: u64,
    pub(crate) n_past_vit: u64,
    pub(crate) n_past_fwd: u64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn write_pipeline_stats<W: Write>(
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
        fmt_g(frac(stats.n_past_msv)),
        fmt_fixed1(expected_msv),
        fmt_g(f1)
    )
    .unwrap();
    writeln!(
        out,
        "Passed bias filter:              {:>11}  ({}); expected {} ({})",
        stats.n_past_bias,
        fmt_g(frac(stats.n_past_bias)),
        fmt_fixed1(expected_msv),
        fmt_g(f1)
    )
    .unwrap();
    writeln!(
        out,
        "Passed Vit filter:               {:>11}  ({}); expected {} ({})",
        stats.n_past_vit,
        fmt_g(frac(stats.n_past_vit)),
        fmt_fixed1(expected_vit),
        fmt_g(f2)
    )
    .unwrap();
    writeln!(
        out,
        "Passed Fwd filter:               {:>11}  ({}); expected {} ({})",
        stats.n_past_fwd,
        fmt_g(frac(stats.n_past_fwd)),
        fmt_fixed1(expected_fwd),
        fmt_g(f3)
    )
    .unwrap();
    writeln!(
        out,
        "Initial search space (Z):        {}  {}",
        fmt_width11_0(z),
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
        fmt_width11_0(domz),
        if domz_from_option {
            "[as set by --domZ on cmdline]"
        } else {
            "[number of targets reported over threshold]"
        }
    )
    .unwrap();
}
pub(crate) fn display_name(hit: &Hit, prefer_acc: bool) -> &str {
    if prefer_acc && !hit.acc.is_empty() {
        &hit.acc
    } else {
        &hit.name
    }
}
pub(crate) fn truncate_for_textw(s: &str, width: usize) -> String {
    if width == 0 || s.chars().count() <= width {
        return s.to_string();
    }
    s.chars().take(width).collect()
}
pub(crate) fn mean_match_relative_entropy(hmm: &tongs::Hmm, bg: &Bg) -> f32 {
    let mut sum = 0.0_f32;
    for node in 1..=hmm.m {
        for x in 0..hmm.abc_k {
            let p = hmm.mat[node][x];
            if p > 0.0 && bg.f[x] > 0.0 {
                sum += p * c_log_f64((p / bg.f[x]) as f64) as f32 * (ESL_CONST_LOG2R as f32);
            }
        }
    }
    sum / hmm.m as f32
}
