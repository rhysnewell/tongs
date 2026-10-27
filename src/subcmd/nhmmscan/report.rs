use super::*;

fn display_name(hit: &tongs::tophits::Hit, prefer_acc: bool) -> &str {
    if prefer_acc && !hit.acc.is_empty() {
        &hit.acc
    } else {
        &hit.name
    }
}

pub(super) fn write_nhmmscan_option_header(out: &mut dyn Write, args: &Args, cmdline: &str) {
    if command_line_has_option(cmdline, "-E") {
        writeln!(
            out,
            "# profile reporting threshold:     E-value <= {}",
            fmt_g(args.e_value)
        )
        .unwrap();
    }
    if let Some(score) = args.score_threshold {
        writeln!(
            out,
            "# profile reporting threshold:     score >= {}",
            fmt_g(score)
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--incE") {
        writeln!(
            out,
            "# profile inclusion threshold:     E-value <= {}",
            fmt_g(args.inc_e)
        )
        .unwrap();
    }
    if let Some(score) = args.inc_t {
        writeln!(
            out,
            "# profile inclusion threshold:     score >= {}",
            fmt_g(score)
        )
        .unwrap();
    }
    if args.cut_ga {
        writeln!(out, "# model-specific thresholding:     GA cutoffs").unwrap();
    }
    if args.cut_nc {
        writeln!(out, "# model-specific thresholding:     NC cutoffs").unwrap();
    }
    if args.cut_tc {
        writeln!(out, "# model-specific thresholding:     TC cutoffs").unwrap();
    }
    if args.max {
        writeln!(
            out,
            "# Max sensitivity mode:            on [all heuristic filters off]"
        )
        .unwrap();
    }
    if command_line_has_option(cmdline, "--F1") {
        writeln!(out, "# MSV filter P threshold:       <= {}", fmt_g(args.f1)).unwrap();
    }
    if command_line_has_option(cmdline, "--F2") {
        writeln!(out, "# Vit filter P threshold:       <= {}", fmt_g(args.f2)).unwrap();
    }
    if command_line_has_option(cmdline, "--F3") {
        writeln!(out, "# Fwd filter P threshold:       <= {}", fmt_g(args.f3)).unwrap();
    }
    if args.nobias {
        writeln!(out, "# biased composition HMM filter:   off").unwrap();
    }
    if args.nonull2 {
        writeln!(out, "# null2 bias corrections:          off").unwrap();
    }
    if args.watson {
        writeln!(out, "# search only top strand:          on").unwrap();
    }
    if args.crick {
        writeln!(out, "# search only bottom strand:       on").unwrap();
    }
}

pub(super) fn command_line_has_option(cmdline: &str, option: &str) -> bool {
    let compact_short = option
        .strip_prefix('-')
        .filter(|rest| !rest.starts_with('-') && rest.chars().count() == 1)
        .map(|_| option);
    cmdline.split_whitespace().any(|token| {
        token == option
            || token.starts_with(&format!("{option}="))
            || compact_short
                .is_some_and(|short| token.starts_with(short) && token.len() > short.len())
    })
}

pub(super) fn normalize_nhmmscan_table_cmdline(cmdline: &str) -> String {
    let mut tokens = cmdline.split_whitespace();
    match (tokens.next(), tokens.next()) {
        (Some(_wrapper), Some("nhmmscan")) => {
            let rest = tokens.collect::<Vec<_>>();
            if rest.is_empty() {
                "nhmmscan".to_string()
            } else {
                format!("nhmmscan {}", rest.join(" "))
            }
        }
        _ => cmdline.to_string(),
    }
}
#[derive(Default)]
pub(super) struct PipelineStats {
    pub(super) n_past_msv: u64,
    pub(super) n_past_bias: u64,
    pub(super) n_past_vit: u64,
    pub(super) n_past_fwd: u64,
}

pub(super) struct ScanModelResult {
    pub(super) hits: Vec<tongs::tophits::Hit>,
    pub(super) stats: PipelineStats,
}

pub(super) fn write_nhmmscan_stdout_hits(
    out: &mut dyn Write,
    th: &TopHits,
    z: f64,
    qname: &str,
    prefer_acc: bool,
    noali: bool,
    textw: usize,
) {
    use tongs::tophits::{P7_IS_INCLUDED, P7_IS_REPORTED};

    writeln!(out, "Scores for complete hit:").unwrap();
    writeln!(
        out,
        "    E-value  score  bias  Model     start    end  Description"
    )
    .unwrap();
    writeln!(
        out,
        "    ------- ------ -----  --------  -----  -----  -----------"
    )
    .unwrap();

    let mut have_printed_incthresh = false;
    let mut any_reported = false;
    for hit in &th.hits {
        if hit.flags & P7_IS_REPORTED == 0 {
            continue;
        }
        any_reported = true;
        if hit.flags & P7_IS_INCLUDED == 0 && !have_printed_incthresh {
            writeln!(out, "  ------ inclusion threshold ------").unwrap();
            have_printed_incthresh = true;
        }
        let best_dom = hit
            .dcl
            .iter()
            .max_by(|a, b| a.bitscore.total_cmp(&b.bitscore));
        let (start, end) = best_dom.map(|d| (d.iali, d.jali)).unwrap_or((0, 0));
        writeln!(
            out,
            "  {:>9} {} {}  {:<8} {:>6} {:>6} {}",
            tongs::output::fmt_evalue(z * c_exp_f64(hit.lnp)),
            fmt_score(hit.score),
            fmt_bias(hit.bias),
            display_name(hit, prefer_acc),
            start,
            end,
            if hit.desc.is_empty() { "" } else { &hit.desc },
        )
        .unwrap();
    }

    if !any_reported {
        writeln!(
            out,
            "   [No hits detected that satisfy reporting thresholds]"
        )
        .unwrap();
    }

    writeln!(out).unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Annotation for each hit{}:",
        if noali { " " } else { "  (and alignments)" }
    )
    .unwrap();
    if !any_reported {
        writeln!(
            out,
            "\n   [No targets detected that satisfy reporting thresholds]"
        )
        .unwrap();
        return;
    }

    for hit in &th.hits {
        if hit.flags & P7_IS_REPORTED == 0 {
            continue;
        }
        writeln!(
            out,
            ">> {}  {}",
            display_name(hit, prefer_acc),
            if hit.desc.is_empty() { "" } else { &hit.desc }
        )
        .unwrap();
        if hit.nreported == 0 {
            writeln!(
                out,
                "   [No individual domains that satisfy reporting thresholds (although complete target did)]"
            )
            .unwrap();
            writeln!(out).unwrap();
            continue;
        }
        writeln!(
            out,
            "    score  bias    Evalue   hmmfrom    hmm to     alifrom    ali to      envfrom    env to      mod len      acc"
        )
        .unwrap();
        writeln!(
            out,
            "   ------ ----- ---------   -------   -------    --------- ---------    --------- ---------    ---------    ----"
        )
        .unwrap();
        for dom in hit.dcl.iter().filter(|dom| dom.is_reported) {
            let (hmmfrom, hmmto) = dom
                .ad
                .as_ref()
                .map(|ad| (ad.hmmfrom, ad.hmmto))
                .unwrap_or((0, 0));
            let indicator = if dom.is_included { "!" } else { "?" };
            let hmm_left = if hmmfrom == 1 { '[' } else { '.' };
            let hmm_right = if hmmto == hit.n { ']' } else { '.' };
            let ali_left = '.';
            let ali_right = '.';
            let env_left = '.';
            let env_right = '.';
            let acc = dom.oasc / (1.0 + (dom.jenv - dom.ienv).abs() as f32);
            writeln!(
                out,
                " {} {} {} {:>9}   {:>7}   {:>7} {}{} {:>9} {:>9} {}{} {:>9} {:>9} {}{} {:>9}    {}",
                indicator,
                tongs::output::fmt_score(dom.bitscore),
                tongs::output::fmt_bias(dom.dombias),
                tongs::output::fmt_evalue(z * c_exp_f64(dom.lnp)),
                hmmfrom,
                hmmto,
                hmm_left,
                hmm_right,
                dom.iali,
                dom.jali,
                ali_left,
                ali_right,
                dom.ienv,
                dom.jenv,
                env_left,
                env_right,
                hit.n,
                tongs::output::fmt_fixed2(acc as f64),
            )
            .unwrap();
        }
        writeln!(out).unwrap();

        if !noali {
            writeln!(out, "  Alignment:").unwrap();
            for dom in hit.dcl.iter().filter(|dom| dom.is_reported) {
                writeln!(
                    out,
                    "  score: {} bits",
                    tongs::output::fmt_fixed1(dom.bitscore as f64)
                )
                .unwrap();
                if let Some(ref ad) = dom.ad {
                    tongs::tophits::print_alidisplay_blocks(
                        out,
                        display_name(hit, prefer_acc),
                        qname,
                        ad,
                        None,
                        textw,
                    );
                }
                writeln!(out).unwrap();
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_nhmmscan_pipeline_stats<W: Write + ?Sized>(
    out: &mut W,
    query_residues: u64,
    n_models: u64,
    total_nodes: u64,
    nreported: u64,
    pos_output: u64,
    stats: &PipelineStats,
    f1: f64,
    f2: f64,
    f3: f64,
) {
    let residues_scanned = query_residues.saturating_mul(n_models);
    let frac = |n: u64| {
        if residues_scanned > 0 {
            n as f64 / residues_scanned as f64
        } else {
            0.0
        }
    };
    writeln!(out).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "Internal pipeline statistics summary:").unwrap();
    writeln!(out, "-------------------------------------").unwrap();
    writeln!(
        out,
        "Query sequence(s):               {:>11}  ({} residues searched)",
        1, query_residues
    )
    .unwrap();
    writeln!(
        out,
        "Target model(s):                 {:>11}  ({} nodes)",
        n_models, total_nodes
    )
    .unwrap();
    writeln!(
        out,
        "Residues passing SSV filter:     {:>11}  ({}); expected ({})",
        stats.n_past_msv,
        tongs::output::fmt_g3(frac(stats.n_past_msv)),
        tongs::output::fmt_g3(f1)
    )
    .unwrap();
    writeln!(
        out,
        "Residues passing bias filter:    {:>11}  ({}); expected ({})",
        stats.n_past_bias,
        tongs::output::fmt_g3(frac(stats.n_past_bias)),
        tongs::output::fmt_g3(f1)
    )
    .unwrap();
    writeln!(
        out,
        "Residues passing Vit filter:     {:>11}  ({}); expected ({})",
        stats.n_past_vit,
        tongs::output::fmt_g3(frac(stats.n_past_vit)),
        tongs::output::fmt_g3(f2)
    )
    .unwrap();
    writeln!(
        out,
        "Residues passing Fwd filter:     {:>11}  ({}); expected ({})",
        stats.n_past_fwd,
        tongs::output::fmt_g3(frac(stats.n_past_fwd)),
        tongs::output::fmt_g3(f3)
    )
    .unwrap();
    writeln!(
        out,
        "Total number of hits:            {:>11}  ({})",
        nreported,
        tongs::output::fmt_g3(frac(pos_output))
    )
    .unwrap();
    // Run-time footer (`# CPU time:` / `# Mc/sec:`) is intentionally
    // suppressed for determinism, matching the hmmsearch/phmmer/hmmscan/
    // jackhmmer Rust programs (which also omit these non-deterministic timing
    // lines). C HMMER prints them, but the Rust ports do not.
}

pub(super) fn reported_aligned_residues(th: &TopHits) -> u64 {
    th.hits
        .iter()
        .filter(|hit| hit.flags & tongs::tophits::P7_IS_REPORTED != 0)
        .flat_map(|hit| hit.dcl.iter())
        .filter(|dom| dom.is_reported)
        .map(|dom| ((dom.jali - dom.iali).abs() + 1) as u64)
        .sum()
}
