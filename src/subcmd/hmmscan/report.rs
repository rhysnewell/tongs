use super::*;

fn truncate_for_textw(s: &str, width: usize) -> String {
    if width == 0 || s.chars().count() <= width {
        return s.to_string();
    }
    s.chars().take(width).collect()
}

pub(super) fn write_hmmscan_score_table<W: Write + ?Sized>(
    out: &mut W,
    th: &TopHits,
    z: f64,
    prefer_acc: bool,
    textw: usize,
) {
    use tongs::tophits::P7_IS_REPORTED;

    let model_namew = th
        .hits
        .iter()
        .map(|hit| display_name(hit, prefer_acc).len())
        .max()
        .unwrap_or(0)
        .max(8);
    let descw = if textw > 0 {
        textw.saturating_sub(model_namew + 61).max(32)
    } else {
        0
    };

    writeln!(
        out,
        "Scores for complete sequence (score includes all domains):"
    )
    .unwrap();
    writeln!(
        out,
        "   --- full sequence ---   --- best 1 domain ---    -#dom-"
    )
    .unwrap();
    writeln!(
        out,
        "    E-value  score  bias    E-value  score  bias    exp  N  {:<model_namew$} Description",
        "Model"
    )
    .unwrap();
    writeln!(
        out,
        "    ------- ------ -----    ------- ------ -----   ---- --  {:<model_namew$} -----------",
        "--------"
    )
    .unwrap();

    let mut any_reported = false;
    let mut have_printed_incthresh = false;
    for hit in &th.hits {
        if hit.flags & P7_IS_REPORTED == 0 {
            continue;
        }
        if hit.flags & tongs::tophits::P7_IS_INCLUDED == 0 && !have_printed_incthresh {
            writeln!(out, "  ------ inclusion threshold ------").unwrap();
            have_printed_incthresh = true;
        }
        any_reported = true;
        let evalue = z * c_exp_f64(hit.lnp);
        let best_dom = crate::subcmd::hmmsearch::best_domain(hit);
        let dom_evalue = best_dom.map(|d| z * c_exp_f64(d.lnp)).unwrap_or(evalue);
        let dom_score = best_dom.map(|d| d.bitscore).unwrap_or(hit.score);
        let dom_bias = best_dom.map(|d| d.dombias).unwrap_or(hit.bias);
        writeln!(
            out,
            "  {} {} {}  {} {} {}  {} {:2}  {:<model_namew$}  {}",
            tongs::output::fmt_evalue(evalue),
            tongs::output::fmt_score(hit.score),
            tongs::output::fmt_bias(hit.bias),
            tongs::output::fmt_evalue(dom_evalue),
            tongs::output::fmt_score(dom_score),
            tongs::output::fmt_bias(dom_bias),
            fmt_width5_1(hit.nexpected as f64),
            hit.nreported,
            display_name(hit, prefer_acc),
            truncate_for_textw(&hit.desc, descw),
        )
        .unwrap();
    }

    if !any_reported {
        writeln!(
            out,
            "\n   [No hits detected that satisfy reporting thresholds]"
        )
        .unwrap();
    }
}

pub(super) fn write_hmmscan_domain_annotation(
    out: &mut dyn Write,
    th: &TopHits,
    z: f64,
    domz: f64,
    qname: &str,
    qacc: &str,
    qlen: usize,
    hmms: &[tongs::Hmm],
    prefer_acc: bool,
    noali: bool,
    textw: usize,
) {
    use tongs::tophits::P7_IS_REPORTED;

    writeln!(out).unwrap();
    writeln!(out).unwrap();
    if noali {
        writeln!(out, "Domain annotation for each model:").unwrap();
    } else {
        writeln!(out, "Domain annotation for each model (and alignments):").unwrap();
    }

    if !th.hits.iter().any(|hit| hit.flags & P7_IS_REPORTED != 0) {
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

        writeln!(out, ">> {}  {}", display_name(hit, prefer_acc), hit.desc).unwrap();
        if hit.nreported == 0 {
            writeln!(
                out,
                "   [No individual domains that satisfy reporting thresholds (although complete target did)]"
            )
            .unwrap();
            writeln!(out).unwrap();
            continue;
        }

        writeln!(out, "   #    score  bias  c-Evalue  i-Evalue hmmfrom  hmm to    alifrom  ali to    envfrom  env to     acc").unwrap();
        writeln!(out, " ---   ------ ----- --------- --------- ------- -------    ------- -------    ------- -------    ----").unwrap();

        let mut reported_idx = 0usize;
        for dom in hit.dcl.iter().filter(|dom| dom.is_reported) {
            reported_idx += 1;
            let indicator = if dom.is_included { '!' } else { '?' };
            let (hf, ht) = if let Some(ref ad) = dom.ad {
                (ad.hmmfrom, ad.hmmto)
            } else {
                (1, hit.n)
            };
            let hmm_left = if hf == 1 { '[' } else { '.' };
            let hmm_right = if ht == hit.n { ']' } else { '.' };
            let ali_left = if dom.iali == 1 { '[' } else { '.' };
            let ali_right = if dom.jali == qlen as i64 { ']' } else { '.' };
            let env_left = if dom.ienv == 1 { '[' } else { '.' };
            let env_right = if dom.jenv == qlen as i64 { ']' } else { '.' };
            let acc = dom.oasc / (1.0 + (dom.jenv - dom.ienv).abs() as f32);
            writeln!(
                out,
                " {:3} {} {} {} {:>9} {:>9} {:7} {:7} {}{} {:7} {:7} {}{} {:7} {:7} {}{} {}",
                reported_idx,
                indicator,
                tongs::output::fmt_score(dom.bitscore),
                tongs::output::fmt_bias(dom.dombias),
                tongs::output::fmt_evalue(domz * c_exp_f64(dom.lnp)),
                tongs::output::fmt_evalue(z * c_exp_f64(dom.lnp)),
                hf,
                ht,
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
                tongs::output::fmt_fixed2(acc as f64),
            )
            .unwrap();
        }
        writeln!(out).unwrap();

        if !noali {
            writeln!(out, "  Alignments for each domain:").unwrap();
            let model_cs = hmmscan_model_cs_for_hit(hmms, hit);
            let mut reported_idx = 0usize;
            for dom in hit.dcl.iter().filter(|dom| dom.is_reported) {
                reported_idx += 1;
                writeln!(
                    out,
                    "  == domain {}  score: {} bits;  conditional E-value: {}",
                    reported_idx,
                    tongs::output::fmt_fixed1(dom.bitscore as f64),
                    tongs::output::fmt_evalue(domz * c_exp_f64(dom.lnp)).trim()
                )
                .unwrap();

                if let Some(ref ad) = dom.ad {
                    let cs_line = model_cs.map(|cs| {
                        let mut s = String::with_capacity(ad.model.len());
                        let mut cs_idx = ad.hmmfrom;
                        for ch in ad.model.chars() {
                            if ch == '.' {
                                s.push('.');
                            } else {
                                s.push(if cs_idx < cs.len() {
                                    cs[cs_idx] as char
                                } else {
                                    ' '
                                });
                                cs_idx += 1;
                            }
                        }
                        s
                    });
                    tongs::tophits::print_alidisplay_blocks_acc(
                        out,
                        &hit.name,
                        &hit.acc,
                        qname,
                        qacc,
                        ad,
                        cs_line.as_deref(),
                        textw,
                        prefer_acc,
                    );
                }
                writeln!(out).unwrap();
            }
        }
    }
    writeln!(out).unwrap();
}

fn hmmscan_model_cs_for_hit<'a>(
    hmms: &'a [tongs::Hmm],
    hit: &tongs::tophits::Hit,
) -> Option<&'a [u8]> {
    hmms.iter()
        .find(|hmm| {
            hmm.name == hit.name
                || hmm
                    .acc
                    .as_deref()
                    .is_some_and(|acc| !hit.acc.is_empty() && acc == hit.acc)
        })
        .and_then(|hmm| hmm.cs.as_deref())
}
#[derive(Default)]
pub(super) struct PipelineStats {
    pub(super) n_past_msv: u64,
    pub(super) n_past_bias: u64,
    pub(super) n_past_vit: u64,
    pub(super) n_past_fwd: u64,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_scan_pipeline_stats<W: Write + ?Sized>(
    out: &mut W,
    n_targets: u64,
    total_nodes: u64,
    query_residues: u64,
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
        n_targets, total_nodes
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
