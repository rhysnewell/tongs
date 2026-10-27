use std::io::Write;

use tongs::output::{fmt_bias, fmt_fixed1, fmt_fixed2, fmt_score, fmt_width5_1};
use tongs::tophits::TopHits;

use tongs::subcmd::outputs::truncate_for_textw;

use super::tables::{best_domain, evalue_from_lnp, shown_hit_name};

pub fn write_standard_stdout_tables<W: Write>(
    out: &mut W,
    th: &TopHits,
    model_name: &str,
    model_acc: &str,
    model_len: usize,
    model_cs: Option<&[u8]>,
    z: f64,
    domz: f64,
    textw: usize,
    show_acc: bool,
    noali: bool,
    show_hit_status: bool,
) {
    let target_namew = th
        .hits
        .iter()
        .map(|hit| shown_hit_name(hit, show_acc).len())
        .max()
        .unwrap_or(0)
        .max(8);
    let target_descw = if textw > 0 {
        textw.saturating_sub(target_namew + 61).max(32)
    } else {
        0
    };

    writeln!(
        out,
        "Scores for complete sequences (score includes all domains):"
    )
    .unwrap();
    writeln!(
        out,
        "   --- full sequence ---   --- best 1 domain ---    -#dom-"
    )
    .unwrap();
    writeln!(
        out,
        "    E-value  score  bias    E-value  score  bias    exp  N  {:<target_namew$} Description",
        "Sequence"
    )
    .unwrap();
    writeln!(
        out,
        "    ------- ------ -----    ------- ------ -----   ---- --  {:<target_namew$} -----------",
        "--------"
    )
    .unwrap();

    let mut any_reported = false;
    let mut have_printed_incthresh = false;
    for hit in &th.hits {
        if hit.flags & tongs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }
        if hit.flags & tongs::tophits::P7_IS_INCLUDED == 0 && !have_printed_incthresh {
            writeln!(out, "  ------ inclusion threshold ------").unwrap();
            have_printed_incthresh = true;
        }
        any_reported = true;
        let evalue = evalue_from_lnp(z, hit.lnp);
        let best_dom = best_domain(hit);
        let dom_evalue = best_dom
            .map(|d| evalue_from_lnp(z, d.lnp))
            .unwrap_or(evalue);
        let dom_score = best_dom.map(|d| d.bitscore).unwrap_or(hit.score);
        let dom_bias = best_dom.map(|d| d.dombias).unwrap_or(hit.bias);
        let desc = truncate_for_textw(&hit.desc, target_descw);
        let status = if show_hit_status
            && hit.flags & tongs::tophits::P7_IS_INCLUDED != 0
            && hit.flags & tongs::tophits::P7_IS_NEW != 0
        {
            '+'
        } else if show_hit_status && hit.flags & tongs::tophits::P7_IS_DROPPED != 0 {
            '-'
        } else {
            ' '
        };

        writeln!(
            out,
            "{} {} {} {}  {} {} {}  {} {:2}  {:<target_namew$}  {}",
            status,
            tongs::output::fmt_evalue(evalue),
            fmt_score(hit.score),
            fmt_bias(hit.bias),
            tongs::output::fmt_evalue(dom_evalue),
            fmt_score(dom_score),
            fmt_bias(dom_bias),
            fmt_width5_1(hit.nexpected as f64),
            hit.nreported,
            shown_hit_name(hit, show_acc),
            desc,
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

    writeln!(out).unwrap();
    writeln!(out).unwrap();

    if noali {
        writeln!(out, "Domain annotation for each sequence:").unwrap();
    } else {
        writeln!(out, "Domain annotation for each sequence (and alignments):").unwrap();
    }
    if !any_reported {
        writeln!(
            out,
            "\n   [No targets detected that satisfy reporting thresholds]"
        )
        .unwrap();
    }

    for hit in &th.hits {
        if hit.flags & tongs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }

        let domain_descw = if textw > 0 {
            textw
                .saturating_sub(shown_hit_name(hit, show_acc).len() + 5)
                .max(32)
        } else {
            0
        };
        let desc = truncate_for_textw(&hit.desc, domain_descw);
        writeln!(out, ">> {}  {}", shown_hit_name(hit, show_acc), desc).unwrap();

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
            let c_evalue = evalue_from_lnp(domz, dom.lnp);
            let i_evalue = evalue_from_lnp(z, dom.lnp);
            let indicator = if dom.is_included { '!' } else { '?' };

            let (hf, ht) = if let Some(ref ad) = dom.ad {
                (ad.hmmfrom, ad.hmmto)
            } else {
                (1, model_len)
            };
            let hmm_left = if hf == 1 { '[' } else { '.' };
            let hmm_right = if ht == model_len { ']' } else { '.' };
            let seq_left = if dom.iali == 1 { '[' } else { '.' };
            let seq_right = if dom.jali == hit.n as i64 { ']' } else { '.' };
            let env_left = if dom.ienv == 1 { '[' } else { '.' };
            let env_right = if dom.jenv == hit.n as i64 { ']' } else { '.' };
            let acc = dom.oasc / (1.0 + (dom.jenv - dom.ienv).abs() as f32);

            writeln!(
                out,
                " {:3} {} {} {} {} {} {:7} {:7} {}{} {:7} {:7} {}{} {:7} {:7} {}{} {}",
                reported_idx,
                indicator,
                fmt_score(dom.bitscore),
                fmt_bias(dom.dombias),
                tongs::output::fmt_evalue(c_evalue),
                tongs::output::fmt_evalue(i_evalue),
                hf,
                ht,
                hmm_left,
                hmm_right,
                dom.iali,
                dom.jali,
                seq_left,
                seq_right,
                dom.ienv,
                dom.jenv,
                env_left,
                env_right,
                fmt_fixed2(acc as f64),
            )
            .unwrap();
        }

        writeln!(out).unwrap();

        if !noali {
            writeln!(out, "  Alignments for each domain:").unwrap();
            let mut reported_idx = 0usize;
            for dom in hit.dcl.iter().filter(|dom| dom.is_reported) {
                reported_idx += 1;
                writeln!(
                    out,
                    "  == domain {}  score: {} bits;  conditional E-value: {}",
                    reported_idx,
                    fmt_fixed1(dom.bitscore as f64),
                    tongs::output::fmt_evalue(evalue_from_lnp(domz, dom.lnp)).trim()
                )
                .unwrap();

                if let Some(ref ad) = dom.ad {
                    let mut ad_for_print = ad.clone();
                    if ad_for_print.rfline.is_empty() {
                        ad_for_print.model = ad_for_print.model.to_ascii_lowercase();
                        ad_for_print.mline = ad_for_print.mline.to_ascii_lowercase();
                    }
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
                    // C applies --acc to both model and sequence (p7_alidisplay.c:1176-1180).
                    tongs::tophits::print_alidisplay_blocks_acc(
                        out,
                        model_name,
                        model_acc,
                        &hit.name,
                        &hit.acc,
                        &ad_for_print,
                        cs_line.as_deref(),
                        textw,
                        show_acc,
                    );
                }
                writeln!(out).unwrap();
            }
        }
    }

    writeln!(out).unwrap();
    writeln!(out).unwrap();
}
