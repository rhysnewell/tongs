use std::fs::File;
use std::io::{BufWriter, Write};

use tongs::alphabet::Alphabet;
use tongs::output::fmt_width5_1;
use tongs::search::{Query, Scan};
use tongs::tophits::TopHits;

use super::tables::{best_domain, evalue_from_lnp, shown_hit_name, truncate_for_textw};
use super::{
    write_ali_output, write_domtblout, write_pfamtblout_with_pipeline, write_tblout, Args,
};

pub(super) struct Sinks {
    pub out: Box<dyn Write>,
    pub tblout: Option<BufWriter<File>>,
    pub domtblout: Option<BufWriter<File>>,
    pub pfamtblout: Option<BufWriter<File>>,
    pub ali: Option<BufWriter<File>>,
}

pub(super) fn write_query_header(out: &mut dyn Write, hmm: &tongs::Hmm) {
    writeln!(out, "Query:       {}  [M={}]", hmm.name, hmm.m).unwrap();
    if let Some(acc) = hmm.acc.as_deref().filter(|acc| !acc.is_empty()) {
        writeln!(out, "Accession:   {}", acc).unwrap();
    }
    if let Some(desc) = hmm.desc.as_deref().filter(|desc| !desc.is_empty()) {
        writeln!(out, "Description: {}", desc).unwrap();
    }
}

pub(super) fn write(
    sinks: &mut Sinks,
    args: &Args,
    query_idx: usize,
    query: &Query,
    scan: Scan,
    n_targets: u64,
    total_residues: u64,
) {
    let hmm = query.hmm;
    let abc = Alphabet::new(hmm.abc_type);
    let textw = if args.notextw { 0 } else { args.textw };
    let out = &mut sinks.out;
    write_query_header(out, hmm);

    let mut pli = query.pli.fork();
    pli.n_past_msv = scan.n_past_msv;
    pli.n_past_bias = scan.n_past_bias;
    pli.n_past_vit = scan.n_past_vit;
    pli.n_past_fwd = scan.n_past_fwd;
    pli.n_targets = n_targets;
    let mut th = TopHits::new();
    th.hits = scan.hits;

    // Set Z (database size)
    let z = match pli.z_setby {
        tongs::pipeline::ZSetBy::Option => pli.z,
        tongs::pipeline::ZSetBy::Ntargets => pli.n_targets as f64,
    };

    // Sort and threshold (first pass with domz = z for sequence-level reporting)
    th.sort_by_sortkey();
    th.threshold(
        &pli, z, z, // temporary domz = z for first pass
    );

    // Set domz: user-specified, or auto = nreported from first pass
    let domz = match pli.domz_setby {
        tongs::pipeline::ZSetBy::Option => pli.domz,
        tongs::pipeline::ZSetBy::Ntargets => th.nreported as f64,
    };
    // Re-threshold with correct domz for domain-level E-values
    if domz != z {
        th.threshold(&pli, z, domz);
    }

    let show_acc = args.show_acc;
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

    // Per-sequence hit table
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
        let desc = if hit.desc.is_empty() { "" } else { &hit.desc };
        let desc = truncate_for_textw(desc, target_descw);

        writeln!(
            out,
            "  {} {} {}  {} {} {}  {} {:2}  {:<target_namew$}  {}",
            tongs::output::fmt_evalue(evalue),
            tongs::output::fmt_score(hit.score),
            tongs::output::fmt_bias(hit.bias),
            tongs::output::fmt_evalue(dom_evalue),
            tongs::output::fmt_score(dom_score),
            tongs::output::fmt_bias(dom_bias),
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

    // Domain annotation for each sequence. C HMMER still prints the domain
    // table under --noali; only the alignment blocks are suppressed.
    if args.noali {
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
            let indicator = if dom.is_included {
                '!'
            } else if dom.is_reported {
                '?'
            } else {
                '?'
            };

            let (hf, ht) = if let Some(ref ad) = dom.ad {
                (ad.hmmfrom, ad.hmmto)
            } else {
                (1, hmm.m)
            };
            // Boundary indicators
            let hmm_left = if hf == 1 { '[' } else { '.' };
            let hmm_right = if ht == hmm.m { ']' } else { '.' };
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
                tongs::output::fmt_score(dom.bitscore),
                tongs::output::fmt_bias(dom.dombias),
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
                tongs::output::fmt_fixed2(acc as f64),
            )
            .unwrap();
        }

        writeln!(out).unwrap();

        // Text alignments for each domain
        if !args.noali {
            writeln!(out, "  Alignments for each domain:").unwrap();
            let mut reported_idx = 0usize;
            for dom in hit.dcl.iter().filter(|dom| dom.is_reported) {
                reported_idx += 1;
                writeln!(
                    out,
                    "  == domain {}  score: {} bits;  conditional E-value: {}",
                    reported_idx,
                    tongs::output::fmt_fixed1(dom.bitscore as f64),
                    tongs::output::fmt_evalue(evalue_from_lnp(domz, dom.lnp)).trim()
                )
                .unwrap();

                if let Some(ref ad) = dom.ad {
                    // Build CS line from hmm.cs over the alignment span.
                    let cs_line = hmm.cs.as_ref().map(|cs| {
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
                        &hmm.name,
                        hmm.acc.as_deref().unwrap_or(""),
                        &hit.name,
                        &hit.acc,
                        ad,
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

    // Statistics
    let expected_msv = (pli.f1 * pli.n_targets as f64).max(0.0);
    let expected_vit = (pli.f2 * pli.n_targets as f64).max(0.0);
    let expected_fwd = (pli.f3 * pli.n_targets as f64).max(0.0);
    let frac_msv = if pli.n_targets > 0 {
        pli.n_past_msv as f64 / pli.n_targets as f64
    } else {
        0.0
    };
    let frac_vit = if pli.n_targets > 0 {
        pli.n_past_vit as f64 / pli.n_targets as f64
    } else {
        0.0
    };
    let frac_fwd = if pli.n_targets > 0 {
        pli.n_past_fwd as f64 / pli.n_targets as f64
    } else {
        0.0
    };
    let frac_bias = if pli.n_targets > 0 {
        pli.n_past_bias as f64 / pli.n_targets as f64
    } else {
        0.0
    };

    writeln!(out, "Internal pipeline statistics summary:").unwrap();
    writeln!(out, "-------------------------------------").unwrap();
    writeln!(
        out,
        "Query model(s):                  {:>11}  ({} nodes)",
        1, hmm.m
    )
    .unwrap();
    writeln!(
        out,
        "Target sequences:                {:>11}  ({} residues searched)",
        pli.n_targets, total_residues
    )
    .unwrap();
    writeln!(
        out,
        "Passed MSV filter:               {:>11}  ({}); expected {} ({})",
        pli.n_past_msv,
        tongs::output::fmt_g(frac_msv),
        tongs::output::fmt_fixed1(expected_msv),
        tongs::output::fmt_g(pli.f1)
    )
    .unwrap();
    writeln!(
        out,
        "Passed bias filter:              {:>11}  ({}); expected {} ({})",
        pli.n_past_bias,
        tongs::output::fmt_g(frac_bias),
        tongs::output::fmt_fixed1(expected_msv),
        tongs::output::fmt_g(pli.f1)
    )
    .unwrap();
    writeln!(
        out,
        "Passed Vit filter:               {:>11}  ({}); expected {} ({})",
        pli.n_past_vit,
        tongs::output::fmt_g(frac_vit),
        tongs::output::fmt_fixed1(expected_vit),
        tongs::output::fmt_g(pli.f2)
    )
    .unwrap();
    writeln!(
        out,
        "Passed Fwd filter:               {:>11}  ({}); expected {} ({})",
        pli.n_past_fwd,
        tongs::output::fmt_g(frac_fwd),
        tongs::output::fmt_fixed1(expected_fwd),
        tongs::output::fmt_g(pli.f3)
    )
    .unwrap();
    writeln!(
        out,
        "Initial search space (Z):        {}  {}",
        tongs::output::fmt_width11_0(z),
        if pli.z_setby == tongs::pipeline::ZSetBy::Option {
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
        if pli.domz_setby == tongs::pipeline::ZSetBy::Option {
            "[as set by --domZ on cmdline]"
        } else {
            "[number of targets reported over threshold]"
        }
    )
    .unwrap();

    // Write tabular output
    if let Some(f) = sinks.tblout.as_mut() {
        write_tblout(f, &hmm.name, hmm.acc.as_deref(), &th, z, query_idx == 0);
    }
    if let Some(f) = sinks.domtblout.as_mut() {
        write_domtblout(
            f,
            &hmm.name,
            hmm.acc.as_deref(),
            hmm.m,
            &th,
            z,
            domz,
            query_idx == 0,
        );
    }
    if let Some(f) = sinks.pfamtblout.as_mut() {
        write_pfamtblout_with_pipeline(f, &hmm.name, hmm.acc.as_deref(), hmm.m, &th, &pli, z, domz);
    }
    writeln!(out, "//").unwrap();
    if let Some(f) = sinks.ali.as_mut() {
        // C hmmsearch.c:554-572: after the "//" line, build the
        // included-domain MSA, write it, then echo the confirmation line
        // (carrying the MSA's nseq) to stdout.
        let nseq = write_ali_output(
            f,
            &abc,
            hmm.m,
            &hmm.name,
            hmm.acc.as_deref(),
            hmm.desc.as_deref(),
            "hmmsearch (HMMER 3.4)",
            &th,
            textw,
        );
        match nseq {
            Some(nseq) => writeln!(
                out,
                "# Alignment of {} hits satisfying inclusion thresholds saved to: {}",
                nseq,
                args.ali_outfile.as_ref().unwrap().display()
            )
            .unwrap(),
            None => writeln!(
                out,
                "# No hits satisfy inclusion thresholds; no alignment saved"
            )
            .unwrap(),
        }
    }
}
