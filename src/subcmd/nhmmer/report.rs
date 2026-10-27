use super::*;

pub(super) fn write_hit_scores(out: &mut dyn Write, th: &TopHits, args: &Args) {
    let namew = th
        .hits
        .iter()
        .filter(|h| h.flags & tongs::tophits::P7_IS_REPORTED != 0)
        .map(|h| h.name.len())
        .max()
        .unwrap_or(8)
        .max(8);
    let posw = th
        .hits
        .iter()
        .filter(|h| h.flags & tongs::tophits::P7_IS_REPORTED != 0)
        .flat_map(|h| {
            h.dcl.iter().flat_map(|d| {
                [d.iali, d.jali, d.ienv, d.jenv]
                    .into_iter()
                    .map(|v| v.abs().to_string().len())
            })
        })
        .max()
        .unwrap_or(6)
        .max(6);

    writeln!(out, "Scores for complete hits:").unwrap();
    writeln!(
        out,
        "  {:>9} {:>6} {:>5}  {:<namew$} {:>posw$} {:>posw$}  Description",
        "E-value",
        " score",
        " bias",
        "Sequence",
        "start",
        "end",
        namew = namew,
        posw = posw,
    )
    .unwrap();
    writeln!(
        out,
        "  {:>9} {:>6} {:>5}  {:<namew$} {:>posw$} {:>posw$}  -----------",
        "-------",
        "------",
        "-----",
        "--------",
        "-----",
        "-----",
        namew = namew,
        posw = posw,
    )
    .unwrap();

    let mut have_printed_incthresh = false;
    for hit in &th.hits {
        if hit.flags & tongs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }
        // Print "inclusion threshold" separator when transitioning from
        // included to reported-but-not-included hits (match C
        // p7_tophits.c:1181).
        if hit.flags & tongs::tophits::P7_IS_INCLUDED == 0 && !have_printed_incthresh {
            writeln!(out, "  ------ inclusion threshold ------").unwrap();
            have_printed_incthresh = true;
        }
        let best_dom = hit.dcl.first();
        let (iali, jali) = best_dom.map(|d| (d.iali, d.jali)).unwrap_or((0, 0));
        let dom_bias_bits = best_dom.map(|d| d.dombias).unwrap_or(hit.bias);
        // Match C's "%c %9.2g %6.1f %5.1f  %-*s %*ld %*ld" — newness,
        // then E-value in %g format.
        writeln!(
            out,
            "  {} {} {}  {:<namew$} {:>posw$} {:>posw$} {}",
            fmt_evalue(c_exp_f64(hit.lnp)),
            tongs::output::fmt_score(hit.score),
            tongs::output::fmt_bias(dom_bias_bits),
            if args.show_acc && !hit.acc.is_empty() {
                &hit.acc
            } else {
                &hit.name
            },
            iali,
            jali,
            if hit.desc.is_empty() { "" } else { &hit.desc },
            namew = namew,
            posw = posw,
        )
        .unwrap();
    }

    if th.nreported == 0 {
        // C p7_tophits_Targets (p7_tophits.c:1245) writes "No hits".
        // The separate "No targets detected" message is emitted later
        // after the Annotation section header (p7_tophits.c:1471).
        writeln!(
            out,
            "\n   [No hits detected that satisfy reporting thresholds]"
        )
        .unwrap();
    }
}

pub(super) fn write_hit_annotations(
    out: &mut dyn Write,
    th: &TopHits,
    hmm: &Hmm,
    args: &Args,
    linewidth: usize,
) {
    // Per-hit annotation block (C p7_tophits_Domains with long_targets).
    writeln!(out).unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        // C p7_tophits.c:1277: "Annotation for each hit %s:\n" where %s is
        // " (and alignments)" or "". The literal space after "hit" stays, so
        // the --noali line is "...hit :" (space before colon) and the
        // alignment line is "...hit  (and alignments):" (colon, no space).
        "Annotation for each hit {}:",
        if args.noali { "" } else { " (and alignments)" }
    )
    .unwrap();
    // C p7_tophits.c:1471: emit "No targets detected" when nothing was
    // reported. Matches C's separate no-targets message in the Domains
    // section (distinct from the "No hits detected" message in the
    // Scores/Targets section above).
    if th.nreported == 0 {
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
        writeln!(
            out,
            ">> {}  {}",
            if args.show_acc && !hit.acc.is_empty() {
                &hit.acc
            } else {
                &hit.name
            },
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
            "   {:>6} {:>5} {:>9} {:>9} {:>9} {:>2} {:>9} {:>9} {:>2} {:>9} {:>9}    {:>9} {:>2} {:>4}",
            "score",
            "bias",
            "  Evalue",
            "hmmfrom",
            "hmm to",
            "  ",
            " alifrom ",
            " ali to ",
            "  ",
            " envfrom ",
            " env to ",
            "  sq len ",
            "  ",
            "acc"
        )
        .unwrap();
        writeln!(
            out,
            "   {:>6} {:>5} {:>9} {:>9} {:>9} {:>2} {:>9} {:>9} {:>2} {:>9} {:>9}    {:>9} {:>2} {:>4}",
            "------",
            "-----",
            "---------",
            "-------",
            "-------",
            "  ",
            "---------",
            "---------",
            "  ",
            "---------",
            "---------",
            "---------",
            "  ",
            "----"
        )
        .unwrap();

        for dom in &hit.dcl {
            if !dom.is_reported {
                continue;
            }
            let (hmmfrom, hmmto, sqfrom, sqto) = if let Some(ref ad) = dom.ad {
                (ad.hmmfrom, ad.hmmto, ad.sqfrom, ad.sqto)
            } else {
                (0, 0, 0, 0)
            };
            let hmmfrom_bracket = if hmmfrom == 1 { '[' } else { '.' };
            let hmmto_bracket = if hmmto == hmm.m { ']' } else { '.' };
            let sqfrom_bracket = if sqfrom == 1 { '[' } else { '.' };
            let sqto_bracket = if sqto as i64 == hit.n as i64 {
                ']'
            } else {
                '.'
            };
            let envfrom_bracket = if dom.ienv == 1 { '[' } else { '.' };
            let envto_bracket = if dom.jenv == hit.n as i64 { ']' } else { '.' };
            let env_span = (dom.jenv - dom.ienv).abs() as f32;
            let acc_val = dom.oasc / (1.0 + env_span);
            writeln!(
                out,
                " {} {} {} {} {:>9} {:>9} {}{} {:>9} {:>9} {}{} {:>9} {:>9} {}{} {:>9}    {}",
                if dom.is_included { '!' } else { '?' },
                tongs::output::fmt_score(dom.bitscore),
                tongs::output::fmt_bias(dom.dombias),
                fmt_evalue(c_exp_f64(dom.lnp)),
                hmmfrom,
                hmmto,
                hmmfrom_bracket,
                hmmto_bracket,
                sqfrom,
                sqto,
                sqfrom_bracket,
                sqto_bracket,
                dom.ienv,
                dom.jenv,
                envfrom_bracket,
                envto_bracket,
                hit.n,
                tongs::output::fmt_width4_2(acc_val as f64),
            )
            .unwrap();
        }

        // Alignment block.
        if !args.noali {
            for dom in &hit.dcl {
                if !dom.is_reported {
                    continue;
                }
                if let Some(ref ad) = dom.ad {
                    if !ad.model.is_empty() {
                        writeln!(out).unwrap();
                        writeln!(out, "  Alignment:").unwrap();
                        writeln!(
                            out,
                            "  score: {} bits",
                            tongs::output::fmt_fixed1(dom.bitscore as f64)
                        )
                        .unwrap();
                        // Build CS line from hmm.cs over positions hmmfrom..=hmmto.
                        let cs_line = hmm.cs.as_deref().map(|cs: &[u8]| {
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
                        tongs::tophits::print_alidisplay_blocks(
                            out,
                            &hmm.name,
                            if args.show_acc && !hit.acc.is_empty() {
                                &hit.acc
                            } else {
                                &hit.name
                            },
                            ad,
                            cs_line.as_deref(),
                            linewidth,
                        );
                    }
                }
            }
        }
        writeln!(out).unwrap();
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_pipeline_statistics(
    out: &mut dyn Write,
    hmm: &Hmm,
    sequences: &[Sequence],
    residues_searched: f64,
    th: &TopHits,
    pli: &Pipeline,
    msv_counter: &AtomicU64,
    bias_counter: &AtomicU64,
    vit_counter: &AtomicU64,
    fwd_counter: &AtomicU64,
) {
    // Internal pipeline statistics summary.
    writeln!(out).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "Internal pipeline statistics summary:").unwrap();
    writeln!(out, "-------------------------------------").unwrap();
    writeln!(
        out,
        "Query model(s):                            {}  ({} nodes)",
        1, hmm.m
    )
    .unwrap();
    writeln!(
        out,
        "Target sequences:                          {}  ({} residues searched)",
        sequences.len(),
        residues_searched as u64
    )
    .unwrap();
    let msv_count = msv_counter.load(std::sync::atomic::Ordering::Relaxed);
    let bias_count = bias_counter.load(std::sync::atomic::Ordering::Relaxed);
    let vit_count = vit_counter.load(std::sync::atomic::Ordering::Relaxed);
    let fwd_count = fwd_counter.load(std::sync::atomic::Ordering::Relaxed);
    // pos_output = sum of aligned residues across reported domains
    // (matches C p7_pipeline.c:1985 `pli->pos_output / nres`).
    let mut pos_output: u64 = 0;
    for hit in &th.hits {
        if hit.flags & tongs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }
        for dom in &hit.dcl {
            if dom.is_reported {
                pos_output += ((dom.jali - dom.iali).abs() + 1) as u64;
            }
        }
    }
    let denom = residues_searched.max(1.0);
    let g3 = |v: f64| -> String { fmt_g3(v) };
    writeln!(
        out,
        "Residues passing SSV filter: {:>15}  ({}); expected ({})",
        msv_count,
        g3(msv_count as f64 / denom),
        g3(pli.f1)
    )
    .unwrap();
    writeln!(
        out,
        "Residues passing bias filter:{:>15}  ({}); expected ({})",
        bias_count,
        g3(bias_count as f64 / denom),
        g3(pli.f1)
    )
    .unwrap();
    writeln!(
        out,
        "Residues passing Vit filter: {:>15}  ({}); expected ({})",
        vit_count,
        g3(vit_count as f64 / denom),
        g3(pli.f2)
    )
    .unwrap();
    writeln!(
        out,
        "Residues passing Fwd filter: {:>15}  ({}); expected ({})",
        fwd_count,
        g3(fwd_count as f64 / denom),
        g3(pli.f3)
    )
    .unwrap();
    writeln!(
        out,
        "Total number of hits:        {:>15}  ({})",
        th.nreported,
        g3(pos_output as f64 / denom)
    )
    .unwrap();
    // Run-time footer (`# CPU time:` / `# Mc/sec:`) is intentionally
    // suppressed for determinism, matching the hmmsearch/phmmer/hmmscan/
    // jackhmmer Rust programs (which also omit these non-deterministic
    // timing lines). C HMMER prints them, but the Rust ports do not.
}

pub(super) fn write_included_alignment(
    out: &mut dyn Write,
    f: &mut std::fs::File,
    th: &TopHits,
    abc: &Alphabet,
    hmm: &Hmm,
    args: &Args,
) {
    if let Some(mut msa) =
        // p7_DEFAULT (nhmmer.c:1174): unused consensus columns are dropped.
        tongs::tophits::included_alignment(th, abc, hmm.m, None, false, &hmm.name)
    {
        msa.acc = hmm.acc.clone();
        msa.desc = hmm.desc.clone();
        msa.author = Some("nhmmer (HMMER 3.4)".to_string());
        // nhmmer.c:1181-1182: Stockholm (200-column blocks) when
        // textw > 0, Pfam (one unwrapped block) under --notextw.
        let textw = if args.notextw { 0 } else { args.textw };
        crate::subcmd::jackhmmer::write_tophits_alignment_msa_stockholm(f, &msa, textw);
        writeln!(
            out,
            "# Alignment of {} hits satisfying inclusion thresholds saved to: {}",
            msa.nseq,
            args.ali_outfile.as_ref().unwrap().display()
        )
        .unwrap();
    } else {
        writeln!(
            out,
            "# No hits satisfy inclusion thresholds; no alignment saved"
        )
        .unwrap();
    }
}
