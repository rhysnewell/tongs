use super::*;

/// Write C-nhmmer-style per-hit tabular output (`--tblout`).
///
/// Mirrors the `long_targets=TRUE` branch of `p7_tophits_TabularTargets` in
/// `hmmer/src/p7_tophits.c:1610`. Emits the header trio, one row per reported
/// hit (target/query name+acc, hmm/ali/env coords, seq length, strand,
/// E-value, score, bias, description), then a footer block with program /
/// version / query file / target file / option settings / cwd / date.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_nhmmer_tblout<W: std::io::Write>(
    f: &mut W,
    program: &str,
    pipeline_mode: &str,
    qname: &str,
    qacc: Option<&str>,
    th: &tongs::tophits::TopHits,
    hmmfile: &std::path::Path,
    seqdb: &std::path::Path,
    cmdline: &str,
    show_header: bool,
    write_footer: bool,
    evalue_scale: f64,
    length_header: &str,
) {
    use tongs::tophits::P7_IS_REPORTED;
    // Column widths sized dynamically to mirror C p7_tophits.c:1612-1616:
    //   tnamew = ESL_MAX(20, GetMaxNameLength)
    //   qnamew = ESL_MAX(20, strlen(qname))
    //   qaccw  = (qacc != NULL) ? ESL_MAX(10, strlen(qacc)) : 10
    //   taccw  = ESL_MAX(10, GetMaxAccessionLength)
    //   posw   = ESL_MAX(7, GetMaxPositionLength)   (long_targets always true here)
    // All GetMax* helpers iterate over *all* hits (h->unsrt), not just reported.
    let namew = th
        .hits
        .iter()
        .map(|h| h.name.len())
        .max()
        .unwrap_or(0)
        .max(20);
    let qw = qname.len().max(20);
    let qaccw = qacc.map(|s| s.len()).unwrap_or(0).max(10);
    let taccw = th
        .hits
        .iter()
        .map(|h| h.acc.len())
        .max()
        .unwrap_or(0)
        .max(10);
    // p7_tophits_GetMaxPositionLength: max decimal-digit length of iali/jali
    // over hits with dcl[0].iali > 0, min 7.
    let posw = th
        .hits
        .iter()
        .filter_map(|h| h.dcl.first())
        .filter(|d| d.iali > 0)
        .map(|d| {
            let a = d.iali.to_string().len();
            let b = d.jali.to_string().len();
            a.max(b)
        })
        .max()
        .unwrap_or(0)
        .max(7);

    // Mirror C p7_tophits.c:1623: `#%-*s ...` where width=namew-1, value has
    // leading space (` target name`). Net effect: `#` + 19-char field = 20.
    if show_header {
        // C p7_tophits.c:1623 long_targets header:
        //   "#%-*s %-*s %-*s %-*s %s %s %*s %*s %*s %*s %*s %6s %9s %6s %5s  %s\n"
        // widths: tnamew-1, taccw, qnamew, qaccw, then "hmmfrom"/"hmm to" plain,
        // then posw x5, then %6s strand, %9s E-value, %6s score, %5s bias.
        writeln!(
            f,
            "#{tname:<tnw$} {tacc:<taccw$} {qn:<qnw$} {qacc:<qaccw$} hmmfrom hmm to {af:>posw$} {at:>posw$} {ef:>posw$} {et:>posw$} {sl:>posw$} {strand:>6} {ev:>9} {sc:>6} {bi:>5}  description of target",
            tname = " target name",
            tacc = "accession",
            qn = "query name",
            qacc = "accession",
            af = "alifrom",
            at = " ali to",
            ef = "envfrom",
            et = " env to",
            sl = length_header,
            strand = "strand",
            ev = "  E-value",
            sc = " score",
            bi = " bias",
            tnw = namew - 1,
            qnw = qw,
            taccw = taccw,
            qaccw = qaccw,
            posw = posw,
        )
        .unwrap();
        // C p7_tophits.c:1626 long_targets dash line:
        //   "#%*s %*s %*s %*s %s %s %*s %*s %*s %*s %*s %6s %9s %6s %5s %s\n"
        // Right-justified (%*s) FIXED-LENGTH dash literals padded with spaces to
        // the column width; NOT fill-character dashes. The literals are:
        //   tnamew-1 -> 19 dashes, taccw -> 10, qnamew -> 20, qaccw -> 10,
        //   hmmfrom/hmm to (%s) -> 7 each (plain), posw x5 -> 7 each,
        //   %6s -> 6, %9s -> 9, %6s -> 6, %5s -> 5, %s -> 21.
        writeln!(
            f,
            "#{tname:>tnw$} {tacc:>taccw$} {qn:>qnw$} {qacc:>qaccw$} ------- ------- {af:>posw$} {at:>posw$} {ef:>posw$} {et:>posw$} {sl:>posw$} {strand:>6} {ev:>9} {sc:>6} {bi:>5} ---------------------",
            tname = "-------------------",
            tacc = "----------",
            qn = "--------------------",
            qacc = "----------",
            af = "-------",
            at = "-------",
            ef = "-------",
            et = "-------",
            sl = "-------",
            strand = "------",
            ev = "---------",
            sc = "------",
            bi = "-----",
            tnw = namew - 1,
            qnw = qw,
            taccw = taccw,
            qaccw = qaccw,
            posw = posw,
        )
        .unwrap();
    }

    for hit in &th.hits {
        if hit.flags & P7_IS_REPORTED == 0 {
            continue;
        }
        let best = match hit.dcl.first() {
            Some(d) => d,
            None => continue,
        };
        let (hmmfrom, hmm_to, alifrom, ali_to) = if let Some(ref ad) = best.ad {
            (ad.hmmfrom, ad.hmmto, ad.sqfrom, ad.sqto)
        } else {
            (0, 0, 0, 0)
        };
        // C passes the 6-char literal "   +  " / "   -  " to %6s.
        let strand_field = if best.iali <= best.jali {
            "   +  "
        } else {
            "   -  "
        };
        let acc_s = hit.acc.as_str();
        let acc_display: &str = if acc_s.is_empty() { "-" } else { acc_s };
        let qacc_display = qacc.filter(|s| !s.is_empty()).unwrap_or("-");
        let desc_display = if hit.desc.is_empty() {
            "-"
        } else {
            hit.desc.as_str()
        };
        let ev_str = fmt_evalue(evalue_scale * c_exp_f64(hit.lnp));
        // C p7_tophits.c:1649 long_targets row:
        //   "%-*s %-*s %-*s %-*s %7d %7d %*PRId64 x5 %6s %9.2g %6.1f %5.1f  %s"
        writeln!(
            f,
            "{tname:<namew$} {tacc:<taccw$} {qn:<qw$} {qacc:<qaccw$} {hf:>7} {ht:>7} {af:>posw$} {at:>posw$} {ef:>posw$} {et:>posw$} {sl:>posw$} {strand} {ev:>9} {sc} {bi}  {desc}",
            tname = hit.name,
            tacc = acc_display,
            qn = qname,
            qacc = qacc_display,
            hf = hmmfrom,
            ht = hmm_to,
            af = alifrom,
            at = ali_to,
            ef = best.ienv,
            et = best.jenv,
            sl = hit.n,
            strand = strand_field,
            ev = ev_str,
            sc = tongs::output::fmt_score(hit.score),
            bi = tongs::output::fmt_bias(best.dombias),
            desc = desc_display,
            namew = namew,
            qw = qw,
            taccw = taccw,
            qaccw = qaccw,
            posw = posw,
        )
        .unwrap();
    }
    if !write_footer {
        return;
    }

    // Footer
    writeln!(f, "#").unwrap();
    writeln!(f, "# Program:         {}", program).unwrap();
    writeln!(f, "# Version:         3.4 (Aug 2023)").unwrap();
    writeln!(f, "# Pipeline mode:   {}", pipeline_mode).unwrap();
    writeln!(f, "# Query file:      {}", hmmfile.display()).unwrap();
    writeln!(f, "# Target file:     {}", seqdb.display()).unwrap();
    writeln!(f, "# Option settings: {} ", cmdline).unwrap();
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| String::new());
    writeln!(f, "# Current dir:     {}", cwd).unwrap();
    let date_str = tongs::output::format_hmmer_date(std::time::SystemTime::now());
    writeln!(f, "# Date:            {}", date_str).unwrap();
    writeln!(f, "# [ok]").unwrap();
}

/// Write C-nhmmscan-style Dfam tabular output (`--dfamtblout`).
///
/// This is the long-target/DNA table used by C `nhmmscan`, not the protein
/// Pfam-style two-section table used by `hmmscan --pfamtblout`.
pub(crate) fn write_nhmmer_dfamtblout<W: std::io::Write>(
    f: &mut W,
    qname: &str,
    qacc: Option<&str>,
    length_header: &str,
    th: &tongs::tophits::TopHits,
    evalue_scale: f64,
) {
    use tongs::tophits::P7_IS_REPORTED;

    let tnamew = th
        .hits
        .iter()
        .map(|h| h.name.len())
        .max()
        .unwrap_or(0)
        .max(20);
    let taccw = th
        .hits
        .iter()
        .map(|h| if h.acc.is_empty() { 1 } else { h.acc.len() })
        .max()
        .unwrap_or(0)
        .max(20);
    let qnamew = qname.len().max(20);
    let posw = th
        .hits
        .iter()
        .flat_map(|hit| {
            hit.dcl.iter().flat_map(|dom| {
                [
                    dom.iali.unsigned_abs() as usize,
                    dom.jali.unsigned_abs() as usize,
                    dom.ienv.unsigned_abs() as usize,
                    dom.jenv.unsigned_abs() as usize,
                    hit.n,
                ]
            })
        })
        .map(|pos| pos.to_string().len())
        .max()
        .unwrap_or(0)
        .max(7);
    let tname_hdrw = tnamew - 1;

    writeln!(f, "# hit scores").unwrap();
    writeln!(f, "# ----------").unwrap();
    writeln!(f, "#").unwrap();
    writeln!(
        f,
        "# {:<tname_hdrw$} {:<taccw$} {:<qnamew$} {:>6} {:>9} {:>5}  hmm-st  hmm-en {:>6} {:>posw$} {:>posw$} {:>posw$} {:>posw$} {:>posw$}   description of target",
        "target name",
        "acc",
        "query name",
        "bits",
        "  e-value",
        " bias",
        "strand",
        "ali-st",
        "ali-en",
        "env-st",
        "env-en",
        length_header
    )
    .unwrap();
    writeln!(
        f,
        "# {:<tname_hdrw$} {:<taccw$} {:<qnamew$} {:>6} {:>9} {:>5} ------- ------- {:>6} {:>posw$} {:>posw$} {:>posw$} {:>posw$} {:>posw$}   ---------------------",
        "-------------------",
        "-------------------",
        "-------------------",
        "------",
        "---------",
        "-----",
        "------",
        "-------",
        "-------",
        "-------",
        "-------",
        "-------"
    )
    .unwrap();

    for hit in &th.hits {
        if hit.flags & P7_IS_REPORTED == 0 {
            continue;
        }
        let Some(dom) = hit.dcl.first() else {
            continue;
        };
        let (hmmfrom, hmmto, alifrom, alito) = if let Some(ref ad) = dom.ad {
            (ad.hmmfrom, ad.hmmto, ad.sqfrom, ad.sqto)
        } else {
            (0, 0, dom.iali as usize, dom.jali as usize)
        };
        let strand = if dom.iali <= dom.jali {
            "   +  "
        } else {
            "   -  "
        };
        let acc = qacc
            .filter(|acc| !acc.is_empty())
            .or_else(|| (!hit.acc.is_empty()).then_some(hit.acc.as_str()))
            .unwrap_or("-");
        let desc = if hit.desc.is_empty() {
            "-"
        } else {
            hit.desc.as_str()
        };
        writeln!(
            f,
            "{:<tnamew$}  {:<taccw$} {:<qnamew$} {} {:>9} {} {:>7} {:>7} {} {:>posw$} {:>posw$} {:>posw$} {:>posw$} {:>posw$}   {}",
            hit.name,
            acc,
            qname,
            tongs::output::fmt_score(hit.score),
            fmt_evalue(evalue_scale * c_exp_f64(hit.lnp)),
            tongs::output::fmt_bias(dom.dombias),
            hmmfrom,
            hmmto,
            strand,
            alifrom,
            alito,
            dom.ienv,
            dom.jenv,
            hit.n,
            desc,
        )
        .unwrap();
    }
}

pub(super) fn search_dfam_query_accession(qacc: Option<&str>) -> &str {
    qacc.filter(|acc| !acc.is_empty()).unwrap_or("-")
}

/// Write nhmmer `--aliscoresout` rows.
///
/// C stores these values in `Domain::scores_per_pos`; keep nhmmer's side-channel
/// local by recomputing the same displayed-column contribution from the
/// alignment display and configured profile when the artifact is written.
pub(super) fn write_nhmmer_aliscoresout<W: std::io::Write>(
    f: &mut W,
    qname: &str,
    th: &tongs::tophits::TopHits,
    gm: &Profile,
    abc: &Alphabet,
) {
    use tongs::tophits::P7_IS_REPORTED;

    for hit in &th.hits {
        if hit.flags & P7_IS_REPORTED == 0 {
            continue;
        }
        let Some(dom) = hit.dcl.first() else {
            continue;
        };
        let Some(ad) = dom.ad.as_ref() else {
            continue;
        };
        write!(f, "{} {} {} {} :", qname, hit.name, dom.iali, dom.jali).unwrap();
        for score in nhmmer_alignment_scores_per_pos(ad, gm, abc) {
            match score {
                Some(score) => write!(f, " {:.3}", score).unwrap(),
                None => write!(f, " >").unwrap(),
            }
        }
        writeln!(f).unwrap();
    }
}

fn nhmmer_alignment_scores_per_pos(
    ad: &tongs::tophits::AliDisplay,
    gm: &Profile,
    abc: &Alphabet,
) -> Vec<Option<f32>> {
    use tongs::profile::{P7P_DD, P7P_DM, P7P_II, P7P_IM, P7P_MD, P7P_MI, P7P_MM};

    let model = ad.model.as_bytes();
    let aseq = ad.aseq.as_bytes();
    let n = model.len().min(aseq.len());
    let mut scores = vec![Some(0.0_f32); n];
    let mut j = ad.hmmfrom.saturating_sub(1);
    let mut k = 0usize;

    while k < n {
        if model[k] != b'.' && aseq[k] != b'-' {
            j += 1;
            let x = residue_code_for_alignment_score(abc, aseq[k]) as usize;
            let emission = gm.msc(j, x);
            let mm = if j == 1 { 0.0 } else { gm.tsc(j - 1, P7P_MM) };
            scores[k] = finite_aliscore_bits(emission + mm);
            k += 1;
        } else if model[k] == b'.' {
            scores[k] = None;
            let mut sc = gm.tsc(j, P7P_MI);
            k += 1;
            while k < n && model[k] == b'.' {
                scores[k] = None;
                sc += gm.tsc(j, P7P_II);
                k += 1;
            }
            sc += gm.tsc(j, P7P_IM) - gm.tsc(j, P7P_MM);
            scores[k - 1] = finite_aliscore_bits(sc);
        } else if aseq[k] == b'-' {
            scores[k] = None;
            let mut sc = gm.tsc(j, P7P_MD);
            j += 1;
            k += 1;
            while k < n && aseq[k] == b'-' {
                scores[k] = None;
                sc += gm.tsc(j, P7P_DD);
                j += 1;
                k += 1;
            }
            sc += gm.tsc(j, P7P_DM) - gm.tsc(j, P7P_MM);
            scores[k - 1] = finite_aliscore_bits(sc);
        } else {
            scores[k] = None;
            k += 1;
        }
    }

    scores
}

fn finite_aliscore_bits(score_nats: f32) -> Option<f32> {
    use tongs::util::cmath::ESL_CONST_LOG2R;

    if score_nats.is_finite() {
        Some(score_nats * ESL_CONST_LOG2R as f32)
    } else {
        None
    }
}

fn residue_code_for_alignment_score(abc: &Alphabet, residue: u8) -> u8 {
    let code = abc.digitize_symbol(residue);
    if (code as usize) < abc.kp {
        code
    } else {
        abc.unknown_code()
    }
}
