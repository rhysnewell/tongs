use std::io::Write;

use tongs::alphabet::Alphabet;
use tongs::tophits::{Domain, Hit, TopHits};
use tongs::util::cmath::c_exp_f64;

/// Write per-sequence tabular output (`--tblout`).
///
/// Mirrors the non-long_target branch of C `p7_tophits_TabularTargets`
/// (`hmmer/src/p7_tophits.c`): a header trio (column widths sized to the
/// longest target name / accession and the query name / accession,
/// minimums 20/10/20/10) followed by one row per reported hit with full
/// sequence and best-domain stats.
pub fn write_tblout<W: Write>(
    f: &mut W,
    qname: &str,
    qacc: Option<&str>,
    th: &TopHits,
    z: f64,
    show_header: bool,
) {
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
        .max(10);
    let qnamew = qname.len().max(20);
    let tname_hdrw = tnamew - 1;
    let qacc_s = qacc.filter(|s| !s.is_empty()).unwrap_or("-");
    let qaccw = qacc_s.len().max(10);

    if show_header {
        writeln!(
            f,
            "#{:>w$} {:>22} {:>22} {:>33}",
            "",
            "--- full sequence ----",
            "--- best 1 domain ----",
            "--- domain number estimation ----",
            w = tnamew + qnamew + taccw + qaccw + 2
        )
        .unwrap();
        writeln!(
            f,
            "#{:<tname_hdrw$} {:<taccw$} {:<qnamew$} {:<qaccw$} {:>9} {:>6} {:>5} {:>9} {:>6} {:>5} {:>5} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} description of target",
            " target name",
            "accession",
            "query name",
            "accession",
            "  E-value",
            " score",
            " bias",
            "  E-value",
            " score",
            " bias",
            "exp",
            "reg",
            "clu",
            " ov",
            "env",
            "dom",
            "rep",
            "inc"
        )
        .unwrap();
        writeln!(
            f,
            "#{:>tname_hdrw$} {:>taccw$} {:>qnamew$} {:>qaccw$} {:>9} {:>6} {:>5} {:>9} {:>6} {:>5} {:>5} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} ---------------------",
            "-------------------",
            "----------",
            "--------------------",
            "----------",
            "---------",
            "------",
            "-----",
            "---------",
            "------",
            "-----",
            "---",
            "---",
            "---",
            "---",
            "---",
            "---",
            "---",
            "---"
        )
        .unwrap();
    }

    for hit in &th.hits {
        if hit.flags & tongs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }
        let evalue = evalue_from_lnp(z, hit.lnp);
        let best_dom = best_domain(hit);
        let dom_evalue = best_dom
            .map(|d| evalue_from_lnp(z, d.lnp))
            .unwrap_or(evalue);
        let dom_score = best_dom.map(|d| d.bitscore).unwrap_or(hit.score);
        let dom_bias = best_dom.map(|d| d.dombias).unwrap_or(hit.bias);

        writeln!(
            f,
            "{:<tnamew$} {:<taccw$} {:<qnamew$} {:<qaccw$} {} {} {} {} {} {} {:>5} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {:>3} {}",
            hit.name,
            if hit.acc.is_empty() { "-" } else { &hit.acc },
            qname,
            qacc_s,
            tongs::output::fmt_evalue(evalue),
            tongs::output::fmt_score(hit.score),
            tongs::output::fmt_bias(hit.bias),
            tongs::output::fmt_evalue(dom_evalue),
            tongs::output::fmt_score(dom_score),
            tongs::output::fmt_bias(dom_bias),
            tongs::output::fmt_width5_1(hit.nexpected as f64),
            hit.nregions,
            hit.nclustered,
            hit.noverlaps,
            hit.nenvelopes,
            hit.ndom,
            hit.nreported,
            hit.nincluded,
            if hit.desc.is_empty() { "-" } else { &hit.desc },
        ).unwrap();
    }
}

pub fn best_domain(hit: &Hit) -> Option<&Domain> {
    // C p7_pipeline.c:1110 uses a strict `>` scan, so the FIRST domain with the
    // maximum bitscore wins on a tie. `Iterator::max_by` keeps the LAST max, so
    // fold manually replacing only on a strict increase.
    hit.dcl
        .iter()
        .fold(None, |best: Option<&Domain>, d| match best {
            Some(b) if b.bitscore >= d.bitscore => Some(b),
            _ => Some(d),
        })
}

#[inline]
pub(super) fn evalue_from_lnp(scale: f64, lnp: f64) -> f64 {
    scale * c_exp_f64(lnp)
}

pub(super) fn shown_hit_name(hit: &Hit, show_acc: bool) -> &str {
    if show_acc && !hit.acc.is_empty() {
        &hit.acc
    } else {
        &hit.name
    }
}

pub(super) fn truncate_for_textw(s: &str, width: usize) -> String {
    if width == 0 {
        return s.to_string();
    }
    s.chars().take(width).collect()
}

/// Write per-domain tabular output (`--domtblout`).
///
/// Mirrors C `p7_tophits_TabularDomains` (`hmmer/src/p7_tophits.c`). Emits one
/// row per reported domain with full-sequence stats plus per-domain c-Evalue,
/// i-Evalue, score, bias, and hmm/ali/env coordinate triples. Column widths
/// auto-fit target names and accessions, with 20/10 minimums.
pub fn write_domtblout<W: Write>(
    f: &mut W,
    qname: &str,
    qacc: Option<&str>,
    qlen: usize,
    th: &TopHits,
    z: f64,
    domz: f64,
    show_header: bool,
) {
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
        .max(10);
    let qnamew = qname.len().max(20);
    let tname_hdrw = tnamew - 1;
    let qacc_s = qacc.filter(|s| !s.is_empty()).unwrap_or("-");
    let qaccw = qacc_s.len().max(10);

    if show_header {
        writeln!(
            f,
            "#{:>w$} {:>22} {:>40} {:>11} {:>11} {:>11}",
            "",
            "--- full sequence ---",
            "-------------- this domain -------------",
            "hmm coord",
            "ali coord",
            "env coord",
            w = tnamew + qnamew + 15 + taccw + qaccw - 1
        )
        .unwrap();
        writeln!(
            f,
            "#{:<tname_hdrw$} {:<taccw$} {:>5} {:<qnamew$} {:<qaccw$} {:>5} {:>9} {:>6} {:>5} {:>3} {:>3} {:>9} {:>9} {:>6} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>4} description of target",
            " target name",
            "accession",
            "tlen",
            "query name",
            "accession",
            "qlen",
            "E-value",
            "score",
            "bias",
            "#",
            "of",
            "c-Evalue",
            "i-Evalue",
            "score",
            "bias",
            "from",
            "to",
            "from",
            "to",
            "from",
            "to",
            "acc"
        )
        .unwrap();
        writeln!(
            f,
            "#{:>tname_hdrw$} {:>taccw$} {:>5} {:>qnamew$} {:>qaccw$} {:>5} {:>9} {:>6} {:>5} {:>3} {:>3} {:>9} {:>9} {:>6} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>4} ---------------------",
            "-------------------",
            "----------",
            "-----",
            "--------------------",
            "----------",
            "-----",
            "---------",
            "------",
            "-----",
            "---",
            "---",
            "---------",
            "---------",
            "------",
            "-----",
            "-----",
            "-----",
            "-----",
            "-----",
            "-----",
            "-----",
            "----"
        )
        .unwrap();
    }

    for hit in &th.hits {
        if hit.flags & tongs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }
        let evalue = evalue_from_lnp(z, hit.lnp);

        let mut reported_idx = 0usize;
        for dom in &hit.dcl {
            if !dom.is_reported {
                continue;
            }
            reported_idx += 1;
            let c_evalue = evalue_from_lnp(domz, dom.lnp);
            let i_evalue = evalue_from_lnp(z, dom.lnp);
            let (hmmfrom, hmmto, acc) = if let Some(ref ad) = dom.ad {
                let acc = dom.oasc / (1.0 + (dom.jenv - dom.ienv).abs() as f32);
                (ad.hmmfrom, ad.hmmto, acc)
            } else {
                (1, qlen, 0.0)
            };

            writeln!(
            f,
                "{:<tnamew$} {:<taccw$} {:>5} {:<qnamew$} {:<qaccw$} {:>5} {} {} {} {:>3} {:>3} {} {} {} {} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {} {}",
                hit.name,
                if hit.acc.is_empty() { "-" } else { &hit.acc },
                hit.n,
                qname,
                qacc_s,
                qlen,
                tongs::output::fmt_evalue(evalue),
                tongs::output::fmt_score(hit.score),
                tongs::output::fmt_bias(hit.bias),
                reported_idx,
                hit.nreported,
                tongs::output::fmt_evalue(c_evalue),
                tongs::output::fmt_evalue(i_evalue),
                tongs::output::fmt_score(dom.bitscore),
                tongs::output::fmt_bias(dom.dombias),
                hmmfrom,
                hmmto,
                dom.iali,
                dom.jali,
                dom.ienv,
                dom.jenv,
                tongs::output::fmt_width4_2(acc as f64),
                if hit.desc.is_empty() { "-" } else { &hit.desc },
            ).unwrap();
        }
    }
}

pub fn write_table_footer<W: Write>(
    f: &mut W,
    program: &str,
    pipeline_mode: &str,
    query_file: &std::path::Path,
    target_file: &std::path::Path,
    cmdline: &str,
) {
    writeln!(f, "#").unwrap();
    writeln!(f, "# Program:         {}", program).unwrap();
    writeln!(f, "# Version:         3.4 (Aug 2023)").unwrap();
    writeln!(f, "# Pipeline mode:   {}", pipeline_mode).unwrap();
    writeln!(f, "# Query file:      {}", query_file.display()).unwrap();
    writeln!(f, "# Target file:     {}", target_file.display()).unwrap();
    writeln!(f, "# Option settings: {} ", cmdline).unwrap();
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| String::new());
    writeln!(f, "# Current dir:     {}", cwd).unwrap();
    writeln!(
        f,
        "# Date:            {}",
        tongs::output::format_hmmer_date(std::time::SystemTime::now())
    )
    .unwrap();
    writeln!(f, "# [ok]").unwrap();
}

pub(super) fn normalize_hmmsearch_table_cmdline(cmdline: &str) -> String {
    let mut tokens = cmdline.split_whitespace();
    match (tokens.next(), tokens.next()) {
        (Some(_wrapper), Some("hmmsearch" | "search")) => {
            let rest = tokens.collect::<Vec<_>>();
            if rest.is_empty() {
                "hmmsearch".to_string()
            } else {
                format!("hmmsearch {}", rest.join(" "))
            }
        }
        _ => cmdline.to_string(),
    }
}

/// Build and write the `-A` multiple alignment of all included domains.
///
/// Faithful port of the `-A` block in `hmmer/src/hmmsearch.c:554-572` and
/// `hmmer/src/phmmer.c:620-638`: it calls `p7_tophits_Alignment(th, abc,
/// NULL,NULL, 0, p7_ALL_CONSENSUS_COLS, &msa)` to assemble a real
/// column-aligned MSA of every included domain (sequence names `name/from-to`,
/// `#=GS DE [subseq from]` lines, `#=GC RF`, `#=GR PP`, all aligned to the model
/// consensus columns), sets `#=GF ID/AC/DE/AU`, then writes Stockholm
/// (`textw > 0`) or Pfam (`textw == 0`). Returns the number of sequences in the
/// MSA when one was produced (so the caller can print the matching
/// `# Alignment of N hits ... saved to:` line), or `None` if no hits satisfy
/// the inclusion thresholds.
///
/// `name`/`acc`/`desc` are the per-tool GF annotation (hmmsearch: model
/// name/acc/desc; phmmer: the model/query name plus optional query acc/desc),
/// and `author` is e.g. `"hmmsearch (HMMER 3.4)"` / `"phmmer (HMMER 3.4)"`.
///
/// Reuses the same `tophits::included_alignment` + Stockholm writer that
/// `jackhmmer` uses. As in C, `textw > 0` writes Stockholm in 200-column
/// blocks and `--notextw` (`textw == 0`) writes one unwrapped Pfam block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_ali_output<W: Write>(
    f: &mut W,
    abc: &Alphabet,
    model_len: usize,
    name: &str,
    acc: Option<&str>,
    desc: Option<&str>,
    author: &str,
    th: &TopHits,
    textw: usize,
) -> Option<usize> {
    // p7_ALL_CONSENSUS_COLS (hmmsearch.c:559, phmmer.c:623).
    let mut msa = tongs::tophits::included_alignment(th, abc, model_len, None, true, name)?;
    msa.acc = acc.filter(|s| !s.is_empty()).map(|s| s.to_string());
    msa.desc = desc.filter(|s| !s.is_empty()).map(|s| s.to_string());
    msa.author = Some(author.to_string());
    let nseq = msa.nseq;
    crate::subcmd::jackhmmer::write_tophits_alignment_msa_stockholm(f, &msa, textw);
    Some(nseq)
}
