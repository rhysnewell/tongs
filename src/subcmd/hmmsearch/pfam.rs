use std::io::Write;

use tongs::pipeline::Pipeline;
use tongs::tophits::{Domain, Hit, TopHits};

use super::tables::evalue_from_lnp;

pub fn write_pfamtblout<W: Write>(
    f: &mut W,
    _qname: &str,
    _qacc: Option<&str>,
    qlen: usize,
    th: &TopHits,
    z: f64,
    _domz: f64,
) {
    let pli = Pipeline::new();
    write_pfamtblout_with_pipeline(f, _qname, _qacc, qlen, th, &pli, z, _domz);
}

pub(crate) fn write_pfamtblout_with_pipeline<W: Write>(
    f: &mut W,
    _qname: &str,
    _qacc: Option<&str>,
    qlen: usize,
    th: &TopHits,
    pli: &Pipeline,
    z: f64,
    _domz: f64,
) {
    let tnamew = th
        .hits
        .iter()
        .map(|h| h.name.len())
        .max()
        .unwrap_or(0)
        .max(20);

    writeln!(f, "# Sequence scores").unwrap();
    writeln!(f, "# ---------------").unwrap();
    writeln!(f, "#").unwrap();
    writeln!(
        f,
        "# {:<tname_hdrw$} {:>6} {:>9} {:>3} {:>5} {:>5}    description",
        "name",
        " bits",
        "  E-value",
        "n",
        "exp",
        " bias",
        tname_hdrw = tnamew - 1,
    )
    .unwrap();
    writeln!(
        f,
        "# {:>tname_hdrw$} {:>6} {:>9} {:>3} {:>5} {:>5}    ---------------------",
        "-------------------",
        "------",
        "---------",
        "---",
        "-----",
        "-----",
        tname_hdrw = tnamew - 1,
    )
    .unwrap();

    for hit in &th.hits {
        if hit.flags & tongs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }
        let evalue = evalue_from_lnp(z, hit.lnp);
        writeln!(
            f,
            "{:<tnamew$}  {} {:>9} {:>3} {:>5} {}    {}",
            hit.name,
            tongs::output::fmt_score(hit.score),
            tongs::output::fmt_evalue(evalue),
            hit.ndom,
            tongs::output::fmt_width5_1(hit.nexpected as f64),
            tongs::output::fmt_bias(hit.bias),
            if hit.desc.is_empty() { "-" } else { &hit.desc },
        )
        .unwrap();
    }

    writeln!(f).unwrap();
    writeln!(f, "# Domain scores").unwrap();
    writeln!(f, "# -------------").unwrap();
    writeln!(f, "#").unwrap();
    writeln!(
        f,
        "# {:<tname_hdrw$} {:>6} {:>9} {:>5} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}     description",
        " name",
        "bits",
        "E-value",
        "hit",
        "bias",
        "env-st",
        "env-en",
        "ali-st",
        "ali-en",
        "hmm-st",
        "hmm-en",
        tname_hdrw = tnamew - 1,
    )
    .unwrap();
    writeln!(
        f,
        "# {:>tname_hdrw$} {:>6} {:>9} {:>5} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}      ---------------------",
        "-------------------",
        "------",
        "---------",
        "-----",
        "-----",
        "------",
        "------",
        "------",
        "------",
        "------",
        "------",
        tname_hdrw = tnamew - 1,
    )
    .unwrap();

    // Build one pseudo-hit per *reported* domain, recording `ndom_reported`:
    // the 1-based ordinal of this domain among the reported domains of its
    // parent hit (C p7_tophits.c:1897-1906, `domhit->ndom = ndomReported`).
    // This is the value printed in the `hit` column — NOT the absolute dcl
    // index (which would skip over any unreported intervening domains).
    let mut reported_domains = Vec::new();
    for (hit_idx, hit) in th.hits.iter().enumerate() {
        if hit.flags & tongs::tophits::P7_IS_REPORTED == 0 {
            continue;
        }
        let mut ndom_reported = 0usize;
        for dom in hit.dcl.iter() {
            if dom.is_reported {
                ndom_reported += 1;
                reported_domains.push((hit_idx, ndom_reported, hit, dom));
            }
        }
    }
    reported_domains.sort_by(|a, b| compare_pfamtblout_domain_pseudo_hits(a, b, pli));

    for (_hit_idx, ndom_reported, hit, dom) in reported_domains {
        let i_evalue = evalue_from_lnp(z, dom.lnp);
        let (hmmfrom, hmmto) = if let Some(ref ad) = dom.ad {
            (ad.hmmfrom, ad.hmmto)
        } else {
            (1, qlen)
        };
        writeln!(
            f,
            "{:<tnamew$}  {} {:>9} {:>5} {} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}     {}",
            hit.name,
            tongs::output::fmt_score(dom.bitscore),
            tongs::output::fmt_evalue(i_evalue),
            ndom_reported,
            tongs::output::fmt_bias(dom.dombias),
            dom.ienv,
            dom.jenv,
            dom.iali,
            dom.jali,
            hmmfrom,
            hmmto,
            if hit.desc.is_empty() { "-" } else { &hit.desc },
        )
        .unwrap();
    }
}

fn compare_pfamtblout_domain_pseudo_hits(
    a: &(usize, usize, &Hit, &Domain),
    b: &(usize, usize, &Hit, &Domain),
    pli: &Pipeline,
) -> std::cmp::Ordering {
    // C p7_tophits.c:1910 keys the pfamtblout domain pseudo-hits purely on
    // pli->inc_by_E: `sortkey = inc_by_E ? -lnP : bitscore`, sorted descending.
    // Expressed as an ascending key (matching this module's convention): use
    // `lnp` when inc_by_e (asc lnp == desc -lnP) else `-bitscore` (asc -bits ==
    // desc bits). This differs from the shared `pli.hit_sortkey`, which keys on
    // the broader `score_sort_active()` predicate; we key locally on inc_by_e to
    // avoid touching pipeline.rs.
    let pfam_key = |bitscore: f32, lnp: f64| -> f64 {
        if pli.inc_by_e {
            lnp
        } else {
            -(bitscore as f64)
        }
    };
    let a_key = pfam_key(a.3.bitscore, a.3.lnp);
    let b_key = pfam_key(b.3.bitscore, b.3.lnp);
    a_key
        .partial_cmp(&b_key)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.2.name.cmp(&b.2.name))
        .then_with(|| {
            let a_dir = if a.3.iali < a.3.jali { 1 } else { -1 };
            let b_dir = if b.3.iali < b.3.jali { 1 } else { -1 };
            if a_dir != b_dir {
                b_dir.cmp(&a_dir)
            } else {
                a.3.iali.cmp(&b.3.iali)
            }
        })
}
