use super::*;
use tongs::tophits::{P7_IS_INCLUDED, P7_IS_REPORTED};

fn domain(bitscore: f32, reported: bool) -> Domain {
    Domain {
        iali: 1,
        jali: 10,
        ienv: 1,
        jenv: 10,
        bitscore,
        lnp: -1.0,
        dombias: 0.0,
        oasc: 0.0,
        envsc: 0.0,
        domcorrection: 0.0,
        is_reported: reported,
        is_included: reported,
        ad: None,
    }
}

fn domain_with_alignment(bitscore: f32, reported: bool, iali: i64, jali: i64) -> Domain {
    let mut dom = domain(bitscore, reported);
    dom.iali = iali;
    dom.jali = jali;
    dom.ienv = iali.min(jali);
    dom.jenv = iali.max(jali);
    dom
}

fn hit_with_name(name: &str, domains: Vec<Domain>, flags: u32) -> Hit {
    Hit {
        name: name.to_string(),
        acc: String::new(),
        desc: String::new(),
        n: 100,
        sortkey: -10.0,
        score: 10.0,
        bias: 0.0,
        pre_score: 10.0,
        sum_score: 10.0,
        lnp: -1.0,
        pre_lnp: -1.0,
        sum_lnp: -1.0,
        nexpected: 1.0,
        nregions: 0,
        nclustered: 0,
        noverlaps: 0,
        nenvelopes: domains.len(),
        ndom: domains.len(),
        nreported: domains.iter().filter(|dom| dom.is_reported).count(),
        nincluded: domains.iter().filter(|dom| dom.is_included).count(),
        dcl: domains,
        flags,
        seqidx: 0,
        subseq_start: 0,
    }
}

fn hit(domains: Vec<Domain>) -> Hit {
    hit_with_name("target", domains, P7_IS_REPORTED | P7_IS_INCLUDED)
}

#[test]
fn best_domain_summary_uses_true_best_domain_even_if_unreported() {
    let hit = hit(vec![domain(40.0, false), domain(12.0, true)]);

    let best = best_domain(&hit).expect("domain");

    assert_eq!(best.bitscore, 40.0);
}

#[test]
fn tblout_width_includes_unreported_hits() {
    let mut th = TopHits::new();
    th.hits.push(hit_with_name(
        "reported",
        vec![domain(12.0, true)],
        P7_IS_REPORTED | P7_IS_INCLUDED,
    ));
    th.hits.push(hit_with_name(
        "unreported-target-name-widens-columns",
        vec![domain(10.0, false)],
        0,
    ));

    let mut out = Vec::new();
    write_tblout(&mut out, "query", None, &th, 1.0, true);
    let out = String::from_utf8(out).unwrap();

    assert!(out.contains("reported                              -"));
}

#[test]
fn pfamtblout_uses_dynamic_name_width_and_ndom_count() {
    let mut th = TopHits::new();
    th.hits.push(hit_with_name(
        "long-reported-target-name",
        vec![domain(20.0, true), domain(10.0, false)],
        P7_IS_REPORTED | P7_IS_INCLUDED,
    ));

    let mut out = Vec::new();
    let pli = Pipeline::new();
    write_pfamtblout_with_pipeline(&mut out, "query", None, 42, &th, &pli, 1.0, 1.0);
    let out = String::from_utf8(out).unwrap();

    assert!(out.contains("long-reported-target-name    10.0"));
    assert!(out.contains("long-reported-target-name    20.0"));
    assert!(out.lines().any(|line| {
        line.starts_with("long-reported-target-name") && line.split_whitespace().nth(3) == Some("2")
    }));
}

#[test]
fn pfamtblout_domain_pseudo_hits_use_c_tie_breakers() {
    let mut th = TopHits::new();
    th.hits.push(hit_with_name(
        "target",
        vec![
            domain_with_alignment(20.0, true, 30, 20),
            domain_with_alignment(20.0, true, 5, 15),
        ],
        P7_IS_REPORTED | P7_IS_INCLUDED,
    ));

    let mut out = Vec::new();
    let pli = Pipeline::new();
    write_pfamtblout_with_pipeline(&mut out, "query", None, 42, &th, &pli, 1.0, 1.0);
    let out = String::from_utf8(out).unwrap();
    let rows: Vec<&str> = out
        .lines()
        .filter(|line| line.starts_with("target"))
        .collect();

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[1].split_whitespace().nth(3), Some("2"));
    assert_eq!(rows[2].split_whitespace().nth(3), Some("1"));
}

#[test]
fn pfamtblout_domain_pseudo_hits_use_live_score_sort_mode() {
    let mut low_lnp_high_score = domain(50.0, true);
    low_lnp_high_score.lnp = -1.0;
    let mut high_lnp_low_score = domain(10.0, true);
    high_lnp_low_score.lnp = -100.0;

    let mut th = TopHits::new();
    th.hits.push(hit_with_name(
        "target",
        vec![low_lnp_high_score, high_lnp_low_score],
        P7_IS_REPORTED | P7_IS_INCLUDED,
    ));

    let mut out = Vec::new();
    let mut pli = Pipeline::new();
    pli.inc_t = Some(20.0);
    pli.inc_by_e = false;
    write_pfamtblout_with_pipeline(&mut out, "query", None, 42, &th, &pli, 1.0, 1.0);
    let out = String::from_utf8(out).unwrap();
    let rows: Vec<&str> = out
        .lines()
        .filter(|line| line.starts_with("target"))
        .collect();

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[1].split_whitespace().nth(1), Some("50.0"));
    assert_eq!(rows[2].split_whitespace().nth(1), Some("10.0"));
}

#[test]
fn hmmsearch_accepts_negative_space_separated_f_values() {
    // C --F1/--F2/--F3 are eslARG_REAL with no range, so C accepts the
    // space-separated negative form. allow_hyphen_values keeps clap from
    // treating "-0.5" as an unknown flag.
    let args = Args::try_parse_from([
        "hmmsearch",
        "--F1",
        "-0.5",
        "--F2",
        "-1e-3",
        "--F3",
        "-2",
        "model.hmm",
        "targets.fa",
    ])
    .unwrap();
    assert_eq!(args.f1, -0.5);
    assert_eq!(args.f2, -1e-3);
    assert_eq!(args.f3, -2.0);
}

#[test]
fn hmmsearch_threshold_preserves_f64_precision() {
    // C's -T/--domT/--incT/--incdomT are eslARG_REAL (double). Rust stores
    // them as f64 too, so a value with >7 significant digits must NOT be
    // rounded to f32. 12.3456789 is not exactly representable in f32; the
    // f32-rounded value differs from the exact f64 parse, so this asserts
    // the parsed threshold kept full double precision (matching C).
    let val = "12.3456789";
    let exact: f64 = val.parse().unwrap();
    let args = Args::try_parse_from([
        "hmmsearch",
        "-T",
        val,
        "--domT",
        val,
        "--incT",
        val,
        "--incdomT",
        val,
        "model.hmm",
        "targets.fa",
    ])
    .unwrap();
    // Round-trip at f64: the parsed values equal the exact f64 parse.
    assert_eq!(args.score_threshold, Some(exact));
    assert_eq!(args.dom_t, Some(exact));
    assert_eq!(args.inc_t, Some(exact));
    assert_eq!(args.inc_dom_t, Some(exact));
    // And the f64 parse is strictly more precise than the f32 round-trip
    // would have been (proving precision is not lost to f32).
    assert_ne!(exact, (val.parse::<f32>().unwrap()) as f64);
}
