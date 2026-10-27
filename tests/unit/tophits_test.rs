use super::*;
use crate::pipeline::Pipeline;

fn test_domain(bitscore: f32, lnp: f64) -> Domain {
    Domain {
        iali: 1,
        jali: 10,
        ienv: 1,
        jenv: 10,
        bitscore,
        lnp,
        dombias: 0.0,
        oasc: 0.0,
        envsc: 0.0,
        domcorrection: 0.0,
        is_reported: true,
        is_included: true,
        ad: None,
    }
}

fn test_hit(name: &str, score: f32, lnp: f64, domains: Vec<Domain>) -> Hit {
    Hit {
        name: name.to_string(),
        acc: String::new(),
        desc: String::new(),
        n: 100,
        sortkey: 123.0,
        score,
        bias: 0.0,
        pre_score: score,
        sum_score: score,
        lnp,
        pre_lnp: lnp,
        sum_lnp: lnp,
        nexpected: 1.0,
        nregions: 0,
        nclustered: 0,
        noverlaps: 0,
        nenvelopes: domains.len(),
        ndom: domains.len(),
        nreported: 9,
        nincluded: 9,
        dcl: domains,
        flags: P7_IS_REPORTED | P7_IS_INCLUDED | P7_IS_DROPPED,
        seqidx: 0,
        subseq_start: 0,
    }
}

#[test]
fn threshold_clears_previous_state_and_gates_domains_by_parent() {
    let mut pli = Pipeline::new();
    pli.e_value_threshold = 1.0;
    pli.inc_e = 0.01;
    pli.dom_e_value_threshold = 1.0;
    pli.inc_dome = 1.0;

    let mut th = TopHits::new();
    th.hits.push(test_hit(
        "included",
        20.0,
        -8.0,
        vec![test_domain(10.0, -8.0)],
    ));
    th.hits.push(test_hit(
        "reported-only",
        20.0,
        -2.0,
        vec![test_domain(10.0, -8.0)],
    ));
    th.hits.push(test_hit(
        "dropped",
        20.0,
        1.0,
        vec![test_domain(10.0, -8.0)],
    ));

    th.threshold(&pli, 1.0, 1.0);

    assert_eq!(th.nreported, 2);
    assert_eq!(th.nincluded, 1);

    assert!(th.hits[0].flags & P7_IS_REPORTED != 0);
    assert!(th.hits[0].flags & P7_IS_INCLUDED != 0);
    assert_eq!(th.hits[0].nreported, 1);
    assert_eq!(th.hits[0].nincluded, 1);
    assert!(th.hits[0].dcl[0].is_reported);
    assert!(th.hits[0].dcl[0].is_included);

    assert!(th.hits[1].flags & P7_IS_REPORTED != 0);
    assert_eq!(th.hits[1].flags & P7_IS_INCLUDED, 0);
    assert_eq!(th.hits[1].nreported, 1);
    assert_eq!(th.hits[1].nincluded, 0);
    assert!(th.hits[1].dcl[0].is_reported);
    assert!(!th.hits[1].dcl[0].is_included);

    assert_eq!(th.hits[2].flags & P7_IS_REPORTED, 0);
    assert_eq!(th.hits[2].flags & P7_IS_INCLUDED, 0);
    assert!(th.hits[2].flags & P7_IS_DROPPED != 0);
    assert_eq!(th.hits[2].nreported, 0);
    assert_eq!(th.hits[2].nincluded, 0);
    assert!(!th.hits[2].dcl[0].is_reported);
    assert!(!th.hits[2].dcl[0].is_included);
}

#[test]
fn domain_inclusion_is_independent_from_domain_reporting_threshold() {
    let mut pli = Pipeline::new();
    pli.e_value_threshold = 1.0;
    pli.inc_e = 1.0;
    pli.dom_e_value_threshold = 0.001;
    pli.inc_dome = 0.1;

    let mut th = TopHits::new();
    th.hits.push(test_hit(
        "included-target",
        20.0,
        0.01_f64.ln(),
        vec![test_domain(10.0, 0.01_f64.ln())],
    ));

    th.threshold(&pli, 1.0, 1.0);

    assert_eq!(th.nreported, 1);
    assert_eq!(th.nincluded, 1);
    assert_eq!(th.hits[0].nreported, 0);
    assert_eq!(th.hits[0].nincluded, 1);
    assert!(!th.hits[0].dcl[0].is_reported);
    assert!(th.hits[0].dcl[0].is_included);
}

#[test]
fn workaround_bug_h74_suppresses_lower_scoring_duplicate_domain() {
    let mut pli = Pipeline::new();
    pli.e_value_threshold = 1.0;
    pli.inc_e = 1.0;
    pli.dom_e_value_threshold = 1.0;
    pli.inc_dome = 1.0;

    // Two domains with identical iali/jali (the h74 collision); the hit is
    // flagged with overlapping envelopes (noverlaps != 0).
    let mut dom_hi = test_domain(20.0, 0.001_f64.ln());
    dom_hi.iali = 5;
    dom_hi.jali = 50;
    let mut dom_lo = test_domain(8.0, 0.01_f64.ln());
    dom_lo.iali = 5;
    dom_lo.jali = 50;

    let mut hit = test_hit("dup", 25.0, 0.001_f64.ln(), vec![dom_hi, dom_lo]);
    hit.noverlaps = 1;

    let mut th = TopHits::new();
    th.hits.push(hit);

    th.threshold(&pli, 1.0, 1.0);

    // Both domains pass the thresholds, but h74 suppresses the lower-scoring
    // duplicate (dcl[1], bitscore 8.0). Only the higher-scoring one remains.
    assert!(th.hits[0].dcl[0].is_reported);
    assert!(th.hits[0].dcl[0].is_included);
    assert!(!th.hits[0].dcl[1].is_reported);
    assert!(!th.hits[0].dcl[1].is_included);
    assert_eq!(th.hits[0].nreported, 1);
    assert_eq!(th.hits[0].nincluded, 1);
    // Target-level counts are unchanged by the workaround.
    assert_eq!(th.nreported, 1);
    assert_eq!(th.nincluded, 1);
}

#[test]
fn workaround_bug_h74_ignored_without_overlap_flag() {
    let mut pli = Pipeline::new();
    pli.e_value_threshold = 1.0;
    pli.inc_e = 1.0;
    pli.dom_e_value_threshold = 1.0;
    pli.inc_dome = 1.0;

    // Identical iali/jali but noverlaps == 0: workaround must NOT fire.
    let mut dom_a = test_domain(20.0, 0.001_f64.ln());
    dom_a.iali = 5;
    dom_a.jali = 50;
    let mut dom_b = test_domain(8.0, 0.01_f64.ln());
    dom_b.iali = 5;
    dom_b.jali = 50;

    let mut hit = test_hit("nodup", 25.0, 0.001_f64.ln(), vec![dom_a, dom_b]);
    hit.noverlaps = 0;

    let mut th = TopHits::new();
    th.hits.push(hit);

    th.threshold(&pli, 1.0, 1.0);

    assert!(th.hits[0].dcl[0].is_reported);
    assert!(th.hits[0].dcl[1].is_reported);
    assert_eq!(th.hits[0].nreported, 2);
    assert_eq!(th.hits[0].nincluded, 2);
}

#[test]
fn seqidx_alipos_sort_matches_c_longtarget_order() {
    let mut plus_short = test_hit("z", 10.0, -1.0, vec![test_domain(10.0, -1.0)]);
    plus_short.seqidx = 1;
    plus_short.dcl[0].iali = 10;
    plus_short.dcl[0].jali = 20;
    let mut minus = test_hit("a", 10.0, -1.0, vec![test_domain(10.0, -1.0)]);
    minus.seqidx = 1;
    minus.dcl[0].iali = 25;
    minus.dcl[0].jali = 10;
    let mut plus_long = test_hit("b", 10.0, -1.0, vec![test_domain(10.0, -1.0)]);
    plus_long.seqidx = 1;
    plus_long.dcl[0].iali = 10;
    plus_long.dcl[0].jali = 30;

    let mut th = TopHits::new();
    th.hits = vec![minus, plus_short, plus_long];
    th.sort_by_seqidx_and_alipos();

    assert_eq!(th.hits[0].name, "b");
    assert_eq!(th.hits[1].name, "z");
    assert_eq!(th.hits[2].name, "a");
}

#[test]
fn modelname_alipos_sort_groups_nhmmscan_models_like_c() {
    let mut model_b = test_hit("model-b", 10.0, -1.0, vec![test_domain(10.0, -1.0)]);
    model_b.dcl[0].iali = 5;
    model_b.dcl[0].jali = 20;
    let mut model_a_minus = test_hit("model-a", 10.0, -1.0, vec![test_domain(10.0, -1.0)]);
    model_a_minus.dcl[0].iali = 20;
    model_a_minus.dcl[0].jali = 5;
    let mut model_a_plus = test_hit("model-a", 10.0, -1.0, vec![test_domain(10.0, -1.0)]);
    model_a_plus.dcl[0].iali = 5;
    model_a_plus.dcl[0].jali = 20;

    let mut th = TopHits::new();
    th.hits = vec![model_b, model_a_minus, model_a_plus];
    th.sort_by_modelname_and_alipos();

    assert_eq!(th.hits[0].name, "model-a");
    assert!(th.hits[0].dcl[0].iali < th.hits[0].dcl[0].jali);
    assert_eq!(th.hits[1].name, "model-a");
    assert!(th.hits[1].dcl[0].iali > th.hits[1].dcl[0].jali);
    assert_eq!(th.hits[2].name, "model-b");
}

#[test]
fn inclusion_score_thresholds_use_negative_score_sortkeys_for_ascending_sort() {
    let mut pli = Pipeline::new();
    pli.inc_by_e = false;
    pli.inc_t = Some(0.0);

    let mut th = TopHits::new();
    th.hits
        .push(test_hit("low", 10.0, -20.0, vec![test_domain(10.0, -20.0)]));
    th.hits
        .push(test_hit("high", 50.0, -1.0, vec![test_domain(10.0, -1.0)]));

    th.threshold(&pli, 1.0, 1.0);
    assert_eq!(th.hits[0].name, "high");
    assert_eq!(th.hits[0].sortkey, -50.0);
    assert_eq!(th.hits[1].name, "low");
    assert_eq!(th.hits[1].sortkey, -10.0);
}

#[test]
fn long_target_threshold_uses_caller_supplied_z() {
    let mut pli = Pipeline::new();
    pli.long_target = true;
    pli.e_value_threshold = 0.2;
    pli.inc_e = 0.2;
    pli.dom_e_value_threshold = 0.2;
    pli.inc_dome = 0.2;

    let lnp = 0.1_f64.ln();
    let mut th = TopHits::new();
    th.hits
        .push(test_hit("nhmmer", 20.0, lnp, vec![test_domain(10.0, lnp)]));

    th.threshold(&pli, 1.0, 1.0);
    assert_eq!(th.nreported, 1);
    assert_eq!(th.nincluded, 1);
    assert_eq!(th.hits[0].nreported, 1);
    assert_eq!(th.hits[0].nincluded, 1);

    th.threshold(&pli, 10.0, 10.0);
    assert_eq!(th.nreported, 0);
    assert_eq!(th.nincluded, 0);
    assert_eq!(th.hits[0].nreported, 0);
    assert_eq!(th.hits[0].nincluded, 0);
}

#[test]
fn bit_cutoff_threshold_counts_preassigned_model_specific_flags() {
    let mut pli = Pipeline::new();
    pli.use_bit_cutoffs = crate::pipeline::BitCutoff::GA;
    pli.t = Some(100.0);
    pli.inc_t = Some(100.0);
    pli.dom_t = Some(100.0);
    pli.inc_dom_t = Some(100.0);
    pli.by_e = false;
    pli.inc_by_e = false;
    pli.dom_by_e = false;
    pli.incdom_by_e = false;

    let mut flagged = test_hit("kept", 10.0, -1.0, vec![test_domain(5.0, -1.0)]);
    flagged.flags = P7_IS_REPORTED | P7_IS_INCLUDED;
    flagged.dcl[0].is_reported = true;
    flagged.dcl[0].is_included = true;

    let mut unflagged = test_hit(
        "not-recomputed",
        200.0,
        -100.0,
        vec![test_domain(200.0, -100.0)],
    );
    unflagged.flags = 0;
    unflagged.dcl[0].is_reported = false;
    unflagged.dcl[0].is_included = false;

    let mut th = TopHits::new();
    th.hits.push(unflagged);
    th.hits.push(flagged);

    th.threshold(&pli, 1.0, 1.0);

    assert_eq!(th.nreported, 1);
    assert_eq!(th.nincluded, 1);
    assert_eq!(th.hits[0].name, "not-recomputed");
    assert_eq!(th.hits[0].flags & P7_IS_REPORTED, 0);
    assert_eq!(th.hits[0].nreported, 0);
    assert_eq!(th.hits[1].name, "kept");
    assert!(th.hits[1].flags & P7_IS_REPORTED != 0);
    assert!(th.hits[1].flags & P7_IS_INCLUDED != 0);
    assert_eq!(th.hits[1].nreported, 1);
    assert_eq!(th.hits[1].nincluded, 1);
}

#[test]
fn bit_cutoff_threshold_does_not_fall_back_to_evalues_when_no_hits_preassigned() {
    let mut pli = Pipeline::new();
    pli.use_bit_cutoffs = crate::pipeline::BitCutoff::GA;
    pli.e_value_threshold = 10.0;
    pli.inc_e = 10.0;

    let mut th = TopHits::new();
    let mut hit = test_hit(
        "low-evalue-but-below-cutoff",
        200.0,
        -100.0,
        vec![test_domain(200.0, -100.0)],
    );
    hit.flags = 0;
    hit.dcl[0].is_reported = false;
    hit.dcl[0].is_included = false;
    th.hits.push(hit);

    th.threshold(&pli, 1.0, 1.0);

    assert_eq!(th.nreported, 0);
    assert_eq!(th.nincluded, 0);
    assert_eq!(th.hits[0].flags & P7_IS_REPORTED, 0);
    assert_eq!(th.hits[0].flags & P7_IS_INCLUDED, 0);
    assert_eq!(th.hits[0].nreported, 0);
    assert_eq!(th.hits[0].nincluded, 0);
}
