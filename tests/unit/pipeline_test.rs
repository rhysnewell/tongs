use super::*;
use crate::alphabet::Alphabet;
use crate::bg::Bg;
use crate::profile::{profile_config, P7_LOCAL};
use crate::sequence::Sequence;
use crate::simd::oprofile::OProfile;
use crate::tophits::TopHits;
use std::path::Path;

#[test]
fn target_reportable_uses_explicit_z_before_admitting_hit() {
    let mut pli = Pipeline::new();
    pli.z = 1_000.0;
    pli.z_setby = ZSetBy::Option;
    pli.e_value_threshold = 1.0;

    assert!(!pli.target_reportable(10.0, (0.01_f64).ln()));
    assert!(pli.target_reportable(10.0, (0.0001_f64).ln()));
}

#[test]
fn target_reportable_long_target_uses_prescaled_pvalue() {
    let mut pli = Pipeline::new();
    pli.long_target = true;
    pli.z = 1_000.0;
    pli.z_setby = ZSetBy::Option;
    pli.e_value_threshold = 1.0;

    assert!(pli.target_reportable(10.0, (0.01_f64).ln()));
}

/// Smoke test: pipeline produces at least one positive-score hit on a perfectly matching dsq.
#[test]
fn test_pipeline_finds_hit() {
    let hmm = crate::hmmfile::read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/20aa.hmm"
    )))
    .unwrap()
    .into_iter()
    .next()
    .unwrap();
    let abc = Alphabet::new(hmm.abc_type);
    let mut bg = Bg::new(&abc);
    let mut gm = Profile::new(hmm.m, &abc);
    profile_config(&hmm, &bg, &mut gm, 20, P7_LOCAL);
    let mut om = OProfile::convert(&gm);

    let mut pli = Pipeline::new();
    pli.new_model(&gm);
    let mut th = TopHits::new();

    let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");
    let sq = Sequence {
        name: "test".to_string(),
        acc: String::new(),
        desc: String::new(),
        dsq,
        n: 20,
        l: 20,
        taxid: -1,
    };

    bg.set_length(sq.n);

    // For this short perfectly-matching sequence, use --max to bypass filters
    pli.do_max = true;
    let hit = pli.run(&mut gm, &mut om, &bg, &hmm, &sq, &mut th);
    assert!(hit, "Pipeline should find a hit for matching sequence");
    assert_eq!(th.hits.len(), 1);
    assert!(th.hits[0].score > 0.0);
}

/// Regression test: ld > l (envelopes overlap or extend) must not produce NaN/+inf.
#[test]
fn reconstruction_score_allows_domain_coverage_to_exceed_sequence_length() {
    let score = reconstruction_score_nats(12.0, 100, 140);
    let expected = 12.0_f64 + (100_i64 - 140_i64) as f64 * c_log_f64(100.0_f64 / 103.0_f64);
    assert!((score as f64 - expected).abs() < 1.0e-6);
    assert!(score.is_finite());
    assert!(score > 12.0);
}
