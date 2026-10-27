use super::*;
use crate::alphabet::Alphabet;
use crate::bg::Bg;
use std::path::Path;

/// Smoke test: build an OProfile from a real test HMM and check basic invariants.
#[test]
fn test_oprofile_convert() {
    let hmm = crate::hmmfile::read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/20aa.hmm"
    )))
    .unwrap()
    .into_iter()
    .next()
    .unwrap();
    let abc = Alphabet::new(hmm.abc_type);
    let bg = Bg::new(&abc);
    let mut gm = Profile::new(hmm.m, &abc);
    profile_config(&hmm, &bg, &mut gm, 400, P7_LOCAL);

    let om = OProfile::convert(&gm);
    assert_eq!(om.m, 20);
    assert!(om.scale_b > 4.0 && om.scale_b < 5.0);
    assert_eq!(om.base_b, 190);
    assert!(om.bias_b > 0);
}

/// Validate the striped segment-length macro for byte-precision vectors.
#[test]
fn test_nqb() {
    assert_eq!(nqb(20), 2);
    assert_eq!(nqb(16), 2);
    assert_eq!(nqb(17), 2);
    assert_eq!(nqb(32), 2);
    assert_eq!(nqb(33), 3);
    assert_eq!(nqb(100), 7);
}
