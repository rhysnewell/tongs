use super::*;
use crate::alphabet::Alphabet;
use crate::bg::Bg;
use crate::dp::generic_viterbi::g_viterbi;
use crate::dp::gmx::Gmx;
use std::path::Path;

/// Sanity check: a Viterbi traceback over the 20aa toy HMM produces a
/// well-formed S...T trace covering plausible model and sequence coords.
#[test]
fn test_traceback_basic() {
    crate::logsum::p7_flogsuminit();
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
    profile_config(&hmm, &bg, &mut gm, 20, P7_LOCAL);

    let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");
    let l = dsq.len() - 2;

    let mut gx = Gmx::new(gm.m, l);
    g_viterbi(&dsq, l, &gm, &mut gx);

    let tr = g_trace(&dsq, l, &gm, &gx);
    assert!(tr.n > 0, "Trace should be non-empty");

    // Should contain S...B...M...E...C...T
    assert_eq!(tr.st[0], State::S);
    assert_eq!(*tr.st.last().unwrap(), State::T);

    // Check domain coords
    let coords = tr.domain_coords();
    assert!(coords.is_some());
    let (hf, ht, sf, st) = coords.unwrap();
    assert!(hf >= 1 && ht <= 20);
    assert!(sf >= 1 && st <= 20);
}

/// `alignment_display` produces nonempty model/aseq strings of equal length
/// for a basic 20aa Viterbi traceback.
#[test]
fn test_alignment_display() {
    crate::logsum::p7_flogsuminit();
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
    profile_config(&hmm, &bg, &mut gm, 20, P7_LOCAL);

    let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWY");
    let l = dsq.len() - 2;

    let mut gx = Gmx::new(gm.m, l);
    g_viterbi(&dsq, l, &gm, &mut gx);

    let tr = g_trace(&dsq, l, &gm, &gx);
    let ad = alignment_display(&tr, &dsq, &hmm, &abc);
    assert!(ad.is_some());
    let ad = ad.unwrap();
    assert!(!ad.model.is_empty());
    assert!(!ad.aseq.is_empty());
    assert_eq!(ad.model.len(), ad.aseq.len());
}
