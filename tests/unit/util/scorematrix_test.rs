use super::*;
use crate::bg::Bg;

/// Reference values produced by the C library itself, at
/// `EddyRivasLab/hmmer@9acd8b6758a0` + `EddyRivasLab/easel@07ca83b`, via
/// `esl_scorematrix_Set` -> `esl_scorematrix_ProbifyGivenBG` ->
/// `esl_scorematrix_JointToConditionalOnQuery` with `p7_bg_Create()`
/// frequencies. Regenerate with `examples/dump_scorematrix.rs` and the
/// matching C driver; every value below is required to match C bit-for-bit,
/// not approximately.
fn conditionals(abc_type: AlphabetType, mxname: &str) -> (f64, Vec<Vec<f64>>, ScoreMatrix) {
    let abc = Alphabet::new(abc_type);
    let bg = Bg::new(&abc);
    let mut s = ScoreMatrix::create(&abc);
    s.set(mxname, &abc).expect("built-in should exist");
    let f: Vec<f64> = bg.f[..abc.k].iter().map(|&x| x as f64).collect();
    let (lambda, mut q) = s.probify_given_bg(&f, &f, &abc).expect("probify");
    joint_to_conditional_on_query(&abc, &mut q);
    (lambda, q, s)
}

#[test]
fn blosum62_matches_c_bitwise() {
    let (lambda, q, s) = conditionals(AlphabetType::Amino, "BLOSUM62");
    assert_eq!(s.k, 20);
    assert_eq!(s.kp, 29);
    assert_eq!(s.max(), 11);
    assert_eq!(s.min(), -4);
    assert_eq!(s.nc, 24);
    assert_eq!(s.outorder, "ARNDCQEGHILKMFPSTWYVBZX*");
    assert_eq!(lambda, 0.3179513041122926);
    // Row 0 (A), first three conditionals P(b|A).
    assert_eq!(q[0][0], 0.2781509053632306);
    assert_eq!(q[0][1], 0.015001883092402036);
    assert_eq!(q[0][2], 0.028042164632627264);
    // Row 26 is X, the fully degenerate amino code (Kp-3); it exercises
    // set_degenerate_probs over all 20 canonical residues.
    assert_eq!(q[26][0], 0.07962498007361424);
    assert_eq!(q[26][19], 0.02906496839179966);
}

#[test]
fn pam30_matches_c_bitwise() {
    let (lambda, _q, _s) = conditionals(AlphabetType::Amino, "PAM30");
    assert_eq!(lambda, 0.33737805564946816);
}

/// The nucleotide path the issue is about: DNA1 is the default built-in for
/// non-amino alphabets (p7_builder.c:299).
#[test]
fn dna1_matches_c_bitwise() {
    let (lambda, q, s) = conditionals(AlphabetType::Dna, "DNA1");
    assert_eq!(s.k, 4);
    assert_eq!(s.kp, 18);
    assert_eq!(s.max(), 46);
    assert_eq!(s.nc, 15);
    assert_eq!(s.outorder, "ACGTRYMKSWHBVDN");
    assert_eq!(lambda, 0.019946343577260536);
    assert_eq!(
        &q[0][..4],
        &[
            0.5685936949998399,
            0.13256631746409928,
            0.14941999376803045,
            0.14941999376803045
        ][..]
    );
    assert_eq!(
        &q[3][..4],
        &[
            0.14796749122354302,
            0.17706372356741848,
            0.13392244688996965,
            0.5410463383190688
        ][..]
    );
    // Row 15 is N (Kp-3), the fully degenerate nucleotide code.
    assert_eq!(
        &q[15][..4],
        &[
            0.24902562725605873,
            0.2428844666608442,
            0.25661975297719786,
            0.25147015310589926
        ][..]
    );
}

/// RNA uses the same DNA1 table and uniform background, so it must land on
/// exactly the same numbers as DNA.
#[test]
fn rna_matches_dna_exactly() {
    let (ld, qd, _) = conditionals(AlphabetType::Dna, "DNA1");
    let (lr, qr, _) = conditionals(AlphabetType::Rna, "DNA1");
    assert_eq!(ld, lr);
    assert_eq!(qd, qr);
}

#[test]
fn unknown_builtin_is_not_found() {
    let abc = Alphabet::new(AlphabetType::Amino);
    let mut s = ScoreMatrix::create(&abc);
    assert!(matches!(
        s.set("NOSUCHMATRIX", &abc),
        Err(ScoreMatrixError::NotFound(_))
    ));
    // DNA1 is a nucleotide built-in; it is not available for amino.
    assert!(matches!(
        s.set("DNA1", &abc),
        Err(ScoreMatrixError::NotFound(_))
    ));
    // ...and BLOSUM62 is not available for DNA.
    let dna = Alphabet::new(AlphabetType::Dna);
    let mut s = ScoreMatrix::create(&dna);
    assert!(matches!(
        s.set("BLOSUM62", &dna),
        Err(ScoreMatrixError::NotFound(_))
    ));
}

/// `esl_scorematrix_Read` round-trip: writing DNA1's canonical block out in
/// matrix-file form and reading it back must reproduce those scores.
#[test]
fn read_parses_labelled_matrix_file() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = "# a comment\n\
                    \x20  A   C   G   T\n\
                    A  41 -32 -26 -26\n\
                    C -32  39 -38 -17\n\
                    G -26 -38  46 -31\n\
                    T -26 -17 -31  39\n";
    let s = ScoreMatrix::read(text, Some("/tmp/custom-dna.mx"), &abc).expect("read");
    assert_eq!(s.nc, 4);
    assert_eq!(s.outorder, "ACGT");
    assert_eq!(s.s[0][0], 41);
    assert_eq!(s.s[2][2], 46);
    assert_eq!(s.s[1][3], -17);
    assert_eq!(s.name.as_deref(), Some("custom-dna.mx"));
    assert_eq!(s.path.as_deref(), Some("/tmp/custom-dna.mx"));
}

#[test]
fn read_rejects_missing_canonical_column() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = "  A   C   G\nA 1 0 0\nC 0 1 0\nG 0 0 1\n";
    let err = ScoreMatrix::read(text, None, &abc).unwrap_err();
    assert!(
        matches!(&err, ScoreMatrixError::Invalid(m) if m.contains("Expected to see a column")),
        "unexpected: {err}"
    );
}
