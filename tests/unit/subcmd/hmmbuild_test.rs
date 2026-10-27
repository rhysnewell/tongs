use super::*;

#[test]
fn hmmbuild_parses_summary_output_and_alphabet_assertions() {
    let args = Args::try_parse_from([
        "hmmbuild",
        "-o",
        "summary.txt",
        "--amino",
        "out.hmm",
        "in.sto",
    ])
    .unwrap();

    assert_eq!(args.summary_out, Some(PathBuf::from("summary.txt")));
    assert!(args.amino);
    assert_eq!(args.hmmfile, PathBuf::from("out.hmm"));
    assert_eq!(args.msafile, PathBuf::from("in.sto"));
}

#[test]
fn hmmbuild_rejects_conflicting_alphabet_flags() {
    // C: ALPHOPTS toggle group -> "Options --amino and --dna conflict".
    assert!(Args::try_parse_from(["hmmbuild", "--amino", "--dna", "out.hmm", "in.sto"]).is_err());
    assert!(Args::try_parse_from(["hmmbuild", "--dna", "--rna", "out.hmm", "in.sto"]).is_err());
    assert!(Args::try_parse_from(["hmmbuild", "--amino", "--rna", "out.hmm", "in.sto"]).is_err());
}

#[test]
fn hmmbuild_rejects_fast_and_hand_together() {
    // C: CONOPTS toggle group -> "Options --fast and --hand conflict".
    assert!(Args::try_parse_from(["hmmbuild", "--fast", "--hand", "out.hmm", "in.sto"]).is_err());
}

#[test]
fn hmmbuild_rejects_out_of_range_symfrac_and_fragthresh() {
    // C: --symfrac/--fragthresh range "0<=x<=1".
    assert!(Args::try_parse_from(["hmmbuild", "--symfrac", "2", "out.hmm", "in.sto"]).is_err());
    assert!(Args::try_parse_from(["hmmbuild", "--symfrac", "-0.1", "out.hmm", "in.sto"]).is_err());
    assert!(Args::try_parse_from(["hmmbuild", "--fragthresh", "2", "out.hmm", "in.sto"]).is_err());
    assert!(Args::try_parse_from(["hmmbuild", "--fragthresh", "-1", "out.hmm", "in.sto"]).is_err());
    // Boundary values are accepted (0<=x<=1 is inclusive).
    assert!(Args::try_parse_from(["hmmbuild", "--symfrac", "0", "out.hmm", "in.sto"]).is_ok());
    assert!(Args::try_parse_from(["hmmbuild", "--symfrac", "1", "out.hmm", "in.sto"]).is_ok());
}

#[test]
fn hmmbuild_symfrac_requires_fast_not_hand() {
    // C: --symfrac reqs --fast; with --hand (which toggles --fast off) it errors.
    assert!(Args::try_parse_from([
        "hmmbuild",
        "--hand",
        "--symfrac",
        "0.6",
        "out.hmm",
        "in.sto"
    ])
    .is_err());
    // --symfrac alone (fast is default-on) is fine.
    assert!(Args::try_parse_from(["hmmbuild", "--symfrac", "0.6", "out.hmm", "in.sto"]).is_ok());
}
