use super::*;
use tongs::alphabet::AlphabetType;
use tongs::Hmm;

#[test]
fn nhmmscan_positionals_match_c_order() {
    let args = Args::try_parse_from(["nhmmscan", "models.hmm", "queries.fa"]).unwrap();

    assert_eq!(args.hmmdb, PathBuf::from("models.hmm"));
    assert_eq!(args.seqfile, PathBuf::from("queries.fa"));
}

#[test]
fn nhmmscan_parses_c_output_options() {
    let args = Args::try_parse_from([
        "nhmmscan",
        "-o",
        "out.txt",
        "--tblout",
        "hits.tbl",
        "--dfamtblout",
        "dfam.tbl",
        "--acc",
        "--noali",
        "--notextw",
        "models.hmm",
        "queries.fa",
    ])
    .unwrap();

    assert_eq!(args.output, Some(PathBuf::from("out.txt")));
    assert_eq!(args.tblout, Some(PathBuf::from("hits.tbl")));
    assert_eq!(args.dfamtblout, Some(PathBuf::from("dfam.tbl")));
    assert!(args.acc);
    assert!(args.noali);
    assert!(args.notextw);
}

#[test]
fn nhmmscan_parses_c_threshold_options() {
    let args = Args::try_parse_from([
        "nhmmscan",
        "-T",
        "42",
        "--incT",
        "40",
        "-Z",
        "45000000",
        "--seed",
        "7",
        "models.hmm",
        "queries.fa",
    ])
    .unwrap();
    assert_eq!(args.score_threshold, Some(42.0));
    assert_eq!(args.inc_t, Some(40.0));
    assert_eq!(args.z_value, Some(45_000_000.0));
    assert_eq!(args.seed, 7);

    let args = Args::try_parse_from(["nhmmscan", "--cut_ga", "models.hmm", "queries.fa"]).unwrap();
    assert_eq!(selected_bit_cutoff(&args), Some(BitCutoff::GA));

    assert!(Args::try_parse_from(["nhmmscan", "--domZ", "1", "models.hmm", "queries.fa"]).is_err());
    assert!(Args::try_parse_from(["nhmmscan", "--domE", "1", "models.hmm", "queries.fa"]).is_err());
}

#[test]
fn nhmmscan_accepts_negative_space_separated_f_values() {
    // C --F1/--F2/--F3 are eslARG_REAL with no range; the space-separated
    // negative form is accepted. allow_hyphen_values matches C instead of
    // treating "-0.5" as an unknown flag.
    let args = Args::try_parse_from([
        "nhmmscan",
        "--F1",
        "-0.5",
        "--F2",
        "-1e-3",
        "--F3",
        "-2",
        "models.hmm",
        "queries.fa",
    ])
    .unwrap();
    assert_eq!(args.f1, -0.5);
    assert_eq!(args.f2, -1e-3);
    assert_eq!(args.f3, -2.0);
}

#[test]
fn nhmmscan_long_target_adjustment_folds_model_search_space_into_lnp() {
    let search_space_ln = tongs::util::cmath::c_log_f64(12.0);
    let got = nhmmscan_long_target_ln_adjustment(800, 200, search_space_ln);
    let expected = tongs::util::cmath::c_log_f64(48.0);

    assert!((got - expected).abs() < 1e-12);
}

#[test]
fn nhmmscan_uses_pressed_database_rna_alphabet() {
    let hmms = vec![Hmm::new(4, AlphabetType::Rna, 4)];
    let abc = nucleotide_alphabet_from_hmms(&hmms).unwrap();
    assert_eq!(abc.abc_type, AlphabetType::Rna);
}

#[test]
fn nhmmscan_rejects_protein_pressed_database_alphabet() {
    let hmms = vec![Hmm::new(4, AlphabetType::Amino, 20)];
    let err = nucleotide_alphabet_from_hmms(&hmms).unwrap_err();
    assert!(err.contains("requires a DNA or RNA HMM database"));
}
