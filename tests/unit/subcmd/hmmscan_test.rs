use super::*;
use tongs::alphabet::AlphabetType;
use tongs::Hmm;

#[test]
fn hmmscan_positionals_match_c_order() {
    let args = Args::try_parse_from(["hmmscan", "models.hmm", "queries.fa"]).unwrap();

    assert_eq!(args.hmmdb, PathBuf::from("models.hmm"));
    assert_eq!(args.seqfile, PathBuf::from("queries.fa"));
}

#[test]
fn hmmscan_parses_c_output_options() {
    let args = Args::try_parse_from([
        "hmmscan",
        "-o",
        "out.txt",
        "--tblout",
        "hits.tbl",
        "--domtblout",
        "domains.tbl",
        "--pfamtblout",
        "pfam.tbl",
        "--acc",
        "--noali",
        "--textw",
        "140",
        "models.hmm",
        "queries.fa",
    ])
    .unwrap();

    assert_eq!(args.output, Some(PathBuf::from("out.txt")));
    assert_eq!(args.tblout, Some(PathBuf::from("hits.tbl")));
    assert_eq!(args.domtblout, Some(PathBuf::from("domains.tbl")));
    assert_eq!(args.pfamtblout, Some(PathBuf::from("pfam.tbl")));
    assert!(args.acc);
    assert!(args.noali);
    assert_eq!(args.textw, Some(140));
}

#[test]
fn hmmscan_parses_model_specific_cutoff_options() {
    let args = Args::try_parse_from(["hmmscan", "--cut_ga", "models.hmm", "queries.fa"]).unwrap();
    assert!(args.cut_ga);
    assert_eq!(selected_bit_cutoff(&args), Some(BitCutoff::GA));

    let args = Args::try_parse_from(["hmmscan", "--cut_tc", "models.hmm", "queries.fa"]).unwrap();
    assert!(args.cut_tc);
    assert_eq!(selected_bit_cutoff(&args), Some(BitCutoff::TC));

    let args = Args::try_parse_from(["hmmscan", "--cut_nc", "models.hmm", "queries.fa"]).unwrap();
    assert!(args.cut_nc);
    assert_eq!(selected_bit_cutoff(&args), Some(BitCutoff::NC));
}

#[test]
fn hmmscan_parses_c_threshold_options() {
    let args = Args::try_parse_from([
        "hmmscan",
        "-T",
        "42",
        "--domT",
        "3",
        "--incT",
        "40",
        "--incdomT",
        "2",
        "-Z",
        "99",
        "--domZ",
        "7",
        "models.hmm",
        "queries.fa",
    ])
    .unwrap();
    assert_eq!(args.score_threshold, Some(42.0));
    assert_eq!(args.dom_t, Some(3.0));
    assert_eq!(args.inc_t, Some(40.0));
    assert_eq!(args.inc_dom_t, Some(2.0));
    assert_eq!(args.z_value, Some(99.0));
    assert_eq!(args.domz_value, Some(7.0));
}

#[test]
fn hmmscan_parses_c_acceleration_options() {
    let args = Args::try_parse_from([
        "hmmscan",
        "--max",
        "--nonull2",
        "--seed",
        "7",
        "models.hmm",
        "queries.fa",
    ])
    .unwrap();
    assert!(args.max);
    assert!(args.nonull2);
    assert_eq!(args.seed, 7);

    let args = Args::try_parse_from([
        "hmmscan",
        "--F1",
        "0.1",
        "--F2",
        "0.2",
        "--F3",
        "0.3",
        "--nobias",
        "models.hmm",
        "queries.fa",
    ])
    .unwrap();
    assert_eq!(args.f1, 0.1);
    assert_eq!(args.f2, 0.2);
    assert_eq!(args.f3, 0.3);
    assert!(args.nobias);
}

#[test]
fn hmmscan_accepts_negative_space_separated_f_values() {
    // C --F1/--F2/--F3 are eslARG_REAL with no range, so the
    // space-separated negative form is accepted (parses past arg handling).
    // allow_hyphen_values lets clap match C instead of treating "-0.5" as
    // an unknown flag.
    let args = Args::try_parse_from([
        "hmmscan",
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
fn hmmscan_uses_pressed_database_alphabet() {
    let hmms = vec![Hmm::new(4, AlphabetType::Dna, 4)];
    let abc = alphabet_from_hmms(&hmms).unwrap();
    assert_eq!(abc.abc_type, AlphabetType::Dna);
}
