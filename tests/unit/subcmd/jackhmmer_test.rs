use super::*;

#[test]
fn jackhmmer_parses_c_output_file_option() {
    let args =
        Args::try_parse_from(["jackhmmer", "-o", "out.txt", "query.fa", "targets.fa"]).unwrap();
    assert_eq!(args.output, Some(PathBuf::from("out.txt")));
    assert_eq!(args.seqfile, PathBuf::from("query.fa"));
    assert_eq!(args.seqdb, PathBuf::from("targets.fa"));
}

#[test]
fn jackhmmer_parses_c_alignment_output_option() {
    let args =
        Args::try_parse_from(["jackhmmer", "-A", "hits.sto", "query.fa", "targets.fa"]).unwrap();
    assert_eq!(args.ali_outfile, Some(PathBuf::from("hits.sto")));
    assert_eq!(args.seqfile, PathBuf::from("query.fa"));
    assert_eq!(args.seqdb, PathBuf::from("targets.fa"));
}

#[test]
fn jackhmmer_parses_c_threshold_and_acceleration_options() {
    let args = Args::try_parse_from([
        "jackhmmer",
        "-T",
        "25",
        "--domT",
        "3",
        "--incT",
        "20",
        "--incdomT",
        "4",
        "-Z",
        "99",
        "--domZ",
        "7",
        "--seed",
        "5",
        "query.fa",
        "targets.fa",
    ])
    .unwrap();
    assert_eq!(args.score_threshold, Some(25.0));
    assert_eq!(args.dom_t, Some(3.0));
    assert_eq!(args.inc_t, Some(20.0));
    assert_eq!(args.incdom_t, Some(4.0));
    assert_eq!(args.z_value, Some(99.0));
    assert_eq!(args.domz_value, Some(7.0));
    assert_eq!(args.seed, 5);
    assert_eq!(args.em_l, 200);

    let args = Args::try_parse_from([
        "jackhmmer",
        "--F1",
        "0.1",
        "--F2",
        "0.2",
        "--F3",
        "0.3",
        "query.fa",
        "targets.fa",
    ])
    .unwrap();
    assert_eq!(args.f1, 0.1);
    assert_eq!(args.f2, 0.2);
    assert_eq!(args.f3, 0.3);
}

#[test]
fn jackhmmer_accepts_negative_space_separated_f_values() {
    // C --F1/--F2/--F3 have no range; the space-separated negative form is
    // accepted. allow_hyphen_values matches that instead of rejecting
    // "-0.5" as an unknown flag.
    let args = Args::try_parse_from([
        "jackhmmer",
        "--F1",
        "-0.5",
        "--F2",
        "-1e-3",
        "--F3",
        "-2",
        "query.fa",
        "targets.fa",
    ])
    .unwrap();
    assert_eq!(args.f1, -0.5);
    assert_eq!(args.f2, -1e-3);
    assert_eq!(args.f3, -2.0);
}

#[test]
fn jackhmmer_parses_single_sequence_calibration_options() {
    let args = Args::try_parse_from([
        "jackhmmer",
        "--seed",
        "7",
        "--EmL",
        "101",
        "--EmN",
        "20",
        "--EvL",
        "102",
        "--EvN",
        "21",
        "--EfL",
        "103",
        "--EfN",
        "22",
        "--Eft",
        "0.03",
        "query.fa",
        "targets.fa",
    ])
    .unwrap();

    assert_eq!(args.seed, 7);
    assert_eq!(args.em_l, 101);
    assert_eq!(args.em_n, 20);
    assert_eq!(args.ev_l, 102);
    assert_eq!(args.ev_n, 21);
    assert_eq!(args.ef_l, 103);
    assert_eq!(args.ef_n, 22);
    assert_eq!(args.eft, 0.03);
}
