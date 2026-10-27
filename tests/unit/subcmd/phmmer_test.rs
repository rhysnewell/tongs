use super::*;

#[test]
fn phmmer_parses_single_sequence_calibration_options() {
    let args = Args::try_parse_from([
        "phmmer",
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

#[test]
fn phmmer_accepts_negative_space_separated_f_values() {
    // C --F1/--F2/--F3 have no range; C accepts the space-separated
    // negative form. allow_hyphen_values matches that.
    let args = Args::try_parse_from([
        "phmmer",
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
