use super::*;

#[test]
fn hmmemit_accepts_c_mode_options_but_rejects_incompatible_pairs() {
    assert!(Args::try_parse_from(["hmmemit", "-c", "model.hmm"]).is_ok());
    assert!(Args::try_parse_from(["hmmemit", "-C", "--minl", "0.4", "model.hmm"]).is_ok());
    assert!(Args::try_parse_from(["hmmemit", "-a", "-N", "3", "model.hmm"]).is_ok());
    assert!(Args::try_parse_from(["hmmemit", "-p", "-L", "20", "--unilocal", "model.hmm"]).is_ok());
    let args = Args::try_parse_from(["hmmemit", "-N", "3", "model.hmm"]).unwrap();
    assert_eq!(args.n.map(NonZeroUsize::get), Some(3));
}

#[test]
fn hmmemit_c_is_short_only_no_consensus_alias() {
    // C (hmmemit.c:32): -c is short-only. The short form works; the
    // previously-erroneous --consensus long alias must be rejected.
    assert!(Args::try_parse_from(["hmmemit", "-c", "model.hmm"]).is_ok());
    assert!(Args::try_parse_from(["hmmemit", "--consensus", "model.hmm"]).is_err());
}

#[test]
fn hmmemit_rejects_nonpositive_n() {
    assert!(Args::try_parse_from(["hmmemit", "-N", "0", "model.hmm"]).is_err());
}

#[test]
fn hmmemit_emit_mode_toggle_is_mutually_exclusive() {
    // C EMITOPTS "-a,-c,-C,-p": all pairs conflict at parse time.
    assert!(Args::try_parse_from(["hmmemit", "-a", "-c", "model.hmm"]).is_err());
    assert!(Args::try_parse_from(["hmmemit", "-a", "-C", "model.hmm"]).is_err());
    assert!(Args::try_parse_from(["hmmemit", "-a", "-p", "model.hmm"]).is_err());
    assert!(Args::try_parse_from(["hmmemit", "-c", "-C", "model.hmm"]).is_err());
    assert!(Args::try_parse_from(["hmmemit", "-c", "-p", "model.hmm"]).is_err());
    assert!(Args::try_parse_from(["hmmemit", "-C", "-p", "model.hmm"]).is_err());
}

#[test]
fn hmmemit_n_incompatible_with_consensus_modes() {
    // C: -N incomp "-c,-C". -N with -a / -p / alone is allowed.
    assert!(Args::try_parse_from(["hmmemit", "-N", "3", "-c", "model.hmm"]).is_err());
    assert!(Args::try_parse_from(["hmmemit", "-N", "3", "-C", "model.hmm"]).is_err());
    assert!(Args::try_parse_from(["hmmemit", "-N", "3", "-a", "model.hmm"]).is_ok());
    assert!(Args::try_parse_from(["hmmemit", "-N", "3", "-p", "model.hmm"]).is_ok());
}

#[test]
fn hmmemit_mode_flags_toggle_and_require_profile() {
    // C MODEOPTS: the four mode flags are mutually exclusive...
    assert!(Args::try_parse_from(["hmmemit", "-p", "--local", "--glocal", "model.hmm"]).is_err());
    assert!(
        Args::try_parse_from(["hmmemit", "-p", "--unilocal", "--uniglocal", "model.hmm"]).is_err()
    );
    // ...and require -p.
    assert!(Args::try_parse_from(["hmmemit", "--local", "model.hmm"]).is_err());
    assert!(Args::try_parse_from(["hmmemit", "--glocal", "model.hmm"]).is_err());
    assert!(Args::try_parse_from(["hmmemit", "-p", "--glocal", "model.hmm"]).is_ok());
}

#[test]
fn hmmemit_minl_minu_range_is_validated() {
    // C range "0<=x<=1".
    assert!(Args::try_parse_from(["hmmemit", "-C", "--minl", "5", "model.hmm"]).is_err());
    assert!(Args::try_parse_from(["hmmemit", "-C", "--minu", "-0.1", "model.hmm"]).is_err());
    assert!(Args::try_parse_from(["hmmemit", "-C", "--minl", "0.0", "model.hmm"]).is_ok());
    assert!(Args::try_parse_from(["hmmemit", "-C", "--minu", "1.0", "model.hmm"]).is_ok());
}

#[test]
fn rejustify_splits_inserts_like_easel() {
    // Easel gives nleft=1 for two residues in a width-3 insert run, so the other flushes right.
    let abc = Alphabet::amino();
    let inscount = vec![0usize, 3usize, 0usize];
    let matmap = vec![0usize, 0usize, 4usize];
    let mut row = b"Ass.-".to_vec(); // left-justified inserts before rejustify
    rejustify_insertions(&abc, &mut row, 2, &inscount, &matmap);
    assert_eq!(&row, b"As.s-");
}

#[test]
fn simple_consensus_uses_masked_unknown_code() {
    let mut hmm = Hmm::new(2, tongs::alphabet::AlphabetType::Amino, 20);
    hmm.name = "x".to_string();
    hmm.mat[1][0] = 1.0;
    hmm.mat[2][1] = 1.0;
    hmm.mm = Some(vec![b' ', b'.', b'm', b'\0']);
    let abc = Alphabet::amino();
    assert_eq!(simple_consensus(&hmm, &abc), b"AX");
}

#[test]
fn fancy_consensus_applies_thresholds_and_case() {
    let mut hmm = Hmm::new(3, tongs::alphabet::AlphabetType::Amino, 20);
    hmm.mat[1][0] = 0.2;
    hmm.mat[2][1] = 0.5;
    hmm.mat[3][2] = 0.9;
    let abc = Alphabet::amino();
    assert_eq!(fancy_consensus(&hmm, &abc, 0.4, 0.8), b"xcD");
}

#[test]
fn core_alignment_renders_stockholm_shape() {
    let mut hmm = Hmm::new(2, tongs::alphabet::AlphabetType::Amino, 20);
    hmm.t[0][MM] = 1.0;
    hmm.t[1][MM] = 1.0;
    hmm.t[2][MM] = 1.0;
    hmm.mat[1][0] = 1.0;
    hmm.mat[2][1] = 1.0;
    let abc = Alphabet::amino();
    let mut rng = MersenneTwister::new(1);
    let (seq, trace) = core_emit(&hmm, &abc, &mut rng);
    let mut out = Vec::new();
    write_stockholm_alignment(
        &mut out,
        &hmm,
        &abc,
        &[("h-sample1".to_string(), seq, trace)],
    );
    let text = String::from_utf8(out).unwrap();
    assert!(text.starts_with("# STOCKHOLM 1.0\n\n"));
    assert!(text.contains("h-sample1 AC"));
    assert!(text.contains("#=GC RF   xx"));
    assert!(text.ends_with("//\n"));
}

#[test]
fn profile_emit_matches_c_fixed_seed_fn3_modes() {
    let hmm = tongs::hmmfile::read_hmm_file(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/tutorial/fn3.hmm"
    )))
    .unwrap()
    .into_iter()
    .next()
    .unwrap();
    let abc = Alphabet::new(hmm.abc_type);
    let bg = Bg::new(&abc);

    let cases = [
            (
                P7_LOCAL,
                b"DIYCALVTEVALHMLTKFRHDAEEAKIHVTLLKTMEYGLQGTQSPHPSGISESHLVALWFRSIGMILGKKAYYKRETQQLVNEFTIAAQGRRYTQEELTEDDNTSTEYEVHQQWVANTTEI".as_slice(),
            ),
            (
                P7_UNILOCAL,
                b"DIYCALVTEVALHMLTKFRHDGADGAGHFTVDKIVPPYAADGQGTQSPHPSGISESHLVALWFRSIGMILGKKAYYKRETQQLVNEFTIAAQGRRYTQEELTEDLDMVR".as_slice(),
            ),
            (
                P7_GLOCAL,
                b"DIYCALVTEVALHMLTKFRHDIDKPTILKSKEAHERELTLQWSPSQYSGGSRDSMFKVTYSAFNSSKSQKITVEEKGPQYAITYLNAEVGFALKVQTVRDEGTGDWHMVRTMEVAHHVVVCNVTDNKVYVSWAKARAPNARNTFYRLVYKPSNSMHMWKERIRKSNHGTSQAVSDEGLLEGEQYGIKVSAVTPNLPQPQSRWLMVKPHMEELG".as_slice(),
            ),
            (
                P7_UNIGLOCAL,
                b"DIYCALVTEVALHMLTKFRHDGADGAGHFTVDKIVPPYAADGQGTQSPKETSDPQALVVYSAKSVTLNWNHPEEGIRNYSGFYYSLDEVEAAPGPNSTQDEETEDTFGDGVLKVAGLVVVANYTFKLTYVSGADFRNA".as_slice(),
            ),
        ];

    for (mode, expected) in cases {
        let mut gm = Profile::new(hmm.m, &abc);
        profile_config(&hmm, &bg, &mut gm, 25, mode);
        let mut rng = MersenneTwister::new(7);
        let (seq, trace) = profile_emit(&hmm, &gm, &bg, &mut rng);
        assert_eq!(sequence_text(&abc, &seq), expected);
        assert_eq!(trace.st.first(), Some(&State::S));
        assert_eq!(trace.st.last(), Some(&State::T));
    }
}
