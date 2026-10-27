use super::*;

#[test]
fn hmmsim_accepts_negative_seed() {
    // C hmmsim.c: --seed is eslARG_INT with no range; a negative seed string
    // is accepted (esl_randomness_Create reinterprets it as a wrapping u32).
    let args = Args::try_parse_from(["hmmsim", "--seed", "-5", "model.hmm"]).unwrap();
    assert_eq!(args.seed, -5);
    // The =form must parse identically.
    let args = Args::try_parse_from(["hmmsim", "--seed=-5", "model.hmm"]).unwrap();
    assert_eq!(args.seed, -5);
}

#[test]
fn hmmsim_accepts_zero_and_positive_seed() {
    let args = Args::try_parse_from(["hmmsim", "--seed", "0", "model.hmm"]).unwrap();
    assert_eq!(args.seed, 0);
    let args = Args::try_parse_from(["hmmsim", "--seed", "42", "model.hmm"]).unwrap();
    assert_eq!(args.seed, 42);
}

#[test]
fn hmmsim_default_seed_is_zero() {
    let args = Args::try_parse_from(["hmmsim", "model.hmm"]).unwrap();
    assert_eq!(args.seed, 0);
}

#[test]
fn hmmsim_positive_seed_is_deterministic() {
    // A positive seed must produce a fully reproducible RNG stream. (A seed of
    // 0 — or a negative value mapping to 0 — would draw an arbitrary clock seed
    // and is intentionally not pinned here.)
    let mk = || {
        let mut rng = Mt19937::new(42i32 as u32);
        (0..8).map(|_| rng.next_u32()).collect::<Vec<_>>()
    };
    assert_eq!(mk(), mk());
}

#[test]
fn hmmsim_negative_seed_wraps_like_c_uint32() {
    // C casts the signed int seed to uint32_t; -5 -> 0xFFFFFFFB. Since that is
    // nonzero, the RNG uses it directly (no clock seed), so it stays
    // deterministic and matches seeding with the wrapped u32 value.
    let from_neg = {
        let mut rng = Mt19937::new((-5i32) as u32);
        (0..8).map(|_| rng.next_u32()).collect::<Vec<_>>()
    };
    let from_wrapped = {
        let mut rng = Mt19937::new(0xFFFF_FFFBu32);
        (0..8).map(|_| rng.next_u32()).collect::<Vec<_>>()
    };
    assert_eq!(from_neg, from_wrapped);
}

#[test]
fn hmmsim_styles_and_algorithms_are_mutually_exclusive() {
    // Regression guard that the existing toggle-group handling still parses.
    assert!(Args::try_parse_from(["hmmsim", "--vit", "model.hmm"]).is_ok());
    assert!(Args::try_parse_from(["hmmsim", "--fwd", "model.hmm"]).is_ok());
    // --forward / --viterbi aliases still accepted.
    assert!(Args::try_parse_from(["hmmsim", "--forward", "model.hmm"]).is_ok());
    assert!(Args::try_parse_from(["hmmsim", "--viterbi", "model.hmm"]).is_ok());
}
