use super::*;

#[test]
fn hmmalign_rejects_conflicting_alphabet_flags() {
    // C: ALPHOPTS toggle group -> "Options --amino and --dna conflict".
    assert!(Args::try_parse_from(["hmmalign", "--amino", "--dna", "m.hmm", "s.fa"]).is_err());
    assert!(Args::try_parse_from(["hmmalign", "--dna", "--rna", "m.hmm", "s.fa"]).is_err());
    assert!(Args::try_parse_from(["hmmalign", "--amino", "--rna", "m.hmm", "s.fa"]).is_err());
    // A single alphabet flag is fine.
    assert!(Args::try_parse_from(["hmmalign", "--dna", "m.hmm", "s.fa"]).is_ok());
}
