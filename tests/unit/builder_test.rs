use super::*;

#[test]
fn copies_stockholm_cutoffs_to_hmm_flags() {
    let cutoffs = crate::msa::StockholmCutoffs {
        ga: Some([25.0, 24.0]),
        tc: Some([30.0, 29.0]),
        nc: Some([-1.0, -2.0]),
    };
    let mut hmm = Hmm::new(1, AlphabetType::Amino, 20);
    copy_stockholm_cutoffs_to_hmm(cutoffs, &mut hmm);

    assert_eq!(hmm.cutoff[P7_GA1], 25.0);
    assert_eq!(hmm.cutoff[P7_GA2], 24.0);
    assert_eq!(hmm.cutoff[P7_TC1], 30.0);
    assert_eq!(hmm.cutoff[P7_TC2], 29.0);
    assert_eq!(hmm.cutoff[P7_NC1], -1.0);
    assert_eq!(hmm.cutoff[P7_NC2], -2.0);
    assert!(hmm.flags & P7H_GA != 0);
    assert!(hmm.flags & P7H_TC != 0);
    assert!(hmm.flags & P7H_NC != 0);
}

#[test]
fn max_length_from_beta_tracks_tail_mass() {
    let hmm = crate::hmmfile::read_hmm_file_auto(std::path::Path::new("hmmer/tutorial/MADE1.hmm"))
        .unwrap()
        .remove(0);

    let loose = max_length_from_beta(&hmm, 0.5);
    let default = max_length_from_beta(&hmm, DEFAULT_WINDOW_BETA);

    assert!(loose > 0);
    assert!(default > 0);
    assert!(loose < default);
}
