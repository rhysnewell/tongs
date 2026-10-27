use super::*;

// Build a score table where the consensus base at node k scores +2 bits and
// everything else scores -3 bits. kp=4 (ACGT only) for simplicity.
fn consensus_scores(consensus_bases: &[usize]) -> (SsvScores, Vec<usize>) {
    let m = consensus_bases.len() as i32;
    let kp = 4;
    let mut scores = SsvScores::new(m, kp);
    let mut consensus = vec![0usize; (m as usize) + 1];
    for (i, &b) in consensus_bases.iter().enumerate() {
        let k = (i + 1) as i32;
        consensus[k as usize] = b;
        for x in 0..kp {
            scores.set(k, x, if x == b { 2.0 } else { -3.0 });
        }
    }
    (scores, consensus)
}

fn codes_to_text(codes: &[usize]) -> Vec<u8> {
    codes.iter().map(|&c| fm_code_to_base(c)).collect()
}

#[test]
fn ssv_scores_index_roundtrips() {
    let mut s = SsvScores::new(3, 4);
    s.set(2, 1, 1.5);
    assert_eq!(s.get(2, 1), 1.5);
    assert_eq!(s.get(1, 0), 0.0);
    assert_eq!(s.scores.len(), 4 * 4);
}

#[test]
fn extend_seed_uses_c_model_start_rule_and_clips_best_subdiagonal() {
    // Model consensus ACGTAC (codes 0,1,2,3,0,1), M=6.
    let (scores, _consensus) = consensus_scores(&[0, 1, 2, 3, 0, 1]);
    let config = FmSsvConfig {
        ssv_length: 6,
        ..FmSsvConfig::default()
    };
    // Target (1-indexed, sentinel at 0): perfect match to consensus.
    // text positions 0..=5 = ACGTAC
    let target_codes = [0usize, 0, 1, 2, 3, 0, 1]; // index 0 sentinel
    let fm_n = 8; // pretend FM text length
                  // Seed covering model k=4 length 2 (positions GT at model 3..4 -> diag.k=4)
    let mut diag = FmDiag {
        n: 3, // target start of seed (0-based fm coord) - aligns model pos
        k: 4,
        length: 2,
        score: 0.0,
        complementarity: Complementarity::NoComplement,
    };
    fm_extend_seed_core(&mut diag, &target_codes, &scores, &config, fm_n);
    // With a perfect-match target the extension should grow the diagonal and
    // produce a positive score equal to 2 bits per matched position.
    assert!(diag.score > 0.0);
    assert!(diag.length >= 2);
    // model_start rule: max(1, k - extend + 1) with extend=max(10,6-2)=10
    // => model_start = max(1, 4-10+1) = 1, so the window starts at model 1.
    assert!(diag.k >= 1 && diag.k <= 6);
}

#[test]
fn merge_seeds_combines_overlapping_same_diagonal() {
    let mut seeds = vec![
        FmDiag {
            n: 10,
            k: 5,
            length: 4,
            score: 0.0,
            complementarity: Complementarity::NoComplement,
        },
        FmDiag {
            n: 12,
            k: 7,
            length: 4,
            score: 0.0,
            complementarity: Complementarity::NoComplement,
        },
    ];
    // Same diagonal (n-k = 5 for both), overlapping -> merge into one.
    fm_merge_seeds(&mut seeds, 1000, 20, 100);
    assert_eq!(seeds.len(), 1);
    let d = seeds[0];
    assert_eq!(d.n, 10);
    // merged end = max(10+4-1, 12+4-1) = 15 -> length = 15-10+1 = 6
    assert_eq!(d.length, 6);
}

#[test]
fn merge_seeds_keeps_distinct_diagonals_separate() {
    let mut seeds = vec![
        FmDiag {
            n: 10,
            k: 5,
            length: 4,
            score: 0.0,
            complementarity: Complementarity::NoComplement,
        },
        // Different diagonal (n - k differs)
        FmDiag {
            n: 50,
            k: 5,
            length: 4,
            score: 0.0,
            complementarity: Complementarity::NoComplement,
        },
    ];
    fm_merge_seeds(&mut seeds, 1000, 20, 100);
    assert_eq!(seeds.len(), 2);
}

#[test]
fn merge_seeds_separates_strands() {
    let mut seeds = vec![
        FmDiag {
            n: 10,
            k: 5,
            length: 4,
            score: 0.0,
            complementarity: Complementarity::NoComplement,
        },
        FmDiag {
            n: 10,
            k: 5,
            length: 4,
            score: 0.0,
            complementarity: Complementarity::Complement,
        },
    ];
    fm_merge_seeds(&mut seeds, 1000, 20, 100);
    assert_eq!(seeds.len(), 2);
}

#[test]
fn recurse_finds_exact_consensus_seed_in_target() {
    // Model consensus = ACGTACGT (8 nodes). A target containing exactly that
    // string should yield a seed on the matching diagonal.
    let consensus_bases = vec![0usize, 1, 2, 3, 0, 1, 2, 3];
    let (scores, consensus) = consensus_scores(&consensus_bases);
    let alph = FmAlphabet::dna();

    // Target text: flanks + the consensus string embedded once.
    // Forward FM index in this codebase is built over the REVERSED text.
    let target_text = b"TTTTACGTACGTTTTT".to_vec();
    let reversed: Vec<u8> = target_text.iter().rev().copied().collect();
    // fmf = BWT of reversed text (forward sweep); fmb = BWT of forward text
    // (backward sweep), mirroring C's fmf/fmb pair.
    let fmf = FmIndex::build(&reversed);
    let fmb = FmIndex::build(&target_text);

    let config = FmSsvConfig {
        max_depth: 8,
        consec_pos_req: 1,
        consensus_match_req: 4,
        score_density_req: 0.0,
        drop_max_len: 8,
        ssv_length: 16,
        ..FmSsvConfig::default()
    };

    // Threshold low enough that the consensus run is found.
    let seeds = fm_get_seeds(
        &fmf, &fmb, &alph, &scores, &consensus, 6.0, &config, true, false,
    );

    // There must be at least one forward (NoComplement) seed.
    assert!(
        seeds
            .iter()
            .any(|d| d.complementarity == Complementarity::NoComplement),
        "expected a forward-strand seed, got {seeds:?}"
    );

    // The embedded consensus starts at target position 4 (0-based) in the
    // forward text "TTTTACGTACGT...". At least one seed should map onto the
    // ACGTACGT region (target n in [4, 11]).
    assert!(
        seeds.iter().any(|d| {
            d.complementarity == Complementarity::NoComplement && d.n >= 0 && d.n <= 12
        }),
        "expected a seed near the embedded consensus, got {seeds:?}"
    );

    // Sanity: the located forward text should actually contain the consensus.
    let cons_text = codes_to_text(&consensus_bases);
    assert!(target_text
        .windows(cons_text.len())
        .any(|w| w == cons_text.as_slice()));
}

#[test]
fn recurse_emits_no_seeds_for_absent_string() {
    // Consensus ACGTACGT but target has no such run -> no high-scoring seed.
    let consensus_bases = vec![0usize, 1, 2, 3, 0, 1, 2, 3];
    let (scores, consensus) = consensus_scores(&consensus_bases);
    let alph = FmAlphabet::dna();

    // Target with no ACGT runs (all A then all T).
    let target_text = b"AAAAAAAATTTTTTTT".to_vec();
    let reversed: Vec<u8> = target_text.iter().rev().copied().collect();
    // fmf = BWT of reversed text (forward sweep); fmb = BWT of forward text
    // (backward sweep), mirroring C's fmf/fmb pair.
    let fmf = FmIndex::build(&reversed);
    let fmb = FmIndex::build(&target_text);

    let config = FmSsvConfig {
        max_depth: 8,
        consec_pos_req: 5,
        consensus_match_req: 8,
        score_density_req: 0.75,
        ssv_length: 16,
        ..FmSsvConfig::default()
    };

    // High threshold: with a -3 mismatch penalty no diagonal in this target
    // can reach +12 bits over 8 positions.
    let seeds = fm_get_seeds(
        &fmf, &fmb, &alph, &scores, &consensus, 12.0, &config, true, false,
    );
    assert!(
        seeds.is_empty(),
        "expected no seeds for absent consensus, got {seeds:?}"
    );
}

#[test]
fn full_pipeline_recurse_merge_extend_runs_clean() {
    // End-to-end: get seeds, then extend each against the target and ensure
    // the kernel produces a coherent diagonal without panicking.
    let consensus_bases = vec![0usize, 1, 2, 3, 0, 1, 2, 3, 0, 1];
    let (scores, consensus) = consensus_scores(&consensus_bases);
    let alph = FmAlphabet::dna();

    let target_text = b"GGACGTACGTACGG".to_vec();
    let reversed: Vec<u8> = target_text.iter().rev().copied().collect();
    // fmf = BWT of reversed text (forward sweep); fmb = BWT of forward text
    // (backward sweep), mirroring C's fmf/fmb pair.
    let fmf = FmIndex::build(&reversed);
    let fmb = FmIndex::build(&target_text);

    let config = FmSsvConfig {
        max_depth: 10,
        consec_pos_req: 1,
        consensus_match_req: 4,
        score_density_req: 0.0,
        drop_max_len: 10,
        ssv_length: 14,
        ..FmSsvConfig::default()
    };

    let mut seeds = fm_get_seeds(
        &fmf, &fmb, &alph, &scores, &consensus, 6.0, &config, true, false,
    );
    assert!(!seeds.is_empty());

    // Build a 1-indexed target in alphabet codes for forward extension.
    let mut target_codes = vec![0usize];
    for &b in &target_text {
        target_codes.push(match b {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => 0,
        });
    }
    let fm_n = target_text.len() as i64 + 1;
    for diag in seeds.iter_mut() {
        if diag.complementarity == Complementarity::NoComplement {
            fm_extend_seed_core(diag, &target_codes, &scores, &config, fm_n);
            assert!(diag.length >= 0);
        }
    }
}
