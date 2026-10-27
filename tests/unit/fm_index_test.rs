use super::*;

#[test]
fn test_fm_index_basic() {
    let text = b"ACGTACGTACGT";
    let fm = FmIndex::build(text);
    assert_eq!(fm.n, 12);

    // Count exact matches
    let count = fm.count(b"ACGT");
    assert_eq!(count, 3);

    let count = fm.count(b"GGGG");
    assert_eq!(count, 0);
}

#[test]
fn test_fm_index_locate() {
    let text = b"AACGTAACGT";
    let fm = FmIndex::build(text);

    let mut positions = fm.locate(b"AACGT");
    positions.sort();
    assert_eq!(positions, vec![0, 5]);
}

#[test]
fn fm_index_does_not_expose_internal_sentinel() {
    let fm = FmIndex::build(b"ACGT");

    assert_eq!(fm.count(&[0]), 0);
    assert!(fm.locate(&[0]).is_empty());
}

#[test]
fn fm_index_reconstructs_from_serialized_parts() {
    let original = FmIndex::build(b"ACGTACGT");
    let rebuilt = FmIndex::from_parts(
        original.bwt.clone(),
        original.sa.clone(),
        original.c,
        original.n,
    )
    .unwrap();

    let mut positions = rebuilt.locate(b"CGT");
    positions.sort();
    assert_eq!(positions, vec![1, 5]);
}

#[test]
fn sampled_fm_index_locates_like_full_suffix_array() {
    let original = FmIndex::build(b"ACGTACGTGCAACGTACGT");
    let sampled = FmIndex::from_parts_sampled(
        original.bwt.clone(),
        original.sa.clone(),
        original.c,
        original.n,
    )
    .unwrap();

    for pattern in [b"ACG".as_slice(), b"TGC".as_slice(), b"CGTA".as_slice()] {
        let mut expected = original.locate(pattern);
        let mut got = sampled.locate(pattern);
        expected.sort_unstable();
        got.sort_unstable();
        assert_eq!(got, expected, "sampled locate mismatch for {pattern:?}");
    }
}

#[test]
fn sampled_fm_index_samples_suffix_array_by_bwt_row() {
    let original = FmIndex::build(b"ACGTACGTGCAACGTACGT");
    let sample_rate = 3;
    let sampled = FmIndex::from_parts_inner(
        original.bwt.clone(),
        original.sa.clone(),
        original.c,
        original.n,
        Some(sample_rate),
    )
    .unwrap();

    let expected: Vec<(i32, i32)> = original
        .sa
        .iter()
        .enumerate()
        .filter(|(row, _)| row % sample_rate == 0)
        .map(|(row, &pos)| (row as i32, pos))
        .collect();
    let position_sampled: Vec<(i32, i32)> = original
        .sa
        .iter()
        .enumerate()
        .filter(|(_, &pos)| (pos as usize).is_multiple_of(sample_rate))
        .map(|(row, &pos)| (row as i32, pos))
        .collect();

    assert_ne!(position_sampled, expected);
    assert_eq!(sampled.sa_samples, expected);
}

#[test]
fn fm_index_rejects_inconsistent_serialized_parts() {
    let original = FmIndex::build(b"ACGT");
    let err = FmIndex::from_parts(original.bwt, original.sa, original.c, 99).unwrap_err();

    assert!(err.contains("BWT length"));
}

#[test]
fn fm_index_rejects_c_table_that_does_not_match_bwt_counts() {
    let original = FmIndex::build(b"ACGT");
    let mut c = original.c;
    c[b'C' as usize] += 1;

    let err = FmIndex::from_parts(original.bwt, original.sa, c, original.n).unwrap_err();
    assert!(err.contains("C table entry"));
}

#[test]
fn fm_index_rejects_bwt_without_exactly_one_sentinel() {
    let original = FmIndex::build(b"ACGT");
    let mut bwt = original.bwt;
    let sentinel = bwt.iter().position(|&ch| ch == 0).unwrap();
    bwt[sentinel] = b'A';

    let err = FmIndex::from_parts(bwt, original.sa, original.c, original.n).unwrap_err();
    assert!(err.contains("exactly one sentinel"));
}

#[cfg(debug_assertions)]
#[test]
fn fm_index_rejects_suffix_array_duplicate_positions() {
    let original = FmIndex::build(b"ACGT");
    let mut sa = original.sa;
    sa[1] = sa[0];

    let err = FmIndex::from_parts(original.bwt, sa, original.c, original.n).unwrap_err();
    assert!(err.contains("not a permutation"));
}

#[test]
fn fm_index_interval_steps_match_locate_and_prune_missing_prefixes() {
    let fm = FmIndex::build(b"AACGTAACGT");
    let interval = fm
        .prepend_interval(fm.root_interval(), b'T')
        .and_then(|iv| fm.prepend_interval(iv, b'G'))
        .and_then(|iv| fm.prepend_interval(iv, b'C'))
        .and_then(|iv| fm.prepend_interval(iv, b'A'))
        .and_then(|iv| fm.prepend_interval(iv, b'A'))
        .unwrap();

    let mut via_interval = fm.locate_interval(interval);
    via_interval.sort();
    let mut via_pattern = fm.locate(b"AACGT");
    via_pattern.sort();
    assert_eq!(via_interval, via_pattern);

    assert_eq!(interval.len(), 2);
    assert!(fm.prepend_interval(interval, b'G').is_none());
}

/// Forward (left-to-right) bi-directional search must locate the same text
/// positions as the backward-search `locate()`, for several patterns and a
/// pattern that is absent. This validates the half-open port of C's
/// `fm_updateIntervalForward` against a brute-force oracle.
///
/// Mirrors C's two-index setup: the interval math runs on the backward
/// index `fmb` (BWT of the reversed text), while the synchronized forward
/// interval is located on the forward index `fmf` (BWT of the text). See
/// `fm_getSARangeForward` (`fm_general.c`).
#[test]
fn fm_index_forward_search_matches_locate() {
    let text = b"ACGTACGTTTACGGTACAACGTAC";
    let fmf = FmIndex::build(text);
    let rev: Vec<u8> = text.iter().rev().copied().collect();
    let fmb = FmIndex::build(&rev);

    let forward_locate = |pattern: &[u8]| -> Option<Vec<usize>> {
        let (&first, rest) = pattern.split_first().unwrap();
        // Single-char start: C[c]..C[c]+occ_total are identical on fmf/fmb
        // (same character multiset), so init both intervals there.
        let mut bk = fmb.char_interval(first)?;
        let mut fwd = bk;
        for &ch in rest {
            let (nb, nf) = fmb.update_interval_forward(bk, fwd, ch)?;
            bk = nb;
            fwd = nf;
        }
        // The forward interval lives in fmf's coordinate space.
        let mut pos = fmf.locate_interval(fwd);
        pos.sort();
        Some(pos)
    };

    for pattern in [
        &b"ACGT"[..],
        &b"ACG"[..],
        &b"TAC"[..],
        &b"AACGTAC"[..],
        &b"A"[..],
        &b"GGTA"[..],
    ] {
        let mut expected = fmf.locate(pattern);
        expected.sort();
        assert_eq!(
            forward_locate(pattern),
            Some(expected),
            "forward search disagrees with locate() for {:?}",
            std::str::from_utf8(pattern).unwrap()
        );
    }

    // Absent pattern: forward search must terminate with no interval.
    assert_eq!(forward_locate(b"ACGTACGTACGTAC"), None);
    assert!(fmf.locate(b"ACGTACGTACGTAC").is_empty());
}

#[test]
fn fm_index_forward_update_rejects_invalid_forward_interval() {
    let text = b"ACGTACGT";
    let fmb = FmIndex::build(&text.iter().rev().copied().collect::<Vec<_>>());
    let bk = fmb.char_interval(b'A').unwrap();
    let invalid_fwd = FmInterval {
        lo: 0,
        hi: fmb.bwt_len() + 1,
    };

    assert!(fmb.update_interval_forward(bk, invalid_fwd, b'C').is_none());
}

/// Regression: `occ`/`count`/`locate` must be correct at exact block/
/// superblock boundaries. Text lengths ≡ 255 (mod 256) make `bwt.len()` a
/// multiple of FREQ_CNT_B; a missing trailing sample row previously made
/// every search return 0 hits at those lengths.
#[test]
fn fm_index_occ_correct_at_block_boundaries() {
    let bases = b"ACGT";
    // Cover text_len around the 256/512 BWT-length boundaries (bwt = len+1).
    for &text_len in &[254usize, 255, 256, 257, 510, 511, 512, 767] {
        let text: Vec<u8> = (0..text_len).map(|i| bases[(i * 7 + 3) % 4]).collect();
        let fm = FmIndex::build(&text);
        for pat in [
            &b"ACGT"[..],
            &b"A"[..],
            &b"GT"[..],
            &b"TAC"[..],
            &b"ACGTAC"[..],
        ] {
            if pat.len() > text_len {
                continue;
            }
            // Brute-force count of overlapping occurrences.
            let expected = text.windows(pat.len()).filter(|w| *w == pat).count();
            assert_eq!(
                fm.count(pat),
                expected,
                "count({:?}) wrong at text_len={text_len}",
                std::str::from_utf8(pat).unwrap()
            );
            let mut got = fm.locate(pat);
            got.sort_unstable();
            let mut want: Vec<usize> = (0..text_len)
                .filter(|&i| i + pat.len() <= text_len && &text[i..i + pat.len()] == pat)
                .collect();
            want.sort_unstable();
            assert_eq!(
                got,
                want,
                "locate({:?}) wrong at text_len={text_len}",
                std::str::from_utf8(pat).unwrap()
            );
        }
    }
}

#[test]
fn makehmmerdb_occ_tables_do_not_double_count_superblock_boundary() {
    let text: Vec<u8> = (0..70_000).map(|i| b"ACGT"[(i * 11 + 1) % 4]).collect();
    let full = FmIndex::build(&text);
    let term_loc = full.bwt.iter().position(|&ch| ch == 0).unwrap();
    let raw_sa_samples = full
        .sa
        .iter()
        .enumerate()
        .filter_map(|(row, &pos)| {
            (row % 8 == 0).then_some(if pos as usize == text.len() { -1 } else { pos })
        })
        .collect::<Vec<_>>();

    let nb = 1 + full.bwt.len().div_ceil(FREQ_CNT_B);
    let nsb = 1 + full.bwt.len().div_ceil(FREQ_CNT_SB);
    let mut occ_b = vec![0u16; nb * 4];
    let mut occ_sb = vec![0u32; nsb * 4];
    let mut cnt_b = [0u16; 4];
    let mut cnt_sb = [0u32; 4];
    for (j, &base) in full.bwt.iter().enumerate() {
        let code = match base {
            0 | b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => unreachable!(),
        };
        cnt_b[code] += 1;
        cnt_sb[code] += 1;
        let joffset = j + 1;
        if joffset % FREQ_CNT_B == 0 {
            let row = joffset / FREQ_CNT_B;
            occ_b[row * 4..row * 4 + 4].copy_from_slice(&cnt_b);
            if joffset % FREQ_CNT_SB == 0 {
                let row = joffset / FREQ_CNT_SB;
                occ_sb[row * 4..row * 4 + 4].copy_from_slice(&cnt_sb);
                cnt_b = [0; 4];
            }
        }
    }
    occ_b[(nb - 1) * 4..nb * 4].copy_from_slice(&cnt_b);
    occ_sb[(nsb - 1) * 4..nsb * 4].copy_from_slice(&cnt_sb);

    let raw = FmIndex::from_makehmmerdb_raw_parts(
        full.bwt.clone(),
        raw_sa_samples,
        term_loc,
        8,
        FREQ_CNT_B,
        occ_b,
        occ_sb,
    )
    .unwrap();

    for pat in [&b"ACGT"[..], &b"CGTA"[..], &b"TACG"[..], &b"A"[..]] {
        assert_eq!(
            raw.count(pat),
            full.count(pat),
            "count mismatch for {pat:?}"
        );
        let mut got = raw.locate(pat);
        got.sort_unstable();
        let mut want = full.locate(pat);
        want.sort_unstable();
        assert_eq!(got, want, "locate mismatch for {pat:?}");
    }
}
