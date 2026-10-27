use super::*;

fn test_raw_fm_record(text: &[u8], freq_sa: usize, freq_cnt_b: usize) -> NhmmerFmRecord {
    test_raw_fm_record_with_target_text(text, text, freq_sa, freq_cnt_b)
}

fn test_raw_fm_record_with_target_text(
    fm_text: &[u8],
    target_text: &[u8],
    freq_sa: usize,
    freq_cnt_b: usize,
) -> NhmmerFmRecord {
    let fm = FmIndex::build(fm_text);
    let raw_bwt = fm.bwt.clone();
    let term_loc = raw_bwt
        .iter()
        .position(|&ch| ch == 0)
        .expect("test FM index must contain sentinel");
    let raw_sa_samples: Vec<i32> = fm
        .sa
        .iter()
        .enumerate()
        .filter_map(|(row, &pos)| {
            (row % freq_sa == 0).then_some(if pos as usize == fm_text.len() {
                -1
            } else {
                pos
            })
        })
        .collect();

    let mut occ_b_codes = Vec::new();
    let nb = raw_bwt.len().div_ceil(freq_cnt_b) + 1;
    for row in 0..nb {
        let pos = (row * freq_cnt_b).min(raw_bwt.len());
        let sb_start = (pos / 65_536) * 65_536;
        occ_b_codes.extend(test_raw_fm_counts(&raw_bwt[sb_start..pos]));
    }
    let mut occ_sb_codes = Vec::new();
    let nsb = raw_bwt.len().div_ceil(65_536) + 1;
    for row in 0..nsb {
        let pos = (row * 65_536).min(raw_bwt.len());
        occ_sb_codes.extend(
            test_raw_fm_counts(&raw_bwt[..pos])
                .into_iter()
                .map(|count| count as u32),
        );
    }

    NhmmerFmRecord {
        term_loc,
        seq_offset: 0,
        seq_count: 1,
        ambig_offset: 0,
        ambig_count: 0,
        overlap_bases: 0,
        text_bases_len: target_text.len(),
        occ_b_offset: 0,
        occ_b_count: occ_b_codes.len(),
        occ_sb_offset: 0,
        occ_sb_count: occ_sb_codes.len(),
        freq_sa,
        freq_cnt_b,
        bwt_len: raw_bwt.len(),
        packed_bwt_offset: 0,
        packed_bwt_bytes: raw_bwt.len().div_ceil(4),
        packed_bwt: test_pack_raw_bwt(&raw_bwt),
        packed_text_offset: 0,
        packed_text_bytes: target_text.len().div_ceil(4),
        packed_text: test_pack_raw_bwt(target_text),
        raw_sa_offset: 0,
        raw_sa_count: raw_sa_samples.len(),
        raw_sa_samples,
        occ_b_codes,
        occ_sb_codes,
        text: target_text.to_vec(),
    }
}

fn test_raw_fm_counts(bytes: &[u8]) -> [u16; 4] {
    let mut counts = [0u16; 4];
    for &byte in bytes {
        match byte {
            0 | b'A' => counts[0] += 1,
            b'C' => counts[1] += 1,
            b'G' => counts[2] += 1,
            b'T' => counts[3] += 1,
            _ => {}
        }
    }
    counts
}

fn test_pack_raw_bwt(raw_bwt: &[u8]) -> Vec<u8> {
    let mut packed = vec![0u8; raw_bwt.len().div_ceil(4)];
    for (idx, &byte) in raw_bwt.iter().enumerate() {
        let code = match byte {
            0 | b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => unreachable!(),
        };
        packed[idx / 4] |= code << (6 - ((idx & 0x3) * 2));
    }
    packed
}

#[test]
fn search_dfam_query_accession_uses_dash_when_absent() {
    assert_eq!(search_dfam_query_accession(None), "-");
    assert_eq!(search_dfam_query_accession(Some("")), "-");
    assert_eq!(
        search_dfam_query_accession(Some("DF0000629.2")),
        "DF0000629.2"
    );
}

#[test]
fn makehmmerdb_c_string_rejects_embedded_nul() {
    let err = read_nul_string(&mut &b"ab\0cd\0"[..], 5).unwrap_err();
    assert!(err.contains("embedded NUL"));
}

#[test]
fn dfamtblout_uses_dynamic_name_and_position_widths() {
    use tongs::tophits::{AliDisplay, Domain, Hit, TopHits, P7_IS_REPORTED};

    let mut th = TopHits::new();
    th.hits.push(Hit {
        name: "very-long-target-name-for-dfam-width".to_string(),
        acc: "very-long-model-accession".to_string(),
        desc: "description".to_string(),
        n: 1_234_567,
        sortkey: -10.0,
        score: 42.0,
        bias: 0.0,
        pre_score: 42.0,
        sum_score: 42.0,
        lnp: (1.0e-12_f64).ln(),
        pre_lnp: (1.0e-12_f64).ln(),
        sum_lnp: (1.0e-12_f64).ln(),
        nexpected: 1.0,
        nregions: 1,
        nclustered: 1,
        noverlaps: 0,
        nenvelopes: 1,
        ndom: 1,
        nreported: 1,
        nincluded: 1,
        dcl: vec![Domain {
            iali: 1_234_560,
            jali: 1_234_500,
            ienv: 1_234_490,
            jenv: 1_234_567,
            bitscore: 42.0,
            lnp: (1.0e-12_f64).ln(),
            dombias: 0.5,
            oasc: 0.0,
            envsc: 0.0,
            domcorrection: 0.0,
            is_reported: true,
            is_included: true,
            ad: Some(AliDisplay {
                hmmfrom: 12,
                hmmto: 345,
                sqfrom: 1_234_560,
                sqto: 1_234_500,
                ..Default::default()
            }),
        }],
        flags: P7_IS_REPORTED,
        seqidx: 0,
        subseq_start: 0,
    });

    let mut out = Vec::new();
    write_nhmmer_dfamtblout(
        &mut out,
        "very-long-query-name-for-dfam-width",
        Some("DF0000001.1"),
        "sq-len",
        &th,
        1.0,
    );
    let out = String::from_utf8(out).unwrap();

    assert!(out.contains("very-long-target-name-for-dfam-width  DF0000001.1"));
    assert!(out.contains("very-long-query-name-for-dfam-width"));
    let row = out
        .lines()
        .find(|line| !line.starts_with('#') && !line.trim().is_empty())
        .unwrap();
    let fields: Vec<&str> = row.split_whitespace().collect();
    assert_eq!(
        &fields[9..14],
        &["1234560", "1234500", "1234490", "1234567", "1234567"]
    );
}

#[test]
fn nhmmer_rejects_domtblout_like_c() {
    let err = match Args::try_parse_from([
        "nhmmer",
        "--domtblout",
        "domains.tbl",
        "query.hmm",
        "targets.fa",
    ]) {
        Ok(_) => panic!("nhmmer unexpectedly accepted --domtblout"),
        Err(err) => err,
    };

    assert_eq!(err.kind(), clap::error::ErrorKind::UnknownArgument);
}

#[test]
fn nhmmer_parses_window_length_override() {
    let args =
        Args::try_parse_from(["nhmmer", "--w_length", "50", "query.hmm", "targets.fa"]).unwrap();

    assert_eq!(args.w_length, Some(50));
}

#[test]
fn nhmmer_reads_binary_h3m_query_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("query.h3m");
    let mut hmm = tongs::Hmm::new(4, tongs::alphabet::AlphabetType::Dna, 4);
    hmm.name = "dna_query".to_string();
    {
        let mut file = std::fs::File::create(&path).unwrap();
        tongs::hmmfile_binary::write_binary_hmm(&mut file, &hmm).unwrap();
    }

    let hmms = read_hmms(&path).unwrap();
    assert_eq!(hmms.len(), 1);
    assert_eq!(hmms[0].name, "dna_query");
    assert_eq!(hmms[0].abc_type, tongs::alphabet::AlphabetType::Dna);
}

#[test]
fn nhmmer_loads_makehmmerdb_container_index_records_for_seed_lookup() {
    let reversed_text = b"TGCATGCA";
    let fm = FmIndex::build(reversed_text);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"HMMERDB\0");
    bytes.extend_from_slice(MAKEHMMERDB_INDEX_MAGIC);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    for value in [0u64, 0, fm.n as u64, 0, 1, 0, 0, 0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&(fm.bwt.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(fm.sa.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&256u64.to_le_bytes());
    bytes.extend_from_slice(&fm.bwt);
    for &pos in &fm.sa {
        bytes.extend_from_slice(&pos.to_le_bytes());
    }
    for &value in &fm.c {
        bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }

    let records = parse_makehmmerdb_index_extension(&bytes).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].block_id, 0);
    assert_eq!(records[0].kind, 0);
    assert_eq!(records[0].text_start, 0);
    assert_eq!(records[0].text_len, 8);
    assert_eq!(records[0].seq_offset, 0);
    assert_eq!(records[0].seq_count, 1);
    assert_eq!(records[0].ambig_offset, 0);
    assert_eq!(records[0].ambig_count, 0);
    assert_eq!(records[0].overlap_bases, 0);
    let mut positions = records[0].fm.locate(b"TGC");
    positions.sort();
    assert_eq!(positions, vec![0, 4]);
    let blocks = vec![test_raw_fm_record(b"ACGTACGT", 4, 8)];
    validate_makehmmerdb_index_records(&records, &blocks, true, 0).unwrap();
}

#[test]
fn makehmmerdb_index_rejects_huge_record_count_before_allocation() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"HMMERDB\0");
    bytes.extend_from_slice(MAKEHMMERDB_INDEX_MAGIC);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());

    let err = parse_makehmmerdb_index_extension(&bytes).unwrap_err();
    assert!(
        err.contains("makehmmerdb FM-index record table"),
        "unexpected error: {err}"
    );
}

#[test]
fn makehmmerdb_index_rejects_huge_bwt_before_allocation() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"HMMERDB\0");
    bytes.extend_from_slice(MAKEHMMERDB_INDEX_MAGIC);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    for value in [0u64, 0, 8, 0, 1, 0, 0, 0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&4096u64.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&256u64.to_le_bytes());
    bytes.extend(std::iter::repeat_n(0, 256 * 8));

    let err = parse_makehmmerdb_index_extension(&bytes).unwrap_err();
    assert!(
        err.contains("makehmmerdb FM-index record payload"),
        "unexpected error: {err}"
    );
}

fn makehmmerdb_c_metadata_header(freq_sa: u32, freq_cnt_b: u32, seq_count: u32) -> Vec<u8> {
    let mut bytes = vec![1, 0, 4, 2];
    bytes.extend_from_slice(&freq_sa.to_le_bytes());
    bytes.extend_from_slice(&64u32.to_le_bytes());
    bytes.extend_from_slice(&freq_cnt_b.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&seq_count.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes
}

#[test]
fn makehmmerdb_c_metadata_preserves_target_start() {
    let mut bytes = makehmmerdb_c_metadata_header(4, 8, 1);
    bytes.extend_from_slice(&7u32.to_le_bytes());
    bytes.extend_from_slice(&123u64.to_le_bytes());
    bytes.extend_from_slice(&11u32.to_le_bytes());
    bytes.extend_from_slice(&17u32.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&3u16.to_le_bytes());
    bytes.extend_from_slice(&4u16.to_le_bytes());
    bytes.extend_from_slice(b"s0\0");
    bytes.extend_from_slice(b"a\0");
    bytes.extend_from_slice(b"src\0");
    bytes.extend_from_slice(b"desc\0");
    let mut cursor = Cursor::new(bytes.as_slice());

    let (_, _, _, _, seq_meta, ambiguities) =
        read_makehmmerdb_c_metadata_payload(&mut cursor).unwrap();

    assert!(ambiguities.is_empty());
    assert_eq!(seq_meta.len(), 1);
    assert_eq!(seq_meta[0].name, "s0");
    assert_eq!(seq_meta[0].target_start, 123);
    assert_eq!(seq_meta[0].fm_start, 11);
    assert_eq!(seq_meta[0].length, 17);
}

#[test]
fn makehmmerdb_c_metadata_rejects_zero_sampling_frequencies() {
    let zero_sa_bytes = makehmmerdb_c_metadata_header(0, 8, 0);
    let mut zero_sa = Cursor::new(zero_sa_bytes.as_slice());
    let err = read_makehmmerdb_c_metadata_payload(&mut zero_sa).unwrap_err();
    assert!(
        err.contains("suffix-array sampling frequency is zero"),
        "unexpected error: {err}"
    );

    let zero_cnt_bytes = makehmmerdb_c_metadata_header(4, 0, 0);
    let mut zero_cnt = Cursor::new(zero_cnt_bytes.as_slice());
    let err = read_makehmmerdb_c_metadata_payload(&mut zero_cnt).unwrap_err();
    assert!(
        err.contains("occurrence sampling frequency is zero"),
        "unexpected error: {err}"
    );
}

#[test]
fn makehmmerdb_c_metadata_rejects_short_huge_sequence_payload_before_allocation() {
    let bytes = makehmmerdb_c_metadata_header(4, 8, u32::MAX);
    let mut cursor = Cursor::new(bytes.as_slice());

    let err = read_makehmmerdb_c_metadata_payload(&mut cursor).unwrap_err();
    assert!(
        err.contains("makehmmerdb metadata payload is truncated"),
        "unexpected error: {err}"
    );
}

#[test]
fn makehmmerdb_c_metadata_rejects_invalid_ambiguity_coordinates() {
    let mut bytes = makehmmerdb_c_metadata_header(4, 8, 0);
    let ambig_count_offset = 4 + 4 + 4 + 4 + 2 + 4;
    bytes[ambig_count_offset..ambig_count_offset + 4].copy_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(-1i32).to_le_bytes());
    bytes.extend_from_slice(&2i32.to_le_bytes());
    let mut cursor = Cursor::new(bytes.as_slice());

    let err = read_makehmmerdb_c_metadata_payload(&mut cursor).unwrap_err();
    assert!(
        err.contains("invalid coordinates"),
        "unexpected error: {err}"
    );
}

#[test]
fn makehmmerdb_c_fm_record_rejects_zero_sampling_frequencies() {
    let mut empty = Cursor::new([].as_slice());
    let err = read_makehmmerdb_c_fm_record(&mut empty, 0, 8, true, 0, true).unwrap_err();
    assert!(
        err.contains("suffix-array sampling frequency is zero"),
        "unexpected error: {err}"
    );

    let mut empty = Cursor::new([].as_slice());
    let err = read_makehmmerdb_c_fm_record(&mut empty, 4, 0, true, 0, true).unwrap_err();
    assert!(
        err.contains("occurrence sampling frequency is zero"),
        "unexpected error: {err}"
    );
}

#[test]
fn makehmmerdb_c_fm_record_tracks_terminal_symbol_separately_from_bases() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&5u64.to_le_bytes());
    for value in [0u32, 0, 0, 0, 1, 0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&[0b00011011, 0]);
    bytes.extend_from_slice(&[0, 0]);
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend(std::iter::repeat_n(0u8, 8 * 2));
    bytes.extend(std::iter::repeat_n(0u8, 8 * 4));
    let mut cursor = Cursor::new(bytes.as_slice());

    let record = read_makehmmerdb_c_fm_record(&mut cursor, 4, 8, true, 0, true).unwrap();
    assert_eq!(record.text_bases_len, 5);
    assert_eq!(record.text, Vec::<u8>::new());
    assert_eq!(record.packed_text, vec![0b00011011, 0]);
}

#[test]
fn makehmmerdb_c_fm_record_rejects_short_huge_payload_before_allocation() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1024u64.to_le_bytes());
    for value in [0u32, 0, 0, 0, 0, 0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let mut cursor = Cursor::new(bytes.as_slice());

    let err = read_makehmmerdb_c_fm_record(&mut cursor, 4, 8, true, 0, true).unwrap_err();
    assert!(
        err.contains("makehmmerdb FM-index record payload"),
        "unexpected error: {err}"
    );
}

#[test]
fn nhmmer_fm_seed_hits_create_candidate_windows() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = b"TTACGTACAA";
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);

    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: text.len(),
        }],
        fm_ambiguities: Vec::new(),
        fm_index_records: vec![NhmmerFmIndexRecord {
            block_id: 0,
            kind: 0,
            text_start: 0,
            text_len: text.len(),
            text: Arc::from(text.to_vec()),
            packed_text: Arc::from([]),
            mapped_packed_text: None,
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 0,
            fm: FmIndex::build(&reversed_text),
        }],
    };

    let windows = fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 6, false).unwrap();
    assert_eq!(
        windows,
        vec![NhmmerCandidateWindow {
            n: 3,
            length: 4,
            k: 4,
            fm_frame: None,
        }]
    );
}

#[test]
fn nhmmer_fm_c_mapping_helpers_convert_and_map_block_coordinates() {
    let text = b"AAAACCCCGGGGTTTT";
    let record = NhmmerFmIndexRecord {
        block_id: 0,
        kind: 0,
        text_start: 0,
        text_len: text.len(),
        text: Arc::from(text.to_vec()),
        packed_text: Arc::from([]),
        mapped_packed_text: None,
        seq_offset: 0,
        seq_count: 2,
        ambig_offset: 0,
        ambig_count: 1,
        overlap_bases: 0,
        fm: FmIndex::build(&text.iter().rev().copied().collect::<Vec<_>>()),
    };
    let seq_meta = vec![
        NhmmerFmSequenceMeta {
            name: "s0".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: 8,
        },
        NhmmerFmSequenceMeta {
            name: "s1".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 8,
            length: 8,
        },
    ];
    let ambiguities = vec![(5, 6)];

    let watson = fm_convert_range_to_dsq(&record, &ambiguities, 4, 4, false, true).unwrap();
    assert_eq!(&watson[1..=4], &[1, 15, 15, 1]);

    let crick = fm_convert_range_to_dsq(&record, &ambiguities, 0, 4, true, false).unwrap();
    assert_eq!(&crick[1..=4], &[0, 0, 0, 0]);

    let pos = fm_get_original_position(&record, &seq_meta, 4, false, 10).unwrap();
    assert_eq!(
        pos,
        NhmmerFmOriginalPosition {
            segment_id: 1,
            seg_pos: 3,
            crosses_segment: false,
        }
    );

    let crossing = fm_get_original_position(&record, &seq_meta, 4, false, 6).unwrap();
    assert_eq!(crossing.segment_id, 0);
    assert_eq!(crossing.seg_pos, 7);
    assert!(crossing.crosses_segment);
}

#[test]
fn nhmmer_fm_window_from_diag_preserves_original_segment_id() {
    let text = b"AAAACCCCGGGGTTTT";
    let record = NhmmerFmIndexRecord {
        block_id: 0,
        kind: 0,
        text_start: 0,
        text_len: text.len(),
        text: Arc::from(text.to_vec()),
        packed_text: Arc::from([]),
        mapped_packed_text: None,
        seq_offset: 0,
        seq_count: 2,
        ambig_offset: 0,
        ambig_count: 0,
        overlap_bases: 0,
        fm: FmIndex::build(&text.iter().rev().copied().collect::<Vec<_>>()),
    };
    let seq_meta = vec![
        NhmmerFmSequenceMeta {
            name: "s0".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: 8,
        },
        NhmmerFmSequenceMeta {
            name: "s1".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 8,
            length: 8,
        },
    ];
    let diag = crate::simd::fm_ssv::FmDiag {
        n: 10,
        k: 2,
        length: 3,
        score: 7.0,
        complementarity: crate::simd::fm_ssv::Complementarity::NoComplement,
    };

    let window = fm_window_from_diag(&record, &seq_meta, 1, 8, &diag).unwrap();

    assert_eq!(window.window.n, 3);
    assert_eq!(window.window.length, 3);
    assert_eq!(
        window.window.fm_frame.unwrap(),
        NhmmerCandidateFmFrame {
            fm_n: 10,
            id: 1,
            target_len: 8,
            fm_start: 8,
            fm_bwt_len: record.fm.bwt_len() as i64,
        }
    );
}

#[test]
fn nhmmer_fm_seed_ssv_length_changes_candidate_window_span() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = b"ACGTTGCAACGATCGTACGAGTCCATGCTAGGATCCGTAACGTTG";
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(text.len(), AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b'N'; hmm.m + 2]);
    hmm.consensus.as_mut().unwrap()[0] = b' ';
    hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
    for node in 1..=hmm.m {
        for x in 0..abc.k {
            hmm.mat[node][x] = 0.01;
        }
        let ch = text[node - 1];
        let code = abc.inmap[ch as usize] as usize;
        hmm.mat[node][code] = 0.97;
        if (16..=30).contains(&node) {
            hmm.consensus.as_mut().unwrap()[node] = ch;
        }
    }

    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: text.len(),
        }],
        fm_ambiguities: Vec::new(),
        fm_index_records: vec![NhmmerFmIndexRecord {
            block_id: 0,
            kind: 0,
            text_start: 0,
            text_len: text.len(),
            text: Arc::from(text.to_vec()),
            packed_text: Arc::from([]),
            mapped_packed_text: None,
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 0,
            fm: FmIndex::build(&reversed_text),
        }],
    };

    let base_config = NhmmerFmSeedConfig {
        score_threshold_bits: 999.0,
        consensus_match_req: 0,
        ..NhmmerFmSeedConfig::default()
    };
    let short_windows = fm_seed_candidate_windows_with_config(
        &target_db,
        0,
        &hmm,
        &Bg::new(&abc),
        100,
        0.02,
        false,
        NhmmerFmSeedConfig {
            ssv_length: 16,
            ..base_config
        },
    )
    .unwrap();
    let long_windows = fm_seed_candidate_windows_with_config(
        &target_db,
        0,
        &hmm,
        &Bg::new(&abc),
        100,
        0.02,
        false,
        NhmmerFmSeedConfig {
            ssv_length: text.len(),
            ..base_config
        },
    )
    .unwrap();

    assert_eq!(short_windows.len(), 1);
    assert_eq!(long_windows.len(), 1);
    assert!(
            short_windows[0].length < long_windows[0].length,
            "seed_ssv_length should bound FM seed extension before window creation: short={short_windows:?} long={long_windows:?}"
        );
}

#[test]
fn nhmmer_fm_seed_hits_use_configured_consensus_match_run() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = b"TTACGTACGTACAA";
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(13, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![
        b' ', b'A', b'C', b'G', b'T', b'A', b'C', b'G', b'T', b'A', b'C', b'G', b'G', b'G', b' ',
    ]);

    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: text.len(),
        }],
        fm_ambiguities: Vec::new(),
        fm_index_records: vec![NhmmerFmIndexRecord {
            block_id: 0,
            kind: 0,
            text_start: 0,
            text_len: text.len(),
            text: Arc::from(text.to_vec()),
            packed_text: Arc::from([]),
            mapped_packed_text: None,
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 0,
            fm: FmIndex::build(&reversed_text),
        }],
    };

    let windows = fm_seed_candidate_windows_with_config(
        &target_db,
        0,
        &hmm,
        &Bg::new(&abc),
        10,
        0.02,
        false,
        NhmmerFmSeedConfig {
            consensus_match_req: 10,
            ..NhmmerFmSeedConfig::default()
        },
    )
    .unwrap();
    assert_eq!(
        windows,
        vec![NhmmerCandidateWindow {
            n: 3,
            length: 10,
            k: 10,
            fm_frame: None,
        }]
    );
}

#[test]
fn nhmmer_fm_model_consensus_trie_reaches_late_capped_seed() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let motif = b"CCGTAACGTTAC";
    let text = [b"GG".as_slice(), motif, b"GG".as_slice()].concat();
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(&text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(180, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b'A'; hmm.m + 2]);
    hmm.consensus.as_mut().unwrap()[0] = b' ';
    hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
    for (offset, &ch) in motif.iter().enumerate() {
        hmm.consensus.as_mut().unwrap()[150 + offset] = ch;
    }

    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: text.len(),
        }],
        fm_ambiguities: Vec::new(),
        fm_index_records: vec![NhmmerFmIndexRecord {
            block_id: 0,
            kind: 0,
            text_start: 0,
            text_len: text.len(),
            text: Arc::from(text.clone()),
            packed_text: Arc::from([]),
            mapped_packed_text: None,
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 0,
            fm: FmIndex::build(&reversed_text),
        }],
    };

    let windows =
        fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 20, false).unwrap();
    assert!(
            windows
                .iter()
                .any(|window| window.n == 3 && window.length == motif.len() && window.k == 161),
            "FM-pruned consensus recursion should reach a late materialized-seed-capped motif: {windows:?}"
        );
}

#[test]
fn nhmmer_fm_model_consensus_trie_augments_materialized_seed_hits() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let early_motif = b"CCCCCCCCCCCC";
    let late_motif = b"CCGTAACGTTAC";
    let text = [
        b"GG".as_slice(),
        early_motif,
        b"GG".as_slice(),
        late_motif,
        b"GG".as_slice(),
    ]
    .concat();
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(&text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(180, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b'A'; hmm.m + 2]);
    hmm.consensus.as_mut().unwrap()[0] = b' ';
    hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
    for (offset, &ch) in early_motif.iter().enumerate() {
        hmm.consensus.as_mut().unwrap()[1 + offset] = ch;
    }
    for (offset, &ch) in late_motif.iter().enumerate() {
        hmm.consensus.as_mut().unwrap()[150 + offset] = ch;
    }

    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: text.len(),
        }],
        fm_ambiguities: Vec::new(),
        fm_index_records: vec![NhmmerFmIndexRecord {
            block_id: 0,
            kind: 0,
            text_start: 0,
            text_len: text.len(),
            text: Arc::from(text.clone()),
            packed_text: Arc::from([]),
            mapped_packed_text: None,
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 0,
            fm: FmIndex::build(&reversed_text),
        }],
    };

    let windows =
        fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 24, false).unwrap();
    assert!(
        windows
            .iter()
            .any(|window| window.n == 3 && window.length == early_motif.len() && window.k == 12),
        "materialized consensus seed should still create the early window: {windows:?}"
    );
    assert!(
            windows.iter().any(|window| {
                window.n == 17 && window.length == late_motif.len() && window.k == 161
            }),
            "recursive consensus traversal should augment, not wait for an empty materialized result: {windows:?}"
        );
}

#[test]
fn nhmmer_fm_model_consensus_trie_ranks_before_source_cap() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let low_motif = b"AAAAAAAAAAAA";
    let high_motif = b"CCGTAACGTTAC";
    let text = [b"GG".as_slice(), low_motif, b"GG".as_slice(), high_motif].concat();
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();

    let mut hmm = Hmm::new(32, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b'A'; hmm.m + 2]);
    hmm.consensus.as_mut().unwrap()[0] = b' ';
    hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
    for node in 1..=hmm.m {
        for x in 0..abc.k {
            hmm.mat[node][x] = 0.24;
        }
    }
    for (offset, &ch) in high_motif.iter().enumerate() {
        let node = 21 + offset;
        hmm.consensus.as_mut().unwrap()[node] = ch;
        let code = abc.inmap[ch as usize] as usize;
        hmm.mat[node][code] = 0.97;
    }

    let hits = fm_locate_model_consensus_seeds(
        &FmIndex::build(&reversed_text),
        &hmm,
        &Bg::new(&abc),
        false,
        text.len(),
        text.len(),
        1,
        NhmmerFmSeedConfig {
            consensus_match_req: 10,
            ..NhmmerFmSeedConfig::default()
        },
    );
    assert_eq!(hits.len(), 1);
    assert!(
            hits[0].0.model_end >= 30,
            "consensus trie should spend a tight source cap on the strongest target-supported seed first: {hits:?}"
        );
}

#[test]
fn nhmmer_fm_seed_extension_scores_diagonal_flanks() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = b"TTACGTAA";
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(6, AlphabetType::Dna, abc.k);
    for node in 1..=hmm.m {
        for x in 0..abc.k {
            hmm.mat[node][x] = 0.01;
        }
    }
    for (node, ch) in [
        (1, b'A'),
        (2, b'C'),
        (3, b'G'),
        (4, b'T'),
        (5, b'A'),
        (6, b'C'),
    ] {
        let code = abc.inmap[ch as usize] as usize;
        hmm.mat[node][code] = 0.97;
    }

    let seed = NhmmerConsensusSeed {
        bases: b"CGT".to_vec(),
        model_end: 4,
    };
    let (start, len, model_end, score_bits) =
        fm_extend_seed_diagonal(&seq, &hmm, &Bg::new(&abc), &seed, 3, 6);
    assert_eq!((start, len, model_end), (2, 5, 5));
    assert!(
        (score_bits - 9.780284).abs() < 1e-5,
        "unexpected extended diagonal score: {score_bits}"
    );
}

#[test]
fn nhmmer_fm_seed_extension_selects_best_bounded_subdiagonal_like_c() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = b"ACGT";
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
    for node in 1..=hmm.m {
        for x in 0..abc.k {
            hmm.mat[node][x] = 0.01;
        }
    }
    for (node, ch, p) in [
        (1, b'A', 0.97),
        (2, b'C', 0.01),
        (3, b'G', 0.97),
        (4, b'T', 0.97),
    ] {
        let code = abc.inmap[ch as usize] as usize;
        hmm.mat[node][code] = p;
    }

    let seed = NhmmerConsensusSeed {
        bases: text.to_vec(),
        model_end: 4,
    };
    let (start, len, model_end, score_bits) =
        fm_extend_seed_diagonal(&seq, &hmm, &Bg::new(&abc), &seed, 0, 6);
    assert_eq!(
            (start, len, model_end),
            (2, 2, 4),
            "C FM_extendSeed keeps the highest-scoring clipped sub-diagonal before FM_window_from_diag derives k+length-1"
        );
    assert!(score_bits > 0.0);
}

#[test]
fn nhmmer_fm_seed_window_emission_filters_nonpositive_extended_diagonal() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = b"ACGT";
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(text);
    seq.n = text.len();
    seq.l = text.len();

    let seq_meta = NhmmerFmSequenceMeta {
        name: "target".to_string(),
        acc: String::new(),
        desc: String::new(),
        target_start: 1,
        fm_start: 0,
        length: text.len(),
    };
    let record = NhmmerFmIndexRecord {
        block_id: 0,
        kind: 0,
        text_start: 0,
        text_len: text.len(),
        text: Arc::from(text.to_vec()),
        packed_text: Arc::from([]),
        mapped_packed_text: None,
        seq_offset: 0,
        seq_count: 1,
        ambig_offset: 0,
        ambig_count: 0,
        overlap_bases: 0,
        fm: FmIndex::build(&text.iter().rev().copied().collect::<Vec<_>>()),
    };
    let seed = NhmmerConsensusSeed {
        bases: text.to_vec(),
        model_end: text.len(),
    };

    let mut low_hmm = Hmm::new(text.len(), AlphabetType::Dna, abc.k);
    low_hmm.name = "query".to_string();
    for node in 1..=low_hmm.m {
        for x in 0..abc.k {
            low_hmm.mat[node][x] = 0.25;
        }
    }

    let bg = Bg::new(&abc);
    let mut windows = Vec::new();
    fm_push_seed_window(
        &mut windows,
        &[],
        &seq_meta,
        &seq,
        &record,
        &low_hmm,
        &bg,
        false,
        &seed,
        0,
        text.len(),
    );
    assert!(
            windows.is_empty(),
            "C FM_extendSeed only sends SSV-passing positive diagonals to FM_window_from_diag: {windows:?}"
        );

    let mut high_hmm = low_hmm;
    for (offset, &ch) in text.iter().enumerate() {
        let node = offset + 1;
        for x in 0..abc.k {
            high_hmm.mat[node][x] = 0.01;
        }
        let code = abc.inmap[ch as usize] as usize;
        high_hmm.mat[node][code] = 0.97;
    }

    fm_push_seed_window(
        &mut windows,
        &[],
        &seq_meta,
        &seq,
        &record,
        &high_hmm,
        &bg,
        false,
        &seed,
        0,
        text.len(),
    );
    assert_eq!(windows.len(), 1);
    assert_eq!(
        windows[0].window,
        NhmmerCandidateWindow {
            n: 1,
            length: text.len(),
            k: text.len(),
            fm_frame: None,
        }
    );
    assert!(windows[0].score_bits > 0.0);
}

#[test]
fn nhmmer_fm_seed_windows_merge_close_same_diagonal_runs() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = b"TTACGTAC";
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(6, AlphabetType::Dna, abc.k);
    for node in 1..=hmm.m {
        for x in 0..abc.k {
            hmm.mat[node][x] = 0.01;
        }
    }
    for (node, ch) in [
        (1, b'A'),
        (2, b'C'),
        (3, b'G'),
        (4, b'T'),
        (5, b'A'),
        (6, b'C'),
    ] {
        let code = abc.inmap[ch as usize] as usize;
        hmm.mat[node][code] = 0.97;
    }

    let windows = fm_merge_seed_windows(
        vec![
            NhmmerScoredCandidateWindow {
                window: NhmmerCandidateWindow {
                    n: 3,
                    length: 4,
                    k: 4,
                    fm_frame: None,
                },
                score_bits: 0.0,
            },
            NhmmerScoredCandidateWindow {
                window: NhmmerCandidateWindow {
                    n: 5,
                    length: 4,
                    k: 6,
                    fm_frame: None,
                },
                score_bits: 0.0,
            },
        ],
        6,
        &seq,
        &hmm,
        &Bg::new(&abc),
    );

    assert_eq!(windows.len(), 1);
    assert_eq!(
        windows[0].window,
        NhmmerCandidateWindow {
            n: 3,
            length: 6,
            k: 6,
            fm_frame: None,
        }
    );
    assert!(
        (windows[0].score_bits - 11.736341).abs() < 1e-5,
        "merged diagonal should be rescored over its full span: {windows:?}"
    );
}

#[test]
fn nhmmer_fm_seed_hits_use_score_threshold_seed_strings() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let seed_text = b"ACGTACGTACGTACG";
    let text = b"TTACGTACGTACGTACGAA";
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(seed_text.len(), AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b'T'; seed_text.len() + 2]);
    hmm.consensus.as_mut().unwrap()[0] = b' ';
    hmm.consensus.as_mut().unwrap()[seed_text.len() + 1] = b' ';
    for (node, &ch) in seed_text.iter().enumerate() {
        let node = node + 1;
        for x in 0..abc.k {
            hmm.mat[node][x] = 0.01;
        }
        let code = abc.inmap[ch as usize] as usize;
        hmm.mat[node][code] = 0.97;
    }

    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: text.len(),
        }],
        fm_ambiguities: Vec::new(),
        fm_index_records: vec![NhmmerFmIndexRecord {
            block_id: 0,
            kind: 0,
            text_start: 0,
            text_len: text.len(),
            text: Arc::from(text.to_vec()),
            packed_text: Arc::from([]),
            mapped_packed_text: None,
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 0,
            fm: FmIndex::build(&reversed_text),
        }],
    };

    let windows =
        fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 15, false).unwrap();
    assert!(
        windows
            .iter()
            .any(|window| window.n == 3 && window.length == seed_text.len() && window.k >= 8),
        "score-threshold seed should create a bounded FM candidate window: {windows:?}"
    );
}

#[test]
fn nhmmer_fm_model_score_trie_reaches_target_pruned_late_model_seed() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let motif = b"CCGTAACGTTACGTA";
    let text = [b"GG".as_slice(), motif, b"GG".as_slice()].concat();
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(&text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(290, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b'T'; hmm.m + 2]);
    hmm.consensus.as_mut().unwrap()[0] = b' ';
    hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
    for node in 1..=hmm.m {
        for x in 0..abc.k {
            hmm.mat[node][x] = 0.01;
        }
        let ch = if (270..270 + motif.len()).contains(&node) {
            motif[node - 270]
        } else {
            b'A'
        };
        let code = abc.inmap[ch as usize] as usize;
        hmm.mat[node][code] = 0.97;
    }

    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: text.len(),
        }],
        fm_ambiguities: Vec::new(),
        fm_index_records: vec![NhmmerFmIndexRecord {
            block_id: 0,
            kind: 0,
            text_start: 0,
            text_len: text.len(),
            text: Arc::from(text.clone()),
            packed_text: Arc::from([]),
            mapped_packed_text: None,
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 0,
            fm: FmIndex::build(&reversed_text),
        }],
    };

    let windows =
        fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 20, false).unwrap();
    assert!(
        windows.iter().any(|window| {
            let window_start0 = window.n - 1;
            let window_end0 = window_start0 + window.length;
            window_start0 <= 2 && window_end0 >= 2 + motif.len() && window.k >= 277
        }),
        "FM-pruned model score recursion should reach a late target-supported seed: {windows:?}"
    );
}

#[test]
fn nhmmer_fm_model_score_trie_augments_existing_consensus_windows() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let early_motif = b"ACGTACGTACGT";
    let late_motif = b"CCGTAACGTTACGTA";
    let text = [
        b"GG".as_slice(),
        early_motif.as_slice(),
        b"GG".as_slice(),
        late_motif.as_slice(),
        b"GG".as_slice(),
    ]
    .concat();
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(&text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(300, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b'T'; hmm.m + 2]);
    hmm.consensus.as_mut().unwrap()[0] = b' ';
    hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
    for node in 1..=hmm.m {
        for x in 0..abc.k {
            hmm.mat[node][x] = 0.01;
        }
        let code = abc.inmap[b'A' as usize] as usize;
        hmm.mat[node][code] = 0.97;
    }
    for (offset, &ch) in early_motif.iter().enumerate() {
        let node = 1 + offset;
        hmm.consensus.as_mut().unwrap()[node] = ch;
        let code = abc.inmap[ch as usize] as usize;
        hmm.mat[node][code] = 0.97;
    }
    for (offset, &ch) in late_motif.iter().enumerate() {
        let node = 280 + offset;
        let code = abc.inmap[ch as usize] as usize;
        hmm.mat[node][code] = 0.97;
    }

    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: text.len(),
        }],
        fm_ambiguities: Vec::new(),
        fm_index_records: vec![NhmmerFmIndexRecord {
            block_id: 0,
            kind: 0,
            text_start: 0,
            text_len: text.len(),
            text: Arc::from(text.clone()),
            packed_text: Arc::from([]),
            mapped_packed_text: None,
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 0,
            fm: FmIndex::build(&reversed_text),
        }],
    };

    let windows =
        fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 20, false).unwrap();
    assert!(
        windows
            .iter()
            .any(|window| window.n == 3 && window.length >= early_motif.len() && window.k <= 20),
        "consensus tier should still create the early FM window: {windows:?}"
    );
    assert!(
            windows.iter().any(|window| {
                let window_start0 = window.n - 1;
                let window_end0 = window_start0 + window.length;
                window_start0 <= 16 && window_end0 >= 16 + late_motif.len() && window.k >= 294
            }),
            "normal score-trie tier should add the late score-only FM window without waiting for empty earlier tiers: {windows:?}"
        );
}

#[test]
fn nhmmer_fm_model_score_trie_ranks_starts_before_source_cap() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let weak_motif = b"ACGTACGTACGTACG";
    let strong_motif = b"CCGTAACGTTACGTA";
    let text = [
        b"GG".as_slice(),
        weak_motif,
        b"GG".as_slice(),
        strong_motif,
        b"GG".as_slice(),
    ]
    .concat();
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();

    let mut hmm = Hmm::new(45, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b'A'; hmm.m + 2]);
    hmm.consensus.as_mut().unwrap()[0] = b' ';
    hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
    for node in 1..=hmm.m {
        for x in 0..abc.k {
            hmm.mat[node][x] = 0.25;
        }
    }
    for (offset, &ch) in weak_motif.iter().enumerate() {
        let node = 1 + offset;
        let code = abc.inmap[ch as usize] as usize;
        hmm.mat[node][code] = 0.75;
    }
    for (offset, &ch) in strong_motif.iter().enumerate() {
        let node = 25 + offset;
        let code = abc.inmap[ch as usize] as usize;
        hmm.mat[node][code] = 0.97;
    }

    let hits = fm_locate_model_score_seeds(
        &FmIndex::build(&reversed_text),
        &hmm,
        &Bg::new(&abc),
        false,
        text.len(),
        text.len(),
        1,
    );
    assert_eq!(hits.len(), 1);
    assert!(
        hits[0].0.model_end >= 32,
        "tight score-trie source cap should prefer the stronger late seed: {hits:?}"
    );
}

#[test]
fn nhmmer_fm_final_windows_keep_more_than_4096_plausible_seed_windows() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let window_count = NHMMER_FM_WINDOW_LIMIT_REGRESSION_SIZE + 2;
    let text = vec![b'A'; window_count];
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(&text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(1, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b' ', b'A', b' ']);
    let bg = Bg::new(&abc);

    let scored_windows: Vec<NhmmerScoredCandidateWindow> = (0..window_count)
        .map(|idx| NhmmerScoredCandidateWindow {
            window: NhmmerCandidateWindow {
                n: idx + 1,
                length: 1,
                k: 1,
                fm_frame: None,
            },
            score_bits: (window_count - idx) as f32,
        })
        .collect();

    let windows = fm_finalize_seed_windows(scored_windows, 1, &seq, &hmm, &bg, false).unwrap();
    assert_eq!(windows.len(), window_count);
    assert_eq!(windows.first().unwrap().n, 1);
    assert_eq!(windows.last().unwrap().n, window_count);
    assert!(
        windows
            .iter()
            .any(|window| window.n == NHMMER_FM_WINDOW_LIMIT_REGRESSION_SIZE + 1),
        "late plausible FM seed windows must not be dropped before long-target scanning"
    );
}

#[test]
fn nhmmer_fm_consensus_trie_ranks_short_suffix_starts_before_source_cap() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let bg = Bg::new(&abc);
    let early_seed = b"ACGTACGTAC";
    let late_seed = b"CCGTAACGTT";
    let consensus = [early_seed.as_slice(), b"A".as_slice(), late_seed.as_slice()].concat();
    let text = [
        early_seed.as_slice(),
        b"GG".as_slice(),
        late_seed.as_slice(),
    ]
    .concat();
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();

    let mut hmm = Hmm::new(consensus.len(), AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b' '; hmm.m + 2]);
    for (offset, &ch) in consensus.iter().enumerate() {
        hmm.consensus.as_mut().unwrap()[offset + 1] = ch;
    }
    for node in 1..=hmm.m {
        for x in 0..abc.k {
            hmm.mat[node][x] = 0.01;
        }
        let code = abc.inmap[consensus[node - 1] as usize] as usize;
        hmm.mat[node][code] = if node <= early_seed.len() { 0.35 } else { 0.97 };
    }

    let hits = fm_locate_model_consensus_seeds(
        &FmIndex::build(&reversed_text),
        &hmm,
        &bg,
        false,
        text.len(),
        text.len(),
        1,
        NhmmerFmSeedConfig {
            consensus_match_req: 10,
            ..NhmmerFmSeedConfig::default()
        },
    );

    assert_eq!(hits.len(), 1);
    assert_eq!(
            hits[0].0.model_end,
            consensus.len(),
            "tight consensus-trie source cap should rank the high-scoring 10-mer suffix before an earlier weaker seed: {hits:?}"
        );
    assert_eq!(hits[0].0.bases, late_seed);
}

#[test]
fn nhmmer_fm_model_score_reverse_trie_locates_target_supported_seed() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let motif = b"CCGTAACGTTACGTA";
    let text = [b"GG".as_slice(), motif, b"TT".as_slice()].concat();
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();

    let mut hmm = Hmm::new(motif.len(), AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b'A'; hmm.m + 2]);
    hmm.consensus.as_mut().unwrap()[0] = b' ';
    hmm.consensus.as_mut().unwrap()[hmm.m + 1] = b' ';
    for node in 1..=hmm.m {
        for x in 0..abc.k {
            hmm.mat[node][x] = 0.01;
        }
        let code = abc.inmap[motif[node - 1] as usize] as usize;
        hmm.mat[node][code] = 0.97;
    }

    let bg = Bg::new(&abc);
    let (abc, max_depth, lod_bits, _best_suffix, best_prefix, consensus_codes) =
        nhmmer_fm_lod_tables(&hmm, &bg, NhmmerFmSeedConfig::default()).unwrap();
    let fm = FmIndex::build(&reversed_text);
    let mut hits = Vec::new();
    let mut current_rev = Vec::new();
    fm_model_score_seed_reverse_dfs(
        &fm,
        hmm.m,
        hmm.m,
        max_depth,
        false,
        &mut current_rev,
        &lod_bits,
        &best_prefix,
        &consensus_codes,
        &abc,
        &mut hits,
        64,
        NhmmerFmScoreState::new(),
        NhmmerFmSeedConfig::default(),
    );

    assert!(
        hits.iter().any(|(seed, fm_pos)| {
            seed.model_end == motif.len()
                && seed.bases == motif
                && *fm_pos == text.len() - 2 - motif.len()
        }),
        "reverse model-direction score trie should locate the full target-supported seed: {hits:?}"
    );
}

#[test]
fn nhmmer_fm_reverse_score_trie_prefix_interval_matches_seed_orientation() {
    let text = b"TTACGTAA";
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
    let fm = FmIndex::build(&reversed_text);

    let full_seed_interval = fm_interval_for_model_seed(&fm, b"ACGT", false).unwrap();
    let reverse_prefix_interval =
        fm_interval_for_reverse_model_prefix(&fm, b"TGCA", false).unwrap();
    let mut full_positions = fm.locate_interval(full_seed_interval);
    let mut reverse_prefix_positions = fm.locate_interval(reverse_prefix_interval);
    full_positions.sort();
    reverse_prefix_positions.sort();
    assert_eq!(reverse_prefix_positions, full_positions);

    assert!(
        fm_interval_for_reverse_model_prefix(&fm, b"TC", false).is_none(),
        "reverse score recursion should prune a target-absent suffix before deeper model traversal"
    );
}

#[test]
fn nhmmer_score_threshold_seed_strings_include_reverse_model_direction() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let motif = b"CCGTAACGTTACGTA";
    let mut hmm = Hmm::new(motif.len(), AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    for node in 1..=hmm.m {
        for x in 0..abc.k {
            hmm.mat[node][x] = 0.01;
        }
        let code = abc.inmap[motif[node - 1] as usize] as usize;
        hmm.mat[node][code] = 0.97;
    }

    let bg = Bg::new(&abc);
    let (abc, max_depth, lod_bits, _best_suffix, best_prefix, consensus_codes) =
        nhmmer_fm_lod_tables(&hmm, &bg, NhmmerFmSeedConfig::default()).unwrap();
    let mut seeds = Vec::new();
    let mut current_rev = Vec::new();
    nhmmer_score_seed_reverse_dfs(
        hmm.m,
        hmm.m,
        max_depth,
        &mut current_rev,
        &lod_bits,
        &best_prefix,
        &consensus_codes,
        &abc,
        &mut seeds,
        NHMMER_FM_SCORE_SEED_LIMIT,
        NhmmerFmScoreState::new(),
        NhmmerFmSeedConfig::default(),
    );

    assert!(
        seeds
            .iter()
            .any(|seed| seed.model_end == motif.len() && seed.bases == motif),
        "materialized reverse model-direction score seeds should include the full motif: {seeds:?}"
    );
}

#[test]
fn nhmmer_fm_score_trie_prunes_nonpositive_prefixes_like_c() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let motif = b"ACCGTAACGTTACGT";
    let mut hmm = Hmm::new(motif.len(), AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    for x in 0..abc.k {
        hmm.mat[1][x] = 0.25;
    }
    for (offset, &ch) in motif[1..].iter().enumerate() {
        let node = offset + 2;
        for x in 0..abc.k {
            hmm.mat[node][x] = 0.01;
        }
        let code = abc.inmap[ch as usize] as usize;
        hmm.mat[node][code] = 0.97;
    }

    let bg = Bg::new(&abc);
    let (abc, max_depth, lod_bits, best_suffix, _best_prefix, consensus_codes) =
        nhmmer_fm_lod_tables(&hmm, &bg, NhmmerFmSeedConfig::default()).unwrap();

    let mut materialized = Vec::new();
    let mut current = Vec::new();
    nhmmer_score_seed_dfs(
        1,
        1,
        max_depth,
        &mut current,
        &lod_bits,
        &best_suffix,
        &consensus_codes,
        &abc,
        &mut materialized,
        NHMMER_FM_SCORE_SEED_LIMIT,
        NhmmerFmScoreState::new(),
        NhmmerFmSeedConfig::default(),
    );
    assert!(
        materialized.is_empty(),
        "C FM_Recurse prunes branches whose accumulated score is <= 0: {materialized:?}"
    );

    let text = motif;
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
    let fm = FmIndex::build(&reversed_text);
    let mut located = Vec::new();
    let mut current = Vec::new();
    fm_model_score_seed_dfs(
        &fm,
        1,
        1,
        max_depth,
        fm.root_interval(),
        false,
        &mut current,
        &lod_bits,
        &best_suffix,
        &consensus_codes,
        &abc,
        &mut located,
        NHMMER_FM_SCORE_SEED_LIMIT,
        NhmmerFmScoreState::new(),
        NhmmerFmSeedConfig::default(),
    );
    assert!(
        located.is_empty(),
        "FM-backed score recursion should apply the same nonpositive-score pruning: {located:?}"
    );
}

#[test]
fn nhmmer_fm_score_trie_prunes_low_density_unless_consensus_supported() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let motif = b"ACGTAACGTTACGTA";

    let mut low_density_hmm = Hmm::new(motif.len(), AlphabetType::Dna, abc.k);
    low_density_hmm.name = "query".to_string();
    low_density_hmm.consensus = Some(vec![b'T'; low_density_hmm.m + 2]);
    low_density_hmm.consensus.as_mut().unwrap()[0] = b' ';
    low_density_hmm.consensus.as_mut().unwrap()[low_density_hmm.m + 1] = b' ';
    for (offset, &ch) in motif.iter().enumerate() {
        let node = offset + 1;
        for x in 0..abc.k {
            low_density_hmm.mat[node][x] = 0.01;
        }
        let code = abc.inmap[ch as usize] as usize;
        low_density_hmm.mat[node][code] = if node <= 5 { 0.36 } else { 0.97 };
    }

    let bg = Bg::new(&abc);
    let (abc, max_depth, lod_bits, best_suffix, _best_prefix, consensus_codes) =
        nhmmer_fm_lod_tables(&low_density_hmm, &bg, NhmmerFmSeedConfig::default()).unwrap();
    let reversed_text: Vec<u8> = motif.iter().rev().copied().collect();
    let fm = FmIndex::build(&reversed_text);

    let mut materialized = Vec::new();
    let mut current = Vec::new();
    nhmmer_score_seed_dfs(
        1,
        1,
        max_depth,
        &mut current,
        &lod_bits,
        &best_suffix,
        &consensus_codes,
        &abc,
        &mut materialized,
        NHMMER_FM_SCORE_SEED_LIMIT,
        NhmmerFmScoreState::new(),
        NhmmerFmSeedConfig::default(),
    );
    assert!(
            materialized.is_empty(),
            "C FM_Recurse prunes non-consensus branches whose score density falls too low: {materialized:?}"
        );

    let mut located = Vec::new();
    let mut current = Vec::new();
    fm_model_score_seed_dfs(
        &fm,
        1,
        1,
        max_depth,
        fm.root_interval(),
        false,
        &mut current,
        &lod_bits,
        &best_suffix,
        &consensus_codes,
        &abc,
        &mut located,
        NHMMER_FM_SCORE_SEED_LIMIT,
        NhmmerFmScoreState::new(),
        NhmmerFmSeedConfig::default(),
    );
    assert!(
        located.is_empty(),
        "FM-backed recursion should apply the same C score-density prune: {located:?}"
    );

    let mut consensus_hmm = low_density_hmm;
    consensus_hmm.consensus = Some(vec![b' '; consensus_hmm.m + 2]);
    for (offset, &ch) in motif.iter().enumerate() {
        consensus_hmm.consensus.as_mut().unwrap()[offset + 1] = ch;
    }
    let (abc, max_depth, lod_bits, best_suffix, _best_prefix, consensus_codes) =
        nhmmer_fm_lod_tables(&consensus_hmm, &bg, NhmmerFmSeedConfig::default()).unwrap();
    let mut located = Vec::new();
    let mut current = Vec::new();
    fm_model_score_seed_dfs(
        &fm,
        1,
        1,
        max_depth,
        fm.root_interval(),
        false,
        &mut current,
        &lod_bits,
        &best_suffix,
        &consensus_codes,
        &abc,
        &mut located,
        NHMMER_FM_SCORE_SEED_LIMIT,
        NhmmerFmScoreState::new(),
        NhmmerFmSeedConfig::default(),
    );
    assert!(
            located
                .iter()
                .any(|(seed, _)| seed.model_end == motif.len() && seed.bases == motif),
            "a continuous consensus-supported path should bypass the C density prune and remain extendable: {located:?}"
        );
}

#[test]
fn nhmmer_fm_seed_queries_share_intervals_and_prune_missing_branches() {
    let text = b"ACGTACGTAAAA";
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
    let fm = FmIndex::build(&reversed_text);
    let seed_a = NhmmerConsensusSeed {
        bases: b"ACGT".to_vec(),
        model_end: 4,
    };
    let seed_b = NhmmerConsensusSeed {
        bases: b"TCGT".to_vec(),
        model_end: 4,
    };
    let seed_c = NhmmerConsensusSeed {
        bases: b"AAAA".to_vec(),
        model_end: 4,
    };
    let seeds = vec![seed_a, seed_b, seed_c];
    let abc = Alphabet::new(AlphabetType::Dna);
    let hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
    let bg = Bg::new(&abc);
    let record = NhmmerFmIndexRecord {
        block_id: 0,
        kind: 0,
        text_start: 0,
        text_len: 12,
        text: Arc::from(b"ACGTACGTACGT".to_vec()),
        packed_text: Arc::from([]),
        mapped_packed_text: None,
        seq_offset: 0,
        seq_count: 1,
        ambig_offset: 0,
        ambig_count: 0,
        overlap_bases: 0,
        fm,
    };
    let queries = fm_seed_queries_for_record(&seeds, &record, 12, false, &hmm, &bg);
    let mut located: Vec<(Vec<u8>, usize)> =
        fm_locate_seed_queries(&record.fm, &queries, usize::MAX)
            .into_iter()
            .map(|(seed, pos)| (seed.bases.clone(), pos))
            .collect();
    located.sort();

    assert_eq!(
        located,
        vec![
            (b"AAAA".to_vec(), 0),
            (b"ACGT".to_vec(), 4),
            (b"ACGT".to_vec(), 8)
        ]
    );
}

#[test]
fn nhmmer_fm_seed_queries_rank_identical_terminal_seeds_before_source_cap() {
    let text = b"ACGTACGT";
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
    let fm = FmIndex::build(&reversed_text);
    let abc = Alphabet::new(AlphabetType::Dna);
    let bg = Bg::new(&abc);
    let mut hmm = Hmm::new(8, AlphabetType::Dna, abc.k);
    for node in 1..=hmm.m {
        for x in 0..abc.k {
            hmm.mat[node][x] = 0.24;
        }
    }
    for (node, ch, p) in [
        (1, b'A', 0.35),
        (2, b'C', 0.35),
        (3, b'G', 0.35),
        (4, b'T', 0.35),
        (5, b'A', 0.97),
        (6, b'C', 0.97),
        (7, b'G', 0.97),
        (8, b'T', 0.97),
    ] {
        let code = abc.inmap[ch as usize] as usize;
        hmm.mat[node][code] = p;
    }

    let low_seed = NhmmerConsensusSeed {
        bases: b"ACGT".to_vec(),
        model_end: 4,
    };
    let high_seed = NhmmerConsensusSeed {
        bases: b"ACGT".to_vec(),
        model_end: 8,
    };
    let record = NhmmerFmIndexRecord {
        block_id: 0,
        kind: 0,
        text_start: 0,
        text_len: text.len(),
        text: Arc::from(text.to_vec()),
        packed_text: Arc::from([]),
        mapped_packed_text: None,
        seq_offset: 0,
        seq_count: 1,
        ambig_offset: 0,
        ambig_count: 0,
        overlap_bases: 0,
        fm,
    };
    let seeds = vec![low_seed, high_seed];
    let queries = fm_seed_queries_for_record(&seeds, &record, text.len(), false, &hmm, &bg);

    let located = fm_locate_seed_queries(&record.fm, &queries, 1);
    assert_eq!(located.len(), 1);
    assert_eq!(
        located[0].0.model_end, 8,
        "tight FM source cap should prefer the higher-scoring model coordinate"
    );
}

#[test]
fn nhmmer_fm_seed_hits_skip_ambiguous_spans() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = b"TTACGTACAA";
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);

    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: text.len(),
        }],
        fm_ambiguities: vec![(3, 4)],
        fm_index_records: vec![NhmmerFmIndexRecord {
            block_id: 0,
            kind: 0,
            text_start: 0,
            text_len: text.len(),
            text: Arc::from(text.to_vec()),
            packed_text: Arc::from([]),
            mapped_packed_text: None,
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 1,
            overlap_bases: 0,
            fm: FmIndex::build(&reversed_text),
        }],
    };

    assert!(fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 6, false).is_none());
}

#[test]
fn nhmmer_fm_seed_hits_include_block_text_start() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = b"TTACGTACAA";
    let block_text = &text[2..];
    let reversed_block_text: Vec<u8> = block_text.iter().rev().copied().collect();
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);

    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: text.len(),
        }],
        fm_ambiguities: Vec::new(),
        fm_index_records: vec![NhmmerFmIndexRecord {
            block_id: 1,
            kind: 0,
            text_start: 2,
            text_len: block_text.len(),
            text: Arc::from(block_text.to_vec()),
            packed_text: Arc::from([]),
            mapped_packed_text: None,
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 0,
            fm: FmIndex::build(&reversed_block_text),
        }],
    };

    let windows = fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 6, false).unwrap();
    assert_eq!(
        windows,
        vec![NhmmerCandidateWindow {
            n: 1,
            length: 4,
            k: 4,
            fm_frame: None,
        }]
    );
}

#[test]
fn nhmmer_fm_seed_hits_use_block_local_ambiguity_coordinates() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = b"TTACGTACAA";
    let block_text = &text[2..];
    let reversed_block_text: Vec<u8> = block_text.iter().rev().copied().collect();
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(block_text);
    seq.n = block_text.len();
    seq.l = block_text.len();

    let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);

    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: block_text.len(),
        }],
        fm_ambiguities: vec![(1, 2)],
        fm_index_records: vec![NhmmerFmIndexRecord {
            block_id: 1,
            kind: 0,
            text_start: 2,
            text_len: block_text.len(),
            text: Arc::from(block_text.to_vec()),
            packed_text: Arc::from([]),
            mapped_packed_text: None,
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 1,
            overlap_bases: 0,
            fm: FmIndex::build(&reversed_block_text),
        }],
    };

    assert!(fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 6, false).is_none());
}

#[test]
fn nhmmer_fm_seed_hits_skip_wholly_overlapped_block_prefix() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = b"ACGTACAA";
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);

    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: text.len(),
        }],
        fm_ambiguities: Vec::new(),
        fm_index_records: vec![NhmmerFmIndexRecord {
            block_id: 1,
            kind: 0,
            text_start: 7,
            text_len: text.len(),
            text: Arc::from(text.to_vec()),
            packed_text: Arc::from([]),
            mapped_packed_text: None,
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 4,
            fm: FmIndex::build(&reversed_text),
        }],
    };

    assert!(fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 6, false).is_none());
}

#[test]
fn nhmmer_fm_seed_hits_keep_boundary_spanning_overlap_hits() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = b"TACGTCAA";
    let reversed_text: Vec<u8> = text.iter().rev().copied().collect();
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);

    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: text.len(),
        }],
        fm_ambiguities: Vec::new(),
        fm_index_records: vec![NhmmerFmIndexRecord {
            block_id: 1,
            kind: 0,
            text_start: 7,
            text_len: text.len(),
            text: Arc::from(text.to_vec()),
            packed_text: Arc::from([]),
            mapped_packed_text: None,
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 2,
            fm: FmIndex::build(&reversed_text),
        }],
    };

    let windows = fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 6, false).unwrap();
    assert_eq!(
        windows,
        vec![NhmmerCandidateWindow {
            n: 2,
            length: 4,
            k: 4,
            fm_frame: None,
        }]
    );
}

#[test]
fn nhmmer_fm_reverse_seed_hits_create_crick_candidate_windows() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = b"AACCTTAC";
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(3, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b' ', b'A', b'A', b'G', b' ']);

    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: text.len(),
        }],
        fm_ambiguities: Vec::new(),
        fm_index_records: vec![NhmmerFmIndexRecord {
            block_id: 0,
            kind: 1,
            text_start: 0,
            text_len: text.len(),
            text: Arc::from(text.to_vec()),
            packed_text: Arc::from([]),
            mapped_packed_text: None,
            seq_offset: 0,
            seq_count: 1,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: 0,
            fm: FmIndex::build(text),
        }],
    };

    let windows = fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 5, true).unwrap();
    assert_eq!(
        windows,
        vec![NhmmerCandidateWindow {
            n: 3,
            length: 3,
            k: 3,
            fm_frame: None,
        }]
    );
    assert!(fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 5, false).is_none());
}

#[test]
fn nhmmer_raw_c_stream_records_create_seed_indexes() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = b"TTACGTACAA";
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);

    let mut reversed_text = text.to_vec();
    reversed_text.reverse();
    let mut records_by_block = vec![test_raw_fm_record_with_target_text(
        &reversed_text,
        text,
        4,
        8,
    )];
    let mut reverse_records_by_block = vec![Some(test_raw_fm_record(text, 4, 8))];
    let (fm_index_records, _) = build_raw_c_stream_fm_index_records(
        &mut records_by_block,
        &mut reverse_records_by_block,
        false,
        None,
    )
    .unwrap();
    assert_eq!(fm_index_records.len(), 2);
    assert_eq!(fm_index_records[0].kind, 0);
    assert_eq!(fm_index_records[1].kind, 1);

    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: text.len(),
        }],
        fm_ambiguities: Vec::new(),
        fm_index_records,
    };

    let windows = fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 6, false).unwrap();
    assert_eq!(
        windows,
        vec![NhmmerCandidateWindow {
            n: 3,
            length: 4,
            k: 4,
            fm_frame: None,
        }]
    );
}

#[test]
fn nhmmer_raw_c_stream_reverse_record_creates_crick_seed_windows() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let text = b"AACCTTAC";
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(text);
    seq.n = text.len();
    seq.l = text.len();

    let mut hmm = Hmm::new(3, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b' ', b'A', b'A', b'G', b' ']);

    let mut reversed_text = text.to_vec();
    reversed_text.reverse();
    let mut records_by_block = vec![test_raw_fm_record_with_target_text(
        &reversed_text,
        text,
        4,
        8,
    )];
    let mut reverse_records_by_block = vec![Some(test_raw_fm_record(text, 4, 8))];
    let (fm_index_records, _) = build_raw_c_stream_fm_index_records(
        &mut records_by_block,
        &mut reverse_records_by_block,
        false,
        None,
    )
    .unwrap();
    let target_db = NhmmerTargetDb {
        sequences: vec![seq],
        fm_sequence_meta: vec![NhmmerFmSequenceMeta {
            name: "target".to_string(),
            acc: String::new(),
            desc: String::new(),
            target_start: 1,
            fm_start: 0,
            length: text.len(),
        }],
        fm_ambiguities: Vec::new(),
        fm_index_records,
    };

    let windows = fm_seed_candidate_windows(&target_db, 0, &hmm, &Bg::new(&abc), 5, true).unwrap();
    assert_eq!(
        windows,
        vec![NhmmerCandidateWindow {
            n: 3,
            length: 3,
            k: 3,
            fm_frame: None,
        }]
    );
}

#[test]
fn nhmmer_reverse_complements_dna_seed() {
    assert_eq!(reverse_complement_dna_seed(b"AAGT"), b"ACTT");
}

#[test]
fn nhmmer_threshold_config_preserves_model_cutoff_thresholds() {
    let mut outer = Pipeline::new();
    outer.use_bit_cutoffs = tongs::pipeline::BitCutoff::GA;
    outer.by_e = false;
    outer.dom_by_e = false;
    outer.inc_by_e = false;
    outer.incdom_by_e = false;
    outer.t = Some(10.0);
    outer.dom_t = Some(8.0);
    outer.inc_t = Some(10.0);
    outer.inc_dom_t = Some(8.0);

    let mut window = Pipeline::new();
    NhmmerThresholdConfig::from_pipeline(&outer).apply_to(&mut window);

    assert_eq!(window.use_bit_cutoffs, tongs::pipeline::BitCutoff::GA);
    assert!(!window.by_e);
    assert!(!window.dom_by_e);
    assert!(!window.inc_by_e);
    assert!(!window.incdom_by_e);
    assert_eq!(window.t, Some(10.0));
    assert_eq!(window.dom_t, Some(8.0));
    assert_eq!(window.inc_t, Some(10.0));
    assert_eq!(window.inc_dom_t, Some(8.0));
}

fn test_longtarget_domain(reported: bool, included: bool) -> tongs::tophits::Domain {
    tongs::tophits::Domain {
        iali: 1,
        jali: 8,
        ienv: 1,
        jenv: 8,
        bitscore: 10.0,
        lnp: -10.0,
        dombias: 1.0,
        oasc: 0.0,
        envsc: 0.0,
        domcorrection: 0.0,
        is_reported: reported,
        is_included: included,
        ad: None,
    }
}

#[test]
fn nhmmer_longtarget_split_uses_per_domain_bit_cutoff_flags() {
    use tongs::tophits::{Hit, P7_IS_DROPPED, P7_IS_INCLUDED, P7_IS_REPORTED};

    let mut sq = Sequence::new();
    sq.name = "target".to_string();
    sq.n = 100;
    sq.l = 100;

    let src_hit = Hit {
        name: "target".to_string(),
        acc: String::new(),
        desc: String::new(),
        n: 80,
        sortkey: -1.0,
        score: 20.0,
        bias: 0.0,
        pre_score: 20.0,
        sum_score: 20.0,
        lnp: -20.0,
        pre_lnp: -20.0,
        sum_lnp: -20.0,
        nexpected: 2.0,
        nregions: 1,
        nclustered: 1,
        noverlaps: 0,
        nenvelopes: 2,
        ndom: 2,
        nreported: 1,
        nincluded: 1,
        dcl: Vec::new(),
        flags: P7_IS_REPORTED | P7_IS_INCLUDED,
        seqidx: 7,
        subseq_start: 0,
    };
    let mut cutoff_config = NhmmerThresholdConfig::from_pipeline(&Pipeline::new());
    cutoff_config.use_bit_cutoffs = tongs::pipeline::BitCutoff::GA;
    cutoff_config.t = Some(10.0);
    cutoff_config.inc_t = Some(10.0);

    let failed_domain_hit = longtarget_domain_hit(
        &sq,
        &src_hit,
        test_longtarget_domain(false, false),
        cutoff_config,
    );

    assert_eq!(failed_domain_hit.flags & P7_IS_REPORTED, 0);
    assert_eq!(failed_domain_hit.flags & P7_IS_INCLUDED, 0);
    assert!(failed_domain_hit.flags & P7_IS_DROPPED != 0);
    assert_eq!(failed_domain_hit.nreported, 0);
    assert_eq!(failed_domain_hit.nincluded, 0);

    let included_domain_hit = longtarget_domain_hit(
        &sq,
        &src_hit,
        test_longtarget_domain(true, true),
        cutoff_config,
    );

    assert!(included_domain_hit.flags & P7_IS_REPORTED != 0);
    assert!(included_domain_hit.flags & P7_IS_INCLUDED != 0);
    assert_eq!(included_domain_hit.flags & P7_IS_DROPPED, 0);
    assert_eq!(included_domain_hit.nreported, 1);
    assert_eq!(included_domain_hit.nincluded, 1);

    cutoff_config.t = Some(11.0);
    cutoff_config.inc_t = Some(11.0);
    let target_cutoff_failed_hit = longtarget_domain_hit(
        &sq,
        &src_hit,
        test_longtarget_domain(true, true),
        cutoff_config,
    );
    assert_eq!(target_cutoff_failed_hit.flags & P7_IS_REPORTED, 0);
    assert_eq!(target_cutoff_failed_hit.flags & P7_IS_INCLUDED, 0);
    assert!(target_cutoff_failed_hit.flags & P7_IS_DROPPED != 0);
    assert_eq!(target_cutoff_failed_hit.nreported, 0);
    assert_eq!(target_cutoff_failed_hit.nincluded, 0);
}

#[test]
fn nhmmer_fm_empty_window_list_does_not_fallback_to_full_sequence_scan() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let mut seq = Sequence::new();
    seq.name = "target".to_string();
    seq.dsq = abc.digitize(b"ACGTACGTACGT");
    seq.n = 12;
    seq.l = 12;

    let mut hmm = Hmm::new(4, AlphabetType::Dna, abc.k);
    hmm.name = "query".to_string();
    hmm.consensus = Some(vec![b' ', b'A', b'C', b'G', b'T', b' ']);
    for node in 1..=hmm.m {
        for x in 0..abc.k {
            hmm.mat[node][x] = 0.01;
        }
        let ch = hmm.consensus.as_ref().unwrap()[node];
        let code = abc.inmap[ch as usize] as usize;
        hmm.mat[node][code] = 0.97;
    }

    let mut bg = Bg::new(&abc);
    bg.set_filter(hmm.m, &hmm.compo);
    let mut gm = Profile::new(hmm.m, &abc);
    profile::profile_config(&hmm, &bg, &mut gm, 20, P7_LOCAL);
    let om = OProfile::convert(&gm);
    let threshold_config = NhmmerThresholdConfig::from_pipeline(&Pipeline::new());
    let msv_counter = std::sync::atomic::AtomicU64::new(0);
    let bias_counter = std::sync::atomic::AtomicU64::new(0);
    let vit_counter = std::sync::atomic::AtomicU64::new(0);
    let fwd_counter = std::sync::atomic::AtomicU64::new(0);

    let hits = p7_pipeline_longtarget(
        &seq,
        &hmm,
        &gm,
        &om,
        &bg,
        20,
        0.02,
        0.003,
        0.00003,
        false,
        false,
        false,
        42,
        NhmmerBiasWindowLengths::default(),
        threshold_config,
        false,
        &msv_counter,
        &bias_counter,
        &vit_counter,
        &fwd_counter,
        Some(&[]),
        None,
    );

    assert!(
        hits.is_empty(),
        "an explicit empty FM window list should not trigger reconstructed full-sequence scanning"
    );
}
