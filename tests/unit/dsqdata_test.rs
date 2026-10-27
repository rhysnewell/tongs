use super::*;

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir();
    dir.join(format!("hmmerrs_dsqdata_{}_{}", std::process::id(), name))
}

fn cleanup(base: &Path) {
    std::fs::remove_file(base).ok();
    std::fs::remove_file(with_ext(base, "dsqi")).ok();
    std::fs::remove_file(with_ext(base, "dsqm")).ok();
    std::fs::remove_file(with_ext(base, "dsqs")).ok();
}

fn write_le_i64_at(buf: &mut [u8], offset: usize, value: i64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[test]
fn roundtrip_amino_5bit() {
    let abc = Alphabet::amino();
    let seqs = vec![
        Sequence {
            name: "seq1".into(),
            acc: "ACC1".into(),
            desc: "first seq".into(),
            dsq: abc.digitize(b"ACDEFGHIKLMNPQRSTVWY"),
            n: 20,
            l: 20,
            taxid: -1,
        },
        Sequence {
            name: "seq2".into(),
            acc: String::new(),
            desc: String::new(),
            dsq: abc.digitize(b"MKV"),
            n: 3,
            l: 3,
            taxid: -1,
        },
    ];

    let base = tmp("amino");
    write_dsqdata(&base, &seqs, &abc).unwrap();
    let got = read_dsqdata(&base).unwrap();
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].name, "seq1");
    assert_eq!(got[0].acc, "ACC1");
    assert_eq!(got[0].desc, "first seq");
    assert_eq!(got[0].n, 20);
    assert_eq!(got[0].dsq, seqs[0].dsq);
    assert_eq!(got[1].name, "seq2");
    assert_eq!(got[1].n, 3);
    assert_eq!(got[1].dsq, seqs[1].dsq);
    cleanup(&base);
}

#[test]
fn roundtrip_dna_2bit_with_degenerates() {
    let abc = Alphabet::dna();
    // Mix of canonical runs and degenerate residues to exercise both 2-bit
    // and 5-bit packets within one sequence.
    let raw = b"ACGTACGTACGTACGTNNACGTNRYACGTACGTACGT";
    let seqs = vec![Sequence {
        name: "dna1".into(),
        acc: String::new(),
        desc: String::new(),
        dsq: abc.digitize(raw),
        n: raw.len(),
        l: raw.len(),
        taxid: -1,
    }];

    let base = tmp("dna");
    write_dsqdata(&base, &seqs, &abc).unwrap();
    let got = read_dsqdata(&base).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].n, raw.len());
    assert_eq!(got[0].dsq, seqs[0].dsq);
    cleanup(&base);
}

#[test]
fn roundtrip_empty_sequence() {
    let abc = Alphabet::amino();
    let seqs = vec![Sequence {
        name: "empty".into(),
        acc: String::new(),
        desc: String::new(),
        dsq: vec![DSQ_SENTINEL, DSQ_SENTINEL],
        n: 0,
        l: 0,
        taxid: -1,
    }];
    let base = tmp("empty");
    write_dsqdata(&base, &seqs, &abc).unwrap();
    let got = read_dsqdata(&base).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].n, 0);
    assert_eq!(got[0].name, "empty");
    assert_eq!(got[0].dsq, vec![DSQ_SENTINEL, DSQ_SENTINEL]);
    cleanup(&base);
}

#[test]
fn rejects_dsqdata_trailing_sidecar_payload_after_final_record() {
    let records = [IndexRecord {
        metadata_end: 3,
        psq_end: 0,
    }];
    let err = validate_dsqdata_index_extents(
        &records,
        DSQDATA_SIDECAR_HEADER_LEN + 5,
        DSQDATA_SIDECAR_HEADER_LEN + 4,
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("metadata sidecar has trailing bytes"));

    let err = validate_dsqdata_index_extents(
        &records,
        DSQDATA_SIDECAR_HEADER_LEN + 4,
        DSQDATA_SIDECAR_HEADER_LEN + 8,
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("sequence sidecar has trailing packets"));
}

#[test]
fn rejects_zero_sequence_dsqdata_with_payload_bytes() {
    let err = validate_dsqdata_index_extents(
        &[],
        DSQDATA_SIDECAR_HEADER_LEN + 1,
        DSQDATA_SIDECAR_HEADER_LEN,
    )
    .unwrap_err();
    assert!(err.to_string().contains("zero sequences"));
}

#[test]
fn rejects_dsqdata_metadata_padding_after_taxid() {
    let abc = Alphabet::amino();
    let dsq = abc.digitize(b"");
    let packed = pack5(&dsq, 0);
    let metadata = b"empty\0\0\0\xff\xff\xff\xffx".to_vec();
    let metadata_end = metadata.len() as i64 - 1;
    let chunk = RawChunk {
        packed,
        metadata,
        records: vec![IndexRecord {
            metadata_end,
            psq_end: 0,
        }],
    };
    let mut out = Vec::new();
    let err = unpack_chunk(&chunk, true, &mut out).unwrap_err();
    assert!(err
        .to_string()
        .contains("metadata record must end with exactly one taxid"));
}

#[test]
fn pack5_roundtrip_lengths() {
    let abc = Alphabet::amino();
    for len in [0usize, 1, 5, 6, 7, 12, 13, 100] {
        let raw: Vec<u8> = (0..len).map(|i| b"ACDEFGHIKLMNPQRSTVWY"[i % 20]).collect();
        let dsq = abc.digitize(&raw);
        let psq = pack5(&dsq, len);
        let mut out = Vec::new();
        let (l, consumed) = unpack5(&psq, 0, &mut out).unwrap();
        assert_eq!(l, len, "len {}", len);
        assert_eq!(consumed, psq.len());
        assert_eq!(out, &dsq[1..=len]);
    }
}

#[test]
fn bad_magic_rejected() {
    let base = tmp("badmagic");
    // Minimal stub + index with wrong magic.
    std::fs::write(&base, "Easel dsqdata v1 x12345\n").unwrap();
    let mut buf = Vec::new();
    buf.extend_from_slice(&0xdeadbeefu32.to_le_bytes()); // bad magic
    buf.extend_from_slice(&12345u32.to_le_bytes());
    buf.extend_from_slice(&[0u8; 4 * 5 + 8 * 3]);
    std::fs::write(with_ext(&base, "dsqi"), buf).unwrap();
    std::fs::write(with_ext(&base, "dsqm"), Vec::<u8>::new()).unwrap();
    std::fs::write(with_ext(&base, "dsqs"), Vec::<u8>::new()).unwrap();
    let err = read_dsqdata(&base).unwrap_err();
    assert!(err.to_string().contains("bad magic"));
    cleanup(&base);
}

#[test]
fn reads_only_first_dsqdata_stub_line() {
    let base = tmp("stub_first_line_only");
    cleanup(&base);
    let mut stub = b"Easel dsqdata v1 x12345\n".to_vec();
    stub.extend(std::iter::repeat_n(b'x', MAX_DSQDATA_STUB_LINE_LEN + 1));
    std::fs::write(&base, stub).unwrap();

    let err = read_dsqdata(&base).unwrap_err();
    assert!(!err.to_string().contains("maximum supported length"));
    cleanup(&base);
}

#[test]
fn rejects_dsqdata_nseq_that_exceeds_index_records_before_allocation() {
    let base = tmp("bad_nseq");
    cleanup(&base);
    std::fs::write(&base, "Easel dsqdata v1 x12345\n").unwrap();
    let mut idx = Vec::new();
    idx.extend_from_slice(&MAGIC_V1.to_le_bytes());
    idx.extend_from_slice(&12345u32.to_le_bytes());
    idx.extend_from_slice(&(AlphabetType::Amino as u32).to_le_bytes());
    idx.extend_from_slice(&0u32.to_le_bytes());
    idx.extend_from_slice(&0u32.to_le_bytes());
    idx.extend_from_slice(&0u32.to_le_bytes());
    idx.extend_from_slice(&0u32.to_le_bytes());
    idx.extend_from_slice(&0u64.to_le_bytes());
    idx.extend_from_slice(&u64::MAX.to_le_bytes());
    idx.extend_from_slice(&0u64.to_le_bytes());
    std::fs::write(with_ext(&base, "dsqi"), idx).unwrap();

    let err = read_dsqdata(&base).unwrap_err();
    assert!(err.to_string().contains("only 0 index records fit"));
    cleanup(&base);
}

#[test]
fn rejects_dsqdata_nonmonotonic_index_before_loader_allocation() {
    let abc = Alphabet::amino();
    let seqs = vec![
        Sequence {
            name: "seq1".into(),
            acc: String::new(),
            desc: String::new(),
            dsq: abc.digitize(b"ACD"),
            n: 3,
            l: 3,
            taxid: -1,
        },
        Sequence {
            name: "seq2".into(),
            acc: String::new(),
            desc: String::new(),
            dsq: abc.digitize(b"EFG"),
            n: 3,
            l: 3,
            taxid: -1,
        },
    ];

    let base = tmp("bad_index_order");
    write_dsqdata(&base, &seqs, &abc).unwrap();
    let mut idx = std::fs::read(with_ext(&base, "dsqi")).unwrap();
    write_le_i64_at(&mut idx, DSQDATA_INDEX_HEADER_LEN as usize + 16, 0);
    std::fs::write(with_ext(&base, "dsqi"), idx).unwrap();

    let err = read_dsqdata(&base).unwrap_err();
    assert!(err.to_string().contains("not monotonic"));
    cleanup(&base);
}

#[test]
fn rejects_dsqdata_packet_extent_past_sequence_file() {
    let abc = Alphabet::amino();
    let seqs = vec![Sequence {
        name: "seq1".into(),
        acc: String::new(),
        desc: String::new(),
        dsq: abc.digitize(b"ACD"),
        n: 3,
        l: 3,
        taxid: -1,
    }];

    let base = tmp("bad_psq_extent");
    write_dsqdata(&base, &seqs, &abc).unwrap();
    let mut idx = std::fs::read(with_ext(&base, "dsqi")).unwrap();
    write_le_i64_at(&mut idx, DSQDATA_INDEX_HEADER_LEN as usize + 8, 999);
    std::fs::write(with_ext(&base, "dsqi"), idx).unwrap();

    let err = read_dsqdata(&base).unwrap_err();
    assert!(err.to_string().contains("packet extent exceeds"));
    cleanup(&base);
}

/// Full taxid storage round-trip: write sequences carrying distinct
/// (non-(-1) and -1) taxids, read them back, and confirm each taxid is
/// preserved along with the surrounding name/acc/desc/sequence metadata.
/// Exercises both `write_dsqdata` (emits `sq.taxid`) and `unpack_chunk`
/// (stores the parsed int32 into `Sequence::taxid`).
#[test]
fn taxid_metadata_roundtrip() {
    let abc = Alphabet::amino();
    let seqs = vec![
        Sequence {
            name: "taxseq1".into(),
            acc: "AC001".into(),
            desc: "organism A".into(),
            dsq: abc.digitize(b"MKVLWA"),
            n: 6,
            l: 6,
            taxid: 9606,
        },
        Sequence {
            name: "taxseq2".into(),
            acc: String::new(),
            desc: "organism B".into(),
            dsq: abc.digitize(b"ACDEF"),
            n: 5,
            l: 5,
            taxid: -1,
        },
    ];

    let base = tmp("taxid");
    write_dsqdata(&base, &seqs, &abc).unwrap();
    let got = read_dsqdata(&base).unwrap();

    // Metadata (incl. taxid) must survive the write→read cycle intact.
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].name, "taxseq1");
    assert_eq!(got[0].acc, "AC001");
    assert_eq!(got[0].desc, "organism A");
    assert_eq!(got[0].n, 6);
    assert_eq!(got[0].dsq, seqs[0].dsq);
    assert_eq!(got[0].taxid, 9606);
    assert_eq!(got[1].name, "taxseq2");
    assert_eq!(got[1].acc, "");
    assert_eq!(got[1].desc, "organism B");
    assert_eq!(got[1].n, 5);
    assert_eq!(got[1].dsq, seqs[1].dsq);
    assert_eq!(got[1].taxid, -1);

    cleanup(&base);
}

/// Stress the threaded loader by writing and reading more sequences than
/// CHUNK_MAXSEQ in a single database. This exercises the chunked pipeline
/// that spans multiple loader→channel→unpack iterations.
#[test]
fn threaded_loader_multi_chunk() {
    let abc = Alphabet::amino();
    // CHUNK_MAXSEQ is 4096; write 2×CHUNK_MAXSEQ+7 sequences to force
    // three complete loader chunks.
    let n_seqs = CHUNK_MAXSEQ * 2 + 7;
    let amino_chars = b"ACDEFGHIKLMNPQRSTVWY";
    let seqs: Vec<Sequence> = (0..n_seqs)
        .map(|i| {
            let raw: Vec<u8> = (0..((i % 20) + 1))
                .map(|j| amino_chars[(i + j) % 20])
                .collect();
            let n = raw.len();
            Sequence {
                name: format!("seq{}", i),
                acc: String::new(),
                desc: String::new(),
                dsq: abc.digitize(&raw),
                n,
                l: n,
                taxid: -1,
            }
        })
        .collect();

    let base = tmp("multichunk");
    write_dsqdata(&base, &seqs, &abc).unwrap();
    let got = read_dsqdata(&base).unwrap();
    assert_eq!(got.len(), n_seqs);
    for (i, (expected, actual)) in seqs.iter().zip(got.iter()).enumerate() {
        assert_eq!(actual.name, expected.name, "name mismatch at seq {}", i);
        assert_eq!(actual.n, expected.n, "length mismatch at seq {}", i);
        assert_eq!(actual.dsq, expected.dsq, "dsq mismatch at seq {}", i);
    }
    cleanup(&base);
}
