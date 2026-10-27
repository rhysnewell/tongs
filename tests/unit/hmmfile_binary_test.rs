use super::*;
use std::io::Cursor;
use std::process::Command;

fn test_path(relative: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

/// Test helper: run the C `hmmconvert -b` binary on an ASCII HMM and
/// stage the resulting `.h3m` in `dir` for the Rust reader to consume.
fn c_binary_hmm_from_text(text_hmm: &Path, dir: &tempfile::TempDir) -> std::path::PathBuf {
    let output = Command::new(test_path("hmmer/src/hmmconvert"))
        .arg("-b")
        .arg(text_hmm)
        .output()
        .expect("failed to run C hmmconvert");
    assert!(
        output.status.success(),
        "C hmmconvert failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let path = dir.path().join("converted.h3m");
    std::fs::write(&path, output.stdout).unwrap();
    path
}

#[test]
fn rejects_unknown_binary_alphabet_code() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC_3F.to_ne_bytes());
    buf.extend_from_slice(&0i32.to_ne_bytes()); // flags
    buf.extend_from_slice(&1i32.to_ne_bytes()); // M
    buf.extend_from_slice(&99i32.to_ne_bytes()); // invalid alphabet code

    let err = read_binary_hmm(&mut Cursor::new(buf)).unwrap_err();
    assert!(
        matches!(err, HmmerError::Format(msg) if msg.contains("Unknown binary HMM alphabet code"))
    );
}

#[test]
fn rejects_nonpositive_binary_model_length_before_allocating() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC_3F.to_ne_bytes());
    buf.extend_from_slice(&0i32.to_ne_bytes()); // flags
    buf.extend_from_slice(&(-1i32).to_ne_bytes()); // invalid M
    buf.extend_from_slice(&3i32.to_ne_bytes()); // amino

    let err = read_binary_hmm(&mut Cursor::new(buf)).unwrap_err();
    assert!(
        matches!(err, HmmerError::Format(msg) if msg.contains("Invalid binary HMM model length"))
    );
}

#[test]
fn rejects_excessive_binary_model_length_before_allocating() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC_3F.to_ne_bytes());
    buf.extend_from_slice(&0i32.to_ne_bytes()); // flags
    buf.extend_from_slice(&(MAX_BINARY_HMM_MODEL_LENGTH + 1).to_ne_bytes()); // invalid M
    buf.extend_from_slice(&3i32.to_ne_bytes()); // amino

    let err = read_binary_hmm(&mut Cursor::new(buf)).unwrap_err();
    assert!(
        matches!(err, HmmerError::Format(msg) if msg.contains("Invalid binary HMM model length"))
    );
}

#[test]
fn rejects_invalid_binary_string_length() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&MAGIC_3F.to_ne_bytes());
    buf.extend_from_slice(&0i32.to_ne_bytes()); // flags
    buf.extend_from_slice(&1i32.to_ne_bytes()); // M
    buf.extend_from_slice(&3i32.to_ne_bytes()); // amino
    for _ in 0..20 {
        buf.extend_from_slice(&0f32.to_ne_bytes()); // mat[1]
    }
    for _ in 0..40 {
        buf.extend_from_slice(&0f32.to_ne_bytes()); // ins[0..1]
    }
    for _ in 0..14 {
        buf.extend_from_slice(&0f32.to_ne_bytes()); // t[0..1]
    }
    buf.extend_from_slice(&(-2i32).to_ne_bytes()); // non-positive len: C read_bin_string returns NULL

    // C treats any len<=0 as a NULL string (no error at the string level), then
    // rejects the record because the required NAME is absent (p7_hmmfile.c:1565,1654).
    let err = read_binary_hmm(&mut Cursor::new(buf)).unwrap_err();
    assert!(matches!(err, HmmerError::Format(msg) if msg.contains("empty required NAME")));
}

#[test]
fn rejects_binary_string_without_trailing_nul() {
    let mut buf = Cursor::new([3i32.to_ne_bytes().as_slice(), b"abc"].concat());

    let err = read_string(&mut buf).unwrap_err();
    assert!(err.to_string().contains("missing trailing NUL"));
}

#[test]
fn rejects_binary_string_with_embedded_nul() {
    let mut buf = Cursor::new([4i32.to_ne_bytes().as_slice(), b"a\0b\0"].concat());

    let err = read_string(&mut buf).unwrap_err();
    assert!(err.to_string().contains("embedded NUL"));
}

#[test]
fn writer_rejects_binary_string_with_embedded_nul() {
    let mut hmm = crate::hmmfile::read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/20aa.hmm"
    )))
    .unwrap()
    .remove(0);
    hmm.name = "bad\0name".to_string();

    let err = write_binary_hmm(&mut Vec::new(), &hmm).unwrap_err();
    assert!(err.to_string().contains("embedded NUL"));
}

#[test]
fn rejects_excessive_binary_string_length_before_allocation() {
    let mut buf = Cursor::new(((MAX_BINARY_HMM_STRING_LEN + 1) as i32).to_ne_bytes());

    let err = read_string(&mut buf).unwrap_err();
    assert!(err.to_string().contains("too large"));
}

#[test]
fn rejects_binary_hmm_with_empty_required_name() {
    let mut hmm = crate::hmmfile::read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/20aa.hmm"
    )))
    .unwrap()
    .remove(0);
    hmm.name.clear();
    let mut buf = Vec::new();
    write_binary_hmm(&mut buf, &hmm).unwrap();

    let err = read_binary_hmm(&mut Cursor::new(buf)).unwrap_err();
    assert!(err.to_string().contains("empty required NAME"));
}

#[test]
fn test_roundtrip_binary() {
    // Read an HMM from ASCII, write to binary, read back
    let hmms = crate::hmmfile::read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/20aa.hmm"
    )))
    .unwrap();
    let hmm = &hmms[0];

    // Write to binary
    let mut buf = Vec::new();
    write_binary_hmm(&mut buf, hmm).unwrap();

    // Read back
    let mut cursor = Cursor::new(&buf);
    let hmm2 = read_binary_hmm(&mut cursor).unwrap().unwrap();

    assert_eq!(hmm.name, hmm2.name);
    assert_eq!(hmm.m, hmm2.m);
    assert_eq!(hmm.abc_type, hmm2.abc_type);

    // Compare match emissions
    for node in 1..=hmm.m {
        for x in 0..hmm.abc_k {
            assert!(
                (hmm.mat[node][x] - hmm2.mat[node][x]).abs() < 1e-6,
                "mat[{}][{}] mismatch",
                node,
                x
            );
        }
    }
}

/// Regression: the Rust binary writer must be byte-for-byte identical to
/// C `hmmconvert -b`. A prior bug wrote a trailing space (0x20) instead of
/// the NUL terminator at index M+1 of the per-node annotation arrays
/// (rf/mm/cons/cs/ca), diverging by one byte just before the
/// comlog/nseq/ctime region. C keeps array[M+1] == '\0' (p7_hmmfile.c:1560-1562)
/// and dumps the full M+2 bytes verbatim (p7_hmmfile.c:1037-1041).
// The reference C reads probabilities with the host expf, which matches ours only on glibc.
#[cfg(target_os = "linux")]
#[test]
fn binary_write_matches_c_hmmconvert_byte_for_byte() {
    // Exercise models with CONS/CS/RF/MAP/COMPO and ctime/comlog/nseq fields.
    for rel in [
        "test_data/Pkinase_pfam.hmm", // CONS annotation, exposed the trailing-NUL bug
        "hmmer/tutorial/fn3.hmm",     // CS + MAP + DATE + NSEQ + EFFN
        "hmmer/tutorial/MADE1.hmm",   // DNA, RF + CONS + max_length (3/c+)
    ] {
        let text_path = test_path(rel);
        if !text_path.exists() {
            continue;
        }
        let dir = tempfile::tempdir().unwrap();
        let c_h3m = std::fs::read(c_binary_hmm_from_text(&text_path, &dir)).unwrap();

        let hmms = crate::hmmfile::read_hmm_file(&text_path).unwrap();
        let mut rust_h3m = Vec::new();
        for hmm in &hmms {
            write_binary_hmm(&mut rust_h3m, hmm).unwrap();
        }

        assert_eq!(
            rust_h3m.len(),
            c_h3m.len(),
            "{rel}: length differs (rust {} vs C {})",
            rust_h3m.len(),
            c_h3m.len()
        );
        if let Some(off) = rust_h3m.iter().zip(&c_h3m).position(|(a, b)| a != b) {
            panic!(
                "{rel}: first byte divergence at offset {off}: rust=0x{:02x} c=0x{:02x}",
                rust_h3m[off], c_h3m[off]
            );
        }
    }
}

#[test]
fn read_binary_hmm_file_rejects_mixed_alphabet_records() {
    let amino = crate::hmmfile::read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/20aa.hmm"
    )))
    .unwrap()
    .remove(0);
    let dna = crate::hmmfile::read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_data/mapali/ecori-rebuilt.hmm"
    )))
    .unwrap()
    .remove(0);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.h3m");
    let mut file = std::fs::File::create(&path).unwrap();
    write_binary_hmm(&mut file, &amino).unwrap();
    write_binary_hmm(&mut file, &dna).unwrap();
    drop(file);

    let err = read_binary_hmm_file(&path).unwrap_err();

    assert!(err.to_string().contains("mixed alphabets"));
}

#[test]
fn binary_writer_normalizes_acc_and_desc_flags_from_fields() {
    let mut hmm = crate::hmmfile::read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/20aa.hmm"
    )))
    .unwrap()
    .remove(0);
    hmm.acc = Some("ACC123".to_string());
    hmm.desc = Some("description".to_string());
    hmm.flags &= !(P7H_ACC | P7H_DESC);

    let mut buf = Vec::new();
    write_binary_hmm(&mut buf, &hmm).unwrap();
    let read = read_binary_hmm(&mut Cursor::new(buf)).unwrap().unwrap();
    assert_eq!(read.acc.as_deref(), Some("ACC123"));
    assert_eq!(read.desc.as_deref(), Some("description"));

    let mut no_meta = hmm;
    no_meta.acc = None;
    no_meta.desc = None;
    no_meta.flags |= P7H_ACC | P7H_DESC;
    let mut buf = Vec::new();
    write_binary_hmm(&mut buf, &no_meta).unwrap();
    let read = read_binary_hmm(&mut Cursor::new(buf)).unwrap().unwrap();
    assert_eq!(read.acc, None);
    assert_eq!(read.desc, None);
}

#[test]
fn reads_c_hmmer_binary_with_raw_annotation_layout() {
    let text_path = test_path("test_data/gecco_cluster1_hmms.hmm");
    let expected = crate::hmmfile::read_hmm_file(&text_path).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let h3m_path = c_binary_hmm_from_text(&text_path, &dir);
    let actual = read_binary_hmm_file(&h3m_path).unwrap();

    assert_eq!(expected.len(), actual.len());
    for (expected, actual) in expected.iter().zip(actual.iter()) {
        assert_eq!(expected.name, actual.name);
        assert_eq!(expected.acc, actual.acc);
        assert_eq!(expected.m, actual.m);
        assert_eq!(expected.abc_type, actual.abc_type);
        assert_eq!(
            expected.consensus.as_ref().map(Vec::len),
            actual.consensus.as_ref().map(Vec::len)
        );
        assert_eq!(
            expected.rf.as_ref().map(Vec::len),
            actual.rf.as_ref().map(Vec::len)
        );
        assert_eq!(
            expected.map.as_ref().map(Vec::len),
            actual.map.as_ref().map(Vec::len)
        );

        for node in 1..=expected.m {
            for x in 0..expected.abc_k {
                assert!(
                    (expected.mat[node][x] - actual.mat[node][x]).abs() < 1e-6,
                    "{} mat[{}][{}] mismatch",
                    expected.name,
                    node,
                    x
                );
            }
        }
    }
}

#[test]
#[ignore = "requires GECCO's full Pfam.h3m fixture"]
fn reads_gecco_full_pfam_h3m() {
    let path = Path::new("/data/henriksson/github/claude/gecco-rs/data/Pfam.h3m");
    let hmms = read_binary_hmm_file(path).unwrap();
    assert_eq!(hmms.len(), 2766);
    assert!(hmms.iter().all(|hmm| !hmm.name.is_empty()));
    assert!(hmms.iter().all(|hmm| hmm.m > 0));
}
