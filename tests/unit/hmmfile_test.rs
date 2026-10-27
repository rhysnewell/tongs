use super::*;
use std::io::{BufReader, Cursor};
use std::path::Path;

struct OneByteReader {
    data: Cursor<Vec<u8>>,
}

impl std::io::Read for OneByteReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let mut byte = [0u8; 1];
        let n = self.data.read(&mut byte)?;
        if n == 1 {
            buf[0] = byte[0];
        }
        Ok(n)
    }
}

fn read_hmms_from_str(s: &str) -> HmmerResult<Vec<Hmm>> {
    read_hmms(BufReader::new(Cursor::new(s.as_bytes())))
}

fn minimal_hmm(name: &str, abc_type: AlphabetType) -> Hmm {
    let abc = Alphabet::new(abc_type);
    let mut hmm = Hmm::new(1, abc_type, abc.k);
    hmm.name = name.to_string();
    hmm
}

#[test]
fn rejects_excessive_ascii_model_length_before_allocating() {
    let input = format!(
            "HMMER3/f\nNAME  huge\nLENG  {}\nALPH  amino\nHMM          A C D E F G H I K L M N P Q R S T V W Y\n",
            MAX_HMM_MODEL_LENGTH + 1
        );

    let err = read_hmms_from_str(&input).unwrap_err();
    assert!(err.to_string().contains("Invalid model length"));
}

#[test]
fn rejects_ascii_hmm_line_that_exceeds_cap() {
    let mut input = b"HMMER3/f ".to_vec();
    input.extend(std::iter::repeat_n(b'x', MAX_ASCII_HMM_LINE_LEN));

    let err = read_hmms(BufReader::new(Cursor::new(input))).unwrap_err();
    assert!(err.to_string().contains("maximum supported length"));
}

#[test]
fn fetch_one_sequential_mirrors_c_onefetch_on_mixed_alphabet() {
    // fn3 (amino) followed by MADE1 (DNA): C's onefetch break-on-match
    // succeeds on the first record but fails on the second / on a miss.
    let dir = tempfile::tempdir().unwrap();
    let mixed = dir.path().join("mixed.hmm");
    let mut bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/tutorial/fn3.hmm"
    ))
    .unwrap();
    bytes.extend_from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/tutorial/MADE1.hmm"
        ))
        .unwrap(),
    );
    std::fs::write(&mixed, bytes).unwrap();

    // First (amino) record: found, never reads the DNA record.
    match fetch_one_sequential(&mixed, "fn3").unwrap() {
        FetchOutcome::Found(hmm) => {
            assert_eq!(hmm.name, "fn3");
            assert_eq!(hmm.abc_type, AlphabetType::Amino);
        }
        FetchOutcome::NotFound => panic!("fn3 should be found"),
    }

    // Second (DNA) record: scan threads the alphabet change → error.
    let made1 = fetch_one_sequential(&mixed, "MADE1");
    assert!(made1.is_err(), "MADE1 should hit the alphabet mismatch");

    // Missing key forces reading the whole file → same alphabet error.
    let miss = fetch_one_sequential(&mixed, "NOSUCH");
    assert!(miss.is_err(), "missing key must hit the alphabet mismatch");

    // Uniform-alphabet file: missing key is a clean NotFound, not an error.
    let uniform = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/tutorial/fn3.hmm"
    ));
    assert!(matches!(
        fetch_one_sequential(uniform, "NOSUCH").unwrap(),
        FetchOutcome::NotFound
    ));
    assert!(matches!(
        fetch_one_sequential(uniform, "fn3").unwrap(),
        FetchOutcome::Found(_)
    ));
}

#[test]
fn test_read_20aa_hmm() {
    let hmms = read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/20aa.hmm"
    )))
    .unwrap();
    assert_eq!(hmms.len(), 1);
    let hmm = &hmms[0];
    assert_eq!(hmm.name, "test");
    assert_eq!(hmm.m, 20);
    assert_eq!(hmm.abc_type, AlphabetType::Amino);
    assert_eq!(hmm.abc_k, 20);
    assert_eq!(hmm.nseq, 10);
    assert!((hmm.eff_nseq - 1.958008).abs() < 1e-5);
    assert!(hmm.flags & P7H_STATS != 0);
    assert!(hmm.flags & P7H_RF != 0);
    assert!(hmm.flags & P7H_CONS != 0);
    assert!(hmm.flags & P7H_MAP != 0);

    // Check that match emissions were converted from -ln(p) to probability
    // Node 1 first value (A emission) was 0.33153 in file -> exp(-0.33153) ≈ 0.7182
    assert!(hmm.mat[1][0] > 0.0 && hmm.mat[1][0] < 1.0);

    // Check E-value params
    assert!((hmm.evparam[P7_MMU] - (-6.4582)).abs() < 1e-3);
    assert!((hmm.evparam[P7_MLAMBDA] - 0.72049).abs() < 1e-4);
}

#[test]
fn test_read_fn3_hmm() {
    let hmms = read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/tutorial/fn3.hmm"
    )))
    .unwrap();
    assert_eq!(hmms.len(), 1);
    let hmm = &hmms[0];
    assert_eq!(hmm.name, "fn3");
    assert!(hmm.m > 50); // fn3 is a medium-sized domain
}

#[test]
fn auto_reader_detects_binary_hmm_from_short_stream_reads() {
    let hmm = read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/tutorial/fn3.hmm"
    )))
    .unwrap()
    .remove(0);
    let mut bytes = Vec::new();
    crate::hmmfile_binary::write_binary_hmm(&mut bytes, &hmm).unwrap();

    let reader = OneByteReader {
        data: Cursor::new(bytes),
    };
    let hmms = read_hmms_auto(BufReader::new(reader)).unwrap();
    assert_eq!(hmms.len(), 1);
    assert_eq!(hmms[0].name, "fn3");
    assert_eq!(hmms[0].m, 86);
}

#[test]
fn binary_annotation_arrays_use_space_sentinel_at_index_0_like_c() {
    // C HMMER's binary writer (`p7_hmmfile_WriteBinary`, p7_hmmfile.c:1037-
    // 1041) emits the raw `M+2`-byte annotation arrays verbatim. Its ASCII
    // reader (`read_asc30hmm`) sets index 0 of rf/mm/consensus/cs to ' '
    // (0x20). Verified empirically against the C `hmmconvert -b` output for
    // fn3: the CONS and CS arrays are byte-identical to Rust's, with a 0x20
    // sentinel at index 0 (NOT a NUL). The audit's F4 claim that C writes a
    // NUL here does not reproduce against the HMMER 3.4 build; C writes a
    // space. This test locks the confirmed C-parity sentinel in.
    let hmm = read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/tutorial/fn3.hmm"
    )))
    .unwrap()
    .remove(0);

    // ASCII reader matches C: index 0 == ' ' (0x20).
    let cons = hmm.consensus.as_ref().expect("fn3 has CONS yes");
    assert_eq!(cons[0], b' ', "CONS index 0 must be space (matches C .h3m)");
    let cs = hmm.cs.as_ref().expect("fn3 has CS yes");
    assert_eq!(cs[0], b' ', "CS index 0 must be space (matches C .h3m)");

    // The binary writer must emit those sentinels verbatim. Round-trip
    // through the binary reader (which preserves the bytes verbatim).
    let mut bytes = Vec::new();
    crate::hmmfile_binary::write_binary_hmm(&mut bytes, &hmm).unwrap();
    let rt = crate::hmmfile_binary::read_binary_hmm(&mut Cursor::new(bytes))
        .unwrap()
        .unwrap();
    assert_eq!(rt.consensus.as_ref().unwrap()[0], b' ');
    assert_eq!(rt.cs.as_ref().unwrap()[0], b' ');
}

#[test]
fn rejects_hmm_alphabet_header_that_disagrees_with_alph() {
    let mut text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/20aa.hmm"
    ))
    .unwrap();
    text = text.replacen("HMM          A        C", "HMM          C        A", 1);

    let err = read_hmms_from_str(&text).unwrap_err();
    assert!(err.to_string().contains("HMM alphabet header symbol 1"));
}

#[test]
fn rejects_non_exact_hmmer3_magic_tags() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/20aa.hmm"
    ))
    .unwrap()
    .replacen("HMMER3/e", "HMMER3/foo", 1);

    let err = read_hmms_from_str(&text).unwrap_err();
    assert!(
        matches!(err, HmmerError::Format(msg) if msg.contains("Unsupported HMMER3 ASCII magic tag"))
    );
}

#[test]
fn parses_alphabet_and_stats_tokens_case_insensitively() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/20aa.hmm"
    ))
    .unwrap()
    .replacen("ALPH  amino", "ALPH  AMINO", 1)
    .replacen("STATS LOCAL MSV", "STATS local mSv", 1)
    .replacen("STATS LOCAL VITERBI", "STATS LoCaL vItErBi", 1)
    .replacen("STATS LOCAL FORWARD", "STATS LOCAL forward", 1);

    let hmm = read_hmms_from_str(&text).unwrap().remove(0);
    assert_eq!(hmm.abc_type, AlphabetType::Amino);
    assert!(hmm.flags & P7H_STATS != 0);
    assert!((hmm.evparam[P7_MMU] - (-6.4582)).abs() < 1e-3);
    assert!((hmm.evparam[P7_FTAU] - (-4.5231)).abs() < 1e-3);
}

#[test]
fn writer_emits_pfam_cutoff_headers() {
    let mut hmm = minimal_hmm("with_cutoffs", AlphabetType::Amino);
    hmm.cutoff[P7_GA1] = 25.0;
    hmm.cutoff[P7_GA2] = 24.5;
    hmm.cutoff[P7_TC1] = 30.0;
    hmm.cutoff[P7_TC2] = 29.5;
    hmm.cutoff[P7_NC1] = -1.0;
    hmm.cutoff[P7_NC2] = -2.0;
    hmm.flags |= P7H_GA | P7H_TC | P7H_NC;

    let mut buf = Vec::new();
    write_hmm(&mut buf, &hmm).unwrap();
    let text = String::from_utf8(buf).unwrap();
    assert!(text.contains("GA    25.00 24.50\n"), "{text}");
    assert!(text.contains("TC    30.00 29.50\n"), "{text}");
    assert!(text.contains("NC    -1.00 -2.00\n"), "{text}");
}

#[test]
fn writer_pads_stats_local_lines_to_c_field_widths() {
    // C p7_hmmfile.c writes STATS LOCAL lines as
    //   "STATS LOCAL MSV      %8.4f %8.5f\n" (and VITERBI/FORWARD).
    // A mu with magnitude >= 10 produces an 8-char value; C's %8.4f emits
    // no leading pad, so the field stays width 8. Rust must match exactly.
    let mut hmm = minimal_hmm("bigmu", AlphabetType::Amino);
    hmm.flags |= P7H_STATS;
    hmm.evparam[P7_MMU] = -10.8752;
    hmm.evparam[P7_MLAMBDA] = 0.70247;
    hmm.evparam[P7_VMU] = -11.6882;
    hmm.evparam[P7_VLAMBDA] = 0.70247;
    hmm.evparam[P7_FTAU] = -5.2290;
    hmm.evparam[P7_FLAMBDA] = 0.70247;

    let mut buf = Vec::new();
    write_hmm(&mut buf, &hmm).unwrap();
    let text = String::from_utf8(buf).unwrap();

    // Byte-for-byte against C `hmmconvert -a` on a model with mu <= -10.
    assert!(
        text.contains("STATS LOCAL MSV      -10.8752  0.70247\n"),
        "{text}"
    );
    assert!(
        text.contains("STATS LOCAL VITERBI  -11.6882  0.70247\n"),
        "{text}"
    );
    // 7-char mu still right-justifies in width 8 (one leading pad space).
    assert!(
        text.contains("STATS LOCAL FORWARD   -5.2290  0.70247\n"),
        "{text}"
    );
}

#[test]
fn writer_uses_hmmer3f_for_match_mask_annotation() {
    let mut hmm = read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/20aa.hmm"
    )))
    .unwrap()
    .remove(0);
    hmm.flags |= P7H_MMASK;

    let mut buf = Vec::new();
    write_hmm(&mut buf, &hmm).unwrap();
    assert!(String::from_utf8(buf).unwrap().starts_with("HMMER3/f "));
}

#[test]
fn parses_hmmer3e_map_annotations_without_mm_column() {
    let hmms = read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/RRM_1.hmm"
    )))
    .unwrap();
    let hmm = &hmms[0];
    assert_eq!(hmm.name, "RRM_1");
    assert_eq!(hmm.map.as_ref().unwrap()[1], 1);
    assert_eq!(hmm.consensus.as_ref().unwrap()[1], b'l');
    assert!(hmm.rf.is_none());
    assert_eq!(hmm.cs.as_ref().unwrap()[1], b'E');
}

#[test]
fn parses_hmmer3a_records_written_by_rust() {
    let hmm = read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/20aa.hmm"
    )))
    .unwrap()
    .remove(0);
    let mut buf = Vec::new();
    write_hmm_with_format(&mut buf, &hmm, HmmAsciiFormat::Hmmer3a).unwrap();
    assert!(String::from_utf8_lossy(&buf).starts_with("HMMER3/a "));

    let hmms = read_hmms(BufReader::new(Cursor::new(buf))).unwrap();
    assert_eq!(hmms.len(), 1);
    assert_eq!(hmms[0].name, "test");
    assert_eq!(hmms[0].m, 20);
    assert_eq!(hmms[0].abc_type, AlphabetType::Amino);
}

#[test]
fn parses_hmmer3f_map_annotations_with_mm_placeholder() {
    let hmms = read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_data/gecco_cluster1_hmms.hmm"
    )))
    .unwrap();
    let hmm = &hmms[0];
    assert_eq!(hmm.name, "Alpha-amylase");
    assert_eq!(hmm.map.as_ref().unwrap()[1], 1);
    assert_eq!(hmm.consensus.as_ref().unwrap()[1], b'G');
    assert!(hmm.rf.is_none());
    assert!(hmm.mm.is_none());
    assert_eq!(hmm.cs.as_ref().unwrap()[1], b'-');
}

#[test]
fn test_read_multiple_hmms() {
    // minipfam has multiple HMMs
    let hmms = read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/minipfam.hmm"
    )))
    .unwrap();
    assert!(hmms.len() > 1, "Expected multiple HMMs in minipfam");
}

#[test]
fn rejects_ascii_hmm_database_with_mixed_alphabets() {
    let mut text = Vec::new();
    write_hmm(&mut text, &minimal_hmm("protein", AlphabetType::Amino)).unwrap();
    write_hmm(&mut text, &minimal_hmm("dna", AlphabetType::Dna)).unwrap();

    let err = read_hmms(BufReader::new(Cursor::new(text))).unwrap_err();
    assert!(
        matches!(err, HmmerError::Format(msg) if msg.contains("ASCII HMM file contains mixed alphabets"))
    );
}

#[test]
fn rejects_bad_ga_cutoff_instead_of_defaulting() {
    let err = read_hmms_from_str(
        "HMMER3/f [3.4 | Aug 2023]\nNAME  x\nLENG  1\nALPH  amino\nGA    nope\n",
    )
    .unwrap_err();
    assert!(matches!(err, HmmerError::Format(msg) if msg.contains("Bad GA cutoff")));
}

#[test]
fn rejects_invalid_yes_no_header_values() {
    let original = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_data/gecco_cluster1_hmms.hmm"
    ))
    .unwrap();
    for key in ["RF", "MM", "CONS", "CS", "MAP"] {
        let yes_needle = format!("{key:<5} yes");
        let no_needle = format!("{key:<5} no");
        let needle = if original.contains(&yes_needle) {
            yes_needle
        } else {
            no_needle
        };
        let mutated = original.replacen(&needle, &format!("{key:<5} maybe"), 1);
        assert_ne!(mutated, original, "test fixture did not contain {needle}");
        let err = read_hmms_from_str(&mutated).unwrap_err();
        assert!(
            matches!(&err, HmmerError::Format(msg) if msg.contains(&format!("Bad {key} value"))),
            "{err}"
        );
    }
}

#[test]
fn writer_preserves_comlog_lines() {
    let mut hmm = minimal_hmm("with_com", AlphabetType::Amino);
    hmm.comlog = Some("first command\nsecond command".to_string());
    let mut buf = Vec::new();
    write_hmm(&mut buf, &hmm).unwrap();
    let text = String::from_utf8(buf).unwrap();
    assert!(text.contains("COM   [1] first command\n"), "{text}");
    assert!(text.contains("COM   [2] second command\n"), "{text}");
}

#[test]
fn rejects_mixed_ascii_hmmer3_versions() {
    let mut text = Vec::new();
    write_hmm(&mut text, &minimal_hmm("first", AlphabetType::Amino)).unwrap();
    let second = {
        let mut buf = Vec::new();
        write_hmm(&mut buf, &minimal_hmm("second", AlphabetType::Amino)).unwrap();
        String::from_utf8(buf)
            .unwrap()
            .replacen("HMMER3/f", "HMMER3/e", 1)
    };
    text.extend_from_slice(second.as_bytes());

    let err = read_hmms(BufReader::new(Cursor::new(text))).unwrap_err();
    assert!(
        matches!(&err, HmmerError::Format(msg) if msg.contains("mixed HMMER3 format versions")),
        "{err}"
    );
}

#[test]
fn parses_pfam_semicolon_terminated_cutoffs() {
    assert_eq!(parse_cutoff_value("GA", "22;").unwrap(), 22.0);
    assert_eq!(parse_cutoff_value("TC", "20.7;").unwrap(), 20.7);
    assert_eq!(parse_cutoff_value("NC", "29.6").unwrap(), 29.6);

    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/tutorial/fn3.hmm"
    ))
    .unwrap()
    .replace("GA    8.00 7.20", "GA    22;")
    .replace("TC    8.00 7.20", "TC    23.5;")
    .replace("NC    7.90 7.90", "NC    -1.25;");

    let hmms = read_hmms_from_str(&text).unwrap();
    let hmm = &hmms[0];
    assert_eq!(hmm.cutoff[P7_GA1], 22.0);
    assert_eq!(hmm.cutoff[P7_GA2], 22.0);
    assert_eq!(hmm.cutoff[P7_TC1], 23.5);
    assert_eq!(hmm.cutoff[P7_TC2], 23.5);
    assert_eq!(hmm.cutoff[P7_NC1], -1.25);
    assert_eq!(hmm.cutoff[P7_NC2], -1.25);
    assert!(hmm.flags & P7H_GA != 0);
    assert!(hmm.flags & P7H_TC != 0);
    assert!(hmm.flags & P7H_NC != 0);
}

#[test]
fn rejects_truncated_compo_line() {
    let text = "\
HMMER3/f [3.4 | Aug 2023]
NAME  x
LENG  1
ALPH  DNA
HMM          A        C        G        T
            m->m     m->i     m->d     i->m     i->i     d->m     d->d
COMPO   1.38629  1.38629
        1.38629  1.38629  1.38629  1.38629
        0.00000        *        *  0.00000        *        *        *
     1  1.38629  1.38629  1.38629  1.38629
        1.38629  1.38629  1.38629  1.38629
              *        *        *  0.00000        *        *        *
//
";
    let err = read_hmms_from_str(text).unwrap_err();
    assert!(
        matches!(err, HmmerError::Format(msg) if msg.contains("COMPO line has too few fields"))
    );
}
