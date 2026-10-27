use super::*;
use crate::alphabet::Alphabet;
use std::path::Path;

#[test]
fn test_read_fasta() {
    let abc = Alphabet::amino();
    let mut sqf = open_seq_file(
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/testsuite/20aa-alitest.fa"
        )),
        &abc,
    )
    .unwrap();

    let mut sq = Sequence::new();
    let mut count = 0;
    while sqf.read(&mut sq).unwrap() {
        count += 1;
        assert!(sq.n > 0);
        assert!(!sq.name.is_empty());
        sq.reuse();
    }
    assert!(count > 0, "Should read at least one sequence");
}

#[test]
fn test_read_uniprot() {
    let abc = Alphabet::amino();
    let mut sqf = open_seq_file(
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/tutorial/7LESS_DROME"
        )),
        &abc,
    )
    .unwrap();

    let mut sq = Sequence::new();
    assert!(sqf.read(&mut sq).unwrap());
    assert_eq!(sq.name, "7LESS_DROME");
    assert_eq!(sq.acc, "P13368");
    // Multi-line UniProt DE: Easel (esl_sqio_ascii.c) strips only the 5-char
    // "DE   " prefix and trailing whitespace, preserving the continuation
    // line's leading whitespace, then esl_sq_AppendDesc joins with a single
    // space. The 7LESS_DROME record is:
    //   DE   RecName: Full=Protein sevenless;
    //   DE            EC=2.7.10.1;
    // The continuation keeps 9 leading spaces (12 spaces after "DE" minus
    // the 3 prefix spaces) and AppendDesc adds 1 separator space => 10
    // spaces before "EC". Verified byte-for-byte against C phmmer.
    assert_eq!(
        sq.desc,
        "RecName: Full=Protein sevenless;          EC=2.7.10.1;"
    );
    assert!(
        sq.n > 2500,
        "7LESS_DROME should be >2500 residues, got {}",
        sq.n
    );
}

#[test]
fn test_read_globins() {
    let abc = Alphabet::amino();
    let mut sqf = open_seq_file(
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/hmmer/tutorial/globins45.fa"
        )),
        &abc,
    )
    .unwrap();

    let mut sq = Sequence::new();
    let mut count = 0;
    while sqf.read(&mut sq).unwrap() {
        count += 1;
        assert!(
            sq.n > 100,
            "Globin seq should be >100 residues, got {}",
            sq.n
        );
        sq.reuse();
    }
    assert!(
        count >= 40,
        "Should read ~45 globin sequences, got {}",
        count
    );
}

fn read_fasta_text(text: &str) -> HmmerResult<Sequence> {
    let abc = Alphabet::amino();
    let mut sqf = SeqFile::new(text.as_bytes(), abc);
    let mut sq = Sequence::new();
    sqf.read(&mut sq)?;
    Ok(sq)
}

#[test]
fn test_read_fasta_rejects_hash() {
    let err = read_fasta_text(">seq\nAC#D\n").unwrap_err();
    assert!(matches!(err, HmmerError::Format(_)));
}

#[test]
fn test_read_fasta_rejects_digit() {
    let err = read_fasta_text(">seq\nACD1EF\n").unwrap_err();
    assert!(matches!(err, HmmerError::Format(_)));
}

#[test]
fn test_read_fasta_rejects_gap() {
    let err = read_fasta_text(">seq\nAC-D\n").unwrap_err();
    assert!(matches!(err, HmmerError::Format(_)));
}

#[test]
fn test_read_fasta_accepts_dot_and_underscore_gaps() {
    // C digital FASTA read (inmap_fasta) accepts '.' and '_' as gap residues
    // while rejecting '-'. Verified: `esl-seqstat --amino` reports 6 residues
    // for `>seq\nAC.DEF` and `>seq\nAC_DEF`, but errors on `AC-DEF`.
    let dot = read_fasta_text(">seq\nAC.DEF\n").unwrap();
    assert_eq!(dot.n, 6);
    let under = read_fasta_text(">seq\nAC_DEF\n").unwrap();
    assert_eq!(under.n, 6);
    // '-' is still rejected (matches inmap_fasta's '-' -> ILLEGAL override).
    assert!(matches!(
        read_fasta_text(">seq\nAC-DEF\n"),
        Err(HmmerError::Format(_))
    ));
}

#[test]
fn test_read_fasta_accepts_nonresidue_star() {
    let sq = read_fasta_text(">seq\nAC*D\n").unwrap();
    assert_eq!(sq.n, 4);
}

#[test]
fn test_read_fasta_ignores_whitespace() {
    let sq = read_fasta_text(">seq\nAC D\tEF\n").unwrap();
    assert_eq!(sq.n, 5);
}

#[test]
fn test_read_fasta_truncates_description_at_ctrl_a() {
    let sq = read_fasta_text(">seq first description\x01second record title\nACDE\n").unwrap();
    assert_eq!(sq.name, "seq");
    assert_eq!(sq.desc, "first description");
}

#[test]
fn test_read_fasta_rejects_preamble_text() {
    let fasta = b"not a sequence record\n>seq1\nACDE\n";
    let abc = Alphabet::amino();
    let mut reader = SeqFile::new(&fasta[..], abc);
    let mut sq = Sequence::new();
    let err = reader.read(&mut sq).unwrap_err();
    assert!(err
        .to_string()
        .contains("unrecognized sequence file record start"));
}

#[test]
fn test_read_fasta_accepts_empty_record() {
    let sq = read_fasta_text(">empty\n").unwrap();
    assert_eq!(sq.name, "empty");
    assert_eq!(sq.n, 0);
    assert_eq!(sq.dsq, vec![DSQ_SENTINEL, DSQ_SENTINEL]);
}

#[test]
fn test_read_fasta_accepts_empty_record_before_next_header() {
    let abc = Alphabet::amino();
    let mut reader = SeqFile::new(&b">empty\n>seq\nACD\n"[..], abc);
    let mut sq = Sequence::new();

    assert!(reader.read(&mut sq).unwrap());
    assert_eq!(sq.name, "empty");
    assert_eq!(sq.n, 0);

    assert!(reader.read(&mut sq).unwrap());
    assert_eq!(sq.name, "seq");
    assert_eq!(sq.n, 3);
}

#[test]
fn test_read_genbank_multiline_definition() {
    let genbank = b"LOCUS       seq1 4 bp DNA\nDEFINITION  first line\n            second line\nACCESSION   ABC123\nORIGIN\n        1 acgt\n//\n";
    let abc = Alphabet::dna();
    let mut reader = SeqFile::new(&genbank[..], abc);
    let mut sq = Sequence::new();
    assert!(reader.read(&mut sq).unwrap());
    assert_eq!(sq.desc, "first line second line");
    assert_eq!(sq.acc, "ABC123");
    assert_eq!(sq.n, 4);
}

#[test]
fn test_read_genbank_with_asserted_format() {
    let genbank = b"LOCUS       seq1 4 bp DNA\nORIGIN\n        1 acgt\n//\n";
    let abc = Alphabet::dna();
    let mut reader = SeqFile::new(&genbank[..], abc).with_format(SequenceFormat::GenBank);
    let mut sq = Sequence::new();
    assert!(reader.read(&mut sq).unwrap());
    assert_eq!(sq.name, "seq1");
    assert_eq!(sq.n, 4);
}

#[test]
fn test_read_stockholm_autodetects_without_asserted_format() {
    let stockholm = b"# STOCKHOLM 1.0\nseq1 ACDE\nseq2 A-D-\n//\n";
    let abc = Alphabet::amino();
    let mut reader = SeqFile::new(&stockholm[..], abc);
    let mut sq = Sequence::new();
    assert!(reader.read(&mut sq).unwrap());
    assert_eq!(sq.name, "seq1");
    assert_eq!(sq.n, 4);
    assert!(reader.read(&mut sq).unwrap());
    assert_eq!(sq.name, "seq2");
    assert_eq!(sq.n, 2);
    assert!(!reader.read(&mut sq).unwrap());
}

#[test]
fn test_asserted_format_rejects_other_record_start() {
    let uniprot = b"ID   seq1\nSQ   SEQUENCE 4 AA;\nACDE\n//\n";
    let abc = Alphabet::amino();
    let mut reader = SeqFile::new(&uniprot[..], abc).with_format(SequenceFormat::Fasta);
    let mut sq = Sequence::new();
    let err = reader.read(&mut sq).unwrap_err();
    assert!(err
        .to_string()
        .contains("unrecognized sequence file record start"));
}

#[test]
fn test_read_fasta_rejects_overlong_header_line() {
    let mut fasta = Vec::new();
    fasta.push(b'>');
    fasta.extend(std::iter::repeat_n(b'a', MAX_SEQUENCE_LINE_LEN));
    fasta.push(b'\n');
    let abc = Alphabet::amino();
    let mut reader = SeqFile::new(&fasta[..], abc);
    let mut sq = Sequence::new();
    let err = reader.read(&mut sq).unwrap_err();
    assert!(err.to_string().contains("sequence file line exceeds"));
}

#[test]
fn test_read_fasta_rejects_overlong_sequence_line() {
    let mut fasta = b">seq\n".to_vec();
    fasta.extend(std::iter::repeat_n(b'A', MAX_SEQUENCE_LINE_LEN + 1));
    fasta.push(b'\n');
    let abc = Alphabet::amino();
    let mut reader = SeqFile::new(&fasta[..], abc);
    let mut sq = Sequence::new();
    let err = reader.read(&mut sq).unwrap_err();
    assert!(err.to_string().contains("sequence file line exceeds"));
}
