use super::*;

#[test]
fn test_read_20aa_stockholm() {
    let msas = read_stockholm(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/20aa.sto"
    )))
    .unwrap();
    assert_eq!(msas.len(), 1);
    let msa = &msas[0];
    assert_eq!(msa.name, "test");
    assert_eq!(msa.nseq, 10);
    assert_eq!(msa.alen, 20);
    assert!(msa.rf.is_some());
}

#[test]
fn test_read_globins4_stockholm() {
    let msas = read_stockholm(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/tutorial/globins4.sto"
    )))
    .unwrap();
    assert_eq!(msas.len(), 1);
    let msa = &msas[0];
    assert_eq!(msa.nseq, 4);
    assert!(msa.alen > 100);
}

#[test]
fn stockholm_de_and_au_last_line_wins() {
    // C `esl_msa_SetDesc`/`SetAuthor` (called from stockholm_parse_gf) free
    // and overwrite the field: the LAST #=GF DE / #=GF AU line wins. Verified
    // with `esl-reformat stockholm` on this input (emits "second part" /
    // "Author Two" only).
    let input = b"# STOCKHOLM 1.0\n#=GF ID testaln\n#=GF DE first part\n#=GF DE second part\n#=GF AU Author One\n#=GF AU Author Two\nseq1 ACDEF\nseq2 ACDEG\n//\n";
    let msas = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap();
    let msa = &msas[0];
    assert_eq!(msa.desc.as_deref(), Some("second part"));
    assert_eq!(msa.author.as_deref(), Some("Author Two"));
}

#[test]
fn rejects_unterminated_stockholm_block() {
    let input = b"# STOCKHOLM 1.0\nseq AC\n";
    let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
    assert!(err.to_string().contains("missing // terminator"));
}

#[test]
fn rejects_mismatched_stockholm_row_lengths() {
    let input = b"# STOCKHOLM 1.0\nseq1 AC\nseq2 A\n//\n";
    let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
    assert!(err.to_string().contains("aligned length"));
}

#[test]
fn parses_stockholm_description_and_pp_annotations() {
    let input =
        b"# STOCKHOLM 1.0\n#=GS s1 DE first desc\ns1 AC\n#=GR s1 PP 9*\n#=GC PP_cons 8*\n//\n";
    let msas = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap();
    let msa = &msas[0];
    assert_eq!(msa.sqdesc[0], "first desc");
    assert!(msa.weights.is_none());
    assert_eq!(msa.pp[0].as_deref(), Some(&b"9*"[..]));
    assert_eq!(msa.pp_cons.as_deref(), Some(&b"8*"[..]));
}

#[test]
fn parses_stockholm_author_and_cutoffs() {
    let input = b"# STOCKHOLM 1.0\n#=GF AU first author\n#=GF AU second author\n#=GF GA 25.0 24.5\n#=GF TC 30.0;\n#=GF NC -1.0 -2.0;\ns1 AC\n//\n";
    let records = read_stockholm_preserved_from_reader(BufReader::new(&input[..])).unwrap();
    let record = &records[0];
    // C `esl_msa_SetAuthor` overwrites: the LAST #=GF AU line wins (verified
    // with `esl-reformat stockholm`).
    assert_eq!(record.msa.author.as_deref(), Some("second author"));
    assert_eq!(record.cutoffs.ga, Some([25.0, 24.5]));
    assert_eq!(record.cutoffs.tc, Some([30.0, 30.0]));
    assert_eq!(record.cutoffs.nc, Some([-1.0, -2.0]));
}

#[test]
fn parses_stockholm_sequence_weights() {
    let input = b"# STOCKHOLM 1.0\n#=GS s1 WT 0.25\n#=GS s2 WT 1.75\ns1 AC\ns2 AC\n//\n";
    let msas = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap();
    assert_eq!(msas[0].weights.as_deref(), Some(&[0.25, 1.75][..]));

    let input = b"# STOCKHOLM 1.0\n#=GS s1 WT 0.25\ns1 AC\ns2 AC\n//\n";
    let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
    assert!(err.to_string().contains("without a weight for s2"));
}

#[test]
fn rejects_stockholm_id_with_extra_fields() {
    let input = b"# STOCKHOLM 1.0\n#=GF ID test with cruft\ns1 AC\n//\n";
    let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
    assert!(err.to_string().contains("exactly one name token"));
}

#[test]
fn rejects_duplicate_stockholm_sequence_descriptions() {
    let input = b"# STOCKHOLM 1.0\n#=GS s1 DE first\n#=GS s1 DE second\ns1 AC\n//\n";
    let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
    assert!(err.to_string().contains("more than one DE annotation"));
}

#[test]
fn rejects_invalid_stockholm_pp_annotations() {
    let input = b"# STOCKHOLM 1.0\ns1 AC\n#=GR s1 PP 9A\n//\n";
    let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
    assert!(err.to_string().contains("invalid PP character"));

    let input = b"# STOCKHOLM 1.0\ns1 AC\n#=GR missing PP 9*\n//\n";
    let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
    assert!(err.to_string().contains("unknown sequence"));
}

#[test]
fn rejects_stockholm_duplicate_sequence_row_in_block() {
    let input = b"# STOCKHOLM 1.0\ns1 AC\ns1 GU\n//\n";
    let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
    assert!(err.to_string().contains("more than once in the same block"));
}

#[test]
fn rejects_stockholm_multiblock_order_mismatch() {
    let input = b"# STOCKHOLM 1.0\ns1 AC\ns2 AC\n\ns2 GU\ns1 GU\n//\n";
    let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
    assert!(err.to_string().contains("out of order"));
}

#[test]
fn rejects_stockholm_multiblock_gc_order_mismatch() {
    let input =
            b"# STOCKHOLM 1.0\ns1 AC\n#=GC RF xx\n#=GC PP_cons 99\n\ns1 GU\n#=GC PP_cons 99\n#=GC RF xx\n//\n";
    let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
    assert!(err.to_string().contains("out of order"));
    assert!(err.to_string().contains("#=GC PP_cons"));
}

#[test]
fn rejects_stockholm_multiblock_gr_order_mismatch() {
    let input =
            b"# STOCKHOLM 1.0\ns1 AC\ns2 AC\n#=GR s1 PP 99\n#=GR s2 PP 88\n\ns1 GU\ns2 GU\n#=GR s2 PP 88\n#=GR s1 PP 99\n//\n";
    let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
    assert!(err.to_string().contains("out of order"));
    assert!(err.to_string().contains("#=GR s2 PP"));
}

#[test]
fn accepts_stockholm_multiblock_same_order() {
    let input = b"# STOCKHOLM 1.0\ns1 AC\ns2 AC\n\ns1 GU\ns2 GU\n//\n";
    let msas = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap();
    assert_eq!(msas[0].aseq[0], b"ACGU");
    assert_eq!(msas[0].aseq[1], b"ACGU");
}

#[test]
fn parses_a2m_consensus_and_insert_columns() {
    let input = b">s1 first\nGAATTC\n>s2 second\nGAaATTC\n";
    let mut reader = &input[..];
    let msas = read_a2m_from_reader(&mut reader, "toy".to_string()).unwrap();
    let msa = &msas[0];
    assert_eq!(msa.name, "toy");
    assert_eq!(msa.sqname, ["s1", "s2"]);
    assert_eq!(msa.sqdesc, ["first", "second"]);
    assert_eq!(msa.alen, 7);
    // Insert-column gaps are '.' (C esl-reformat emits "GA.ATTC"), verified
    // against `esl-reformat afa` on the same A2M input.
    assert_eq!(msa.aseq[0], b"GA.ATTC");
    assert_eq!(msa.aseq[1], b"GAaATTC");
    assert_eq!(msa.rf.as_deref(), Some(&b"xx.xxxx"[..]));
}

#[test]
fn rejects_a2m_rows_with_different_consensus_counts() {
    let input = b">s1\nGAATTC\n>s2\nGAaTTC\n";
    let mut reader = &input[..];
    let err = read_a2m_from_reader(&mut reader, "toy".to_string()).unwrap_err();
    assert!(err.to_string().contains("consensus columns"));
}

#[test]
fn full_reader_buckets_tags_and_writer_reflows() {
    // GF specials (ID/AC/DE) + arbitrary XX; GS DE + arbitrary OS; GR SS/PP
    // (input order PP then SS -> writer emits SS then PP); GC SS_cons/RF/MM
    // (writer order SS_cons, ..., RF, MM, then arbitrary). Written in text
    // mode (abc=None) so residues pass through verbatim.
    let input = b"# STOCKHOLM 1.0\n#=GF ID meta\n#=GF AC PF1\n#=GF DE a desc\n#=GF XX extra gf\n#=GS s1 DE seq desc\n#=GS s1 OS extra gs\ns1 ACDEFG\n#=GR s1 PP 999999\n#=GR s1 SS HHHHHH\n#=GC RF xxxxxx\n#=GC SS_cons <<<<<<\n#=GC MM ......\n//\n";
    let msas = read_stockholm_full_from_reader(BufReader::new(&input[..])).unwrap();
    assert_eq!(msas.len(), 1);
    let m = &msas[0];
    assert_eq!(m.name.as_deref(), Some("meta"));
    assert_eq!(m.acc.as_deref(), Some("PF1"));
    assert_eq!(m.desc.as_deref(), Some("a desc"));
    assert_eq!(m.gf, vec![("XX".to_string(), "extra gf".to_string())]);
    assert_eq!(m.sqdesc[0].as_deref(), Some("seq desc"));
    assert_eq!(m.gs[0].0, "OS");
    assert_eq!(m.ss[0].as_deref(), Some(&b"HHHHHH"[..]));
    assert_eq!(m.pp[0].as_deref(), Some(&b"999999"[..]));
    assert_eq!(m.ss_cons.as_deref(), Some(&b"<<<<<<"[..]));
    assert_eq!(m.rf.as_deref(), Some(&b"xxxxxx"[..]));

    let mut out = Vec::new();
    write_stockholm_full(&mut out, m, 200, None).unwrap();
    let s = String::from_utf8(out).unwrap();
    // GF reflow: arbitrary XX after the specials.
    let idx_de = s.find("#=GF DE a desc\n").unwrap();
    let idx_xx = s.find("#=GF XX extra gf\n").unwrap();
    assert!(idx_de < idx_xx);
    // GS: DE group then OS group, each with a blank line after.
    assert!(s.contains("#=GS s1 DE seq desc\n\n#=GS s1 OS extra gs\n\n"));
    // GR reordered: SS before PP.
    let idx_ss = s.find("#=GR s1 SS").unwrap();
    let idx_pp = s.find("#=GR s1 PP").unwrap();
    assert!(idx_ss < idx_pp);
    // GC reordered: SS_cons, then RF, then MM.
    let i_ss = s.find("#=GC SS_cons").unwrap();
    let i_rf = s.find("#=GC RF").unwrap();
    let i_mm = s.find("#=GC MM").unwrap();
    assert!(i_ss < i_rf && i_rf < i_mm);
    assert!(s.ends_with("//\n"));
}

#[test]
fn full_writer_normalizes_gaps_with_alphabet() {
    // Digital-mode write (abc=Some) normalizes '.'/'_' gap chars to '-' in
    // sequences (esl_abc_TextizeN), matching alimask; #=GC RF stays text.
    let input = b"# STOCKHOLM 1.0\ns1 AC.DEF\ns2 AC_DEF\n#=GC RF xx.xxx\n//\n";
    let m = read_stockholm_full_from_reader(BufReader::new(&input[..]))
        .unwrap()
        .remove(0);
    let mut out = Vec::new();
    write_stockholm_full(&mut out, &m, 200, Some(&crate::alphabet::Alphabet::amino())).unwrap();
    let s = String::from_utf8(out).unwrap();
    // Gap chars '.' and '_' normalize to '-'; margin is driven by #=GC RF
    // (maxgc=2 -> margin=8), so names pad to width 7.
    assert!(s.contains("s1      AC-DEF\n"), "{s}");
    assert!(s.contains("s2      AC-DEF\n"), "{s}");
    // RF annotation is not a digitized sequence; its '.' is preserved.
    assert!(s.contains("#=GC RF xx.xxx\n"), "{s}");
}

#[test]
fn full_writer_blockwraps_at_cpl() {
    let input = b"# STOCKHOLM 1.0\ns1 AAAACCCCGG\ns2 AAAACCCCGG\n//\n";
    let m = read_stockholm_full_from_reader(BufReader::new(&input[..]))
        .unwrap()
        .remove(0);
    let mut out = Vec::new();
    write_stockholm_full(&mut out, &m, 5, None).unwrap();
    let s = String::from_utf8(out).unwrap();
    // alen=10, cpl=5 -> two blocks separated by a blank line.
    assert!(
        s.contains("s1 AAAAC\ns2 AAAAC\n\ns1 CCCGG\ns2 CCCGG\n//\n"),
        "{s}"
    );
}

#[test]
fn rejects_stockholm_junk_preamble() {
    let input = b"junk\n# STOCKHOLM 1.0\ns1 AC\n//\n";
    let err = read_stockholm_from_reader(BufReader::new(&input[..])).unwrap_err();
    assert!(err.to_string().contains("Expected Stockholm header"));
}

#[test]
fn text_msa_reader_rejects_oversized_input_before_full_allocation() {
    let input = b"# STOCKHOLM 1.0\ns1 AC\n//\n";
    let err = read_text_msa_to_string_with_limit(&mut &input[..], 8).unwrap_err();
    assert!(err.to_string().contains("MSA input exceeds 8 bytes"));
}
