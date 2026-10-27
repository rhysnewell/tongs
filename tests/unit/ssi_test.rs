use super::*;

fn fn3_record_with(name: &str, acc: &str) -> Vec<u8> {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/tutorial/fn3.hmm"
    ))
    .unwrap();
    text.replace("NAME  fn3", &format!("NAME  {name}"))
        .replace("ACC   PF00041.13", &format!("ACC   {acc}"))
        .into_bytes()
}

/// Smoke test: building an index for `minipfam.hmm` should index >=10 HMMs
/// and let us look up `14-3-3` by name.
#[test]
fn test_build_index() {
    let idx = Index::build_from_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/testsuite/minipfam.hmm"
    )))
    .unwrap();
    assert!(
        idx.len() >= 10,
        "minipfam should have at least 10 HMMs, got {}",
        idx.len()
    );
    assert!(idx.lookup("14-3-3").is_some());
}

#[test]
fn write_hmm_ssi_emits_easel_v3_header_and_keys() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("two.hmm");
    let mut data = fn3_record_with("b", "PF00002");
    data.extend_from_slice(&fn3_record_with("a", "PF00001"));
    std::fs::write(&hmm_path, data).unwrap();

    let (ssi_path, nprimary, nsecondary) = write_hmm_ssi(&hmm_path).unwrap();
    assert_eq!(nprimary, 2);
    assert_eq!(nsecondary, 2);

    let bytes = std::fs::read(ssi_path).unwrap();
    assert_eq!(
        u32::from_be_bytes(bytes[0..4].try_into().unwrap()),
        SSI_V30_MAGIC
    );
    assert_eq!(
        u32::from_be_bytes(bytes[8..12].try_into().unwrap()),
        SSI_OFFSZ
    );
    assert_eq!(u16::from_be_bytes(bytes[12..14].try_into().unwrap()), 1);
    assert_eq!(u64::from_be_bytes(bytes[14..22].try_into().unwrap()), 2);
    assert_eq!(u64::from_be_bytes(bytes[22..30].try_into().unwrap()), 2);

    let poffset = u64::from_be_bytes(bytes[62..70].try_into().unwrap()) as usize;
    assert_eq!(&bytes[poffset..poffset + 2], b"a\0");
}

#[test]
fn write_hmm_ssi_records_rejects_embedded_nul_key() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("models.hmm");
    let ssi_path = dir.path().join("models.hmm.ssi");

    let err = write_hmm_ssi_records(
        &hmm_path,
        &ssi_path,
        [("bad\0name".to_string(), None, 0)],
        false,
    )
    .unwrap_err();
    assert!(err.to_string().contains("embedded NUL"));
}

#[test]
fn read_hmm_ssi_accepts_32_bit_offsets() {
    let dir = tempfile::tempdir().unwrap();
    let ssi_path = dir.path().join("small.ssi");

    let stored_path = "small.hmm";
    let primary = "alpha";
    let secondary = "PF00001";
    let flen = fixed_len(stored_path);
    let plen = fixed_len(primary);
    let slen = fixed_len(secondary);
    let frecsize = flen + 4 * std::mem::size_of::<u32>();
    let precsize = plen + std::mem::size_of::<u16>() + 2 * SSI_OFFSZ_32 as usize + 8;
    let srecsize = slen + plen;
    let foffset = 9 * std::mem::size_of::<u32>()
        + 2 * std::mem::size_of::<u64>()
        + std::mem::size_of::<u16>()
        + 3 * SSI_OFFSZ_32 as usize;
    let poffset = foffset + frecsize;
    let soffset = poffset + precsize;

    let mut out = std::fs::File::create(&ssi_path).unwrap();
    write_u32(&mut out, SSI_V30_MAGIC).unwrap();
    write_u32(&mut out, 0).unwrap();
    write_u32(&mut out, SSI_OFFSZ_32).unwrap();
    write_u16(&mut out, 1).unwrap();
    write_u64(&mut out, 1).unwrap();
    write_u64(&mut out, 1).unwrap();
    write_u32(&mut out, flen as u32).unwrap();
    write_u32(&mut out, plen as u32).unwrap();
    write_u32(&mut out, slen as u32).unwrap();
    write_u32(&mut out, frecsize as u32).unwrap();
    write_u32(&mut out, precsize as u32).unwrap();
    write_u32(&mut out, srecsize as u32).unwrap();
    write_u32(&mut out, foffset as u32).unwrap();
    write_u32(&mut out, poffset as u32).unwrap();
    write_u32(&mut out, soffset as u32).unwrap();

    write_fixed_str(&mut out, stored_path, flen).unwrap();
    write_u32(&mut out, 0).unwrap();
    write_u32(&mut out, 0).unwrap();
    write_u32(&mut out, 0).unwrap();
    write_u32(&mut out, 0).unwrap();

    write_fixed_str(&mut out, primary, plen).unwrap();
    write_u16(&mut out, 0).unwrap();
    write_u32(&mut out, 1234).unwrap();
    write_u32(&mut out, 0).unwrap();
    write_i64(&mut out, 0).unwrap();

    write_fixed_str(&mut out, secondary, slen).unwrap();
    write_fixed_str(&mut out, primary, plen).unwrap();
    drop(out);

    let on_disk = read_ssi_path(&ssi_path).unwrap();
    assert_eq!(on_disk.indexed_path, PathBuf::from(stored_path));
    assert_eq!(on_disk.lookup(primary), Some(1234));
    assert_eq!(on_disk.lookup(secondary), Some(1234));
}

#[test]
fn read_hmm_ssi_rejects_table_extent_beyond_file() {
    let dir = tempfile::tempdir().unwrap();
    let ssi_path = dir.path().join("bad.ssi");

    let stored_path = "bad.hmm";
    let primary = "alpha";
    let flen = fixed_len(stored_path);
    let plen = fixed_len(primary);
    let slen = 0usize;
    let frecsize = flen + 4 * std::mem::size_of::<u32>();
    let precsize = plen + std::mem::size_of::<u16>() + 2 * SSI_OFFSZ as usize + 8;
    let srecsize = slen + plen;
    let foffset = 9 * std::mem::size_of::<u32>()
        + 2 * std::mem::size_of::<u64>()
        + std::mem::size_of::<u16>()
        + 3 * SSI_OFFSZ as usize;
    let poffset = 9_999u64;
    let soffset = foffset as u64 + frecsize as u64;

    let mut out = std::fs::File::create(&ssi_path).unwrap();
    write_u32(&mut out, SSI_V30_MAGIC).unwrap();
    write_u32(&mut out, 0).unwrap();
    write_u32(&mut out, SSI_OFFSZ).unwrap();
    write_u16(&mut out, 1).unwrap();
    write_u64(&mut out, 1).unwrap();
    write_u64(&mut out, 0).unwrap();
    write_u32(&mut out, flen as u32).unwrap();
    write_u32(&mut out, plen as u32).unwrap();
    write_u32(&mut out, slen as u32).unwrap();
    write_u32(&mut out, frecsize as u32).unwrap();
    write_u32(&mut out, precsize as u32).unwrap();
    write_u32(&mut out, srecsize as u32).unwrap();
    write_u64(&mut out, foffset as u64).unwrap();
    write_u64(&mut out, poffset).unwrap();
    write_u64(&mut out, soffset).unwrap();
    write_fixed_str(&mut out, stored_path, flen).unwrap();
    write_u32(&mut out, 0).unwrap();
    write_u32(&mut out, 0).unwrap();
    write_u32(&mut out, 0).unwrap();
    write_u32(&mut out, 0).unwrap();
    drop(out);

    let err = read_ssi_path(&ssi_path).unwrap_err();
    assert!(err
        .to_string()
        .contains("primary table extent exceeds file length"));
}

#[test]
fn read_hmm_ssi_rejects_unterminated_fixed_string() {
    let err = validate_fixed_string_padding(b"alpha").unwrap_err();
    assert!(err.to_string().contains("missing NUL terminator"));
}

#[test]
fn read_hmm_ssi_rejects_nonzero_fixed_string_padding() {
    let err = validate_fixed_string_padding(b"a\0x").unwrap_err();
    assert!(err.to_string().contains("nonzero padding"));
}

// APFS rejects file names that are not UTF-8.
#[cfg(target_os = "linux")]
#[test]
fn write_hmm_ssi_preserves_non_utf8_path_bytes() {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};

    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir
        .path()
        .join(std::ffi::OsString::from_vec(b"two-\xff.hmm".to_vec()));
    let mut data = fn3_record_with("b", "PF00002");
    data.extend_from_slice(&fn3_record_with("a", "PF00001"));
    std::fs::write(&hmm_path, data).unwrap();

    let (ssi_path, nprimary, nsecondary) = write_hmm_ssi(&hmm_path).unwrap();
    assert_eq!(nprimary, 2);
    assert_eq!(nsecondary, 2);
    assert!(ssi_path.exists());
    assert_eq!(ssi_path, path_with_appended_suffix(&hmm_path, ".ssi"));
    assert!(ssi_path.as_os_str().as_bytes().contains(&0xff));

    // C stores only the basename (esl_FileTail) in the SSI file table, so the
    // round-tripped path is the basename — its non-UTF8 bytes must still survive.
    let basename = PathBuf::from(std::ffi::OsString::from_vec(b"two-\xff.hmm".to_vec()));
    let on_disk = read_hmm_ssi(&hmm_path).unwrap().unwrap();
    assert_eq!(on_disk.indexed_path, basename);
    assert!(on_disk.indexed_path.as_os_str().as_bytes().contains(&0xff));
}

#[test]
fn read_hmm_ssi_rejects_stale_file_table_basename() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("current.hmm");
    let ssi_path = path_with_appended_suffix(&hmm_path, ".ssi");
    write_hmm_ssi_records_with_stored_path(
        &hmm_path,
        Path::new("other.hmm"),
        &ssi_path,
        [("alpha".to_string(), None, 0)],
        false,
    )
    .unwrap();

    let err = read_hmm_ssi(&hmm_path).unwrap_err();
    assert!(err.to_string().contains("was built for"));
}

#[test]
fn index_rejects_missing_record_terminator() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("bad.hmm");
    std::fs::write(&hmm_path, b"HMMER3/f\nNAME  bad\nLENG  1\n").unwrap();

    let err = Index::build_from_hmm_file(&hmm_path).unwrap_err();
    assert!(err.to_string().contains("Missing // terminator"));
}

#[test]
fn index_rejects_terminated_record_that_full_parser_rejects() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("bad.hmm");
    std::fs::write(&hmm_path, b"HMMER3/f\nNAME  bad\nLENG  1\n//\n").unwrap();

    let err = write_hmm_ssi(&hmm_path).unwrap_err();
    assert!(
        err.to_string().contains("Unexpected EOF in HMM header")
            || err.to_string().contains("Missing ALPH")
    );
    assert!(!path_with_appended_suffix(&hmm_path, ".ssi").exists());
}
