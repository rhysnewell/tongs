use super::*;
use crate::alphabet::Alphabet;
use crate::bg::Bg;
use crate::hmmfile;
use crate::hmmfile_binary::write_binary_hmm;
use crate::profile::{self, Profile, P7_LOCAL};
use crate::ssi;
use std::io::Cursor;

fn push_i32(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_ne_bytes());
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_ne_bytes());
}

fn push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_ne_bytes());
}

fn minimal_h3f_record(m: i32, abc_type: i32, name: &str) -> Vec<u8> {
    let mut buf = Vec::new();
    push_u32(&mut buf, V3F_FMAGIC);
    push_i32(&mut buf, m);
    push_i32(&mut buf, abc_type);
    push_i32(&mut buf, name.len() as i32);
    buf.extend_from_slice(name.as_bytes());
    buf.push(0);
    push_i32(&mut buf, 400);
    buf.extend_from_slice(&[1, 2, 3]);
    push_f32(&mut buf, 4.5);
    buf.extend_from_slice(&[5, 6]);

    let k = if abc_type == 3 { 29 } else { 18 };
    let q = (((m as usize).max(1) - 1) / 16 + 1).max(2);
    let msv_bytes = k * (q + P7O_EXTRA_SB + q) * 16;
    buf.extend((0..msv_bytes).map(|i| (i % 251) as u8));

    for i in 0..6 {
        push_f32(&mut buf, 10.0 + i as f32);
    }
    for i in 0..3 {
        buf.extend_from_slice(&(1000_i64 + i).to_ne_bytes());
    }
    for i in 0..P7_MAXABET {
        push_f32(&mut buf, 100.0 + i as f32);
    }
    push_u32(&mut buf, V3F_FMAGIC);
    buf
}

fn fn3_hmm() -> Hmm {
    hmmfile::read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/tutorial/fn3.hmm"
    )))
    .unwrap()
    .remove(0)
}

fn made1_hmm() -> Hmm {
    hmmfile::read_hmm_file(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/hmmer/tutorial/MADE1.hmm"
    )))
    .unwrap()
    .remove(0)
}

fn oprofile_for_press(hmm: &Hmm) -> OProfile {
    let abc = Alphabet::new(hmm.abc_type);
    let mut bg = Bg::new(&abc);
    bg.set_length(400);
    let mut gm = Profile::new(hmm.m, &abc);
    profile::profile_config(hmm, &bg, &mut gm, 400, P7_LOCAL);
    OProfile::convert(&gm)
}

fn write_valid_pressed_set(hmm_path: &Path, hmm: &Hmm) {
    let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
    let h3f = PathBuf::from(format!("{}.h3f", hmm_path.display()));
    let h3p = PathBuf::from(format!("{}.h3p", hmm_path.display()));
    let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
    let om = oprofile_for_press(hmm);

    let mut m = std::fs::File::create(&h3m).unwrap();
    write_binary_hmm(&mut m, hmm).unwrap();
    let mut f = std::fs::File::create(&h3f).unwrap();
    write_h3f_record(&mut f, hmm, &om, [0, 0, 0]).unwrap();
    let mut p = std::fs::File::create(&h3p).unwrap();
    write_h3p_record(&mut p, hmm, &om).unwrap();
    ssi::write_hmm_ssi_records_with_stored_path(
        &h3m,
        hmm_path,
        &h3i,
        [(hmm.name.clone(), hmm.acc.clone(), 0)],
        false,
    )
    .unwrap();
}

fn read_be_u32_at(buf: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(buf[offset..offset + 4].try_into().unwrap())
}

fn write_be_u32_at(buf: &mut [u8], offset: usize, value: u32) {
    buf[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
}

fn read_be_u64_at(buf: &[u8], offset: usize) -> u64 {
    u64::from_be_bytes(buf[offset..offset + 8].try_into().unwrap())
}

fn write_be_u64_at(buf: &mut [u8], offset: usize, value: u64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_be_bytes());
}

fn write_fixed_string_at(buf: &mut [u8], offset: usize, len: usize, value: &str) {
    let dst = &mut buf[offset..offset + len];
    dst.fill(0);
    let bytes = value.as_bytes();
    dst[..bytes.len().min(len)].copy_from_slice(&bytes[..bytes.len().min(len)]);
}

fn write_be_i64_at(buf: &mut [u8], offset: usize, value: i64) {
    buf[offset..offset + 8].copy_from_slice(&value.to_be_bytes());
}

fn h3i_primary_data_offset_position(buf: &[u8]) -> usize {
    let plen = read_be_u32_at(buf, 34) as usize;
    let poffset = read_be_u64_at(buf, 62) as usize;
    poffset + plen + std::mem::size_of::<u16>() + std::mem::size_of::<u64>()
}

fn write_two_record_pressed_set(hmm_path: &Path, first: &Hmm, second: &Hmm) {
    let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
    let h3f = PathBuf::from(format!("{}.h3f", hmm_path.display()));
    let h3p = PathBuf::from(format!("{}.h3p", hmm_path.display()));
    let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
    let om = oprofile_for_press(first);

    let mut h3m_bytes = Vec::new();
    write_binary_hmm(&mut h3m_bytes, first).unwrap();
    let second_offset = h3m_bytes.len() as u64;
    write_binary_hmm(&mut h3m_bytes, second).unwrap();
    std::fs::write(&h3m, h3m_bytes).unwrap();

    let mut f = std::fs::File::create(&h3f).unwrap();
    write_h3f_record(&mut f, first, &om, [0, 0, 0]).unwrap();
    let mut p = std::fs::File::create(&h3p).unwrap();
    write_h3p_record(&mut p, first, &om).unwrap();
    ssi::write_hmm_ssi_records_with_stored_path(
        &h3m,
        hmm_path,
        &h3i,
        [
            (first.name.clone(), first.acc.clone(), 0),
            (second.name.clone(), second.acc.clone(), second_offset),
        ],
        false,
    )
    .unwrap();
}

fn assert_oprofile_exact(actual: &OProfile, expected: &OProfile) {
    assert_eq!(actual.m, expected.m);
    assert_eq!(actual.l, expected.l);
    assert_eq!(actual.name, expected.name);
    assert_eq!(actual.abc_k, expected.abc_k);
    assert_eq!(actual.abc_kp, expected.abc_kp);
    assert_eq!(actual.rbv, expected.rbv);
    assert_eq!(actual.sbv, expected.sbv);
    assert_eq!(actual.tbm_b, expected.tbm_b);
    assert_eq!(actual.tec_b, expected.tec_b);
    assert_eq!(actual.tjb_b, expected.tjb_b);
    assert_eq!(actual.scale_b.to_bits(), expected.scale_b.to_bits());
    assert_eq!(actual.base_b, expected.base_b);
    assert_eq!(actual.bias_b, expected.bias_b);
    assert_eq!(actual.rwv, expected.rwv);
    assert_eq!(actual.twv, expected.twv);
    assert_eq!(actual.xw, expected.xw);
    assert_eq!(actual.scale_w.to_bits(), expected.scale_w.to_bits());
    assert_eq!(actual.base_w, expected.base_w);
    assert_eq!(actual.ddbound_w, expected.ddbound_w);
    assert_eq!(
        actual.ncj_roundoff.to_bits(),
        expected.ncj_roundoff.to_bits()
    );
    assert_eq!(actual.rfv, expected.rfv);
    assert_eq!(actual.tfv, expected.tfv);
    assert_eq!(actual.xf, expected.xf);
    assert_eq!(actual.nj.to_bits(), expected.nj.to_bits());
    assert_eq!(actual.mode, expected.mode);
    assert_eq!(actual.evparam, expected.evparam);
    assert_eq!(actual.cutoff, expected.cutoff);
    assert_eq!(actual.compo, expected.compo);
}

fn msv_score(result: crate::simd::msv_filter::MsvResult) -> Option<u32> {
    match result {
        crate::simd::msv_filter::MsvResult::Ok(sc) => Some(sc.to_bits()),
        crate::simd::msv_filter::MsvResult::Overflow => None,
    }
}

fn vit_score(result: crate::simd::vit_filter::VitResult) -> Option<u32> {
    match result {
        crate::simd::vit_filter::VitResult::Ok(sc) => Some(sc.to_bits()),
        crate::simd::vit_filter::VitResult::Overflow => None,
    }
}

#[test]
fn reads_h3f_msv_scores_with_extra_sbv_vectors() {
    let mut cursor = Cursor::new(minimal_h3f_record(20, 3, "prot"));
    let rec = read_h3f_record(&mut cursor).unwrap().unwrap();

    let q = 2;
    assert_eq!(rec.msv_scores.len(), 29 * (q + P7O_EXTRA_SB + q) * 16);
    assert_eq!(rec.name, "prot");
    assert_eq!(rec.compo[19], 119.0);
    assert!(read_h3f_record(&mut cursor).unwrap().is_none());
}

#[test]
fn reads_all_twenty_composition_floats_for_dna_profiles() {
    let mut cursor = Cursor::new(minimal_h3f_record(7, 2, "dna"));
    let rec = read_h3f_record(&mut cursor).unwrap().unwrap();

    assert_eq!(rec.msv_scores.len(), 18 * (2 + P7O_EXTRA_SB + 2) * 16);
    assert_eq!(rec.compo[0], 100.0);
    assert_eq!(rec.compo[19], 119.0);
}

#[test]
fn read_h3f_rejects_truncated_initial_magic() {
    let mut cursor = Cursor::new(vec![0xb3, 0xe6]);

    let err = match read_h3f_record(&mut cursor) {
        Ok(_) => panic!("truncated .h3f initial magic was accepted"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("failed to fill whole buffer"));
}

#[test]
fn read_h3f_rejects_invalid_header_values() {
    let mut cursor = Cursor::new(minimal_h3f_record(0, 3, "bad"));

    let err = match read_h3f_record(&mut cursor) {
        Ok(_) => panic!("invalid .h3f header was accepted"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("Invalid .h3f model length"));
}

#[test]
fn read_h3f_rejects_excessive_model_length_before_vector_allocation() {
    let mut buf = Vec::new();
    push_u32(&mut buf, V3F_FMAGIC);
    push_i32(&mut buf, MAX_PRESSED_MODEL_LENGTH + 1);
    push_i32(&mut buf, 3);

    let err = match read_h3f_record(&mut Cursor::new(buf)) {
        Ok(_) => panic!("excessive .h3f model length was accepted"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("Invalid .h3f model length"));
}

#[test]
fn read_h3f_rejects_excessive_name_length_before_allocation() {
    let mut buf = Vec::new();
    push_u32(&mut buf, V3F_FMAGIC);
    push_i32(&mut buf, 20);
    push_i32(&mut buf, 3);
    push_i32(&mut buf, (MAX_PRESSED_STRING_LEN + 1) as i32);

    let err = match read_h3f_record(&mut Cursor::new(buf)) {
        Ok(_) => panic!("excessive .h3f name length was accepted"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("name length"));
    assert!(err.to_string().contains("too large"));
}

#[test]
fn read_h3f_rejects_embedded_nul_in_name() {
    let mut record = minimal_h3f_record(20, 3, "prot");
    record[16 + 2] = 0;

    let err = match read_h3f_record(&mut Cursor::new(record)) {
        Ok(_) => panic!("embedded NUL in .h3f name was accepted"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("embedded NUL"));
}

#[test]
fn writer_rejects_pressed_cstr_with_embedded_nul() {
    let err = write_len_prefixed_cstr(&mut Vec::new(), "bad\0name").unwrap_err();
    assert!(err.to_string().contains("embedded NUL"));
}

#[test]
fn read_h3f_rejects_empty_name() {
    let err = match read_h3f_record(&mut Cursor::new(minimal_h3f_record(20, 3, ""))) {
        Ok(_) => panic!("empty .h3f name was accepted"),
        Err(err) => err,
    };
    assert!(err.to_string().contains(".h3f name is empty"));
}

#[test]
fn read_h3p_rejects_missing_name_terminator() {
    let hmm = fn3_hmm();
    let om = oprofile_for_press(&hmm);
    let mut record = Vec::new();
    write_h3p_record(&mut record, &hmm, &om).unwrap();
    let name_len = i32::from_ne_bytes(record[12..16].try_into().unwrap()) as usize;
    record[16 + name_len] = b'X';

    let err = match read_h3p_record(&mut Cursor::new(record)) {
        Ok(_) => panic!("missing .h3p name terminator was accepted"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("missing trailing NUL"));
}

#[test]
fn read_h3p_rejects_empty_name() {
    let mut hmm = fn3_hmm();
    hmm.name.clear();
    let om = oprofile_for_press(&hmm);
    let mut record = Vec::new();
    write_h3p_record(&mut record, &hmm, &om).unwrap();

    let err = match read_h3p_record(&mut Cursor::new(record)) {
        Ok(_) => panic!("empty .h3p name was accepted"),
        Err(err) => err,
    };
    assert!(err.to_string().contains(".h3p name is empty"));
}

#[test]
fn pressed_db_available_rejects_partial_sidecars() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("models.hmm");
    std::fs::write(&hmm, b"HMMER3/f\nNAME  x\nLENG  1\n//\n").unwrap();
    std::fs::write(format!("{}.h3m", hmm.display()), b"partial").unwrap();

    let err = pressed_db_available(&hmm).unwrap_err();
    assert!(err.to_string().contains("Incomplete pressed database"));
}

#[test]
fn pressed_db_available_rejects_bad_h3f_magic() {
    let dir = tempfile::tempdir().unwrap();
    let hmm = dir.path().join("models.hmm");
    std::fs::write(&hmm, b"HMMER3/f\nNAME  x\nLENG  1\n//\n").unwrap();
    std::fs::write(format!("{}.h3f", hmm.display()), 0_u32.to_ne_bytes()).unwrap();
    std::fs::write(format!("{}.h3p", hmm.display()), 0_u32.to_ne_bytes()).unwrap();
    std::fs::write(format!("{}.h3i", hmm.display()), b"index").unwrap();
    std::fs::write(format!("{}.h3m", hmm.display()), b"models").unwrap();

    let err = pressed_db_available(&hmm).unwrap_err();
    assert!(err.to_string().contains("Bad .h3f magic"));
}

#[cfg(unix)]
#[test]
fn pressed_sidecar_paths_preserve_non_utf8_database_path() {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};

    let mut raw = b"/tmp/nonutf8-hmm-".to_vec();
    raw.push(0xff);
    raw.extend_from_slice(b".hmm");
    let hmm_path = PathBuf::from(std::ffi::OsString::from_vec(raw.clone()));

    let paths = pressed_sidecar_paths(&hmm_path);
    for (path, suffix) in paths.iter().zip([".h3f", ".h3p", ".h3i", ".h3m"]) {
        let mut expected = raw.clone();
        expected.extend_from_slice(suffix.as_bytes());
        assert_eq!(path.as_os_str().as_bytes(), expected.as_slice());
    }
}

#[test]
fn pressed_db_available_rejects_mismatched_h3f_h3p_records() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let hmm = fn3_hmm();
    write_valid_pressed_set(&hmm_path, &hmm);

    let mut mismatched = hmm.clone();
    mismatched.name = "stale-profile".to_string();
    let om = oprofile_for_press(&mismatched);
    let mut h3p = std::fs::File::create(format!("{}.h3p", hmm_path.display())).unwrap();
    write_h3p_record(&mut h3p, &mismatched, &om).unwrap();

    let err = pressed_db_available(&hmm_path).unwrap_err();
    assert!(err
        .to_string()
        .contains("Pressed sidecar first-record mismatch"));
}

#[test]
fn pressed_db_available_rejects_mismatched_h3m_record() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let hmm = fn3_hmm();
    write_valid_pressed_set(&hmm_path, &hmm);

    let mut mismatched = hmm.clone();
    mismatched.name = "stale-model".to_string();
    let mut h3m = std::fs::File::create(format!("{}.h3m", hmm_path.display())).unwrap();
    write_binary_hmm(&mut h3m, &mismatched).unwrap();

    let err = pressed_db_available(&hmm_path).unwrap_err();
    assert!(err
        .to_string()
        .contains("key fn3 points to HMM stale-model"));
}

#[test]
fn pressed_db_available_rejects_stale_h3i_file_record() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let hmm = fn3_hmm();
    write_valid_pressed_set(&hmm_path, &hmm);

    let wrong_h3m = dir.path().join("other.h3m");
    let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
    ssi::write_hmm_ssi_records(
        &wrong_h3m,
        &h3i,
        [(hmm.name.clone(), hmm.acc.clone(), 0)],
        true,
    )
    .unwrap();

    let err = pressed_db_available(&hmm_path).unwrap_err();
    assert!(err.to_string().contains("was built for"));
}

#[test]
fn pressed_db_available_rejects_h3i_stale_offset() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let hmm = fn3_hmm();
    write_valid_pressed_set(&hmm_path, &hmm);

    let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
    let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
    ssi::write_hmm_ssi_records_with_stored_path(
        &h3m,
        &hmm_path,
        &h3i,
        [(hmm.name.clone(), hmm.acc.clone(), 1)],
        true,
    )
    .unwrap();

    let err = pressed_db_available(&hmm_path).unwrap_err();
    assert!(err.to_string().contains("Bad binary HMM magic"));
}

#[test]
fn pressed_db_available_rejects_h3i_stale_primary_name() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let hmm = fn3_hmm();
    write_valid_pressed_set(&hmm_path, &hmm);

    let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
    let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
    ssi::write_hmm_ssi_records_with_stored_path(
        &h3m,
        &hmm_path,
        &h3i,
        [("stale".to_string(), None, 0)],
        true,
    )
    .unwrap();

    let err = pressed_db_available(&hmm_path).unwrap_err();
    assert!(err.to_string().contains("key stale points to HMM"));
}

#[test]
fn pressed_db_available_rejects_excessive_h3i_fixed_field_length() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let hmm = fn3_hmm();
    write_valid_pressed_set(&hmm_path, &hmm);

    let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
    let mut bytes = std::fs::read(&h3i).unwrap();
    write_be_u32_at(&mut bytes, 30, (MAX_PRESSED_STRING_LEN + 1) as u32);
    std::fs::write(&h3i, bytes).unwrap();

    let err = pressed_db_available(&hmm_path).unwrap_err();
    assert!(err.to_string().contains("excessive fixed field length"));
}

#[test]
fn pressed_db_available_rejects_h3i_table_extent_past_file() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let hmm = fn3_hmm();
    write_valid_pressed_set(&hmm_path, &hmm);

    let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
    let mut bytes = std::fs::read(&h3i).unwrap();
    let past_end = bytes.len() as u64 + 1;
    write_be_u64_at(&mut bytes, 62, past_end);
    std::fs::write(&h3i, bytes).unwrap();

    let err = pressed_db_available(&hmm_path).unwrap_err();
    assert!(err.to_string().contains("primary table extent exceeds"));
}

#[test]
fn pressed_db_available_rejects_h3i_primary_data_offset() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let hmm = fn3_hmm();
    write_valid_pressed_set(&hmm_path, &hmm);

    let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
    let mut bytes = std::fs::read(&h3i).unwrap();
    let data_offset_pos = h3i_primary_data_offset_position(&bytes);
    write_be_u64_at(&mut bytes, data_offset_pos, 1);
    std::fs::write(&h3i, bytes).unwrap();

    let err = pressed_db_available(&hmm_path).unwrap_err();
    assert!(err.to_string().contains("unsupported data_offset=1"));
}

#[test]
fn pressed_db_available_rejects_h3i_primary_record_len() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let hmm = fn3_hmm();
    write_valid_pressed_set(&hmm_path, &hmm);

    let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
    let mut bytes = std::fs::read(&h3i).unwrap();
    let record_len_pos = h3i_primary_data_offset_position(&bytes) + std::mem::size_of::<u64>();
    write_be_i64_at(&mut bytes, record_len_pos, 7);
    std::fs::write(&h3i, bytes).unwrap();

    let err = pressed_db_available(&hmm_path).unwrap_err();
    assert!(err.to_string().contains("record_len=7"));
}

#[test]
fn pressed_db_available_rejects_duplicate_h3i_secondary_key() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("two.hmm");
    let first = fn3_hmm();
    let mut second = first.clone();
    second.name = "fn3-other".to_string();
    second.acc = Some("PF99999.1".to_string());
    write_two_record_pressed_set(&hmm_path, &first, &second);

    let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
    let mut bytes = std::fs::read(&h3i).unwrap();
    let srecsize = read_be_u32_at(&bytes, 50) as usize;
    let soffset = read_be_u64_at(&bytes, 70) as usize;
    let first_secondary = bytes[soffset..soffset + srecsize].to_vec();
    bytes[soffset + srecsize..soffset + 2 * srecsize].copy_from_slice(&first_secondary);
    std::fs::write(&h3i, bytes).unwrap();

    let err = pressed_db_available(&hmm_path).unwrap_err();
    assert!(err.to_string().contains("duplicate secondary key"));
}

#[test]
fn pressed_db_available_rejects_unsorted_h3i_primary_keys() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("two.hmm");
    let first = fn3_hmm();
    let mut second = first.clone();
    second.name = "fn3-other".to_string();
    second.acc = Some("PF99999.1".to_string());
    write_two_record_pressed_set(&hmm_path, &first, &second);

    let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
    let mut bytes = std::fs::read(&h3i).unwrap();
    let precsize = read_be_u32_at(&bytes, 46) as usize;
    let poffset = read_be_u64_at(&bytes, 62) as usize;
    let first_primary = bytes[poffset..poffset + precsize].to_vec();
    let second_primary = bytes[poffset + precsize..poffset + 2 * precsize].to_vec();
    bytes[poffset..poffset + precsize].copy_from_slice(&second_primary);
    bytes[poffset + precsize..poffset + 2 * precsize].copy_from_slice(&first_primary);
    std::fs::write(&h3i, bytes).unwrap();

    let err = pressed_db_available(&hmm_path).unwrap_err();
    assert!(err
        .to_string()
        .contains("primary keys are not strictly sorted"));
}

#[test]
fn pressed_db_available_rejects_unsorted_h3i_secondary_keys() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("two.hmm");
    let first = fn3_hmm();
    let mut second = first.clone();
    second.name = "fn3-other".to_string();
    second.acc = Some("PF99999.1".to_string());
    write_two_record_pressed_set(&hmm_path, &first, &second);

    let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
    let mut bytes = std::fs::read(&h3i).unwrap();
    let srecsize = read_be_u32_at(&bytes, 50) as usize;
    let soffset = read_be_u64_at(&bytes, 70) as usize;
    let first_secondary = bytes[soffset..soffset + srecsize].to_vec();
    let second_secondary = bytes[soffset + srecsize..soffset + 2 * srecsize].to_vec();
    bytes[soffset..soffset + srecsize].copy_from_slice(&second_secondary);
    bytes[soffset + srecsize..soffset + 2 * srecsize].copy_from_slice(&first_secondary);
    std::fs::write(&h3i, bytes).unwrap();

    let err = pressed_db_available(&hmm_path).unwrap_err();
    assert!(err
        .to_string()
        .contains("secondary keys are not strictly sorted"));
}

#[test]
fn pressed_db_available_rejects_empty_h3i_secondary_key() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let mut hmm = fn3_hmm();
    hmm.acc = None;
    write_valid_pressed_set(&hmm_path, &hmm);

    let h3i = PathBuf::from(format!("{}.h3i", hmm_path.display()));
    let mut bytes = std::fs::read(&h3i).unwrap();
    let plen = read_be_u32_at(&bytes, 34) as usize;
    let soffset = read_be_u64_at(&bytes, 70) as usize;
    write_be_u64_at(&mut bytes, 22, 1);
    write_be_u32_at(&mut bytes, 38, 1);
    write_be_u32_at(&mut bytes, 50, (plen + 1) as u32);
    if bytes.len() < soffset + 1 + plen {
        bytes.resize(soffset + 1 + plen, 0);
    }
    write_fixed_string_at(&mut bytes, soffset + 1, plen, &hmm.name);
    std::fs::write(&h3i, bytes).unwrap();

    let err = pressed_db_available(&hmm_path).unwrap_err();
    assert!(err.to_string().contains("empty secondary key"));
}

#[test]
fn pressed_records_reconstruct_oprofile_exactly() {
    let hmm = fn3_hmm();
    let expected = oprofile_for_press(&hmm);

    let mut h3f = Vec::new();
    write_h3f_record(&mut h3f, &hmm, &expected, [11, 22, 33]).unwrap();
    let mut h3p = Vec::new();
    write_h3p_record(&mut h3p, &hmm, &expected).unwrap();

    let msv = read_h3f_record(&mut Cursor::new(h3f)).unwrap().unwrap();
    let profile = read_h3p_record(&mut Cursor::new(h3p)).unwrap().unwrap();
    let actual = oprofile_from_pressed_records(&msv, &profile).unwrap();

    assert_eq!(msv.offsets, [11, 22, 33]);
    assert_oprofile_exact(&actual, &expected);
}

// === Regression tests for pressed-DB byte parity with C hmmpress ===
// F1: .h3f field 6 must be om.max_length (HMM MAXL), not the configured L.
// F2: .h3p annotation arrays must carry a '\0' (not ' ') sentinel at M+1.
// F3: .h3f compo tail [K..20] must be 0.0 (not COMPO_UNSET) for nucleic.

/// Parse the `max_length` field and the 20 compo floats out of a written
/// `.h3f` record (single record). Returns (max_length, compo).
fn parse_h3f_max_length_and_compo(bytes: &[u8]) -> (i32, [f32; 20]) {
    let mut off = 0usize;
    let rd_i32 = |b: &[u8], o: &mut usize| {
        let v = i32::from_ne_bytes(b[*o..*o + 4].try_into().unwrap());
        *o += 4;
        v
    };
    // magic, M, abc
    let _magic = rd_i32(bytes, &mut off);
    let _m = rd_i32(bytes, &mut off);
    let _abc = rd_i32(bytes, &mut off);
    // name: len-prefixed cstr (len, bytes, NUL)
    let nlen = rd_i32(bytes, &mut off) as usize;
    off += nlen + 1;
    let max_length = rd_i32(bytes, &mut off);
    // compo is the last 20 floats before the 4-byte footer magic.
    let compo_start = bytes.len() - 4 - 20 * 4;
    let mut compo = [0.0f32; 20];
    for (i, c) in compo.iter_mut().enumerate() {
        let o = compo_start + i * 4;
        *c = f32::from_ne_bytes(bytes[o..o + 4].try_into().unwrap());
    }
    (max_length, compo)
}

#[test]
fn h3f_writes_hmm_max_length_not_configured_length() {
    // F1: MADE1 has MAXL 426; configured L is 400. C writes max_length(426).
    let hmm = made1_hmm();
    assert_eq!(hmm.max_length, 426);
    let om = oprofile_for_press(&hmm);
    assert_eq!(om.l, 400, "press configures L=400");
    let mut h3f = Vec::new();
    write_h3f_record(&mut h3f, &hmm, &om, [0, 0, 0]).unwrap();
    let (max_length, _compo) = parse_h3f_max_length_and_compo(&h3f);
    assert_eq!(max_length, 426, "F1: .h3f must store HMM MAXL, not L=400");

    // fn3 (amino, no MAXL) → C writes -1.
    let amino = fn3_hmm();
    assert_eq!(amino.max_length, -1);
    let om_a = oprofile_for_press(&amino);
    let mut h3f_a = Vec::new();
    write_h3f_record(&mut h3f_a, &amino, &om_a, [0, 0, 0]).unwrap();
    let (max_length_a, _) = parse_h3f_max_length_and_compo(&h3f_a);
    assert_eq!(max_length_a, -1, "F1: amino model with no MAXL → -1");
}

#[test]
fn h3f_zero_fills_compo_tail_for_nucleic() {
    // F3: nucleic K=4; compo[4..20] must be 0.0 (C zeroed array), not -1.0.
    let hmm = made1_hmm();
    let om = oprofile_for_press(&hmm);
    let mut h3f = Vec::new();
    write_h3f_record(&mut h3f, &hmm, &om, [0, 0, 0]).unwrap();
    let (_max_length, compo) = parse_h3f_max_length_and_compo(&h3f);
    for (i, &c) in compo.iter().enumerate().skip(4) {
        assert_eq!(c, 0.0, "F3: compo[{i}] tail must be 0.0 for nucleic");
    }
    // [0..4) carry the real composition (finite, not COMPO_UNSET).
    for (i, &c) in compo.iter().enumerate().take(4) {
        assert!(c.is_finite() && c != -1.0, "compo[{i}] should be real");
    }
}

#[test]
fn h3p_annotation_arrays_end_with_nul_sentinel() {
    // F2: each present annotation array of width M+2 must have '\0' at M+1.
    let hmm = fn3_hmm();
    let m = hmm.m;
    let om = oprofile_for_press(&hmm);
    let mut h3p = Vec::new();
    write_h3p_record(&mut h3p, &hmm, &om).unwrap();

    // Re-walk the record header to locate the first annotation array.
    let mut off = 0usize;
    let rd_i32 = |b: &[u8], o: &mut usize| {
        let v = i32::from_ne_bytes(b[*o..*o + 4].try_into().unwrap());
        *o += 4;
        v
    };
    let _magic = rd_i32(&h3p, &mut off);
    let _m = rd_i32(&h3p, &mut off);
    let _abc = rd_i32(&h3p, &mut off);
    let nlen = rd_i32(&h3p, &mut off) as usize; // name
    off += nlen + 1;
    let acclen = rd_i32(&h3p, &mut off) as usize; // acc
    if acclen > 0 {
        off += acclen + 1;
    }
    let desclen = rd_i32(&h3p, &mut off) as usize; // desc
    if desclen > 0 {
        off += desclen + 1;
    }
    // Now four annotation arrays of width m+2: rf, mm, cs, consensus.
    let width = m + 2;
    for arr in 0..4 {
        let base = off + arr * width;
        assert_eq!(
            h3p[base + m + 1],
            0,
            "F2: annotation array {arr} must end with NUL at index M+1"
        );
    }
}

#[test]
fn read_pressed_oprofiles_preserves_h3m_record_order_profile() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let hmm = fn3_hmm();
    let expected = oprofile_for_press(&hmm);
    write_valid_pressed_set(&hmm_path, &hmm);

    let profiles = read_pressed_oprofiles(&hmm_path).unwrap();

    assert_eq!(profiles.len(), 1);
    assert_oprofile_exact(&profiles[0], &expected);
}

#[test]
fn read_pressed_oprofiles_uses_h3f_profile_offset() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
    let h3f = PathBuf::from(format!("{}.h3f", hmm_path.display()));
    let h3p = PathBuf::from(format!("{}.h3p", hmm_path.display()));
    let hmm = fn3_hmm();
    let expected = oprofile_for_press(&hmm);

    let mut m = std::fs::File::create(&h3m).unwrap();
    write_binary_hmm(&mut m, &hmm).unwrap();
    let mut p = std::fs::File::create(&h3p).unwrap();
    p.write_all(b"prefix").unwrap();
    let profile_offset = p.stream_position().unwrap() as i64;
    write_h3p_record(&mut p, &hmm, &expected).unwrap();

    let mut f = std::fs::File::create(&h3f).unwrap();
    write_h3f_record(&mut f, &hmm, &expected, [0, 0, profile_offset]).unwrap();

    let profiles = read_pressed_oprofiles(&hmm_path).unwrap();

    assert_eq!(profiles.len(), 1);
    assert_oprofile_exact(&profiles[0], &expected);
}

#[test]
fn read_pressed_oprofiles_rejects_negative_h3p_offset() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
    let h3f = PathBuf::from(format!("{}.h3f", hmm_path.display()));
    let h3p = PathBuf::from(format!("{}.h3p", hmm_path.display()));
    let hmm = fn3_hmm();
    let expected = oprofile_for_press(&hmm);

    let mut m = std::fs::File::create(&h3m).unwrap();
    write_binary_hmm(&mut m, &hmm).unwrap();
    let mut p = std::fs::File::create(&h3p).unwrap();
    write_h3p_record(&mut p, &hmm, &expected).unwrap();
    let mut f = std::fs::File::create(&h3f).unwrap();
    write_h3f_record(&mut f, &hmm, &expected, [0, 0, -1]).unwrap();

    let err = read_pressed_oprofiles(&hmm_path).unwrap_err();
    assert!(err.to_string().contains("negative .h3p offset"));
}

#[test]
fn read_pressed_oprofiles_rejects_extra_h3p_records() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let hmm = fn3_hmm();
    write_valid_pressed_set(&hmm_path, &hmm);

    let h3p = PathBuf::from(format!("{}.h3p", hmm_path.display()));
    let original = std::fs::read(&h3p).unwrap();
    let mut appended = original.clone();
    appended.extend_from_slice(&original);
    std::fs::write(&h3p, appended).unwrap();

    let err = read_pressed_oprofiles(&hmm_path).unwrap_err();
    assert!(err.to_string().contains("trailing bytes"));
}

#[test]
fn read_pressed_oprofiles_rejects_stale_h3f_foffset() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
    let h3f = PathBuf::from(format!("{}.h3f", hmm_path.display()));
    let h3p = PathBuf::from(format!("{}.h3p", hmm_path.display()));
    let hmm = fn3_hmm();
    let expected = oprofile_for_press(&hmm);

    let mut m = std::fs::File::create(&h3m).unwrap();
    write_binary_hmm(&mut m, &hmm).unwrap();
    let mut p = std::fs::File::create(&h3p).unwrap();
    write_h3p_record(&mut p, &hmm, &expected).unwrap();
    let mut f = std::fs::File::create(&h3f).unwrap();
    write_h3f_record(&mut f, &hmm, &expected, [0, 1, 0]).unwrap();

    let err = read_pressed_oprofiles(&hmm_path).unwrap_err();
    assert!(err.to_string().contains("FOFFSET mismatch"));
}

#[test]
fn read_pressed_oprofiles_rejects_stale_h3f_moffset() {
    let dir = tempfile::tempdir().unwrap();
    let hmm_path = dir.path().join("fn3.hmm");
    let h3m = PathBuf::from(format!("{}.h3m", hmm_path.display()));
    let h3f = PathBuf::from(format!("{}.h3f", hmm_path.display()));
    let h3p = PathBuf::from(format!("{}.h3p", hmm_path.display()));
    let hmm = fn3_hmm();
    let expected = oprofile_for_press(&hmm);

    let mut m = std::fs::File::create(&h3m).unwrap();
    write_binary_hmm(&mut m, &hmm).unwrap();
    let mut p = std::fs::File::create(&h3p).unwrap();
    write_h3p_record(&mut p, &hmm, &expected).unwrap();
    let mut f = std::fs::File::create(&h3f).unwrap();
    write_h3f_record(&mut f, &hmm, &expected, [1, 0, 0]).unwrap();

    let err = read_pressed_oprofiles(&hmm_path).unwrap_err();
    assert!(err.to_string().contains("MOFFSET mismatch"));
}

#[test]
fn validates_pressed_oprofile_order_against_h3m_records() {
    let hmm_a = fn3_hmm();
    let mut hmm_b = hmm_a.clone();
    hmm_b.name = "fn3_alt".to_string();

    let om_a = oprofile_for_press(&hmm_a);
    let om_b = oprofile_for_press(&hmm_b);

    validate_pressed_oprofiles_match_hmms(&[hmm_a.clone(), hmm_b.clone()], &[om_a, om_b]).unwrap();

    let err = validate_pressed_oprofiles_match_hmms(
        &[hmm_a, hmm_b],
        &[
            oprofile_for_press(&fn3_hmm()),
            oprofile_for_press(&fn3_hmm()),
        ],
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("does not match .h3m record 2"),
        "{err}"
    );
}

#[test]
fn reconstructed_oprofile_scores_like_converted_profile() {
    let hmm = fn3_hmm();
    let mut expected = oprofile_for_press(&hmm);

    let mut h3f = Vec::new();
    write_h3f_record(&mut h3f, &hmm, &expected, [11, 22, 33]).unwrap();
    let mut h3p = Vec::new();
    write_h3p_record(&mut h3p, &hmm, &expected).unwrap();

    let msv = read_h3f_record(&mut Cursor::new(h3f)).unwrap().unwrap();
    let profile = read_h3p_record(&mut Cursor::new(h3p)).unwrap().unwrap();
    let mut actual = oprofile_from_pressed_records(&msv, &profile).unwrap();

    let abc = Alphabet::amino();
    let dsq = abc.digitize(b"ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY");
    let l = dsq.len() - 2;
    expected.reconfig_length(l as i32);
    actual.reconfig_length(l as i32);

    unsafe {
        assert_eq!(
            msv_score(crate::simd::msv_filter::p7_msv_filter(&dsq, l, &actual)),
            msv_score(crate::simd::msv_filter::p7_msv_filter(&dsq, l, &expected))
        );
        assert_eq!(
            vit_score(crate::simd::vit_filter::p7_viterbi_filter(&dsq, l, &actual)),
            vit_score(crate::simd::vit_filter::p7_viterbi_filter(
                &dsq, l, &expected
            ))
        );
        assert_eq!(
            crate::simd::fwd_filter::forward_parser(&dsq, l, &actual).to_bits(),
            crate::simd::fwd_filter::forward_parser(&dsq, l, &expected).to_bits()
        );
    }
}

#[test]
fn pressed_records_reconstruction_rejects_stale_pairing() {
    let hmm = fn3_hmm();
    let expected = oprofile_for_press(&hmm);

    let mut h3f = Vec::new();
    write_h3f_record(&mut h3f, &hmm, &expected, [0, 0, 0]).unwrap();
    let mut h3p = Vec::new();
    let mut stale = hmm.clone();
    stale.name = "stale".to_string();
    let stale_om = oprofile_for_press(&stale);
    write_h3p_record(&mut h3p, &stale, &stale_om).unwrap();

    let msv = read_h3f_record(&mut Cursor::new(h3f)).unwrap().unwrap();
    let profile = read_h3p_record(&mut Cursor::new(h3p)).unwrap().unwrap();
    let err = oprofile_from_pressed_records(&msv, &profile).unwrap_err();

    assert!(err.to_string().contains("Pressed profile record mismatch"));
}
