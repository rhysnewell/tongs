use std::path::Path;

use super::*;
use crate::alphabet::Alphabet;
use crate::bg::Bg;
use crate::hmmfile::read_hmm_file;
use crate::profile::{profile_config, Profile, P7_LOCAL};
use crate::sequence::{open_seq_file, Sequence};
use crate::simd::msv_filter::{p7_msv_filter_dp_only_rust_helper, MsvResult};

fn models() -> Vec<crate::hmm::Hmm> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data");
    let mut hmms = read_hmm_file(&dir.join("gecco_cluster1_hmms.hmm")).unwrap();
    for name in [
        "Pkinase_pfam.hmm",
        "WD40_pfam.hmm",
        "Sugar_tr_pfam.hmm",
        "7tm_1_pfam.hmm",
    ] {
        hmms.extend(read_hmm_file(&dir.join(name)).unwrap());
    }
    hmms
}

fn proteins(abc: &Alphabet) -> Vec<Sequence> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data/gecco_cluster1_proteins.faa");
    let mut file = open_seq_file(&path, abc).unwrap();
    let mut all = Vec::new();
    let mut sq = Sequence::new();
    while file.read(&mut sq).unwrap() {
        all.push(std::mem::replace(&mut sq, Sequence::new()));
    }
    all
}

fn outcome(result: &SsvResult) -> Option<u32> {
    match result {
        SsvResult::Ok(score) => Some(score.to_bits()),
        SsvResult::Overflow => Some(u32::MAX),
        SsvResult::NoResult => None,
    }
}

// Every lane width must find the same best diagonal, and where the shortcut answers it must
// agree with the full MSV recurrence it stands in for.
#[test]
fn shortcut_matches_msv_at_every_lane_width() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let mut answered = 0;
    for hmm in models() {
        let abc = Alphabet::new(hmm.abc_type);
        let bg = Bg::new(&abc);
        let mut gm = Profile::new(hmm.m, &abc);
        profile_config(&hmm, &bg, &mut gm, 400, P7_LOCAL);
        let mut om = OProfile::convert(&gm);
        let wide = SsvProfile::striped(&om.sbv, om.m, 32);
        let narrow = SsvProfile::striped(&om.sbv, om.m, 16);
        let mut dp = Vec::new();
        for sq in proteins(&abc) {
            om.reconfig_length(sq.n as i32);
            let short = wide.score(&sq.dsq, sq.n, &om);
            assert_eq!(
                outcome(&short),
                outcome(&narrow.score(&sq.dsq, sq.n, &om)),
                "{} {}",
                hmm.name,
                sq.name
            );
            let full = unsafe { p7_msv_filter_dp_only_rust_helper(&sq.dsq, sq.n, &om, &mut dp) };
            match (short, full) {
                (SsvResult::Ok(short), MsvResult::Ok(long)) => {
                    assert_eq!(short.to_bits(), long.to_bits(), "{} {}", hmm.name, sq.name);
                    answered += 1;
                }
                (SsvResult::Overflow, MsvResult::Overflow) | (SsvResult::NoResult, _) => {}
                _ => panic!("{} {} disagrees with the MSV recurrence", hmm.name, sq.name),
            }
        }
    }
    assert!(answered > 0);
}
