use std::path::Path;

use crate::alphabet::Alphabet;
use crate::bg::Bg;
use crate::hmmfile::read_hmm_file;
use crate::profile::{profile_config, Profile, P7_LOCAL};
use crate::sequence::{open_seq_file, Sequence};
use crate::simd::oprofile::OProfile;

pub(crate) fn profiles() -> Vec<(String, OProfile)> {
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
    hmms.iter()
        .map(|hmm| {
            let abc = Alphabet::new(hmm.abc_type);
            let bg = Bg::new(&abc);
            let mut gm = Profile::new(hmm.m, &abc);
            profile_config(hmm, &bg, &mut gm, 400, P7_LOCAL);
            (hmm.name.clone(), OProfile::convert(&gm))
        })
        .collect()
}

pub(crate) fn proteins() -> Vec<Sequence> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data/gecco_cluster1_proteins.faa");
    let abc = Alphabet::new(crate::alphabet::AlphabetType::Amino);
    let mut file = open_seq_file(&path, &abc).unwrap();
    let mut all = Vec::new();
    let mut sq = Sequence::new();
    while file.read(&mut sq).unwrap() {
        all.push(std::mem::take(&mut sq));
    }
    all
}
