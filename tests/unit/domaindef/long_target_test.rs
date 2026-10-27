use super::*;
use crate::alphabet::AlphabetType;

#[test]
fn local_bg_distributes_degenerates_and_normalizes() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let bg = Bg::new(&abc);
    let dsq = abc.digitize(b"NNNN");

    let local = local_bg(&dsq, 1, 4, &bg, &abc, 4);
    let sum: f32 = local.f.iter().take(abc.k).sum();

    assert!((sum - 1.0).abs() < 1e-6);
    for f in local.f.iter().take(abc.k) {
        assert!(*f > 0.0);
    }
}

#[test]
fn local_bg_falls_back_when_window_has_no_residue_evidence() {
    let abc = Alphabet::new(AlphabetType::Dna);
    let bg = Bg::new(&abc);
    let dsq = vec![DSQ_SENTINEL; 4];

    let local = local_bg(&dsq, 1, 2, &bg, &abc, 2);

    for i in 0..abc.k {
        assert_eq!(local.f[i].to_bits(), bg.f[i].to_bits());
    }
}
