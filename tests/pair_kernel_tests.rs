use std::path::Path;

use tongs::alphabet::{Alphabet, Dsq};
use tongs::simd::bck_filter::{backward, backward_pair, Backward};
use tongs::simd::fwd_filter::{forward, forward_pair, Forward};
use tongs::simd::oprofile::OProfile;
use tongs::simd::probmx::ProbMx;
use tongs::{bg::Bg, hmmfile, profile};

const KINASE: &[u8] = b"MQLVETKGGTFGKVYKARDLKSEMEVAIKQIEHPNVVKLLGACTQGGPLYVLMEYAAKGSLRDLVRR";
const HOMEOBOX: &[u8] = b"RRKRTAYTRYQLLELEKEFHFNRYLTRRRRIEIAHALCLTERQIKIWFQNRRMKWKK";

fn load(name: &str) -> (Alphabet, OProfile) {
    let path = format!("{}/test_data/{name}", env!("CARGO_MANIFEST_DIR"));
    let hmms = hmmfile::read_hmm_file(Path::new(&path)).unwrap();
    let hmm = &hmms[0];
    let abc = Alphabet::new(hmm.abc_type);
    let bg = Bg::new(&abc);
    let mut gm = profile::Profile::new(hmm.m, &abc);
    profile::profile_config(hmm, &bg, &mut gm, 400, profile::P7_LOCAL);
    let om = OProfile::convert(&gm);
    (abc, om)
}

fn target(abc: &Alphabet, motif: &[u8], len: usize, seed: u64) -> Vec<Dsq> {
    let residues = b"ACDEFGHIKLMNPQRSTVWY";
    let mut state = seed;
    let mut seq: Vec<u8> = (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            residues[(state >> 33) as usize % residues.len()]
        })
        .collect();
    for start in (len / 5..len).step_by(motif.len() * 2) {
        let take = motif.len().min(len - start);
        seq[start..start + take].copy_from_slice(&motif[..take]);
    }
    abc.digitize(&seq)
}

fn matrix(om: &OProfile, l: usize, full: bool) -> ProbMx {
    let mut mx = ProbMx::default();
    if full {
        mx.resize_full(om.m, l);
    } else {
        mx.resize_parser(om.m, l);
    }
    mx
}

struct Run {
    fwd: [ProbMx; 2],
    bck: [ProbMx; 2],
    fwd_sc: [f32; 2],
}

fn single(om: &OProfile, dsq: [&[Dsq]; 2], full: [bool; 2]) -> Run {
    let mut run = Run {
        fwd: [0, 1].map(|t| matrix(om, dsq[t].len() - 2, full[t])),
        bck: [0, 1].map(|t| matrix(om, dsq[t].len() - 2, full[t])),
        fwd_sc: [0.0; 2],
    };
    for (t, &dsq) in dsq.iter().enumerate() {
        let l = dsq.len() - 2;
        let fwd = &mut run.fwd[t];
        run.fwd_sc[t] = unsafe { forward(om, &mut Forward::new(dsq, 0, l, om.xf, fwd)) };
        let scales = Some(&run.fwd[t].row_scale[..]);
        let mut lane = Backward::new(dsq, 0, l, om.xf, scales, &mut run.bck[t]);
        unsafe { backward(om, &mut lane) };
    }
    run
}

fn paired(om: &OProfile, dsq: [&[Dsq]; 2], full: [bool; 2]) -> Run {
    let [la, lb] = dsq.map(|d| d.len() - 2);
    let [mut fwd_a, mut fwd_b] = [0, 1].map(|t| matrix(om, dsq[t].len() - 2, full[t]));
    let [mut bck_a, mut bck_b] = [0, 1].map(|t| matrix(om, dsq[t].len() - 2, full[t]));
    let fwd_sc = unsafe {
        let mut a = Forward::new(dsq[0], 0, la, om.xf, &mut fwd_a);
        let mut b = Forward::new(dsq[1], 0, lb, om.xf, &mut fwd_b);
        let (sa, sb) = forward_pair(om, &mut a, &mut b);
        [sa, sb]
    };
    unsafe {
        let mut a = Backward::new(dsq[0], 0, la, om.xf, Some(&fwd_a.row_scale), &mut bck_a);
        let mut b = Backward::new(dsq[1], 0, lb, om.xf, Some(&fwd_b.row_scale), &mut bck_b);
        backward_pair(om, &mut a, &mut b);
    }
    Run {
        fwd: [fwd_a, fwd_b],
        bck: [bck_a, bck_b],
        fwd_sc,
    }
}

fn same_bits(label: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len(), "{label} length");
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "{label}[{i}]: {g:?} against {e:?}"
        );
    }
}

#[test]
fn paired_kernels_match_each_target_alone() {
    let models = [
        ("Pkinase_pfam.hmm", KINASE),
        ("Homeodomain_pfam.hmm", HOMEOBOX),
    ];
    let lengths = [(257, 190), (61, 300), (150, 150), (1, 40)];
    let modes = [[true, true], [false, false], [true, false], [false, true]];
    for (name, motif) in models {
        let (abc, om) = load(name);
        for (la, lb) in lengths {
            let a = target(&abc, motif, la, la as u64);
            let b = target(&abc, motif, lb, 7 * lb as u64 + 3);
            for full in modes {
                let case = format!("{name} {la}x{lb} full {full:?}");
                let lone = single(&om, [&a, &b], full);
                let pair = paired(&om, [&a, &b], full);
                for (t, &keeps_rows) in full.iter().enumerate() {
                    let at = format!("{case} target {t}");
                    same_bits(
                        &format!("{at} forward"),
                        &[pair.fwd_sc[t]],
                        &[lone.fwd_sc[t]],
                    );
                    for (kind, got, want) in [
                        ("fwd", &pair.fwd[t], &lone.fwd[t]),
                        ("bck", &pair.bck[t], &lone.bck[t]),
                    ] {
                        same_bits(&format!("{at} {kind} xmx"), &got.xmx, &want.xmx);
                        same_bits(
                            &format!("{at} {kind} scale"),
                            &got.row_scale,
                            &want.row_scale,
                        );
                        assert_eq!(got.has_own_scales, want.has_own_scales, "{at} {kind}");
                        if keeps_rows {
                            same_bits(
                                &format!("{at} {kind} dp"),
                                &got.striped_dp,
                                &want.striped_dp,
                            );
                        }
                    }
                }
            }
        }
    }
}
