use super::*;
use crate::simd::fixtures::{profiles, proteins};
use crate::simd::msv_filter::{p7_msv_filter_dp_only_rust_helper, MsvResult};

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
    #[cfg(target_arch = "x86_64")]
    let widths = if is_x86_feature_detected!("avx2") { vec![16, 32] } else { vec![16] };
    #[cfg(not(target_arch = "x86_64"))]
    let widths = vec![16];
    let mut answered = 0;
    let targets = proteins();
    for (name, mut om) in profiles() {
        let shortcuts: Vec<_> =
            widths.iter().map(|&lanes| SsvProfile::striped(&om.sbv, om.m, lanes)).collect();
        let mut dp = Vec::new();
        for sq in &targets {
            om.reconfig_length(sq.n as i32);
            let short = shortcuts[0].score(&sq.dsq, sq.n, &om);
            for other in &shortcuts[1..] {
                assert_eq!(
                    outcome(&short),
                    outcome(&other.score(&sq.dsq, sq.n, &om)),
                    "{} {}",
                    name,
                    sq.name
                );
            }
            let full = unsafe { p7_msv_filter_dp_only_rust_helper(&sq.dsq, sq.n, &om, &mut dp) };
            match (short, full) {
                (SsvResult::Ok(short), MsvResult::Ok(long)) => {
                    assert_eq!(short.to_bits(), long.to_bits(), "{} {}", name, sq.name);
                    answered += 1;
                }
                (SsvResult::Overflow, MsvResult::Overflow) | (SsvResult::NoResult, _) => {}
                _ => panic!("{} {} disagrees with the MSV recurrence", name, sq.name),
            }
        }
    }
    assert!(answered > 0);
}
