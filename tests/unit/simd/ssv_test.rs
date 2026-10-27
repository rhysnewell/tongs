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
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let mut answered = 0;
    let targets = proteins();
    for (name, mut om) in profiles() {
        let wide = SsvProfile::striped(&om.sbv, om.m, 32);
        let narrow = SsvProfile::striped(&om.sbv, om.m, 16);
        let mut dp = Vec::new();
        for sq in &targets {
            om.reconfig_length(sq.n as i32);
            let short = wide.score(&sq.dsq, sq.n, &om);
            assert_eq!(
                outcome(&short),
                outcome(&narrow.score(&sq.dsq, sq.n, &om)),
                "{} {}",
                name,
                sq.name
            );
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
