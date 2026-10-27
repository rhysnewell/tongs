use super::*;
use crate::simd::fixtures::{profiles, proteins};
use crate::simd::vit_filter::p7_viterbi_filter;

fn bits(result: VitResult) -> Option<u32> {
    match result {
        VitResult::Ok(score) => Some(score.to_bits()),
        VitResult::Overflow => None,
    }
}

#[test]
fn sixteen_lanes_score_as_eight() {
    if !wide::available() {
        return;
    }
    let targets = proteins();
    let mut dp = Vec::new();
    for (name, mut om) in profiles() {
        let vp = om.vit.clone().unwrap();
        for sq in &targets {
            om.reconfig_length(sq.n as i32);
            let narrow = unsafe { p7_viterbi_filter(&sq.dsq, sq.n, &om) };
            let wide = unsafe { viterbi_filter(&sq.dsq, sq.n, &om, &vp, &mut dp) };
            assert_eq!(bits(narrow), bits(wide), "{name} {}", sq.name);
        }
    }
}
