use crate::alphabet::{Alphabet, Dsq};
use crate::util::cmath::c_logf_to_f32;

pub(super) fn null2_by_trace_striped(
    om: &crate::simd::oprofile::OProfile,
    tr: &crate::trace::Trace,
    zstart: usize,
    zend: usize,
) -> Vec<f32> {
    use crate::trace::State;
    let q = crate::simd::oprofile::nqf(om.m);
    let mut exp_m = vec![0.0_f32; q * 4];
    let mut exp_n = 0.0_f32;
    let mut exp_c = 0.0_f32;
    let mut exp_j = 0.0_f32;
    let mut ld = 0usize;

    for z in zstart..=zend.min(tr.n.saturating_sub(1)) {
        if tr.i[z] == 0 {
            continue;
        }
        ld += 1;
        if tr.k[z] > 0 {
            let k = tr.k[z] - 1;
            exp_m[(k % q) * 4 + k / q] += 1.0;
        } else {
            match tr.st[z] {
                State::N => exp_n += 1.0,
                State::C => exp_c += 1.0,
                State::J => exp_j += 1.0,
                _ => {}
            }
        }
    }

    if ld == 0 {
        return vec![1.0; om.abc_k];
    }

    let norm = 1.0 / ld as f32;
    for val in &mut exp_m {
        *val *= norm;
    }
    let xfactor = exp_n * norm + exp_c * norm + exp_j * norm;

    om.rfv
        .iter()
        .take(om.abc_k)
        .map(|odds| {
            let mut lanes = [0.0_f32; 4];
            for (stripe, expected) in odds.iter().take(q).zip(exp_m.as_chunks::<4>().0) {
                for lane in 0..4 {
                    lanes[lane] += expected[lane] * stripe[lane];
                }
            }
            ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + xfactor
        })
        .collect()
}

pub(super) fn add_null2_correction(
    null2: &[f32],
    abc: &Alphabet,
    dsq: &[Dsq],
    ienv: usize,
    jenv: usize,
    n2sc: &mut [f32],
) -> f32 {
    let log_null2: Vec<f32> = null2.iter().map(|&odds| c_logf_to_f32(odds)).collect();
    let mut correction = 0.0_f32;
    for pos in ienv..=jenv {
        let x = dsq[pos] as usize;
        let sc = match log_null2.get(x) {
            Some(&sc) => sc,
            None => degenerate_odds(null2, abc, x).map_or(0.0, c_logf_to_f32),
        };
        n2sc[pos] = sc;
        correction += sc;
    }
    correction
}

pub(super) fn null2_ratio_for_code(null2: &[f32], abc: &Alphabet, x: usize) -> f32 {
    match null2.get(x) {
        Some(&odds) => odds,
        None => degenerate_odds(null2, abc, x).unwrap_or(1.0),
    }
}

fn degenerate_odds(null2: &[f32], abc: &Alphabet, x: usize) -> Option<f32> {
    if x >= abc.kp.saturating_sub(2) || x == abc.k || abc.ndegen[x] == 0 {
        return None;
    }
    let mut odds = 0.0_f32;
    for (a, &value) in null2.iter().enumerate().take(abc.k) {
        if abc.degen[x][a] {
            odds += value;
        }
    }
    Some(odds / abc.ndegen[x] as f32)
}
