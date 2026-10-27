use crate::alphabet::{Alphabet, Dsq};
use crate::simd::sse::*;
use crate::util::cmath::c_logf_to_f32;

pub(super) fn null2_by_trace_striped(
    om: &crate::simd::oprofile::OProfile,
    tr: &crate::trace::Trace,
    zstart: usize,
    zend: usize,
) -> Vec<f32> {
    use crate::trace::State;
    let q = crate::simd::oprofile::nqf(om.m);
    let mut exp_k = vec![0.0_f32; q * 4];
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
            exp_k[tr.k[z] - 1] += 1.0;
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
    let xfactor = exp_n * norm + exp_c * norm + exp_j * norm;
    let exp_m: Vec<__m128> = (0..q)
        .map(|stripe| {
            let lanes = [0, 1, 2, 3].map(|lane| exp_k[lane * q + stripe] * norm);
            unsafe { _mm_loadu_ps(lanes.as_ptr()) }
        })
        .collect();

    // Four residues per pass keep four sums in registers, each summed in stripe order as C does.
    let (blocks, rest) = om.rfv[..om.abc_k].as_chunks::<4>();
    assert!(rest.is_empty());
    let mut null2 = Vec::with_capacity(om.abc_k);
    for block in blocks {
        let rows = block.each_ref().map(|row| &row[..q]);
        let sums = unsafe {
            let mut acc = [_mm_setzero_ps(); 4];
            for (stripe, &ev) in exp_m.iter().enumerate() {
                for (sv, row) in acc.iter_mut().zip(rows) {
                    *sv = _mm_add_ps(*sv, _mm_mul_ps(ev, _mm_load_ps(row[stripe].as_ptr())));
                }
            }
            acc.map(|sv| {
                let mut lanes = [0.0_f32; 4];
                _mm_storeu_ps(lanes.as_mut_ptr(), sv);
                lanes
            })
        };
        null2.extend(sums.map(|l| ((l[0] + l[1]) + (l[2] + l[3])) + xfactor));
    }
    null2
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
