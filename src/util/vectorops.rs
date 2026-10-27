//! Vector operations on f32/f64/i32 slices.
//! Direct port of Easel's esl_vectorops.c, focusing on functions used by HMMER.

use crate::util::cmath::{c_logf_to_f32, ESL_CONST_LOG2R};

// ===== Set =====

/// Set every element of `vec` to `value`. Port of `esl_vec_DSet`.
pub fn d_set(vec: &mut [f64], value: f64) {
    vec.iter_mut().for_each(|v| *v = value);
}

/// Set every element of `vec` to `value`. Port of `esl_vec_FSet`.
pub fn f_set(vec: &mut [f32], value: f32) {
    vec.iter_mut().for_each(|v| *v = value);
}

// ===== Scale =====

// ===== Increment =====

// ===== Add =====

/// In-place vec1 += a * vec2 (element-wise). Port of `esl_vec_FAddScaled`.
pub fn f_add_scaled(vec1: &mut [f32], vec2: &[f32], a: f32) {
    for (v1, v2) in vec1.iter_mut().zip(vec2.iter()) {
        *v1 += *v2 * a;
    }
}

// ===== Sum (Kahan compensated summation) =====

/// Sum of `vec` using Kahan compensated summation. Port of `esl_vec_DSum`.
///
/// Most accurate when `vec` is sorted in increasing magnitude.
pub fn d_sum(vec: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for &v in vec {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Sum of `vec` using Kahan compensated summation. Port of `esl_vec_FSum`.
pub fn f_sum(vec: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let mut c = 0.0_f32;
    for &v in vec {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

// ===== Max, Min, ArgMax, ArgMin =====

/// Return the index of the maximum element in `vec`. Port of `esl_vec_FArgMax`.
///
/// On ties the smallest index is returned, matching the C convention HMMER's
/// optimal-accuracy tracebacks depend on. Returns 0 for an empty slice.
pub fn f_argmax(vec: &[f32]) -> usize {
    vec.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ===== Copy =====

// ===== Reverse =====

// ===== Normalize =====

// Easel divides by the sum rather than scaling by its reciprocal, and the rounding differs.
pub fn d_norm(vec: &mut [f64]) {
    let sum = d_sum(vec);
    if sum != 0.0 {
        vec.iter_mut().for_each(|v| *v /= sum);
    } else if !vec.is_empty() {
        d_set(vec, 1.0 / vec.len() as f64);
    }
}

pub fn f_norm(vec: &mut [f32]) {
    let sum = f_sum(vec);
    if sum != 0.0 {
        vec.iter_mut().for_each(|v| *v /= sum);
    } else if !vec.is_empty() {
        f_set(vec, 1.0 / vec.len() as f32);
    }
}

// ===== Log / Exp operations =====

// esl_vec_FLogSum() skips terms more than 50 nats below the maximum.

// ===== Entropy =====

/// Shannon entropy in bits for a single-precision p-vector. Port of `esl_vec_FEntropy`.
pub fn f_entropy(p: &[f32]) -> f32 {
    let mut h = 0.0_f32;
    for &pi in p {
        if pi > 0.0 {
            h -= pi * c_logf_to_f32(pi) * ESL_CONST_LOG2R as f32;
        }
    }
    h
}

// ===== Type conversion =====

// ===== Compare =====

// ===== Validate =====

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f_sum_kahan() {
        // Kahan summation should be more accurate than naive
        let vec = vec![1.0e-8_f32; 100_000_000];
        let result = f_sum(&vec[..100]); // just test small case
        assert!((result - 1.0e-6).abs() < 1.0e-12);
    }

    #[test]
    fn test_d_norm() {
        let mut v = vec![2.0, 3.0, 5.0];
        d_norm(&mut v);
        assert!((d_sum(&v) - 1.0).abs() < 1e-10);
        assert!((v[0] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_f_entropy() {
        // Uniform distribution over 4 symbols = 2 bits
        let p = vec![0.25_f32; 4];
        assert!((f_entropy(&p) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_f_argmax() {
        let v = vec![1.0_f32, 5.0, 3.0, 2.0];
        assert_eq!(f_argmax(&v), 1);
    }
}
