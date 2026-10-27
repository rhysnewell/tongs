//! C math shims used where bit parity with HMMER's C code matters.

/// Easel's `eslCONST_LOG2`.
pub const ESL_CONST_LOG2: f64 = std::f64::consts::LN_2;

/// Easel's `eslCONST_LOG2R`.
pub const ESL_CONST_LOG2R: f64 = std::f64::consts::LOG2_E;

/// Natural log wrapper kept to make parity-sensitive call sites explicit.
#[inline]
pub fn c_log_f64(x: f64) -> f64 {
    x.ln()
}

/// `log(double)`, then cast to `f32`, matching C code that calls `log()`.
#[inline]
pub fn c_log_to_f32(x: f64) -> f32 {
    c_log_f64(x) as f32
}

/// Promote `f32` to `f64`, call C-style `log(double)`, then truncate to `f32`.
#[inline]
pub fn c_log_f32_to_f32(x: f32) -> f32 {
    c_log_to_f32(x as f64)
}

/// `f32` natural log wrapper kept to make parity-sensitive call sites explicit.
#[inline]
pub fn c_logf_to_f32(x: f32) -> f32 {
    super::libm_f32::logf(x)
}

/// Exponential wrapper kept to make parity-sensitive call sites explicit.
#[inline]
pub fn c_exp_f64(x: f64) -> f64 {
    x.exp()
}

/// `exp(double)`, then cast to `f32`, matching C code that calls `exp()`.
#[inline]
pub fn c_exp_to_f32(x: f64) -> f32 {
    c_exp_f64(x) as f32
}

/// `f32` exponential wrapper kept to make parity-sensitive call sites explicit.
#[inline]
pub fn c_expf_to_f32(x: f32) -> f32 {
    super::libm_f32::expf(x)
}

/// Power wrapper kept to make parity-sensitive call sites explicit.
#[inline]
pub fn c_pow_f64(x: f64, y: f64) -> f64 {
    x.powf(y)
}

/// Square-root wrapper kept to make parity-sensitive call sites explicit.
#[inline]
pub fn c_sqrt_f64(x: f64) -> f64 {
    x.sqrt()
}
