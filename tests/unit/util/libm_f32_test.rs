use super::*;

// Bits glibc 2.39 returns on x86_64. Both expf inputs come out one ulp low unless the range
// reduction is fused the way the -mfma build fuses it.
#[test]
fn matches_glibc_where_rounding_order_shows() {
    for (x, bits) in [(0x4202_422f_u32, 0x56fc_9f1c_u32), (0xc27c_65d9, 0x11fa_2993)] {
        assert_eq!(expf(f32::from_bits(x)).to_bits(), bits, "expf({:e})", f32::from_bits(x));
    }
    assert_eq!(logf(1.0), 0.0);
    assert_eq!(logf(0.0), f32::NEG_INFINITY);
    assert!(logf(-1.0).is_nan());
    assert_eq!(expf(f32::NEG_INFINITY), 0.0);
}
