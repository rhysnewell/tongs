use super::*;

/// 5-bit pack a digital sequence `dsq[1..=n]` (1-based, sentinel at 0). Returns
/// the packets. Port of `dsqdata_pack5()`.
pub(super) fn pack5(dsq: &[Dsq], n: usize) -> Vec<u32> {
    let mut psq: Vec<u32> = Vec::new();
    let mut r = 1usize; // position in dsq (1-based)
    while r <= n {
        let mut v = FIVEBIT_BIT;
        let mut b: i32 = 25;
        while b >= 0 && r <= n {
            v |= (dsq[r] as u32) << b;
            r += 1;
            b -= 5;
        }
        while b >= 0 {
            v |= 31u32 << b;
            b -= 5;
        }
        if r > n {
            v |= EOD_BIT;
        }
        psq.push(v);
    }
    // n == 0: a single empty EOD sentinel packet (all bits set).
    if psq.is_empty() {
        psq.push(!0u32);
    }
    psq
}

/// 2-bit (+ 5-bit for noncanonicals) pack a digital nucleic sequence.
/// Port of `dsqdata_pack2()`.
pub(super) fn pack2(dsq: &[Dsq], n: usize) -> Vec<u32> {
    let mut psq: Vec<u32> = Vec::new();
    let mut d = 0usize; // position of next degenerate residue (1..=n), n+1 if none
    let mut r = 1usize; // position in dsq (1-based)
    while r <= n {
        // Slide the "next degenerate residue" detector.
        if d < r {
            d = r;
            while d <= n {
                if dsq[d] > 3 {
                    break;
                }
                d += 1;
            }
        }

        let v;
        // Can we 2-bit pack the next 15 residues r..r+14?
        if n - r + 1 >= 15 && d > r + 14 {
            let mut vv = 0u32;
            let mut b: i32 = 28;
            while b >= 0 {
                vv |= (dsq[r] as u32) << b;
                r += 1;
                b -= 2;
            }
            v = vv;
        } else {
            let mut vv = FIVEBIT_BIT;
            let mut b: i32 = 25;
            while b >= 0 && r <= n {
                vv |= (dsq[r] as u32) << b;
                r += 1;
                b -= 5;
            }
            while b >= 0 {
                vv |= 31u32 << b;
                b -= 5;
            }
            v = vv;
        }

        let v = if r > n { v | EOD_BIT } else { v };
        psq.push(v);
    }
    if psq.is_empty() {
        psq.push(!0u32);
    }
    psq
}

/// Unpack one 5-bit-encoded sequence starting at `psq[start..]`. Appends
/// residues to `out` (without sentinels). Returns (L, packets_consumed).
/// Port of `dsqdata_unpack5()`.
pub(super) fn unpack5(
    psq: &[u32],
    start: usize,
    out: &mut Vec<Dsq>,
) -> HmmerResult<(usize, usize)> {
    let mut pos = start;
    let mut l = 0usize;
    let mut v = *psq
        .get(pos)
        .ok_or_else(|| HmmerError::Format("dsqdata: truncated packet stream".into()))?;
    pos += 1;

    while !is_eod(v) {
        for b in [25, 20, 15, 10, 5, 0] {
            out.push(((v >> b) & 31) as Dsq);
            l += 1;
        }
        v = *psq
            .get(pos)
            .ok_or_else(|| HmmerError::Format("dsqdata: truncated packet stream".into()))?;
        pos += 1;
    }
    // EOD packet, possibly partial. Stop at first 0x1f sentinel slot.
    let mut b: i32 = 25;
    while b >= 0 && ((v >> b) & 31) != 31 {
        out.push(((v >> b) & 31) as Dsq);
        l += 1;
        b -= 5;
    }
    Ok((l, pos - start))
}

/// Unpack one 2-bit (mixed 5-bit) encoded sequence. Port of `dsqdata_unpack2()`.
pub(super) fn unpack2(
    psq: &[u32],
    start: usize,
    out: &mut Vec<Dsq>,
) -> HmmerResult<(usize, usize)> {
    let mut pos = start;
    let mut l = 0usize;
    let mut v = *psq
        .get(pos)
        .ok_or_else(|| HmmerError::Format("dsqdata: truncated packet stream".into()))?;
    pos += 1;

    while !is_eod(v) {
        if is_5bit(v) {
            for b in [25, 20, 15, 10, 5, 0] {
                out.push(((v >> b) & 31) as Dsq);
                l += 1;
            }
        } else {
            for b in [28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0] {
                out.push(((v >> b) & 3) as Dsq);
                l += 1;
            }
        }
        v = *psq
            .get(pos)
            .ok_or_else(|| HmmerError::Format("dsqdata: truncated packet stream".into()))?;
        pos += 1;
    }
    // EOD packet. 2-bit EOD is full; 5-bit EOD may be partial.
    if is_5bit(v) {
        let mut b: i32 = 25;
        while b >= 0 && ((v >> b) & 31) != 31 {
            out.push(((v >> b) & 31) as Dsq);
            l += 1;
            b -= 5;
        }
    } else {
        for b in [28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0] {
            out.push(((v >> b) & 3) as Dsq);
            l += 1;
        }
    }
    Ok((l, pos - start))
}
