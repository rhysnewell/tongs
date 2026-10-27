use super::*;

fn win(n: usize, length: usize, complement: bool, id: i32) -> HmmWindow {
    HmmWindow {
        n,
        k: length,
        length,
        score: 0.0,
        target_len: 1_000_000,
        complement,
        id,
        fm_n: -1,
        fm_start: -1,
        fm_bwt_len: 0,
    }
}

// LOW-1: C's do/while always emits at least one tail per oversized window.
#[test]
fn split_long_windows_emits_tail_at_exact_shift_multiple() {
    let max = 80000;
    let overlap = 40000; // shift = 40000
    let w = win(1, max + (max - overlap), false, 0); // 120000
    let out = split_long_windows(&[w], max, overlap);
    assert!(
        out.len() >= 2,
        "expected head plus a tail, got {}",
        out.len()
    );
    assert_eq!(out[0].length, max);
    // new_len: 120000 - 40000 = 80000 (<=max) => final tail length 80000.
    assert_eq!(out[1].length, 80000);
}

#[test]
fn split_long_windows_walks_multiple_tails() {
    let max = 80000;
    let overlap = 40000; // shift 40000
    let w = win(1, 120001, false, 0);
    let out = split_long_windows(&[w], max, overlap);
    // head=80000; iter1 new_len=80001 (>max) push 80000; iter2 new_len=40001
    // (<=max) push 40001, stop.
    assert_eq!(out[0].length, 80000);
    assert_eq!(out[1].length, 80000);
    assert_eq!(out[2].length, 40001);
    assert_eq!(out.len(), 3);

    let w2 = win(1, max + 2 * 40000, false, 0); // 160000
    let out2 = split_long_windows(&[w2], max, overlap);
    // head 80000; iter1 new_len 120000 push 80000; iter2 new_len 80000 push
    // 80000 stop.
    assert_eq!(out2.len(), 3);
    assert!(out2.iter().all(|w| w.length <= max));
}

// MED-2: merges require BOTH complementarity AND id to match.
#[test]
fn merge_respects_id_guard() {
    let mut ws = vec![win(100, 100, false, 0), win(110, 100, false, 1)];
    merge_windows_impl(&mut ws, 0.0);
    assert_eq!(
        ws.len(),
        2,
        "windows from different FM segments must not merge"
    );

    let mut ws2 = vec![win(100, 100, false, 7), win(110, 100, false, 7)];
    merge_windows_impl(&mut ws2, 0.0);
    assert_eq!(ws2.len(), 1, "same-segment overlapping windows must merge");

    let mut ws3 = vec![win(100, 100, false, 3), win(110, 100, true, 3)];
    merge_windows_impl(&mut ws3, 0.0);
    assert_eq!(ws3.len(), 2, "opposite-strand windows must not merge");
}

// MED-2: the complement extension flip only fires in the forward
// concatenated-FM frame (fm_n >= 0); RC-local windows (fm_n=-1) do not flip.
#[test]
fn complement_flip_gated_on_fm_frame() {
    let mut rc_local = win(10, 5, true, 0);
    rc_local.fm_n = -1;
    assert!(!rc_local.needs_complement_extension_flip());

    let mut fm_frame = win(10, 5, true, 0);
    fm_frame.fm_n = 0;
    assert!(fm_frame.needs_complement_extension_flip());

    let mut fwd = win(10, 5, false, 0);
    fwd.fm_n = 0;
    assert!(!fwd.needs_complement_extension_flip());
}

#[test]
fn p7_pli_extend_uses_window_target_len_like_c() {
    let mut w = win(40, 5, false, 0);
    w.target_len = 50;
    let mut windows = vec![w];
    let prefix = vec![0.0; 8];
    let suffix = vec![0.0; 8];

    p7_pli_extend_and_merge_windows(&mut windows, 10, 10, 0.0, &prefix, &suffix);

    assert_eq!(windows[0].n, 39);
    assert_eq!(windows[0].length, 8);
}
