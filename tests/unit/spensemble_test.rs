use super::*;

fn all_pairs(segments: &[SegmentPair], params: &ClusterParams) -> (Vec<usize>, usize) {
    let n = segments.len();
    let mut available: Vec<usize> = (0..n).rev().collect();
    let mut connected = Vec::with_capacity(n);
    let mut assignments = vec![0usize; n];
    let mut nclusters = 0usize;
    while let Some(v) = available.pop() {
        connected.push(v);
        while let Some(v) = connected.pop() {
            assignments[v] = nclusters;
            let mut idx = available.len();
            while idx > 0 {
                idx -= 1;
                if segments_overlap(&segments[v], &segments[available[idx]], params) {
                    let w = available[idx];
                    let last = available.pop().unwrap();
                    if idx < available.len() {
                        available[idx] = last;
                    }
                    connected.push(w);
                }
            }
        }
        nclusters += 1;
    }
    (assignments, nclusters)
}

fn draw(state: &mut u64, below: usize) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state % below as u64) as usize
}

// Envelopes sort stably by start, so cluster numbers decide ties. Any drift from the scan's
// numbering moves a row.
#[test]
fn diagonal_buckets_number_clusters_as_the_all_pairs_scan() {
    let params = ClusterParams::default();
    let mut state = 0x9e37_79b9_7f4a_7c15;
    for case in 0..3000 {
        let n = draw(&mut state, 400) + 1;
        let domains = draw(&mut state, 6) + 1;
        let segments = (0..n)
            .map(|at| {
                let i = 1 + 60 * draw(&mut state, domains) + draw(&mut state, 12);
                let k = 1 + draw(&mut state, 10);
                let span = 20 + draw(&mut state, 30);
                SegmentPair {
                    i,
                    j: i + span,
                    k,
                    m: k + span - draw(&mut state, 8),
                    trace_idx: at / 2,
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(
            single_linkage_assignments(&segments, &params),
            all_pairs(&segments, &params),
            "case {case}"
        );
    }
}
