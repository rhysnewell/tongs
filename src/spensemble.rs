//! Segment pair ensemble clustering for multidomain regions.
//! Simplified port of p7_spensemble.c.

/// A segment pair from a stochastic trace.
#[derive(Debug, Clone)]
pub struct SegmentPair {
    pub i: usize,         // sequence start (1-based)
    pub j: usize,         // sequence end
    pub k: usize,         // HMM start (1-based)
    pub m: usize,         // HMM end
    pub trace_idx: usize, // which trace this came from
}

/// Result of clustering: a significant domain envelope.
#[derive(Debug, Clone)]
pub struct DomainEnvelope {
    pub ienv: usize,
    pub jenv: usize,
    pub kenv: usize,
    pub menv: usize,
    pub posterior: f32, // fraction of traces that contain this domain
}

/// Clustering parameters.
pub struct ClusterParams {
    pub min_overlap: f32,    // minimum overlap fraction (default 0.8)
    pub min_posterior: f32,  // minimum posterior for significance (default 0.25)
    pub min_endpointp: f32,  // minimum endpoint probability (default 0.02)
    pub max_diagdiff: usize, // maximum diagonal difference (default 4)
    pub of_smaller: bool,    // use smaller segment as overlap denominator
}

impl Default for ClusterParams {
    /// HMMER 3 default clustering parameters: 80% overlap, 25% posterior cutoff,
    /// 2% endpoint probability, max 4 diagonals difference, denominator = smaller segment.
    fn default() -> Self {
        ClusterParams {
            min_overlap: 0.8,
            min_posterior: 0.25,
            min_endpointp: 0.02,
            max_diagdiff: 4,
            of_smaller: true,
        }
    }
}

/// Cluster a segment-pair ensemble and define domain envelopes.
/// Port of `p7_spensemble_Cluster()`: single-linkage cluster segment pairs by
/// sequence/HMM overlap and diagonal proximity, keep clusters with posterior
/// >= `min_posterior`, then pick consensus i,j,k,m endpoints whose count meets
/// > the `min_endpointp` threshold. `ntraces` is the total number of stochastic
/// > traces the ensemble came from. Also drops envelopes dominated (>=80% covered
/// > in sequence coords) by a higher-posterior neighbor, matching the post-cluster
/// > pruning in `region_trace_ensemble()`.
pub fn cluster(
    segments: &[SegmentPair],
    ntraces: usize,
    params: &ClusterParams,
) -> Vec<DomainEnvelope> {
    if segments.is_empty() || ntraces == 0 {
        return Vec::new();
    }

    let (assignments, nclusters) = single_linkage_assignments(segments, params);
    let mut clusters = vec![Vec::new(); nclusters];
    for (idx, &cluster_idx) in assignments.iter().enumerate() {
        clusters[cluster_idx].push(idx);
    }

    let mut envelopes = Vec::new();
    for members in clusters {
        let mut trace_count = 0usize;
        let mut last_trace = None;
        for &idx in &members {
            if last_trace != Some(segments[idx].trace_idx) {
                trace_count += 1;
            }
            last_trace = Some(segments[idx].trace_idx);
        }
        let posterior = trace_count as f32 / ntraces as f32;

        if posterior < params.min_posterior {
            continue;
        }

        let mut imin = usize::MAX;
        let mut imax = 0usize;
        let mut jmin = usize::MAX;
        let mut jmax = 0usize;
        let mut kmin = usize::MAX;
        let mut kmax = 0usize;
        let mut mmin = usize::MAX;
        let mut mmax = 0usize;

        for &idx in &members {
            imin = imin.min(segments[idx].i);
            imax = imax.max(segments[idx].i);
            jmin = jmin.min(segments[idx].j);
            jmax = jmax.max(segments[idx].j);
            kmin = kmin.min(segments[idx].k);
            kmax = kmax.max(segments[idx].k);
            mmin = mmin.min(segments[idx].m);
            mmax = mmax.max(segments[idx].m);
        }

        let epc_threshold = ((trace_count as f32) * params.min_endpointp).ceil() as usize;
        let epc_threshold = epc_threshold.max(1);

        let best_i = left_endpoint(&members, segments, |s| s.i, imin, imax, epc_threshold);
        let best_k = left_endpoint(&members, segments, |s| s.k, kmin, kmax, epc_threshold);
        let best_j = right_endpoint(&members, segments, |s| s.j, jmin, jmax, epc_threshold);
        let best_m = right_endpoint(&members, segments, |s| s.m, mmin, mmax, epc_threshold);

        if best_i > best_j || best_k > best_m {
            continue;
        }

        envelopes.push(DomainEnvelope {
            ienv: best_i,
            jenv: best_j,
            kenv: best_k,
            menv: best_m,
            posterior,
        });
    }

    // Sort by sequence position
    envelopes.sort_by_key(|e| e.ienv);

    // Remove dominated domains relative to sequence coords, matching
    // region_trace_ensemble() after p7_spensemble_Cluster().
    let mut dominated = vec![false; envelopes.len()];
    for d in 0..envelopes.len() {
        for d2 in (d + 1)..envelopes.len() {
            let nov = envelopes[d].jenv.min(envelopes[d2].jenv) as isize
                - envelopes[d].ienv.max(envelopes[d2].ienv) as isize
                + 1;
            if nov == 0 {
                break;
            }
            let n = (envelopes[d].jenv - envelopes[d].ienv + 1)
                .min(envelopes[d2].jenv - envelopes[d2].ienv + 1);
            if (nov as f32) / (n as f32) >= 0.8 {
                if envelopes[d].posterior > envelopes[d2].posterior {
                    dominated[d2] = true;
                } else {
                    dominated[d] = true;
                }
            }
        }
    }

    envelopes
        .into_iter()
        .enumerate()
        .filter_map(|(idx, env)| if dominated[idx] { None } else { Some(env) })
        .collect()
}

// Segments link only within max_diagdiff of a start or an end diagonal, so each popped segment
// tests its diagonal neighbours rather than every one left. Links are taken in the order the
// all-pairs scan finds them, from the top of the available list down, so cluster numbers match it.
fn single_linkage_assignments(
    segments: &[SegmentPair],
    params: &ClusterParams,
) -> (Vec<usize>, usize) {
    let n = segments.len();
    let mut available: Vec<usize> = (0..n).rev().collect();
    let mut position: Vec<usize> = (0..n).map(|v| n - 1 - v).collect();
    let mut starts = Diagonals::new(segments.iter().map(|s| s.i as i64 - s.k as i64));
    let mut ends = Diagonals::new(segments.iter().map(|s| s.j as i64 - s.m as i64));
    let reach = params.max_diagdiff as i64;
    let mut connected = Vec::with_capacity(n);
    let mut near = Vec::new();
    let mut assignments = vec![0usize; n];
    let mut nclusters = 0usize;

    while let Some(v) = available.pop() {
        starts.remove(v);
        ends.remove(v);
        connected.push(v);
        while let Some(v) = connected.pop() {
            assignments[v] = nclusters;
            near.clear();
            starts.near(v, reach, &mut near);
            ends.near(v, reach, &mut near);
            near.sort_unstable_by_key(|&w| std::cmp::Reverse(position[w]));
            near.dedup();
            for &w in &near {
                if !segments_overlap(&segments[v], &segments[w], params) {
                    continue;
                }
                let last = available.pop().expect("available is non-empty");
                if position[w] < available.len() {
                    available[position[w]] = last;
                    position[last] = position[w];
                }
                starts.remove(w);
                ends.remove(w);
                connected.push(w);
            }
        }
        nclusters += 1;
    }

    (assignments, nclusters)
}

// Segments bucketed by one diagonal, each bucket a run of `flat` that shrinks as segments leave.
struct Diagonals {
    first: i64,
    of: Vec<i64>,
    start: Vec<usize>,
    len: Vec<usize>,
    flat: Vec<usize>,
    slot: Vec<usize>,
}

impl Diagonals {
    fn new(diagonals: impl Iterator<Item = i64>) -> Self {
        let of = diagonals.collect::<Vec<_>>();
        let first = of.iter().copied().min().unwrap_or(0);
        let span = of.iter().map(|&d| (d - first) as usize + 1).max().unwrap_or(0);
        let mut len = vec![0usize; span];
        for &d in &of {
            len[(d - first) as usize] += 1;
        }
        let mut start = vec![0usize; span];
        for b in 1..span {
            start[b] = start[b - 1] + len[b - 1];
        }
        let mut fill = start.clone();
        let mut flat = vec![0usize; of.len()];
        let mut slot = vec![0usize; of.len()];
        for (id, &d) in of.iter().enumerate() {
            let b = (d - first) as usize;
            flat[fill[b]] = id;
            slot[id] = fill[b];
            fill[b] += 1;
        }
        Diagonals {
            first,
            of,
            start,
            len,
            flat,
            slot,
        }
    }

    fn remove(&mut self, id: usize) {
        let b = (self.of[id] - self.first) as usize;
        self.len[b] -= 1;
        let last = self.start[b] + self.len[b];
        let moved = self.flat[last];
        self.flat[self.slot[id]] = moved;
        self.slot[moved] = self.slot[id];
    }

    fn near(&self, id: usize, reach: i64, out: &mut Vec<usize>) {
        let low = (self.of[id] - reach - self.first).max(0) as usize;
        let high = (self.of[id] + reach - self.first).min(self.len.len() as i64 - 1) as usize;
        for b in low..=high {
            out.extend_from_slice(&self.flat[self.start[b]..self.start[b] + self.len[b]]);
        }
    }
}

/// Single-linkage edge predicate: sufficient overlap in both seq and HMM coords,
/// and either start or end diagonals within `max_diagdiff`. Mirrors C's
/// `link_spsamples()` in `p7_spensemble.c`.
fn segments_overlap(a: &SegmentPair, b: &SegmentPair, params: &ClusterParams) -> bool {
    // Sequence overlap
    let seq_ovl = overlap_fraction(a.i, a.j, b.i, b.j, true, params.of_smaller);
    if seq_ovl < params.min_overlap {
        return false;
    }

    // HMM overlap
    let hmm_ovl = overlap_fraction(a.k, a.m, b.k, b.m, false, params.of_smaller);
    if hmm_ovl < params.min_overlap {
        return false;
    }

    // Diagonal proximity
    let diag_a_start = (a.i as i64) - (a.k as i64);
    let diag_b_start = (b.i as i64) - (b.k as i64);
    let diag_a_end = (a.j as i64) - (a.m as i64);
    let diag_b_end = (b.j as i64) - (b.m as i64);

    let start_diff = (diag_a_start - diag_b_start).unsigned_abs() as usize;
    let end_diff = (diag_a_end - diag_b_end).unsigned_abs() as usize;

    start_diff <= params.max_diagdiff || end_diff <= params.max_diagdiff
}

/// Overlap fraction of two closed intervals, normalized by either the smaller
/// or larger length (set by `of_smaller`). Returns 0.0 if they don't overlap.
fn overlap_fraction(
    a_start: usize,
    a_end: usize,
    b_start: usize,
    b_end: usize,
    include_endpoint: bool,
    of_smaller: bool,
) -> f32 {
    let endpoint = if include_endpoint { 1isize } else { 0isize };
    let overlap_len = a_end.min(b_end) as isize - a_start.max(b_start) as isize + endpoint;
    if overlap_len <= 0 {
        return 0.0;
    }
    let a_len = a_end - a_start + 1;
    let b_len = b_end - b_start + 1;
    let denom = if of_smaller {
        a_len.min(b_len)
    } else {
        a_len.max(b_len)
    };
    overlap_len as f32 / denom as f32
}

/// Pick the leftmost coordinate whose member count meets `threshold` (consensus
/// start endpoint). Falls back to the most-frequent coord if none qualifies.
fn left_endpoint<F>(
    members: &[usize],
    segments: &[SegmentPair],
    coord: F,
    min_coord: usize,
    max_coord: usize,
    threshold: usize,
) -> usize
where
    F: Fn(&SegmentPair) -> usize,
{
    let mut counts = vec![0usize; max_coord - min_coord + 1];
    for &idx in members {
        counts[coord(&segments[idx]) - min_coord] += 1;
    }
    for (offset, &count) in counts.iter().enumerate() {
        if count >= threshold {
            return min_coord + offset;
        }
    }
    min_coord + argmax_first(&counts)
}

/// Pick the rightmost coordinate whose member count meets `threshold` (consensus
/// end endpoint). Falls back to the most-frequent coord if none qualifies.
fn right_endpoint<F>(
    members: &[usize],
    segments: &[SegmentPair],
    coord: F,
    min_coord: usize,
    max_coord: usize,
    threshold: usize,
) -> usize
where
    F: Fn(&SegmentPair) -> usize,
{
    let mut counts = vec![0usize; max_coord - min_coord + 1];
    for &idx in members {
        counts[coord(&segments[idx]) - min_coord] += 1;
    }
    for offset in (0..counts.len()).rev() {
        if counts[offset] >= threshold {
            return min_coord + offset;
        }
    }
    min_coord + argmax_first(&counts)
}

/// Index of the maximum element in `values`, breaking ties toward the lowest index.
fn argmax_first(values: &[usize]) -> usize {
    let mut best = 0usize;
    let mut best_val = values[0];
    for (idx, &val) in values.iter().enumerate().skip(1) {
        if val > best_val {
            best = idx;
            best_val = val;
        }
    }
    best
}

#[cfg(test)]
#[path = "../tests/unit/spensemble_test.rs"]
mod linkage_test;

#[cfg(test)]
mod tests {
    use super::*;

    /// Three near-identical segments should cluster into one envelope.
    #[test]
    fn test_single_domain() {
        let segments = vec![
            SegmentPair {
                i: 10,
                j: 50,
                k: 1,
                m: 40,
                trace_idx: 0,
            },
            SegmentPair {
                i: 11,
                j: 49,
                k: 1,
                m: 40,
                trace_idx: 1,
            },
            SegmentPair {
                i: 10,
                j: 51,
                k: 1,
                m: 40,
                trace_idx: 2,
            },
        ];
        let params = ClusterParams::default();
        let envs = cluster(&segments, 10, &params);
        assert_eq!(envs.len(), 1);
        assert_eq!(envs[0].ienv, 10);
        assert!(envs[0].posterior >= 0.25);
    }

    /// Two widely separated segment groups should yield two distinct envelopes.
    #[test]
    fn test_two_domains() {
        let segments = vec![
            // Domain 1: positions 10-50
            SegmentPair {
                i: 10,
                j: 50,
                k: 1,
                m: 40,
                trace_idx: 0,
            },
            SegmentPair {
                i: 11,
                j: 49,
                k: 1,
                m: 40,
                trace_idx: 1,
            },
            SegmentPair {
                i: 10,
                j: 50,
                k: 1,
                m: 40,
                trace_idx: 2,
            },
            // Domain 2: positions 100-140
            SegmentPair {
                i: 100,
                j: 140,
                k: 1,
                m: 40,
                trace_idx: 0,
            },
            SegmentPair {
                i: 101,
                j: 139,
                k: 1,
                m: 40,
                trace_idx: 1,
            },
            SegmentPair {
                i: 100,
                j: 140,
                k: 1,
                m: 40,
                trace_idx: 2,
            },
        ];
        let params = ClusterParams::default();
        let envs = cluster(&segments, 10, &params);
        assert_eq!(envs.len(), 2);
        assert!(envs[0].ienv <= 11);
        assert!(envs[1].ienv >= 100);
    }
}
