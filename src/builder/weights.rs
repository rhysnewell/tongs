use super::*;

/// Henikoff position-based sequence weights, normalized to sum to `nseq`.
///
/// Faithful port of Easel's `esl_msaweight_PB_adv()` (esl_msaweight.c:182) as
/// invoked by the HMMER builder (`relative_weights`, p7_builder.c:809) with the
/// default config. Consensus columns are determined by RF annotation when
/// present and `ignore_rf` is false (i.e. `--hand`); otherwise by the symfrac
/// gap-fraction rule over fragment-aware counts (`consensus_by_all`). Counts
/// are collected with HMMER's fragment-span rule so fragments' external gaps
/// are excluded. The per-residue PB rule `1/(r·ct)`, per-sequence `/rlen`
/// normalization, and final scale-to-`nseq` all match the C exactly.
///
/// The deep-alignment subsampling path (`consensus_by_sample`, nseq > 50000) is
/// reproduced bit-faithfully: it samples `nsamp` sequence indices with Easel's
/// 64-bit Mersenne Twister (`esl_rand64`) + Vitter sequential sampling
/// (`esl_rand64_Deal`), marks fragments by the same span rule, and determines
/// consensus columns from the sample (or rejects when too many fragments are
/// seen, falling back to `consensus_by_all`).
pub fn pb_weights(msa: &Msa, abc: &Alphabet, ignore_rf: bool) -> Vec<f64> {
    let nseq = msa.nseq;
    let alen = msa.alen;
    let k = abc.k;
    let kp = abc.kp;

    // Contract: single-sequence MSA gets weight 1.0 (esl_msaweight.c:199).
    if nseq == 1 {
        return vec![1.0];
    }
    if nseq == 0 || alen == 0 {
        return vec![1.0; nseq];
    }

    let ax = msa.digitize(abc);

    // Count matrix ct[apos=0..=alen][a=0..Kp-1]; apos is 1-based, [0][] unused.
    let mut ct = vec![vec![0i32; kp]; alen + 1];
    // Consensus column indices (1-based), in increasing order.
    let mut conscols: Vec<usize> = Vec::with_capacity(alen);

    // Determine consensus columns early if we can. RF takes priority; else, on
    // a deep alignment, sample to determine consensus (esl_msaweight.c:205-207).
    // ncons stays 0 if neither path is used or the sample is rejected.
    match msa.rf.as_ref().filter(|_| !ignore_rf) {
        Some(rf) => consensus_by_rf(rf, abc, alen, &mut conscols),
        None if PB_ALLOW_SAMP && nseq > PB_SAMPTHRESH => {
            consensus_by_sample(&ax, abc, alen, nseq, &mut ct, &mut conscols)
        }
        None => {}
    }

    // Collect count matrix (over all columns, or only consensus columns if
    // already known), excluding fragments' external gaps.
    collect_counts(&ax, abc, alen, nseq, &conscols, &mut ct);

    // If consensus columns weren't determined yet, do it now from <ct>.
    if conscols.is_empty() {
        consensus_by_all(&ct, abc, alen, &mut conscols);
    }

    // If still nothing, that's pathological -- use all columns.
    if conscols.is_empty() {
        conscols.extend(1..=alen);
    }

    let ncons = conscols.len();

    // Count how many different canonical residues are used in each consensus
    // column: r[j] (esl_msaweight.c:224-231).
    let mut r = vec![0i32; ncons];
    for (j, &apos) in conscols.iter().enumerate() {
        for &count in ct[apos].iter().take(k) {
            if count > 0 {
                r[j] += 1;
            }
        }
    }

    // Bump sequence weights using the PB rule (esl_msaweight.c:234-246).
    let mut weights = vec![0.0_f64; nseq];
    for idx in 0..nseq {
        let mut rlen = 0i32;
        for (j, &apos) in conscols.iter().enumerate() {
            let a = ax[idx][apos] as usize;
            if a < k {
                weights[idx] += 1.0 / (r[j] * ct[apos][a]) as f64;
                rlen += 1;
            }
        }
        if rlen > 0 {
            weights[idx] /= rlen as f64;
        }
    }

    // Normalize to sum to 1, then scale to nseq (esl_vec_DNorm + DScale).
    let sum: f64 = weights.iter().sum();
    if sum != 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    } else {
        let uniform = 1.0 / nseq as f64;
        weights.fill(uniform);
    }
    for w in &mut weights {
        *w *= nseq as f64;
    }

    weights
}
fn normalize_weights_to_nseq(weights: &mut [f64], nseq: usize) {
    let sum: f64 = weights.iter().sum();
    if sum > 0.0 {
        let scale = nseq as f64 / sum;
        for weight in weights {
            *weight *= scale;
        }
    } else if nseq > 0 {
        weights.fill(1.0);
    }
}

/// Use MSA-supplied sequence weights exactly as parsed, or 1.0 defaults.
pub fn given_weights(msa: &Msa) -> Vec<f64> {
    msa.weights.clone().unwrap_or_else(|| vec![1.0; msa.nseq])
}

/// BLOSUM relative weights: single-linkage cluster at pairwise identity
/// `max_id`, then divide one unit of weight over each cluster.
pub fn blosum_weights(msa: &Msa, abc: &Alphabet, max_id: f64) -> Vec<f64> {
    let nseq = msa.nseq;
    if nseq <= 1 {
        return vec![1.0; nseq];
    }
    let ax = msa.digitize(abc);
    let mut dsu = DisjointSet::new(nseq);
    for i in 0..nseq {
        for j in i + 1..nseq {
            if pair_identity(abc, &ax[i], &ax[j]) >= max_id {
                dsu.union(i, j);
            }
        }
    }

    let mut cluster_id = vec![usize::MAX; nseq];
    let mut cluster_sizes = Vec::<usize>::new();
    for seq in 0..nseq {
        let root = dsu.find(seq);
        if cluster_id[root] == usize::MAX {
            cluster_id[root] = cluster_sizes.len();
            cluster_sizes.push(0);
        }
        cluster_sizes[cluster_id[root]] += 1;
    }

    let mut weights = vec![0.0; nseq];
    for (seq, weight) in weights.iter_mut().enumerate() {
        let cluster = cluster_id[dsu.find(seq)];
        *weight = 1.0 / cluster_sizes[cluster] as f64;
    }
    normalize_weights_to_nseq(&mut weights, nseq);
    weights
}

/// Number of single-linkage clusters at pairwise identity `max_id`.
///
/// This is the effective-sequence-number calculation used by hmmbuild
/// `--eclust`: any pair with identity >= cutoff links their clusters.
pub fn single_linkage_cluster_count(msa: &Msa, abc: &Alphabet, max_id: f64) -> usize {
    let nseq = msa.nseq;
    if nseq <= 1 {
        return nseq;
    }
    let ax = msa.digitize(abc);
    let mut dsu = DisjointSet::new(nseq);
    for i in 0..nseq {
        for j in i + 1..nseq {
            if pair_identity(abc, &ax[i], &ax[j]) >= max_id {
                dsu.union(i, j);
            }
        }
    }

    let mut nclusters = 0usize;
    for seq in 0..nseq {
        if dsu.find(seq) == seq {
            nclusters += 1;
        }
    }
    nclusters
}

/// Gerstein/Sonnhammer/Chothia tree weights, using UPGMA over fractional
/// pairwise differences, normalized to sum to `nseq`.
pub fn gsc_weights(msa: &Msa, abc: &Alphabet) -> Vec<f64> {
    let nseq = msa.nseq;
    if nseq <= 1 {
        return vec![1.0; nseq];
    }
    let ax = msa.digitize(abc);
    let mut distances = vec![vec![0.0_f64; nseq]; nseq];
    for i in 0..nseq {
        for j in i + 1..nseq {
            let distance = 1.0 - pair_identity(abc, &ax[i], &ax[j]);
            distances[i][j] = distance;
            distances[j][i] = distance;
        }
    }

    let tree = upgma_tree(distances);
    let mut clade_size = vec![0usize; nseq - 1];
    for node in (0..nseq - 1).rev() {
        clade_size[node] = child_clade_size(tree.left[node], &clade_size)
            + child_clade_size(tree.right[node], &clade_size);
    }

    let mut x = vec![0.0_f64; nseq - 1];
    for node in (0..nseq - 1).rev() {
        x[node] = tree.ld[node] + tree.rd[node];
        if tree.left[node] > 0 {
            x[node] += x[tree.left[node] as usize];
        }
        if tree.right[node] > 0 {
            x[node] += x[tree.right[node] as usize];
        }
    }

    let mut weights = vec![0.0_f64; nseq];
    x[0] = 0.0;
    for node in 0..nseq - 1 {
        let mut lw = tree.ld[node];
        if tree.left[node] > 0 {
            lw += x[tree.left[node] as usize];
        }
        let mut rw = tree.rd[node];
        if tree.right[node] > 0 {
            rw += x[tree.right[node] as usize];
        }

        let (lx, rx) = if lw + rw == 0.0 {
            let left_share =
                child_clade_size(tree.left[node], &clade_size) as f64 / clade_size[node] as f64;
            let right_share =
                child_clade_size(tree.right[node], &clade_size) as f64 / clade_size[node] as f64;
            (x[node] * left_share, x[node] * right_share)
        } else {
            (x[node] * lw / (lw + rw), x[node] * rw / (lw + rw))
        };

        assign_gsc_child_weight(tree.left[node], lx + tree.ld[node], &mut x, &mut weights);
        assign_gsc_child_weight(tree.right[node], rx + tree.rd[node], &mut x, &mut weights);
    }
    normalize_weights_to_nseq(&mut weights, nseq);
    weights
}

fn pair_identity(abc: &Alphabet, ax1: &[u8], ax2: &[u8]) -> f64 {
    let mut nid = 0usize;
    let mut len1 = 0usize;
    let mut len2 = 0usize;
    let mut pos = 1usize;
    while pos < ax1.len()
        && pos < ax2.len()
        && ax1[pos] != crate::alphabet::DSQ_SENTINEL
        && ax2[pos] != crate::alphabet::DSQ_SENTINEL
    {
        if abc.is_residue(ax1[pos]) {
            len1 += 1;
        }
        if abc.is_residue(ax2[pos]) {
            len2 += 1;
        }
        if abc.is_residue(ax1[pos]) && abc.is_residue(ax2[pos]) && ax1[pos] == ax2[pos] {
            nid += 1;
        }
        pos += 1;
    }
    let denom = len1.min(len2);
    if denom == 0 {
        0.0
    } else {
        nid as f64 / denom as f64
    }
}

#[derive(Clone)]
pub(super) struct DisjointSet {
    pub(super) parent: Vec<usize>,
    pub(super) rank: Vec<u8>,
}

impl DisjointSet {
    pub(super) fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub(super) fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return;
        }
        if self.rank[root_a] < self.rank[root_b] {
            std::mem::swap(&mut root_a, &mut root_b);
        }
        self.parent[root_b] = root_a;
        if self.rank[root_a] == self.rank[root_b] {
            self.rank[root_a] += 1;
        }
    }
}

pub(super) struct UpgmaTree {
    pub(super) left: Vec<i32>,
    pub(super) right: Vec<i32>,
    pub(super) ld: Vec<f64>,
    pub(super) rd: Vec<f64>,
}

fn upgma_tree(mut distances: Vec<Vec<f64>>) -> UpgmaTree {
    let ntaxa = distances.len();
    let mut idx: Vec<i32> = (0..ntaxa).map(|i| -(i as i32)).collect();
    let mut nin = vec![1usize; ntaxa];
    let mut height = vec![0.0_f64; ntaxa - 1];
    let mut tree = UpgmaTree {
        left: vec![0; ntaxa - 1],
        right: vec![0; ntaxa - 1],
        ld: vec![0.0; ntaxa - 1],
        rd: vec![0.0; ntaxa - 1],
    };

    for n_active in (2..=ntaxa).rev() {
        let mut best_i = 0usize;
        let mut best_j = 1usize;
        let mut min_distance = distances[0][1];
        for (row, row_vals) in distances.iter().enumerate().take(n_active) {
            for (col, &distance) in row_vals.iter().enumerate().take(n_active).skip(row + 1) {
                if distance < min_distance {
                    min_distance = distance;
                    best_i = row;
                    best_j = col;
                }
            }
        }

        let node = n_active - 2;
        tree.left[node] = idx[best_i];
        tree.right[node] = idx[best_j];
        height[node] = min_distance / 2.0;
        tree.ld[node] = height[node];
        tree.rd[node] = height[node];
        if idx[best_i] > 0 {
            tree.ld[node] = (tree.ld[node] - height[idx[best_i] as usize]).max(0.0);
        }
        if idx[best_j] > 0 {
            tree.rd[node] = (tree.rd[node] - height[idx[best_j] as usize]).max(0.0);
        }

        if best_j != n_active - 1 {
            swap_distance_index(&mut distances, best_j, n_active - 1, n_active);
            idx.swap(best_j, n_active - 1);
            nin.swap(best_j, n_active - 1);
        }
        if best_i != n_active - 2 {
            swap_distance_index(&mut distances, best_i, n_active - 2, n_active);
            idx.swap(best_i, n_active - 2);
            nin.swap(best_i, n_active - 2);
        }

        let i = n_active - 2;
        let j = n_active - 1;
        let mut col = 0usize;
        while col < n_active {
            distances[i][col] = (nin[i] as f64 * distances[i][col]
                + nin[j] as f64 * distances[j][col])
                / (nin[i] + nin[j]) as f64;
            distances[col][i] = distances[i][col];
            col += 1;
        }
        nin[i] += nin[j];
        idx[i] = node as i32;
    }

    tree
}

fn swap_distance_index(distances: &mut [Vec<f64>], a: usize, b: usize, n_active: usize) {
    for row in distances.iter_mut().take(n_active) {
        row.swap(a, b);
    }
    distances.swap(a, b);
}

fn child_clade_size(child: i32, clade_size: &[usize]) -> usize {
    if child > 0 {
        clade_size[child as usize]
    } else {
        1
    }
}

fn assign_gsc_child_weight(child: i32, value: f64, x: &mut [f64], weights: &mut [f64]) {
    if child > 0 {
        x[child as usize] = value;
    } else {
        weights[(-child) as usize] = value;
    }
}
