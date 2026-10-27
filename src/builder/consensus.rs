use super::*;

/// Use RF annotation to define consensus columns (esl_msaweight.c:271).
/// A column is consensus unless its RF character maps to the gap symbol
/// (`esl_abc_CIsGap`). `conscols` is filled with 1-based indices.
pub(super) fn consensus_by_rf(rf: &[u8], abc: &Alphabet, alen: usize, conscols: &mut Vec<usize>) {
    let gap = abc.gap_code();
    for apos in 1..=alen {
        let c = rf.get(apos - 1).copied().unwrap_or(b' ');
        if abc.digitize_symbol(c) == gap {
            continue;
        }
        conscols.push(apos);
    }
}

/// Use counts from all sequences to determine consensus (esl_msaweight.c:400).
/// A column is consensus if its gap fraction `ct[K]/tot < symfrac`, where
/// `tot` sums symbol codes `0..Kp-2` (residues + gaps, incl. degeneracies,
/// excl. nonresidue/missing). Float arithmetic mirrors the C cast order.
pub(super) fn consensus_by_all(
    ct: &[Vec<i32>],
    abc: &Alphabet,
    alen: usize,
    conscols: &mut Vec<usize>,
) {
    let k = abc.k;
    let kp = abc.kp;
    for (apos, row) in ct.iter().enumerate().take(alen + 1).skip(1) {
        let tot: i32 = row.iter().take(kp - 2).sum();
        if (row[k] as f32 / tot as f32) < PB_SYMFRAC {
            conscols.push(apos);
        }
    }
}

/// Collect the observed symbol-count matrix `ct[apos][a]`, applying HMMER's
/// fragment-span rule so fragments' external (terminal) gaps are not counted
/// (esl_msaweight.c:434). A sequence is a fragment if its aligned span
/// `rpos-lpos+1 < ceil(fragthresh*alen)`; full-length sequences count columns
/// `1..=alen`, fragments only `lpos..=rpos`. If `conscols` is non-empty, only
/// those columns are counted (a pure optimization that matches C).
pub(super) fn collect_counts(
    ax: &[Vec<u8>],
    abc: &Alphabet,
    alen: usize,
    nseq: usize,
    conscols: &[usize],
    ct: &mut [Vec<i32>],
) {
    // C re-zeros the whole matrix here (esl_mat_ISet, esl_msaweight.c:443), so
    // any counts left over from consensus_by_sample are discarded.
    for row in ct.iter_mut() {
        row.iter_mut().for_each(|c| *c = 0);
    }

    let minspan = (PB_FRAGTHRESH * alen as f32).ceil() as i64;

    let alen_i = alen as i64;
    for ax_seq in ax.iter().take(nseq) {
        // Leftmost / rightmost aligned residue (1..=alen), as 1-based signed
        // indices to mirror C's int loops (lpos may exit at alen+1, rpos at 0).
        let mut lpos: i64 = 1;
        while lpos <= alen_i && !abc.is_residue(ax_seq[lpos as usize]) {
            lpos += 1;
        }
        let mut rpos: i64 = alen_i;
        while rpos >= 1 && !abc.is_residue(ax_seq[rpos as usize]) {
            rpos -= 1;
        }

        // Fragment test: span = rpos-lpos+1. Full-length seqs reset to whole
        // alignment; fragments keep the [lpos,rpos] span. (span <= 0 for the
        // all-gap / empty case, which is < minspan, so treated as a fragment.)
        let span = rpos - lpos + 1;
        if span >= minspan {
            lpos = 1;
            rpos = alen_i;
        }

        if !conscols.is_empty() {
            for &apos in conscols {
                let apos = apos as i64;
                if apos > rpos {
                    break;
                }
                if apos < lpos {
                    continue;
                }
                let a = ax_seq[apos as usize] as usize;
                ct[apos as usize][a] += 1;
            }
        } else {
            let mut apos = lpos;
            while apos <= rpos {
                let a = ax_seq[apos as usize] as usize;
                ct[apos as usize][a] += 1;
                apos += 1;
            }
        }
    }
}

/// Determine consensus columns from a statistical subsample of sequences,
/// for deep alignments (esl_msaweight.c:321, `consensus_by_sample`).
///
/// Faithful port: samples `nsamp` sequence indices (0..nseq-1) without
/// replacement using Easel's 64-bit Mersenne Twister seeded with
/// `eslMSAWEIGHT_RNGSEED` (42) and the Vitter sequential-sampling `Deal`
/// algorithm, then collects observed symbol counts in `ct` over each sampled
/// sequence's fragment span (marking fragments by the `minspan = ceil(fragthresh*alen)`
/// rule). If at most `maxfrag` fragments are seen, consensus columns are those
/// with gap fraction `ct[K]/tot < symfrac`; otherwise sampling is rejected
/// (`conscols` left empty), and the caller falls back to `consensus_by_all`.
///
/// On entry `ct` is overwritten (zeroed then filled); on success `conscols`
/// holds the 1-based consensus-column indices. The `ct` populated here is
/// discarded by `collect_counts`, which re-zeros it — exactly as in C.
pub(super) fn consensus_by_sample(
    ax: &[Vec<u8>],
    abc: &Alphabet,
    alen: usize,
    nseq: usize,
    ct: &mut [Vec<i32>],
    conscols: &mut Vec<usize>,
) {
    let k = abc.k;
    let kp = abc.kp;

    // Zero ct (esl_mat_ISet, esl_msaweight.c:340).
    for row in ct.iter_mut() {
        row.iter_mut().for_each(|c| *c = 0);
    }

    // Sample nsamp indices in 0..nseq-1, sorted ascending (esl_rand64_Deal).
    let mut rng = Rand64::new(PB_RNGSEED);
    let sampidx = esl_rand64_deal(&mut rng, PB_NSAMP as i64, nseq as i64);

    let minspan = (PB_FRAGTHRESH * alen as f32).ceil() as i64;
    let alen_i = alen as i64;
    let mut nfrag = 0usize;

    for &idx64 in &sampidx {
        let idx = idx64 as usize;
        let ax_seq = &ax[idx];

        let mut lpos: i64 = 1;
        while lpos <= alen_i && !abc.is_residue(ax_seq[lpos as usize]) {
            lpos += 1;
        }
        let mut rpos: i64 = alen_i;
        while rpos >= 1 && !abc.is_residue(ax_seq[rpos as usize]) {
            rpos -= 1;
        }
        if rpos - lpos + 1 < minspan {
            nfrag += 1;
        } else {
            lpos = 1;
            rpos = alen_i;
        }

        let mut apos = lpos;
        while apos <= rpos {
            let a = ax_seq[apos as usize] as usize;
            ct[apos as usize][a] += 1;
            apos += 1;
        }
    }

    if nfrag <= PB_MAXFRAG {
        for (apos, row) in ct.iter().enumerate().take(alen + 1).skip(1) {
            let tot: i32 = row.iter().take(kp - 2).sum();
            if (row[k] as f32 / tot as f32) < PB_SYMFRAC {
                conscols.push(apos);
            }
        }
    }
    // else: too many fragments -> reject; conscols stays empty (eslFAIL path).
}

/// Apply the `#=GC MM` model mask to the digitized alignment in place
/// (`do_modelmask`, build.c:222). In every column marked `'m'`, each non-gap,
/// non-missing residue is rewritten to the degenerate "any" symbol (Kp-3).
/// `ax` rows are 1-based (`ax[seq][1..=alen]`); `msa.mm[apos-1]` is the mask.
pub(super) fn do_modelmask(ax: &mut [Vec<u8>], abc: &Alphabet, msa: &Msa) {
    let Some(mm) = msa.mm.as_ref() else {
        return;
    };
    let any = abc.unknown_code(); // Kp-3
    let gap = abc.gap_code(); // K
    let missing = abc.missing_code(); // Kp-1
    for apos in 1..=msa.alen {
        if mm.get(apos - 1).copied() != Some(b'm') {
            continue;
        }
        for row in ax.iter_mut() {
            let c = row[apos];
            if c != gap && c != missing {
                row[apos] = any;
            }
        }
    }
}

/// Transfer rf/mm/cs/ca optional annotation from the MSA to the model
/// (`annotate_model`, build.c:338). Each line, if present in the MSA, is
/// emitted as a model array `hmm.X[0..=M]` where index 0 is a leading space
/// and indices 1..=M hold the source character at each successive match
/// column. The MM line maps `'.'` to `'-'` exactly as C does (build.c:360).
/// (The alignment column map is set separately by the caller, matching the
/// `p7H_MAP` portion of annotate_model.)
pub(super) fn annotate_model(hmm: &mut Hmm, matassign: &[bool], msa: &Msa) {
    let alen = msa.alen;

    if let Some(rf) = msa.rf.as_ref() {
        let mut out = vec![b' '; hmm.m + 2];
        let mut k = 1usize;
        for apos in 1..=alen {
            if matassign[apos - 1] {
                out[k] = rf.get(apos - 1).copied().unwrap_or(b' ');
                k += 1;
            }
        }
        hmm.rf = Some(out);
        hmm.flags |= P7H_RF;
    }

    if let Some(mm) = msa.mm.as_ref() {
        let mut out = vec![b' '; hmm.m + 2];
        let mut k = 1usize;
        for apos in 1..=alen {
            if matassign[apos - 1] {
                let c = mm.get(apos - 1).copied().unwrap_or(b' ');
                out[k] = if c == b'.' { b'-' } else { c };
                k += 1;
            }
        }
        hmm.mm = Some(out);
        hmm.flags |= P7H_MMASK;
    }

    if let Some(ss) = msa.ss_cons.as_ref() {
        let mut out = vec![b' '; hmm.m + 2];
        let mut k = 1usize;
        for apos in 1..=alen {
            if matassign[apos - 1] {
                out[k] = ss.get(apos - 1).copied().unwrap_or(b' ');
                k += 1;
            }
        }
        hmm.cs = Some(out);
        hmm.flags |= P7H_CS;
    }

    if let Some(sa) = msa.sa_cons.as_ref() {
        let mut out = vec![b' '; hmm.m + 2];
        let mut k = 1usize;
        for apos in 1..=alen {
            if matassign[apos - 1] {
                out[k] = sa.get(apos - 1).copied().unwrap_or(b' ');
                k += 1;
            }
        }
        hmm.ca = Some(out);
        hmm.flags |= P7H_CA;
    }
}
