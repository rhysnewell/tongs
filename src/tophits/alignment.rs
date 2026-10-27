use super::*;

/// Build a multiple sequence alignment from all included hits/domains in `th`.
/// Each included domain's `AliDisplay` is back-converted to a (sequence, trace)
/// pair, then a faux MSA is assembled mapping each match column of the model.
/// Port of `p7_tophits_Alignment()` (hmmer/src/p7_tophits.c:1478) with the
/// supporting machinery from Easel's `esl_msa_*` helpers. `extra` lets callers
/// (e.g. `jackhmmer`) prepend the query sequence/trace into the alignment.
/// `all_consensus_cols` is C's `p7_ALL_CONSENSUS_COLS` option flag: when set,
/// every match state gets an alignment column even if no included hit uses it
/// (hmmsearch, phmmer, jackhmmer); when clear (`p7_DEFAULT`, as nhmmer.c:1174
/// passes), match columns with no residues are dropped (tracealign.c:212-213).
/// Returns `None` if no included domains and no `extra` were supplied.
pub fn included_alignment(
    th: &TopHits,
    abc: &Alphabet,
    model_len: usize,
    extra: Option<(&Sequence, &Trace)>,
    all_consensus_cols: bool,
    msa_name: &str,
) -> Option<Msa> {
    let ndom = th
        .hits
        .iter()
        .filter(|hit| hit.flags & P7_IS_INCLUDED != 0)
        .map(|hit| hit.dcl.iter().filter(|dom| dom.is_included).count())
        .sum::<usize>();
    let mut sequences = Vec::new();
    let mut traces = Vec::new();

    if let Some((sq, tr)) = extra {
        sequences.push(sq.clone());
        traces.push(tr.clone());
    }

    if ndom == 0 {
        if sequences.is_empty() {
            return None;
        }
        let (inscount, matuse, matmap, alen) = map_new_msa(model_len, &traces, all_consensus_cols);
        let mut aseq = Vec::with_capacity(sequences.len());
        let mut pp = Vec::with_capacity(sequences.len());
        let mut sqname = Vec::with_capacity(sequences.len());
        let mut pp_totals = vec![0.0_f64; alen];
        let mut pp_counts = vec![0usize; alen];
        for (sq, tr) in sequences.iter().zip(traces.iter()) {
            let mut row = make_text_row(abc, sq, tr, &matuse, &matmap, alen);
            let mut pp_row =
                make_pp_row(tr, &matmap, model_len, alen, &mut pp_totals, &mut pp_counts);
            rejustify_insertions_text(
                &mut row,
                pp_row.as_deref_mut(),
                &inscount,
                &matmap,
                &matuse,
                model_len,
            );
            aseq.push(row);
            pp.push(pp_row);
            sqname.push(sq.name.clone());
        }

        let mut rf = vec![b'.'; alen];
        for k in 1..=model_len {
            if matuse[k] {
                rf[matmap[k] - 1] = b'x';
            }
        }

        return Some(Msa {
            name: msa_name.to_string(),
            acc: None,
            desc: None,
            author: None,
            sqname,
            sqacc: sequences.iter().map(|sq| sq.acc.clone()).collect(),
            sqdesc: sequences.iter().map(|sq| sq.desc.clone()).collect(),
            weights: None,
            pp,
            aseq,
            nseq: sequences.len(),
            alen,
            rf: Some(rf),
            mm: None,
            ss_cons: None,
            sa_cons: None,
            pp_cons: pp_consensus(&pp_totals, &pp_counts),
        });
    }

    for hit in &th.hits {
        if hit.flags & P7_IS_INCLUDED == 0 {
            continue;
        }
        for dom in &hit.dcl {
            if !dom.is_included {
                continue;
            }
            let ad = dom.ad.as_ref()?;
            let (sq, tr) = alidisplay_backconvert(hit, ad, abc);
            sequences.push(sq);
            traces.push(tr);
        }
    }

    if sequences.is_empty() {
        return None;
    }

    let (inscount, matuse, matmap, alen) = map_new_msa(model_len, &traces, all_consensus_cols);
    let mut aseq = Vec::with_capacity(sequences.len());
    let mut pp = Vec::with_capacity(sequences.len());
    let mut sqname = Vec::with_capacity(sequences.len());
    let mut pp_totals = vec![0.0_f64; alen];
    let mut pp_counts = vec![0usize; alen];
    for (sq, tr) in sequences.iter().zip(traces.iter()) {
        let mut row = make_text_row(abc, sq, tr, &matuse, &matmap, alen);
        let mut pp_row = make_pp_row(tr, &matmap, model_len, alen, &mut pp_totals, &mut pp_counts);
        rejustify_insertions_text(
            &mut row,
            pp_row.as_deref_mut(),
            &inscount,
            &matmap,
            &matuse,
            model_len,
        );
        aseq.push(row);
        pp.push(pp_row);
        sqname.push(sq.name.clone());
    }

    let mut rf = vec![b'.'; alen];
    for k in 1..=model_len {
        if matuse[k] {
            rf[matmap[k] - 1] = b'x';
        }
    }

    Some(Msa {
        name: msa_name.to_string(),
        acc: None,
        desc: None,
        author: None,
        sqname,
        sqacc: sequences.iter().map(|sq| sq.acc.clone()).collect(),
        sqdesc: sequences.iter().map(|sq| sq.desc.clone()).collect(),
        weights: None,
        pp,
        aseq,
        nseq: sequences.len(),
        alen,
        rf: Some(rf),
        mm: None,
        ss_cons: None,
        sa_cons: None,
        pp_cons: pp_consensus(&pp_totals, &pp_counts),
    })
}

/// Reconstruct a `(Sequence, Trace)` pair from a printable `AliDisplay`.
/// Walks the alignment columns, dropping gap columns and emitting M/I/D states
/// with sequence indices. Port of `p7_alidisplay_Backconvert()`
/// (hmmer/src/p7_alidisplay.c:1233).
fn alidisplay_backconvert(hit: &Hit, ad: &AliDisplay, abc: &Alphabet) -> (Sequence, Trace) {
    let sub_l = ad.aseq.bytes().filter(|&c| c != b'.' && c != b'-').count();

    let mut sq = Sequence::new();
    // p7_alidisplay.c:1319: "%s/%d-%d" with ad->sqfrom/ad->sqto in alignment
    // orientation, so a minus-strand nhmmer hit reads e.g. "chr/40318-40002".
    sq.name = format!("{}/{}-{}", hit.name, ad.sqfrom, ad.sqto);
    sq.desc = format!(
        "[subseq from] {}",
        if hit.desc.is_empty() {
            hit.name.as_str()
        } else {
            hit.desc.as_str()
        }
    );
    sq.acc = hit.acc.clone();
    sq.n = sub_l;
    sq.l = hit.n;
    sq.dsq = Vec::with_capacity(sub_l + 2);
    sq.dsq.push(crate::alphabet::DSQ_SENTINEL);

    let mut tr = Trace::new();
    tr.append(State::S, 0, 0);
    tr.append(State::N, 0, 0);
    tr.append(State::B, 0, 0);

    let model = ad.model.as_bytes();
    let aseq = ad.aseq.as_bytes();
    let mut k = ad.hmmfrom.saturating_sub(1);
    let mut i = 1usize;
    for a in 0..ad.model.len() {
        let model_gap = model[a] == b'.' || model[a] == b'-';
        let aseq_gap = aseq[a] == b'.' || aseq[a] == b'-';
        let state = if !model_gap {
            k += 1;
            if !aseq_gap {
                State::M
            } else {
                State::D
            }
        } else {
            State::I
        };
        let pp = ad
            .ppline
            .as_bytes()
            .get(a)
            .copied()
            .map(decode_postprob)
            .unwrap_or(0.0);
        tr.append_with_pp(state, k, i, pp);
        match state {
            State::M | State::I => {
                sq.dsq
                    .push(abc.digitize_symbol(aseq[a].to_ascii_uppercase()));
                i += 1;
            }
            State::D => {}
            _ => unreachable!(),
        }
    }
    tr.append_with_pp(State::E, 0, 0, 0.0);
    tr.append_with_pp(State::C, 0, 0, 0.0);
    tr.append_with_pp(State::T, 0, 0, 0.0);
    sq.dsq.push(crate::alphabet::DSQ_SENTINEL);

    (sq, tr)
}

/// Encode a posterior probability as a single ASCII character ('0'..'9' or '*').
/// Port of `p7_alidisplay_EncodePostProb()`.
fn encode_postprob(p: f32) -> u8 {
    let shifted = p as f64 + 0.05_f64;
    if shifted >= 1.0 {
        b'*'
    } else {
        (shifted * 10.0) as u8 + b'0'
    }
}

/// Decode a posterior-probability character back to a float in [0, 1].
/// Inverse of `encode_postprob`; matches `p7_alidisplay_DecodePostProb()`.
fn decode_postprob(pc: u8) -> f32 {
    match pc {
        b'0' => 0.01,
        b'*' => 1.0,
        b'.' => 0.0,
        b'1'..=b'9' => (pc - b'0') as f32 / 10.0,
        _ => 0.0,
    }
}

/// Compute the per-column consensus PP line by averaging recorded values and
/// encoding the mean. Returns `None` if no column had any PP-bearing row.
fn pp_consensus(pp_totals: &[f64], pp_counts: &[usize]) -> Option<Vec<u8>> {
    if pp_counts.iter().all(|&count| count == 0) {
        return None;
    }
    Some(
        pp_totals
            .iter()
            .zip(pp_counts.iter())
            .map(|(&total, &count)| {
                if count > 0 {
                    encode_postprob((total / count as f64) as f32)
                } else {
                    b'.'
                }
            })
            .collect(),
    )
}

/// Plan an MSA layout from a set of traces against a model of length `m`.
/// Returns `(inscount[k], matuse[k], matmap[k], alen)`: per-column max insert
/// length, which match columns are used by at least one trace, the mapping
/// from model node 1..M to its alignment column (1-based), and the total
/// alignment width. Mirrors C `map_new_msa()` in `tracealign.c`. With
/// `all_consensus_cols` every match column is used up front
/// (`p7_ALL_CONSENSUS_COLS`); otherwise only match states some trace visits.
fn map_new_msa(
    m: usize,
    traces: &[Trace],
    all_consensus_cols: bool,
) -> (Vec<usize>, Vec<bool>, Vec<usize>, usize) {
    let mut inscount = vec![0usize; m + 1];
    let mut matuse = vec![all_consensus_cols; m + 1];
    matuse[0] = false;
    let mut insnum = vec![0usize; m + 1];

    for tr in traces {
        insnum.fill(0);
        for z in 1..tr.n {
            match tr.st[z] {
                State::I => insnum[tr.k[z]] += 1,
                State::N if tr.st[z - 1] == State::N => insnum[0] += 1,
                State::C if tr.st[z - 1] == State::C => insnum[m] += 1,
                State::M => matuse[tr.k[z]] = true,
                State::J => panic!("J state unsupported in TopHits alignment"),
                _ => {}
            }
        }
        for k in 0..=m {
            inscount[k] = inscount[k].max(insnum[k]);
        }
    }

    let mut matmap = vec![0usize; m + 1];
    let mut alen = inscount[0];
    for k in 1..=m {
        if matuse[k] {
            matmap[k] = alen + 1;
            alen += 1 + inscount[k];
        } else {
            matmap[k] = alen;
            alen += inscount[k];
        }
    }

    (inscount, matuse, matmap, alen)
}

/// Render one aligned sequence row for `sq`/`tr` into the MSA layout produced
/// by `map_new_msa`. Match states use uppercase, inserts use lowercase, and
/// unused match columns get '-' or '.'. Mirrors `make_text_row()` in C.
fn make_text_row(
    abc: &Alphabet,
    sq: &Sequence,
    tr: &Trace,
    matuse: &[bool],
    matmap: &[usize],
    alen: usize,
) -> Vec<u8> {
    let mut aseq = vec![b'.'; alen];
    for k in 1..matuse.len() {
        if matuse[k] {
            aseq[matmap[k] - 1] = b'-';
        }
    }

    let mut apos = 0usize;
    for z in 0..tr.n {
        match tr.st[z] {
            State::M => {
                let idx = matmap[tr.k[z]] - 1;
                aseq[idx] = (abc.sym[sq.dsq[tr.i[z]] as usize] as char).to_ascii_uppercase() as u8;
                apos = matmap[tr.k[z]];
            }
            State::D => {
                if matuse[tr.k[z]] {
                    aseq[matmap[tr.k[z]] - 1] = b'-';
                }
                apos = matmap[tr.k[z]];
            }
            State::I => {
                if apos < alen {
                    aseq[apos] =
                        (abc.sym[sq.dsq[tr.i[z]] as usize] as char).to_ascii_lowercase() as u8;
                    apos += 1;
                }
            }
            State::N | State::C => {
                if tr.i[z] > 0 && apos < alen {
                    aseq[apos] =
                        (abc.sym[sq.dsq[tr.i[z]] as usize] as char).to_ascii_lowercase() as u8;
                    apos += 1;
                }
            }
            State::E => {
                apos = matmap[matmap.len() - 1];
            }
            _ => {}
        }
    }

    aseq
}

/// Render one PP annotation row aligned to the MSA columns and accumulate the
/// per-column running totals/counts used to build the consensus PP line. Returns
/// `None` if the trace lacks PP annotations.
fn make_pp_row(
    tr: &Trace,
    matmap: &[usize],
    model_len: usize,
    alen: usize,
    pp_totals: &mut [f64],
    pp_counts: &mut [usize],
) -> Option<Vec<u8>> {
    let pp_values = tr.pp.as_ref()?;
    let mut pp = vec![b'.'; alen];
    let mut apos = 0usize;
    for z in 0..tr.n {
        match tr.st[z] {
            State::M => {
                let idx = matmap[tr.k[z]] - 1;
                let value = pp_values[z];
                pp[idx] = encode_postprob(value);
                pp_totals[idx] += value as f64;
                pp_counts[idx] += 1;
                apos = matmap[tr.k[z]];
            }
            State::D => {
                apos = matmap[tr.k[z]];
            }
            State::I => {
                if tr.k[z] != 0 && tr.k[z] != model_len && apos < alen {
                    let value = pp_values[z];
                    pp[apos] = encode_postprob(value);
                    apos += 1;
                }
            }
            State::N | State::C => {
                if tr.i[z] > 0 && apos < alen {
                    let value = pp_values[z];
                    pp[apos] = encode_postprob(value);
                    apos += 1;
                }
            }
            State::E => {
                apos = matmap[model_len];
            }
            _ => {}
        }
    }
    Some(pp)
}

/// Right-justify insert columns so that aligned residues sit flush against the
/// next match column on either side, matching Easel's
/// `rejustify_insertions_text()` used by `esl_msa_FromAliDisplay`-style outputs.
fn rejustify_insertions_text(
    aseq: &mut [u8],
    mut pp: Option<&mut [u8]>,
    inserts: &[usize],
    matmap: &[usize],
    matuse: &[bool],
    m: usize,
) {
    fn is_text_gap(c: u8) -> bool {
        matches!(c, b'.' | b'-' | b'~')
    }

    for k in 0..m {
        if inserts[k] <= 1 {
            continue;
        }

        let start = matmap[k];
        let end = matmap[k + 1] - usize::from(matuse[k + 1]);
        let mut nins = (start..end)
            .filter(|&apos| aseq[apos].is_ascii_alphabetic())
            .count();
        if k == 0 {
            nins = 0;
        } else {
            nins /= 2;
        }

        let floor = (start + nins) as isize;
        let mut opos = end as isize - 1;
        let mut npos = end as isize - 1;
        while opos >= floor {
            if is_text_gap(aseq[opos as usize]) {
                opos -= 1;
                continue;
            }
            aseq[npos as usize] = aseq[opos as usize];
            if let Some(pp) = &mut pp {
                pp[npos as usize] = pp[opos as usize];
            }
            opos -= 1;
            npos -= 1;
        }
        while npos >= floor {
            aseq[npos as usize] = b'.';
            if let Some(pp) = &mut pp {
                pp[npos as usize] = b'.';
            }
            npos -= 1;
        }
    }
}
