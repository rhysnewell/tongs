use super::*;

/// Reconstruct the processed text MSA that HMMER's builder saves with
/// `hmmbuild -O`: mark fragments, build doctored faux traces from the assigned
/// match columns, trace-align them, annotate RF, and rejustify insertions.
pub fn tracealigned_post_msa_rows(
    msa: &Msa,
    abc: &Alphabet,
    model_mask: &[u8],
    fragthresh: f32,
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut ax = msa.digitize(abc);
    mark_fragments_old(&mut ax, abc, msa.alen, fragthresh);
    let matassign: Vec<bool> = model_mask.iter().map(|&ch| ch == b'x').collect();
    let traces = faux_trace_from_msa(&ax, &matassign, abc);
    let model_len = matassign.iter().filter(|&&is_match| is_match).count();
    let (inscount, matuse, matmap, alen) = tracealign_map_new_msa(model_len, &traces);
    let mut rows = tracealign_make_text_rows(abc, &ax, &traces, &matuse, &matmap, model_len, alen);
    tracealign_rejustify_insertions(abc, &mut rows, &inscount, &matmap, &matuse, model_len);
    let rf = tracealign_rf(model_len, alen, &matuse, &matmap);
    (rf, rows)
}

fn tracealign_map_new_msa(
    m: usize,
    traces: &[Trace],
) -> (Vec<usize>, Vec<bool>, Vec<usize>, usize) {
    let mut inscount = vec![0usize; m + 1];
    let mut matuse = vec![false; m + 1];
    let mut insnum = vec![0usize; m + 1];

    for tr in traces {
        insnum.fill(0);
        for z in 1..tr.n {
            match tr.st[z] {
                TraceState::I => insnum[tr.k[z]] += 1,
                TraceState::N if tr.st[z - 1] == TraceState::N => insnum[0] += 1,
                TraceState::C if tr.st[z - 1] == TraceState::C => insnum[m] += 1,
                TraceState::M => matuse[tr.k[z]] = true,
                TraceState::J => panic!("J state unsupported in builder MSA reconstruction"),
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

fn tracealign_make_text_rows(
    abc: &Alphabet,
    ax: &[Vec<u8>],
    traces: &[Trace],
    matuse: &[bool],
    matmap: &[usize],
    m: usize,
    alen: usize,
) -> Vec<Vec<u8>> {
    let mut rows = Vec::with_capacity(traces.len());
    for (idx, tr) in traces.iter().enumerate() {
        let mut row = vec![b'.'; alen];
        for k in 1..=m {
            if matuse[k] {
                row[matmap[k] - 1] = b'-';
            }
        }

        let mut apos = 0usize;
        for z in 0..tr.n {
            match tr.st[z] {
                TraceState::M => {
                    row[matmap[tr.k[z]] - 1] =
                        abc.sym[ax[idx][tr.i[z]] as usize].to_ascii_uppercase();
                    apos = matmap[tr.k[z]];
                }
                TraceState::D => {
                    if matuse[tr.k[z]] {
                        row[matmap[tr.k[z]] - 1] = b'-';
                    }
                    apos = matmap[tr.k[z]];
                }
                TraceState::I => {
                    if apos < alen {
                        row[apos] = abc.sym[ax[idx][tr.i[z]] as usize].to_ascii_lowercase();
                        apos += 1;
                    }
                }
                TraceState::N | TraceState::C => {
                    if tr.i[z] > 0 && apos < alen {
                        row[apos] = abc.sym[ax[idx][tr.i[z]] as usize].to_ascii_lowercase();
                        apos += 1;
                    }
                }
                TraceState::E => {
                    apos = matmap[m];
                }
                TraceState::X => {
                    if z > 0 && tr.st[z - 1] == TraceState::B {
                        let next_k = tr.k[z + 1];
                        for ch in row.iter_mut().take(matmap[next_k]) {
                            *ch = b'~';
                        }
                        apos = matmap[next_k];
                    } else if z + 1 < tr.n && tr.st[z + 1] == TraceState::E {
                        for ch in row.iter_mut().take(alen).skip(apos) {
                            *ch = b'~';
                        }
                    }
                }
                _ => {}
            }
        }
        rows.push(row);
    }
    rows
}

fn tracealign_rejustify_insertions(
    abc: &Alphabet,
    rows: &mut [Vec<u8>],
    inserts: &[usize],
    matmap: &[usize],
    matuse: &[bool],
    m: usize,
) {
    let is_residue = |ch: u8| abc.is_residue(abc.digitize_symbol(ch.to_ascii_uppercase()));
    let is_gap = |ch: u8| matches!(ch, b'.' | b'-' | b'_');

    for row in rows {
        for k in 0..m {
            if inserts[k] <= 1 {
                continue;
            }
            let start = matmap[k];
            let end_exclusive = matmap[k + 1] - usize::from(matuse[k + 1]);
            let mut nins = row[start..end_exclusive]
                .iter()
                .filter(|&&ch| is_residue(ch))
                .count();
            if k == 0 {
                nins = 0;
            } else {
                nins /= 2;
            }

            let mut npos = end_exclusive as isize - 1;
            let mut opos = end_exclusive as isize - 1;
            let stop = (start + nins) as isize;
            while opos >= stop {
                if is_gap(row[opos as usize]) {
                    opos -= 1;
                } else {
                    row[npos as usize] = row[opos as usize];
                    npos -= 1;
                    opos -= 1;
                }
            }
            while npos >= stop {
                row[npos as usize] = b'.';
                npos -= 1;
            }
        }
    }
}

fn tracealign_rf(m: usize, alen: usize, matuse: &[bool], matmap: &[usize]) -> Vec<u8> {
    let mut rf = vec![b'.'; alen];
    for k in 1..=m {
        if matuse[k] {
            rf[matmap[k] - 1] = b'x';
        }
    }
    rf
}
