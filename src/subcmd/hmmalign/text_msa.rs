use super::*;

pub(crate) fn build_text_msa(
    hmm: &tongs::hmm::Hmm,
    abc: &Alphabet,
    bg: &Bg,
    sequences: &[Sequence],
    trim: bool,
) -> Result<TextMsa, Unalignable> {
    let traces = align_traces(hmm, abc, bg, sequences)?;

    let (inscount, matuse, matmap, alen) = map_new_msa(hmm.m, &traces, trim);
    let mut rows = Vec::with_capacity(sequences.len());
    let mut pp_cons_sum = vec![0.0_f32; alen];
    let mut pp_cons_n = vec![0usize; alen];

    for (sq, (tr, pp_z)) in sequences.iter().zip(traces.iter()) {
        let (aseq, ppline) = make_text_row(abc, sq, tr, pp_z, &matuse, &matmap, alen, trim);
        for z in 0..tr.n {
            if tr.st[z] == State::M {
                let apos = matmap[tr.k[z]] - 1;
                pp_cons_sum[apos] += pp_z[z].min(1.0);
                pp_cons_n[apos] += 1;
            }
        }
        rows.push(AlignmentRow {
            name: sq.name.clone(),
            acc: (!sq.acc.is_empty()).then(|| sq.acc.clone()),
            desc: (!sq.desc.is_empty()).then(|| sq.desc.clone()),
            aseq,
            ppline: Some(ppline),
        });
    }

    for row in &mut rows {
        if let Some(ppline) = &mut row.ppline {
            rejustify_insertions_text(&mut row.aseq, ppline, &inscount, &matmap, &matuse, hmm.m);
        }
    }

    let mut rfline = vec![b'.'; alen];
    for k in 1..=hmm.m {
        if matuse[k] {
            rfline[matmap[k] - 1] = b'x';
        }
    }

    let mut pp_cons = String::with_capacity(alen);
    for apos in 0..alen {
        if pp_cons_n[apos] > 0 {
            pp_cons.push(pp_to_char(pp_cons_sum[apos] / pp_cons_n[apos] as f32));
        } else {
            pp_cons.push('.');
        }
    }

    Ok(TextMsa {
        rows,
        rfline: String::from_utf8(rfline).unwrap(),
        pp_cons,
    })
}

/// Variant of `build_text_msa` that also folds in a previously built `--mapali`
/// MSA. Verifies the HMM checksum/map, computes per-column insert widths spanning
/// both the source MSA and the new traces, then emits a merged alignment. Mirrors
/// the `map_alignment()` + main-loop path in hmmalign.c.
pub(super) fn build_text_msa_with_mapali(
    hmm: &tongs::hmm::Hmm,
    abc: &Alphabet,
    bg: &Bg,
    sequences: &[Sequence],
    mapped: &msa::Msa,
    trim: bool,
) -> Result<TextMsa, Unalignable> {
    if hmm.flags & tongs::hmm::P7H_CHKSUM == 0 {
        eprintln!("HMM has no checksum. --mapali unreliable without it.");
        std::process::exit(1);
    }
    if hmm.flags & tongs::hmm::P7H_MAP == 0 || hmm.map.is_none() {
        eprintln!("HMM has no map. --mapali can't work without it.");
        std::process::exit(1);
    }
    let checksum = msa::checksum(mapped, abc);
    if checksum != hmm.checksum {
        eprintln!("--mapali MSA isn't same as the one HMM came from (checksum mismatch)");
        std::process::exit(1);
    }

    let map = hmm.map.as_ref().unwrap();
    let mapped_pairs = faux_from_mapped_msa(mapped, map, abc, hmm.m);
    let traces = align_traces(hmm, abc, bg, sequences)?;

    let mut all_traces = Vec::with_capacity(mapped_pairs.len() + traces.len());
    all_traces.extend(mapped_pairs.iter().map(|(_, tr)| (tr.clone(), Vec::new())));
    all_traces.extend(traces.iter().cloned());

    let (inscount, matuse, matmap, alen) = map_new_msa(hmm.m, &all_traces, trim);

    let mut rows = Vec::with_capacity(mapped.nseq + sequences.len());
    for (idx, (sq, tr)) in mapped_pairs.iter().enumerate() {
        let dummy_pp = vec![0.0_f32; tr.n];
        let (mut aseq, mut ppline) =
            make_text_row(abc, sq, tr, &dummy_pp, &matuse, &matmap, alen, trim);
        rejustify_insertions_text(&mut aseq, &mut ppline, &inscount, &matmap, &matuse, hmm.m);
        rows.push(AlignmentRow {
            name: mapped.sqname[idx].clone(),
            acc: (!mapped.sqacc[idx].is_empty()).then(|| mapped.sqacc[idx].clone()),
            desc: (!mapped.sqdesc[idx].is_empty()).then(|| mapped.sqdesc[idx].clone()),
            aseq,
            ppline: None,
        });
    }

    let mut pp_cons_sum = vec![0.0_f32; alen];
    let mut pp_cons_n = vec![0usize; alen];
    for (sq, (tr, pp_z)) in sequences.iter().zip(traces.iter()) {
        let (mut aseq, mut ppline) = make_text_row(abc, sq, tr, pp_z, &matuse, &matmap, alen, trim);
        rejustify_insertions_text(&mut aseq, &mut ppline, &inscount, &matmap, &matuse, hmm.m);
        for z in 0..tr.n {
            if tr.st[z] == State::M {
                let apos = matmap[tr.k[z]] - 1;
                pp_cons_sum[apos] += pp_z[z].min(1.0);
                pp_cons_n[apos] += 1;
            }
        }
        rows.push(AlignmentRow {
            name: sq.name.clone(),
            acc: (!sq.acc.is_empty()).then(|| sq.acc.clone()),
            desc: (!sq.desc.is_empty()).then(|| sq.desc.clone()),
            aseq,
            ppline: Some(ppline),
        });
    }

    let mut rfline = vec![b'.'; alen];
    for k in 1..=hmm.m {
        if matuse[k] {
            rfline[matmap[k] - 1] = b'x';
        }
    }
    let mut pp_cons = String::with_capacity(alen);
    for apos in 0..alen {
        if pp_cons_n[apos] > 0 {
            pp_cons.push(pp_to_char(pp_cons_sum[apos] / pp_cons_n[apos] as f32));
        } else {
            pp_cons.push('.');
        }
    }

    let mut msa = TextMsa {
        rows,
        rfline: String::from_utf8(rfline).unwrap(),
        pp_cons,
    };
    if trim {
        trim_terminal_residueless_columns(&mut msa);
    }
    Ok(msa)
}

fn faux_from_mapped_msa(
    mapped: &msa::Msa,
    map: &[i32],
    abc: &Alphabet,
    m: usize,
) -> Vec<(Sequence, tongs::trace::Trace)> {
    let mut matassign = vec![false; mapped.alen + 1];
    for k in 1..=m {
        matassign[map[k] as usize] = true;
    }

    (0..mapped.nseq)
        .map(|idx| {
            let mut sq = Sequence::new();
            sq.name = mapped.sqname[idx].clone();
            sq.acc = mapped.sqacc[idx].clone();
            sq.desc = mapped.sqdesc[idx].clone();

            let mut tr = tongs::trace::Trace::new();
            tr.append(State::B, 0, 0);

            let mut k = 0usize;
            let mut rpos = 1usize;
            for (apos0, &ch) in mapped.aseq[idx].iter().enumerate() {
                let apos = apos0 + 1;
                let code = abc.digitize_symbol(ch);
                let is_residue = abc.is_residue(code) || code == abc.nonresidue_code();
                let is_gap = matches!(ch, b'-' | b'.' | b'_');
                let is_missing = code == abc.missing_code();

                if matassign[apos] {
                    k += 1;
                    if is_residue {
                        sq.dsq.push(code);
                        tr.append(State::M, k, rpos);
                        rpos += 1;
                    } else if is_gap {
                        tr.append(State::D, k, 0);
                    } else if is_missing && tr.st.last().copied() != Some(State::X) {
                        tr.append(State::X, k, 0);
                    }
                } else if is_residue {
                    sq.dsq.push(code);
                    tr.append(State::I, k, rpos);
                    rpos += 1;
                } else if is_missing && tr.st.last().copied() != Some(State::X) {
                    tr.append(State::X, k, 0);
                }
            }

            tr.append(State::E, 0, 0);
            tr.m = k;
            tr.l = mapped.alen;
            sq.n = sq.dsq.len() - 1;
            sq.l = sq.n;
            sq.dsq.push(tongs::alphabet::DSQ_SENTINEL);

            (sq, tr)
        })
        .collect()
}

fn trim_terminal_residueless_columns(msa: &mut TextMsa) {
    let alen = msa.rfline.len();
    let first = (0..alen).find(|&apos| {
        msa.rows
            .iter()
            .any(|row| row.aseq.as_bytes()[apos].is_ascii_alphabetic())
    });
    let Some(first) = first else {
        return;
    };
    let last = (0..alen)
        .rfind(|&apos| {
            msa.rows
                .iter()
                .any(|row| row.aseq.as_bytes()[apos].is_ascii_alphabetic())
        })
        .unwrap();
    if first == 0 && last + 1 == alen {
        return;
    }

    for row in &mut msa.rows {
        row.aseq = row.aseq[first..=last].to_string();
        if let Some(ppline) = row.ppline.as_mut() {
            *ppline = ppline[first..=last].to_string();
        }
    }
    msa.rfline = msa.rfline[first..=last].to_string();
    msa.pp_cons = msa.pp_cons[first..=last].to_string();
}

/// Build the merged-MSA coordinate system (insert widths, matuse flags, matmap,
/// total `alen`) from a set of traces. Mirrors Easel's `esl_msa_MapNew` /
/// HMMER's `p7_tracealign_Seqs()` map-construction step.
fn map_new_msa(
    m: usize,
    traces: &[AlignedTrace],
    trim: bool,
) -> (Vec<usize>, Vec<bool>, Vec<usize>, usize) {
    let mut inscount = vec![0usize; m + 1];
    let mut matuse = vec![true; m + 1];
    matuse[0] = false;
    let mut insnum = vec![0usize; m + 1];

    for (tr, _) in traces {
        insnum.fill(0);
        for z in 1..tr.n {
            match tr.st[z] {
                State::I => insnum[tr.k[z]] += 1,
                State::N if tr.st[z - 1] == State::N => insnum[0] += 1,
                State::C if tr.st[z - 1] == State::C => insnum[m] += 1,
                State::M => matuse[tr.k[z]] = true,
                State::J => panic!("J state unsupported in hmmalign MSA construction"),
                _ => {}
            }
        }
        for k in 0..=m {
            inscount[k] = inscount[k].max(insnum[k]);
        }
    }

    if trim {
        inscount[0] = 0;
        inscount[m] = 0;
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

/// Render one sequence's trace as text-aligned residues plus the matching PP
/// (posterior-probability) annotation row, using `matmap` to place match-state
/// residues and contiguous insert positions for I/N/C states.
#[allow(clippy::too_many_arguments)]
fn make_text_row(
    abc: &Alphabet,
    sq: &Sequence,
    tr: &tongs::trace::Trace,
    pp_z: &[f32],
    matuse: &[bool],
    matmap: &[usize],
    alen: usize,
    trim: bool,
) -> (String, String) {
    let mut aseq = vec![b'.'; alen];
    let mut ppline = vec![b'.'; alen];
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
                ppline[idx] = pp_to_char(pp_z[z].min(1.0)) as u8;
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
                    ppline[apos] = pp_to_char(pp_z[z].min(1.0)) as u8;
                    apos += 1;
                }
            }
            State::N | State::C => {
                if !trim && tr.i[z] > 0 && apos < alen {
                    aseq[apos] =
                        (abc.sym[sq.dsq[tr.i[z]] as usize] as char).to_ascii_lowercase() as u8;
                    ppline[apos] = pp_to_char(pp_z[z].min(1.0)) as u8;
                    apos += 1;
                }
            }
            State::E => {
                apos = matmap[matmap.len() - 1];
            }
            _ => {}
        }
    }

    (
        String::from_utf8(aseq).unwrap(),
        String::from_utf8(ppline).unwrap(),
    )
}

/// Map an N/C special state to the matching column index in the special-states
/// (XMX) portion of the posterior decoding matrix.
pub(super) fn state_pp_index(st: State) -> usize {
    match st {
        State::N => tongs::dp::gmx::P7G_N,
        State::C => tongs::dp::gmx::P7G_C,
        _ => unreachable!(),
    }
}

/// Right-justify the second half of each insert region in `aseq`/`ppline` so
/// inserts pack toward the flanking match columns. Replicates HMMER's
/// `rejustify_inserts_text` behavior.
fn rejustify_insertions_text(
    aseq: &mut String,
    ppline: &mut String,
    inserts: &[usize],
    matmap: &[usize],
    matuse: &[bool],
    m: usize,
) {
    fn is_text_gap(c: u8) -> bool {
        matches!(c, b'.' | b'-' | b'~')
    }

    let mut aseq_bytes = aseq.as_bytes().to_vec();
    let mut pp_bytes = ppline.as_bytes().to_vec();

    for k in 0..m {
        if inserts[k] <= 1 {
            continue;
        }

        let start = matmap[k];
        let end = matmap[k + 1] - usize::from(matuse[k + 1]);
        let mut nins = (start..end)
            .filter(|&apos| aseq_bytes[apos].is_ascii_alphabetic())
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
            if is_text_gap(aseq_bytes[opos as usize]) {
                opos -= 1;
                continue;
            }
            aseq_bytes[npos as usize] = aseq_bytes[opos as usize];
            pp_bytes[npos as usize] = pp_bytes[opos as usize];
            opos -= 1;
            npos -= 1;
        }
        while npos >= floor {
            aseq_bytes[npos as usize] = b'.';
            pp_bytes[npos as usize] = b'.';
            npos -= 1;
        }
    }

    *aseq = String::from_utf8(aseq_bytes).unwrap();
    *ppline = String::from_utf8(pp_bytes).unwrap();
}

/// Convert a posterior probability in [0,1] to its single-character bin
/// (`0`-`9`, `*`, or `.`) used in Stockholm PP annotation lines.
fn pp_to_char(pp: f32) -> char {
    let p = pp.clamp(0.0, 1.0);
    if p >= 0.95 {
        '*'
    } else if p >= 0.85 {
        '9'
    } else if p >= 0.75 {
        '8'
    } else if p >= 0.65 {
        '7'
    } else if p >= 0.55 {
        '6'
    } else if p >= 0.45 {
        '5'
    } else if p >= 0.35 {
        '4'
    } else if p >= 0.25 {
        '3'
    } else if p >= 0.15 {
        '2'
    } else if p >= 0.05 {
        '1'
    } else if p > 0.0 {
        '0'
    } else {
        '.'
    }
}
