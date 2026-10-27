use super::*;

pub(super) fn write_fasta(out: &mut dyn Write, name: &str, seq: &[u8]) {
    writeln!(out, ">{}", name).unwrap();
    for chunk in seq.chunks(60) {
        writeln!(out, "{}", std::str::from_utf8(chunk).unwrap()).unwrap();
    }
}

pub(super) fn sequence_text(abc: &Alphabet, seq: &Sequence) -> Vec<u8> {
    seq.dsq[1..=seq.n]
        .iter()
        .map(|&x| abc.sym[x as usize])
        .collect()
}

/// Write the sampled core-model alignment as interleaved Stockholm, matching
/// C's `emit_alignment()` (`hmmemit.c`) which builds an MSA via
/// `p7_tracealign_Seqs(..., p7_ALL_CONSENSUS_COLS, ...)` and writes it with
/// `esl_msafile_Write(ofp, msa, eslMSAFILE_STOCKHOLM)`.
///
/// Reproduces the Easel Stockholm writer layout (`esl_msafile_stockholm.c`,
/// `stockholm_write`): residues wrapped at [`STOCKHOLM_CPL`] columns into
/// interleaved blocks separated by a blank line, a left margin sized to the
/// widest of `maxname+1` and `maxgc+6` (`maxgc == 2` for the RF tag), and a
/// `#=GC RF` consensus line per block. Inserts are rejustified to match Easel's
/// `rejustify_insertions_digital()` (`tracealign.c`).
pub(super) fn write_stockholm_alignment(
    out: &mut dyn Write,
    hmm: &Hmm,
    abc: &Alphabet,
    samples: &[(String, Sequence, Trace)],
) {
    let traces: Vec<&Trace> = samples.iter().map(|(_, _, tr)| tr).collect();
    let (inscount, matmap, alen) = map_core_alignment(hmm.m, &traces);

    // Build full-width rows, then rejustify inserts exactly as Easel does.
    let rows: Vec<Vec<u8>> = samples
        .iter()
        .map(|(_, seq, trace)| {
            let mut row = render_core_alignment_row(abc, seq, trace, &inscount, &matmap, alen);
            rejustify_insertions(abc, &mut row, hmm.m, &inscount, &matmap);
            row
        })
        .collect();
    let rf = core_rf_line(hmm.m, &inscount, &matmap, alen);

    // Easel margin = max(maxname+1, maxgc+6). maxgc = 2 (the "RF" tag, clamped
    // to a minimum of 2 when msa->rf is present). Sequence lines are `%-*s `
    // with width margin-1; the `#=GC RF` line is `#=GC %-*s ` with the RF tag
    // padded to margin-6 — both land the residues at column `margin`.
    let maxname = samples
        .iter()
        .map(|(name, _, _)| name.len())
        .max()
        .unwrap_or(0);
    let maxgc = 2usize;
    let margin = (maxname + 1).max(maxgc + 6);

    writeln!(out, "# STOCKHOLM 1.0").unwrap();
    writeln!(out).unwrap();

    let mut currpos = 0usize;
    loop {
        let acpl = (alen - currpos).min(STOCKHOLM_CPL);
        if currpos > 0 {
            writeln!(out).unwrap();
        }
        for ((name, _, _), row) in samples.iter().zip(rows.iter()) {
            writeln!(
                out,
                "{:<width$} {}",
                name,
                std::str::from_utf8(&row[currpos..currpos + acpl]).unwrap(),
                width = margin - 1
            )
            .unwrap();
        }
        writeln!(
            out,
            "#=GC {:<width$} {}",
            "RF",
            std::str::from_utf8(&rf[currpos..currpos + acpl]).unwrap(),
            width = margin - 6
        )
        .unwrap();

        currpos += acpl;
        if currpos >= alen {
            break;
        }
    }
    writeln!(out, "//").unwrap();
}

fn map_core_alignment(m: usize, traces: &[&Trace]) -> (Vec<usize>, Vec<usize>, usize) {
    let mut inscount = vec![0usize; m + 1];
    let mut insnum = vec![0usize; m + 1];
    for tr in traces {
        insnum.fill(0);
        for z in 1..tr.n {
            if tr.st[z] == State::I {
                insnum[tr.k[z]] += 1;
            }
        }
        for k in 0..=m {
            inscount[k] = inscount[k].max(insnum[k]);
        }
    }

    let mut matmap = vec![0usize; m + 1];
    let mut alen = inscount[0];
    for k in 1..=m {
        matmap[k] = alen;
        alen += 1 + inscount[k];
    }
    (inscount, matmap, alen)
}

fn render_core_alignment_row(
    abc: &Alphabet,
    sq: &Sequence,
    tr: &Trace,
    inscount: &[usize],
    matmap: &[usize],
    alen: usize,
) -> Vec<u8> {
    let mut row = vec![b'.'; alen];
    for k in 1..matmap.len() {
        row[matmap[k]] = b'-';
    }
    let mut used_insert = vec![0usize; inscount.len()];
    for z in 0..tr.n {
        match tr.st[z] {
            State::M => row[matmap[tr.k[z]]] = abc.sym[sq.dsq[tr.i[z]] as usize],
            State::D => row[matmap[tr.k[z]]] = b'-',
            State::I => {
                let base = if tr.k[z] == 0 { 0 } else { matmap[tr.k[z]] + 1 };
                let col = base + used_insert[tr.k[z]];
                row[col] = (abc.sym[sq.dsq[tr.i[z]] as usize] as char).to_ascii_lowercase() as u8;
                used_insert[tr.k[z]] += 1;
            }
            _ => {}
        }
    }
    row
}

/// Rejustify the inserted residues of a single alignment row, mirroring Easel's
/// `rejustify_insertions_digital()` (`tracealign.c`). For each node `k` whose
/// insert column run is longer than 1, residues are split in half: the first
/// `nres/2` stay left-justified and the remainder are pushed flush-right within
/// the run (gaps fall in the middle). The N-terminal run (`k == 0`) is fully
/// right-justified (`nres -> 0` left), and the C-terminal run (`k == m`) is left
/// untouched (the C loop runs `k = 0 .. m-1`).
///
/// Columns are 0-based here: match node `k` (`k >= 1`) occupies `matmap[k]` and
/// its insert run is `matmap[k]+1 ..= matmap[k]+inscount[k]`; the N-terminal run
/// is `0 ..= inscount[0]-1`.
pub(super) fn rejustify_insertions(
    abc: &Alphabet,
    row: &mut [u8],
    m: usize,
    inscount: &[usize],
    matmap: &[usize],
) {
    let is_residue = |c: u8| {
        let up = c.to_ascii_uppercase();
        up != b'.' && up != b'-' && abc.sym[..abc.k].contains(&up)
    };

    for k in 0..m {
        if inscount[k] <= 1 {
            continue;
        }
        // Insert run [lo, hi] inclusive, 0-based.
        let (lo, hi) = if k == 0 {
            (0usize, inscount[0] - 1)
        } else {
            (matmap[k] + 1, matmap[k] + inscount[k])
        };

        let nres = row[lo..=hi].iter().filter(|&&c| is_residue(c)).count();
        // N-terminus is fully right-justified; otherwise split in half (the
        // number of residues that stay left-justified).
        let nleft = if k == 0 { 0 } else { nres / 2 };
        let split = lo + nleft; // first column that receives right-justified content

        // Pack residues toward the right end (hi down to split), then gap-fill.
        let mut npos = hi as isize;
        let mut opos = hi as isize;
        while opos >= split as isize {
            if !is_residue(row[opos as usize]) {
                opos -= 1;
            } else {
                row[npos as usize] = row[opos as usize];
                npos -= 1;
                opos -= 1;
            }
        }
        while npos >= split as isize {
            row[npos as usize] = b'.';
            npos -= 1;
        }
    }
}

fn core_rf_line(m: usize, inscount: &[usize], matmap: &[usize], alen: usize) -> Vec<u8> {
    let mut rf = vec![b'.'; alen];
    for k in 1..=m {
        rf[matmap[k]] = b'x';
        for col in matmap[k] + 1..matmap[k] + 1 + inscount[k] {
            rf[col] = b'.';
        }
    }
    rf
}

pub(super) fn masked_node(hmm: &Hmm, node: usize) -> bool {
    hmm.mm
        .as_ref()
        .and_then(|mm| mm.get(node))
        .is_some_and(|&c| c == b'm')
}

pub(super) fn argmax(xs: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if x > best_v {
            best_i = i;
            best_v = x;
        }
    }
    best_i
}
