use super::*;

/// Emit a Stockholm 1.0 representation of the assembled alignment, including
/// per-row PP annotation and `#=GC PP_cons` / `#=GC RF` consensus lines.
///
/// Faithful port of Easel's `stockholm_write` (`esl_msafile_stockholm.c`) for
/// the subset of annotation that `hmmalign` produces: per-row `#=GR <name> PP`,
/// `#=GC PP_cons`, `#=GC RF`, and optional `#=GS <name> AC/DE`. Stockholm format
/// wraps the alignment in 200-column blocks; the left margin width matches C's
/// `margin` computation so sequence, GR, and GC lines stay in register.
///
/// ## Intentional annotation scope — why no SS/SA/MM/#=GF lines
///
/// `hmmalign` builds its MSA via `p7_tracealign_Seqs()` in tracealign.c, which
/// calls three annotation helpers and then sets per-sequence names/accessions/descs:
///
///   1. `annotate_rf()` → sets `msa->rf` ('x'/'.' column mask) — **emitted** as `#=GC RF`
///   2. `annotate_mm()` → sets `msa->mm` only when `hmm->mm != NULL` (model mask).
///      Standard HMMs from `hmmbuild` never carry a model mask, so `msa->mm` is NULL
///      in practice for all `hmmalign` runs; the `#=GC MM` line is therefore never
///      produced. **Not emitted** (correctly).
///   3. `annotate_posterior_probability()` → sets `msa->pp[i]` per sequence and
///      `msa->pp_cons` — **emitted** as `#=GR <name> PP` and `#=GC PP_cons`.
///
/// Fields that `p7_tracealign_Seqs()` never sets on the output MSA for `hmmalign`:
///   - `msa->name/acc/desc/au` → NULL  → no `#=GF ID/AC/DE/AU` lines (correct)
///   - `msa->ss_cons/sa_cons`  → NULL  → no `#=GC SS_cons/SA_cons` lines (correct)
///   - `msa->ss[i]/sa[i]`      → NULL  → no `#=GR <name> SS/SA` lines (correct)
///   - `msa->cutset[]`         → unset → no `#=GF GA/TC/NC` lines (correct)
///
/// The per-sequence `sqacc` and `sqdesc` arrays are set from `sq[i]->acc` and
/// `sq[i]->desc`, which come from the input FASTA/sequence headers — **emitted** as
/// `#=GS <name> AC` / `#=GS <name> DE` when non-empty.
///
/// Conclusion: the Rust writers emit exactly what C's `hmmalign` + `stockholm_write`
/// would emit given the annotation that `p7_tracealign_Seqs()` places on the MSA.
/// The "missing" SS/SA/MM/GF-cutoff lines are absent because `hmmalign` itself never
/// produces them — this is a non-gap, not an omission in the Rust port.
pub(super) fn write_stockholm(out: &mut dyn Write, msa: &TextMsa) {
    write_stockholm_blocked(out, msa, 200);
}

/// Shared Stockholm/Pfam writer; `cpl` is the residues-per-line block width
/// (200 for Stockholm, alen for Pfam — Easel's `stockholm_write` <cpl> arg).
pub(super) fn write_stockholm_blocked(out: &mut dyn Write, msa: &TextMsa, cpl: usize) {
    let maxname = msa.rows.iter().map(|row| row.name.len()).max().unwrap_or(0);

    // maxgc: GC tags emitted here are PP_cons (7) and RF (2) => 7.
    let mut maxgc = 0usize;
    maxgc = maxgc.max(7); // PP_cons
    maxgc = maxgc.max(2); // RF
                          // maxgr: GR tag emitted here is PP (2).
    let maxgr = 2usize;

    let mut margin = maxname + 1;
    if maxgc > 0 && maxgc + 6 > margin {
        margin = maxgc + 6;
    }
    if maxgr > 0 && maxname + maxgr + 7 > margin {
        margin = maxname + maxgr + 7;
    }

    writeln!(out, "# STOCKHOLM 1.0").unwrap();

    // GF section: hmmalign's MSA carries no #=GF annotation, but Easel always
    // closes the GF section with a blank line.
    writeln!(out).unwrap();

    // GS section: per-sequence AC and DE blocks, each terminated by a blank line
    // when present (matching Easel's separate msa->sqacc / msa->sqdesc loops).
    if msa.rows.iter().any(|row| row.acc.is_some()) {
        for row in &msa.rows {
            if let Some(acc) = &row.acc {
                if !acc.is_empty() {
                    writeln!(out, "#=GS {:<width$} AC {}", row.name, acc, width = maxname).unwrap();
                }
            }
        }
        writeln!(out).unwrap();
    }
    if msa.rows.iter().any(|row| row.desc.is_some()) {
        for row in &msa.rows {
            if let Some(desc) = &row.desc {
                if !desc.is_empty() {
                    writeln!(
                        out,
                        "#=GS {:<width$} DE {}",
                        row.name,
                        desc,
                        width = maxname
                    )
                    .unwrap();
                }
            }
        }
        writeln!(out).unwrap();
    }

    // Alignment section, in <cpl>-column blocks.
    let alen = msa.rfline.len();
    let mut currpos = 0usize;
    while currpos < alen {
        let end = (currpos + cpl).min(alen);
        if currpos > 0 {
            writeln!(out).unwrap();
        }
        for row in &msa.rows {
            writeln!(
                out,
                "{:<width$} {}",
                row.name,
                &row.aseq[currpos..end],
                width = margin - 1
            )
            .unwrap();
            if let Some(ppline) = &row.ppline {
                writeln!(
                    out,
                    "#=GR {:<namew$} {:<tagw$} {}",
                    row.name,
                    "PP",
                    &ppline[currpos..end],
                    namew = maxname,
                    tagw = margin - maxname - 7
                )
                .unwrap();
            }
        }
        writeln!(
            out,
            "#=GC {:<tagw$} {}",
            "PP_cons",
            &msa.pp_cons[currpos..end],
            tagw = margin - 6
        )
        .unwrap();
        writeln!(
            out,
            "#=GC {:<tagw$} {}",
            "RF",
            &msa.rfline[currpos..end],
            tagw = margin - 6
        )
        .unwrap();
        currpos = end;
    }
    writeln!(out, "//").unwrap();
}

/// Emit the alignment in A2M format: consensus columns uppercase / `-`,
/// insert columns lowercase (insert gaps suppressed); `O`/`o` mapped to `X`/`x`
/// for amino alphabets.
pub(crate) fn write_a2m(out: &mut dyn Write, msa: &TextMsa, is_amino: bool) {
    for row in &msa.rows {
        // esl_msafile_a2m_Write (esl_msafile_a2m.c:418-420): ">name", then
        // " acc" if the sequence has an accession, then " desc" if it has a
        // description.
        write!(out, ">{}", row.name).unwrap();
        if let Some(acc) = row.acc.as_deref().filter(|acc| !acc.is_empty()) {
            write!(out, " {acc}").unwrap();
        }
        if let Some(desc) = row.desc.as_deref().filter(|desc| !desc.is_empty()) {
            write!(out, " {desc}").unwrap();
        }
        writeln!(out).unwrap();
        let mut seq = String::with_capacity(row.aseq.len());
        for (ch, rf) in row.aseq.chars().zip(msa.rfline.chars()) {
            let is_consensus = rf.is_ascii_alphanumeric();
            let out_ch = if is_consensus {
                if ch.is_ascii_alphabetic() {
                    let mut up = ch.to_ascii_uppercase();
                    if is_amino && up == 'O' {
                        up = 'X';
                    }
                    Some(up)
                } else {
                    Some('-')
                }
            } else if ch.is_ascii_alphabetic() {
                let mut low = ch.to_ascii_lowercase();
                if is_amino && low == 'o' {
                    low = 'x';
                }
                Some(low)
            } else {
                None
            };
            if let Some(ch) = out_ch {
                seq.push(ch);
            }
        }
        for chunk in seq.as_bytes().chunks(60) {
            writeln!(out, "{}", std::str::from_utf8(chunk).unwrap()).unwrap();
        }
    }
}

/// Easel version string, used in the CLUSTAL-like header line. Matches the
/// `EASEL_VERSION` substituted by `hmmer/configure.ac` (currently "0.49").
pub(super) const EASEL_VERSION: &str = "0.49";

/// Borrow the assembled `TextMsa` rows as the generic `msa::WriteRow` view used
/// by the reusable text-MSA writers in `src/msa.rs`.
pub(super) fn write_rows(msa: &TextMsa) -> Vec<msa::WriteRow<'_>> {
    msa.rows
        .iter()
        .map(|row| msa::WriteRow {
            name: &row.name,
            acc: row.acc.as_deref(),
            desc: row.desc.as_deref(),
            aseq: &row.aseq,
        })
        .collect()
}
