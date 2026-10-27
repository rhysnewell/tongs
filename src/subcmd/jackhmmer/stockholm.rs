use super::*;

/// Render an in-memory `Msa` as a Stockholm 1.0 file, emitting `#=GF`,
/// per-sequence `#=GS` descriptions, residue rows, per-row `#=GR ... PP`,
/// and `#=GC PP_cons`/`#=GC RF` consensus annotation.
///
/// Column widths are computed exactly as Easel's `stockholm_write()`
/// (hmmer/easel/esl_msafile_stockholm.c:1069): the left margin is
/// `max(maxname+1, maxgc+6, maxname+maxgr+7)`, `#=GF` tags pad to `maxgf`
/// (>=2), `#=GS` name fields to `maxname`, residue-row names to `margin-1`,
/// `#=GR` to `maxname` + a `margin-maxname-7` tag field, and `#=GC` to
/// `margin-6`. This is written as a single unwrapped block, i.e. equivalent to
/// Easel's Pfam format (`cpl == alen`); Easel's default Stockholm wraps at 200
/// aligned residues per block, so for alignments wider than 200 columns the
/// block layout would differ (the per-line annotation is identical).
///
/// When `fold_inserts_to_match` is `true`, residue rows are uppercased and
/// insert-gap '.' characters are rewritten to '-'. This reproduces how
/// `jackhmmer` renders its `-A`/checkpoint MSAs: jackhmmer's MSA carries the
/// real Viterbi-traced query (`p7_tophits_Alignment(th, abc, &qsq, &qtr, 1,
/// ...)`), whose presence collapses the round's insert columns into the
/// model's consensus (match) columns, so C emits '-'/uppercase there. phmmer
/// and hmmsearch build the same MSA without the query (`inc_n == 0`) and keep
/// the raw insert representation ('.'/lowercase), so they pass `false`.
pub(crate) fn write_stockholm_msa(out: &mut dyn Write, msa: &Msa) {
    write_stockholm_msa_inner(out, msa, true, msa.alen.max(1))
}

/// C's `esl_msafile_stockholm_Write()` block width: 200 aligned columns per
/// block for `eslMSAFILE_STOCKHOLM`, one unwrapped block (`cpl == alen`) for
/// `eslMSAFILE_PFAM` (esl_msafile_stockholm.c:355-360). The search programs
/// pick the format from `textw`: Stockholm when `textw > 0`, Pfam under
/// `--notextw` (hmmsearch.c:566-567, phmmer.c:630-631, jackhmmer.c:723-724).
pub(crate) fn stockholm_cpl_for_textw(textw: usize, alen: usize) -> usize {
    if textw > 0 {
        200
    } else {
        alen.max(1)
    }
}

/// As [`write_stockholm_msa`], with the block width chosen from `textw` the
/// way C's `-A` output does.
pub(crate) fn write_stockholm_msa_for_textw(out: &mut dyn Write, msa: &Msa, textw: usize) {
    write_stockholm_msa_inner(out, msa, true, stockholm_cpl_for_textw(textw, msa.alen))
}

/// Same Stockholm writer, but emitting the raw `p7_tophits_Alignment` text
/// (insert columns kept as '.'/lowercase). Used by phmmer and hmmsearch `-A`,
/// which build the MSA without the query (`inc_n == 0`) and so match C's
/// `esl_msafile_Write` byte-for-byte without any insert folding. `textw`
/// selects Stockholm (200-column blocks) or Pfam (unwrapped) as in C.
pub(crate) fn write_tophits_alignment_msa_stockholm(out: &mut dyn Write, msa: &Msa, textw: usize) {
    write_stockholm_msa_inner(out, msa, false, stockholm_cpl_for_textw(textw, msa.alen))
}

fn write_stockholm_msa_inner(
    out: &mut dyn Write,
    msa: &Msa,
    fold_inserts_to_match: bool,
    cpl: usize,
) {
    let maxname = msa.sqname.iter().map(|name| name.len()).max().unwrap_or(0);
    // maxgf: longest #=GF tag. We only ever emit ID/AC/DE/AU, all 2 chars,
    // and Easel floors maxgf at 2.
    let maxgf = 2usize;
    // maxgc: PP_cons => 7, RF => 2.
    let mut maxgc = 0usize;
    if msa.pp_cons.is_some() {
        maxgc = maxgc.max(7);
    }
    if msa.rf.is_some() {
        maxgc = maxgc.max(2);
    }
    // maxgr: PP rows => 2.
    let maxgr = if msa.pp.iter().any(|pp| pp.is_some()) {
        2usize
    } else {
        0
    };
    let mut margin = maxname + 1;
    if maxgc > 0 && maxgc + 6 > margin {
        margin = maxgc + 6;
    }
    if maxgr > 0 && maxname + maxgr + 7 > margin {
        margin = maxname + maxgr + 7;
    }

    writeln!(out, "# STOCKHOLM 1.0").unwrap();
    if !msa.name.is_empty() {
        writeln!(out, "#=GF {:<maxgf$} {}", "ID", msa.name).unwrap();
    }
    if let Some(acc) = &msa.acc {
        if !acc.is_empty() {
            writeln!(out, "#=GF {:<maxgf$} {}", "AC", acc).unwrap();
        }
    }
    if let Some(desc) = &msa.desc {
        if !desc.is_empty() {
            writeln!(out, "#=GF {:<maxgf$} {}", "DE", desc).unwrap();
        }
    }
    if let Some(author) = &msa.author {
        if !author.is_empty() {
            writeln!(out, "#=GF {:<maxgf$} {}", "AU", author).unwrap();
        }
    }
    writeln!(out).unwrap();

    // GS section, in Easel's order: AC block, then DE block, each followed by
    // a blank line (esl_msafile_stockholm.c:1183-1199). C emits the AC block
    // whenever the msa has an sqacc array, which esl_msa_SetSeqAccession
    // allocates on the first sequence that carries an accession.
    if !msa.sqacc.iter().all(|acc| acc.is_empty()) {
        for (name, acc) in msa.sqname.iter().zip(msa.sqacc.iter()) {
            if !acc.is_empty() {
                writeln!(out, "#=GS {:<maxname$} AC {}", name, acc).unwrap();
            }
        }
        writeln!(out).unwrap();
    }
    if !msa.sqdesc.iter().all(|desc| desc.is_empty()) {
        for (name, desc) in msa.sqname.iter().zip(msa.sqdesc.iter()) {
            if !desc.is_empty() {
                writeln!(out, "#=GS {:<maxname$} DE {}", name, desc).unwrap();
            }
        }
        writeln!(out).unwrap();
    }

    let gr_tag_width = margin.saturating_sub(maxname + 7);
    let cpl = cpl.max(1);
    let mut pos = 0usize;
    while pos < msa.alen {
        let end = (pos + cpl).min(msa.alen);
        for ((name, row), pp) in msa.sqname.iter().zip(msa.aseq.iter()).zip(msa.pp.iter()) {
            // The builder (`tophits::included_alignment`) emits Easel's digital-MSA
            // text form, where match-column gaps are '-', insert-column gaps are
            // '.', and inserted residues are lowercase. C's `esl_msafile_Write`
            // textizes those bytes unchanged (phmmer/hmmsearch path). jackhmmer's
            // MSA instead collapses inserts into the consensus, so we optionally
            // uppercase and fold '.'->'-' to match its C rendering.
            let rendered: std::borrow::Cow<'_, [u8]> = if fold_inserts_to_match {
                std::borrow::Cow::Owned(
                    row[pos..end]
                        .iter()
                        .map(|&ch| match ch {
                            b'.' => b'-',
                            b'a'..=b'z' => ch.to_ascii_uppercase(),
                            _ => ch,
                        })
                        .collect(),
                )
            } else {
                std::borrow::Cow::Borrowed(&row[pos..end])
            };
            writeln!(
                out,
                "{:<width$} {}",
                name,
                String::from_utf8_lossy(&rendered),
                width = margin - 1
            )
            .unwrap();
            if let Some(pp) = pp {
                writeln!(
                    out,
                    "#=GR {:<maxname$} {:<gr_tag_width$} {}",
                    name,
                    "PP",
                    String::from_utf8_lossy(&pp[pos..end]),
                )
                .unwrap();
            }
        }
        if let Some(pp_cons) = &msa.pp_cons {
            writeln!(
                out,
                "#=GC {:<width$} {}",
                "PP_cons",
                String::from_utf8_lossy(&pp_cons[pos..end]),
                width = margin - 6
            )
            .unwrap();
        }
        if let Some(rf) = &msa.rf {
            writeln!(
                out,
                "#=GC {:<width$} {}",
                "RF",
                String::from_utf8_lossy(&rf[pos..end]),
                width = margin - 6
            )
            .unwrap();
        }
        pos = end;
        if pos < msa.alen {
            writeln!(out).unwrap();
        }
    }
    writeln!(out, "//").unwrap();
}
