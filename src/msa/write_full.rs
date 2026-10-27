use super::*;

/// Serialize a [`FullStockholm`] to `out`, byte-for-byte matching Easel's
/// `stockholm_write(fp, msa, cpl)` (`esl_msafile_stockholm.c`).
///
/// `cpl` is the characters-per-line block width (200 for `eslMSAFILE_STOCKHOLM`,
/// as `esl_msafile_Write` passes for `alimask`).
///
/// If `abc` is `Some`, the aligned sequences are textized through it (the
/// digital-mode path, line 1231: `esl_abc_TextizeN`): each residue is mapped to
/// its canonical symbol and every gap character (`.`, `_`, `-`) normalizes to
/// `-`. This matches `alimask`, which opens the alignment with an autodetected
/// alphabet (digital mode). If `abc` is `None`, sequences are written verbatim
/// (text mode, line 1232). Annotation lines (`#=GR`/`#=GC`) are always text and
/// never normalized, exactly as in C.
pub fn write_stockholm_full(
    out: &mut dyn std::io::Write,
    msa: &FullStockholm,
    cpl: usize,
    abc: Option<&crate::alphabet::Alphabet>,
) -> std::io::Result<()> {
    // In digital mode, sequences are textized through the alphabet: residues
    // map to canonical symbols and all gap chars normalize to '-'. Annotation
    // (#=GR/#=GC) stays text in either mode, so we only transform aseq.
    let textized: Vec<Vec<u8>> = match abc {
        Some(abc) => msa
            .aseq
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&ch| {
                        if ch == b'-' || ch == b'.' || ch == b'_' {
                            abc.sym[abc.gap_code() as usize]
                        } else {
                            let code = abc.digitize_symbol(ch);
                            if code == crate::alphabet::DSQ_ILLEGAL
                                || code == crate::alphabet::DSQ_IGNORED
                            {
                                ch
                            } else {
                                abc.sym[code as usize]
                            }
                        }
                    })
                    .collect()
            })
            .collect(),
        None => msa.aseq.clone(),
    };

    // Unique-name check (esl_msa_CheckUniqueNames). If names collide, C forces
    // a "<i>|" prefix of width uniqwidth.
    let mut uniqwidth = 0usize;
    let make_unique = {
        let mut seen = HashSet::new();
        let dup = msa.sqname.iter().any(|n| !seen.insert(n.as_str()));
        if dup {
            let mut t = msa.sqname.len();
            while t != 0 {
                uniqwidth += 1;
                t /= 10;
            }
            uniqwidth += 1; // includes the '|'
        }
        dup
    };

    let maxname = max_width(msa.sqname.iter().map(|n| n.len()));

    let mut maxgf = max_width(msa.gf.iter().map(|(t, _)| t.len()));
    if maxgf < 2 {
        maxgf = 2;
    }

    let mut maxgc = max_width(msa.gc.iter().map(|(t, _)| t.len()));
    if (msa.rf.is_some() || msa.mm.is_some()) && maxgc < 2 {
        maxgc = 2;
    }
    if (msa.ss_cons.is_some() || msa.sa_cons.is_some() || msa.pp_cons.is_some()) && maxgc < 7 {
        maxgc = 7;
    }

    let mut maxgr = max_width(msa.gr.iter().map(|(t, _)| t.len()));
    let any_ss = msa.ss.iter().any(Option::is_some);
    let any_sa = msa.sa.iter().any(Option::is_some);
    let any_pp = msa.pp.iter().any(Option::is_some);
    if (any_ss || any_sa || any_pp) && maxgr < 2 {
        maxgr = 2;
    }

    let mut margin = uniqwidth + maxname + 1;
    if maxgc > 0 && maxgc + 6 > margin {
        margin = maxgc + 6;
    }
    if maxgr > 0 && uniqwidth + maxname + maxgr + 7 > margin {
        margin = uniqwidth + maxname + maxgr + 7;
    }

    // Helper: write a left-justified seqname (with optional uniqizing prefix).
    let write_name =
        |out: &mut dyn std::io::Write, i: usize, field: usize| -> std::io::Result<()> {
            if make_unique {
                // "%0*d|%-*s" with uniqwidth-1 and field
                write!(
                    out,
                    "{:0width$}|{:<field$}",
                    i,
                    msa.sqname[i],
                    width = uniqwidth - 1,
                    field = field
                )
            } else {
                write!(out, "{:<field$}", msa.sqname[i], field = field)
            }
        };

    writeln!(out, "# STOCKHOLM 1.0")?;
    if make_unique {
        writeln!(
            out,
            "# WARNING: seq names have been made unique by adding a prefix of \"<seq#>|\""
        )?;
    }

    // Comments
    for c in &msa.comments {
        writeln!(out, "#{c}")?;
    }
    if !msa.comments.is_empty() {
        writeln!(out)?;
    }

    // GF section
    if let Some(name) = &msa.name {
        writeln!(out, "#=GF {:<maxgf$} {}", "ID", name)?;
    }
    if let Some(acc) = &msa.acc {
        writeln!(out, "#=GF {:<maxgf$} {}", "AC", acc)?;
    }
    if let Some(desc) = &msa.desc {
        writeln!(out, "#=GF {:<maxgf$} {}", "DE", desc)?;
    }
    if let Some(au) = &msa.au {
        writeln!(out, "#=GF {:<maxgf$} {}", "AU", au)?;
    }
    write_cutoff(out, "GA", msa.ga, maxgf)?;
    write_cutoff(out, "NC", msa.nc, maxgf)?;
    write_cutoff(out, "TC", msa.tc, maxgf)?;
    for (tag, value) in &msa.gf {
        writeln!(out, "#=GF {:<maxgf$} {}", tag, value)?;
    }
    writeln!(out)?;

    // GS section
    if msa.has_wgts {
        for i in 0..msa.sqname.len() {
            write!(out, "#=GS ")?;
            write_name(out, i, maxname)?;
            writeln!(out, " WT {:.2}", msa.wgt[i])?;
        }
        writeln!(out)?;
    }
    if msa.sqacc.iter().any(Option::is_some) {
        for i in 0..msa.sqname.len() {
            if let Some(acc) = &msa.sqacc[i] {
                write!(out, "#=GS ")?;
                write_name(out, i, maxname)?;
                writeln!(out, " AC {}", acc)?;
            }
        }
        writeln!(out)?;
    }
    if msa.sqdesc.iter().any(Option::is_some) {
        for i in 0..msa.sqname.len() {
            if let Some(de) = &msa.sqdesc[i] {
                write!(out, "#=GS ")?;
                write_name(out, i, maxname)?;
                writeln!(out, " DE {}", de)?;
            }
        }
        writeln!(out)?;
    }
    for (tag, vals) in &msa.gs {
        let gslen = tag.len();
        for (i, v) in vals.iter().enumerate() {
            if let Some(v) = v {
                for tok in v.split('\n') {
                    write!(out, "#=GS ")?;
                    write_name(out, i, maxname)?;
                    writeln!(out, " {:<gslen$} {}", tag, tok)?;
                }
            }
        }
        writeln!(out)?;
    }

    // Alignment blocks
    let alen = msa.alen;
    let mut currpos = 0usize;
    let mut first_block = true;
    while currpos < alen {
        let acpl = (alen - currpos).min(cpl);
        let end = currpos + acpl;
        if !first_block {
            writeln!(out)?;
        }
        first_block = false;

        for i in 0..msa.sqname.len() {
            write_name(out, i, margin - uniqwidth - 1)?;
            out.write_all(b" ")?;
            out.write_all(&textized[i][currpos..end])?;
            writeln!(out)?;

            write_gr(
                out,
                msa,
                i,
                "SS",
                &msa.ss[i],
                maxname,
                margin,
                uniqwidth,
                currpos,
                end,
                make_unique,
            )?;
            write_gr(
                out,
                msa,
                i,
                "SA",
                &msa.sa[i],
                maxname,
                margin,
                uniqwidth,
                currpos,
                end,
                make_unique,
            )?;
            write_gr(
                out,
                msa,
                i,
                "PP",
                &msa.pp[i],
                maxname,
                margin,
                uniqwidth,
                currpos,
                end,
                make_unique,
            )?;
            for (tag, vals) in &msa.gr {
                write_gr(
                    out,
                    msa,
                    i,
                    tag,
                    &vals[i],
                    maxname,
                    margin,
                    uniqwidth,
                    currpos,
                    end,
                    make_unique,
                )?;
            }
        }

        write_gc(out, "SS_cons", &msa.ss_cons, margin, currpos, end)?;
        write_gc(out, "SA_cons", &msa.sa_cons, margin, currpos, end)?;
        write_gc(out, "PP_cons", &msa.pp_cons, margin, currpos, end)?;
        write_gc(out, "RF", &msa.rf, margin, currpos, end)?;
        write_gc(out, "MM", &msa.mm, margin, currpos, end)?;
        for (tag, value) in &msa.gc {
            write_gc_bytes(out, tag, value, margin, currpos, end)?;
        }

        currpos += cpl;
    }
    writeln!(out, "//")?;
    Ok(())
}

fn write_cutoff(
    out: &mut dyn std::io::Write,
    tag: &str,
    cut: Option<(f32, Option<f32>)>,
    maxgf: usize,
) -> std::io::Result<()> {
    if let Some((a, b)) = cut {
        match b {
            Some(b) => writeln!(out, "#=GF {:<maxgf$} {:.1} {:.1}", tag, a, b),
            None => writeln!(out, "#=GF {:<maxgf$} {:.1}", tag, a),
        }
    } else {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn write_gr(
    out: &mut dyn std::io::Write,
    msa: &FullStockholm,
    i: usize,
    tag: &str,
    data: &Option<Vec<u8>>,
    maxname: usize,
    margin: usize,
    uniqwidth: usize,
    currpos: usize,
    end: usize,
    make_unique: bool,
) -> std::io::Result<()> {
    let Some(data) = data else { return Ok(()) };
    let tagfield = margin - maxname - uniqwidth - 7;
    write!(out, "#=GR ")?;
    if make_unique {
        write!(
            out,
            "{:0width$}|{:<field$}",
            i,
            msa.sqname[i],
            width = uniqwidth - 1,
            field = maxname
        )?;
    } else {
        write!(out, "{:<maxname$}", msa.sqname[i])?;
    }
    write!(out, " {:<tagfield$} ", tag)?;
    out.write_all(&data[currpos..end])?;
    writeln!(out)
}

fn write_gc(
    out: &mut dyn std::io::Write,
    tag: &str,
    data: &Option<Vec<u8>>,
    margin: usize,
    currpos: usize,
    end: usize,
) -> std::io::Result<()> {
    match data {
        Some(d) => write_gc_bytes(out, tag, d, margin, currpos, end),
        None => Ok(()),
    }
}

fn write_gc_bytes(
    out: &mut dyn std::io::Write,
    tag: &str,
    data: &[u8],
    margin: usize,
    currpos: usize,
    end: usize,
) -> std::io::Result<()> {
    let field = margin - 6;
    write!(out, "#=GC {:<field$} ", tag)?;
    out.write_all(&data[currpos..end])?;
    writeln!(out)
}
