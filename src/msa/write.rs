/// A single aligned row to be serialized by the text-mode MSA writers below.
///
/// These writers operate on already-assembled text alignment rows (as produced
/// by `hmmalign`), independent of the [`Msa`] reader struct. `aseq` holds the
/// aligned text (consensus columns + insert columns, with gap characters).
pub struct WriteRow<'a> {
    pub name: &'a str,
    pub acc: Option<&'a str>,
    pub desc: Option<&'a str>,
    pub aseq: &'a str,
}

/// Write an aligned FASTA (AFA) alignment to `out`.
///
/// Faithful port of Easel's text-mode `esl_msafile_afa_Write`: each record is a
/// `>name [acc] [desc]` header followed by the aligned sequence wrapped at 60
/// columns per line. Aligned text (including gap characters) is emitted verbatim.
pub fn write_afa(out: &mut dyn std::io::Write, rows: &[WriteRow<'_>]) -> std::io::Result<()> {
    for row in rows {
        write!(out, ">{}", row.name)?;
        if let Some(acc) = row.acc {
            if !acc.is_empty() {
                write!(out, " {}", acc)?;
            }
        }
        if let Some(desc) = row.desc {
            if !desc.is_empty() {
                write!(out, " {}", desc)?;
            }
        }
        writeln!(out)?;
        for chunk in row.aseq.as_bytes().chunks(60) {
            out.write_all(chunk)?;
            writeln!(out)?;
        }
    }
    Ok(())
}

/// Write a CLUSTAL (or CLUSTAL-like) alignment to `out`.
///
/// Faithful port of Easel's text-mode `esl_msafile_clustal_Write`. Emits the
/// magic header (`CLUSTAL 2.1 ...` or `EASEL (...) ...`), then interleaved
/// blocks of 60 columns. Each block is preceded by a blank line, and each block
/// ends with a consensus line: in text mode (where the alphabet is unknown),
/// only `*` (a column with exactly one distinct alphabetic symbol) and ` `
/// appear, matching `make_text_consensus_line`.
pub fn write_clustal(
    out: &mut dyn std::io::Write,
    rows: &[WriteRow<'_>],
    clustallike: bool,
    easel_version: &str,
) -> std::io::Result<()> {
    let cpl = 60usize;
    let alen = rows.first().map(|r| r.aseq.len()).unwrap_or(0);
    let maxnamelen = rows.iter().map(|r| r.name.len()).max().unwrap_or(0);
    let consline = text_consensus_line(rows, alen);

    if clustallike {
        writeln!(out, "EASEL ({}) multiple sequence alignment", easel_version)?;
    } else {
        writeln!(out, "CLUSTAL 2.1 multiple sequence alignment")?;
    }

    let mut apos = 0usize;
    while apos < alen {
        let end = (apos + cpl).min(alen);
        writeln!(out)?;
        for row in rows {
            let seg = &row.aseq.as_bytes()[apos..end];
            write!(out, "{:<width$} ", row.name, width = maxnamelen)?;
            out.write_all(seg)?;
            writeln!(out)?;
        }
        write!(out, "{:<width$} ", "", width = maxnamelen)?;
        out.write_all(&consline[apos..end])?;
        writeln!(out)?;
        apos += cpl;
    }
    Ok(())
}

/// Build the text-mode CLUSTAL consensus line (`esl_msafile_clustal.c`
/// `make_text_consensus_line`). A column gets `*` iff exactly one distinct
/// uppercased A-Z symbol appears in it (no gaps or non-alphabetic symbols);
/// otherwise ` `.
fn text_consensus_line(rows: &[WriteRow<'_>], alen: usize) -> Vec<u8> {
    let mut v = vec![0u32; alen];
    for row in rows {
        for (apos, &ch) in row.aseq.as_bytes().iter().enumerate().take(alen) {
            let x = ch.to_ascii_uppercase() as i32 - b'A' as i32;
            if (0..26).contains(&x) {
                v[apos] |= 1 << x;
            } else {
                v[apos] |= 1 << 26;
            }
        }
    }
    let maxv = (1u32 << 26) - 1;
    let mut consline = vec![b' '; alen];
    for apos in 0..alen {
        let nbits = v[apos].count_ones();
        consline[apos] = if nbits == 1 && v[apos] < maxv {
            b'*'
        } else {
            b' '
        };
    }
    consline
}

/// Write a PSIBLAST alignment to `out`.
///
/// Faithful port of the text-mode `esl_msafile_psiblast_Write`: interleaved
/// blocks of 60 columns, `name  aseq` rows (names left-justified to a common
/// width followed by two spaces), blocks separated by a blank line.
///
/// Each column is classified as consensus or insert from `rf` (a column is
/// consensus iff `rf[col]` is alphanumeric; if `rf` is `None`, the first row is
/// used as the reference, matching Easel). Within each column an alphanumeric
/// symbol is uppercased in consensus columns and lowercased in insert columns;
/// every non-residue character becomes `-`.
pub fn write_psiblast(
    out: &mut dyn std::io::Write,
    rows: &[WriteRow<'_>],
    rf: Option<&str>,
) -> std::io::Result<()> {
    let cpl = 60usize;
    let alen = rows.first().map(|r| r.aseq.len()).unwrap_or(0);
    let maxnamelen = rows.iter().map(|r| r.name.len()).max().unwrap_or(0);
    let rf_bytes = rf.map(|s| s.as_bytes());
    let first = rows.first().map(|r| r.aseq.as_bytes());

    let is_consensus = |col: usize| -> bool {
        match rf_bytes {
            Some(rf) => rf.get(col).is_some_and(|c| c.is_ascii_alphanumeric()),
            None => first.is_some_and(|f| f.get(col).is_some_and(|c| c.is_ascii_alphanumeric())),
        }
    };

    let mut pos = 0usize;
    while pos < alen {
        let end = (pos + cpl).min(alen);
        for row in rows {
            let seq = row.aseq.as_bytes();
            let mut buf = Vec::with_capacity(end - pos);
            for (offset, &sym) in seq[pos..end].iter().enumerate() {
                let col = pos + offset;
                let is_residue = sym.is_ascii_alphanumeric();
                let ch = if is_consensus(col) {
                    if is_residue {
                        sym.to_ascii_uppercase()
                    } else {
                        b'-'
                    }
                } else if is_residue {
                    sym.to_ascii_lowercase()
                } else {
                    b'-'
                };
                buf.push(ch);
            }
            write!(out, "{:<width$}  ", row.name, width = maxnamelen)?;
            out.write_all(&buf)?;
            writeln!(out)?;
        }
        if end < alen {
            writeln!(out)?;
        }
        pos += cpl;
    }
    Ok(())
}

/// Write a SELEX alignment to `out`.
///
/// Faithful port of the text-mode core of `esl_msafile_selex_Write`: interleaved
/// blocks of 60 columns, each row written as `name aseq` with names
/// left-justified to a common width (minimum 4, to accommodate `#=RF` etc.).
/// Blocks after the first are separated by a blank line. An optional `rf`
/// reference line is written as `#=RF` at the top of each block.
pub fn write_selex(
    out: &mut dyn std::io::Write,
    rows: &[WriteRow<'_>],
    rf: Option<&str>,
) -> std::io::Result<()> {
    let cpl = 60usize;
    let alen = rows.first().map(|r| r.aseq.len()).unwrap_or(0);
    let maxnamelen = rows.iter().map(|r| r.name.len()).max().unwrap_or(0).max(4);

    let mut apos = 0usize;
    while apos < alen {
        let end = (apos + cpl).min(alen);
        if apos != 0 {
            writeln!(out)?;
        }
        if let Some(rf) = rf {
            write!(out, "{:<width$} ", "#=RF", width = maxnamelen)?;
            out.write_all(&rf.as_bytes()[apos..end.min(rf.len())])?;
            writeln!(out)?;
        }
        for row in rows {
            write!(out, "{:<width$} ", row.name, width = maxnamelen)?;
            out.write_all(&row.aseq.as_bytes()[apos..end])?;
            writeln!(out)?;
        }
        apos += cpl;
    }
    Ok(())
}

/// Write a PHYLIP alignment to `out` (interleaved if `sequential` is false,
/// sequential otherwise).
///
/// Faithful port of the text-mode `phylip_interleaved_Write` /
/// `phylip_sequential_Write`. Header is ` <nseq> <alen>`. Names are truncated /
/// padded to `namewidth` (strict PHYLIP = 10) and only appear on the first line
/// of each sequence. Residues per line default to 60. Text symbols are rectified
/// per `phylip_rectify_output_seq_text`: lowercase to uppercase, `.`/`_`/space to
/// `-`, and `~` to `?`.
pub fn write_phylip(
    out: &mut dyn std::io::Write,
    rows: &[WriteRow<'_>],
    sequential: bool,
) -> std::io::Result<()> {
    let rpl = 60usize;
    let namewidth = 10usize;
    let alen = rows.first().map(|r| r.aseq.len()).unwrap_or(0);

    let rectify = |seg: &[u8]| -> Vec<u8> {
        seg.iter()
            .map(|&ch| {
                let ch = ch.to_ascii_uppercase();
                match ch {
                    b'.' | b'_' | b' ' => b'-',
                    b'~' => b'?',
                    other => other,
                }
            })
            .collect()
    };
    let padded_name = |name: &str| -> Vec<u8> {
        let mut bytes = name.as_bytes().to_vec();
        bytes.truncate(namewidth);
        while bytes.len() < namewidth {
            bytes.push(b' ');
        }
        bytes
    };

    if sequential {
        writeln!(out, " {} {}", rows.len(), alen)?;
        for row in rows {
            let seq = row.aseq.as_bytes();
            let mut apos = 0usize;
            while apos < alen {
                let end = (apos + rpl).min(alen);
                let seg = rectify(&seq[apos..end]);
                if apos == 0 {
                    out.write_all(&padded_name(row.name))?;
                    write!(out, " ")?;
                }
                out.write_all(&seg)?;
                writeln!(out)?;
                apos += rpl;
            }
        }
    } else {
        write!(out, " {} {}", rows.len(), alen)?;
        let mut apos = 0usize;
        while apos < alen {
            let end = (apos + rpl).min(alen);
            writeln!(out)?;
            for row in rows {
                let seg = rectify(&row.aseq.as_bytes()[apos..end]);
                if apos == 0 {
                    out.write_all(&padded_name(row.name))?;
                    write!(out, " ")?;
                }
                out.write_all(&seg)?;
                writeln!(out)?;
            }
            apos += rpl;
        }
    }
    Ok(())
}
