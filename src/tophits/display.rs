use super::*;

/// Print one alignment display as ASCII blocks, matching C
/// `p7_nontranslated_alidisplay_Print()` (p7_alidisplay.c:1162). Splits the
/// alignment into blocks of `aliwidth = linewidth - namewidth - 2*coordwidth -
/// 5` (bounded below by `min_aliwidth=40`). When `linewidth==0` the whole
/// alignment is printed in one block.
pub fn print_alidisplay_blocks(
    out: &mut dyn std::io::Write,
    hmm_name: &str,
    seq_name: &str,
    ad: &AliDisplay,
    cs_line: Option<&str>,
    linewidth: usize,
) {
    // Back-compat wrapper: never prefers accessions (no --acc handling).
    print_alidisplay_blocks_acc(
        out, hmm_name, "", seq_name, "", ad, cs_line, linewidth, false,
    );
}

/// Like [`print_alidisplay_blocks`] but honors the `--acc` option
/// (C `p7_nontranslated_alidisplay_Print`'s `show_accessions` argument,
/// p7_alidisplay.c:1162,1176-1180): when `show_accessions` is set and the
/// corresponding accession is non-empty, the accession is shown in place of
/// the name and `namewidth` is derived from the chosen strings.
#[allow(clippy::too_many_arguments)]
pub fn print_alidisplay_blocks_acc(
    out: &mut dyn std::io::Write,
    hmm_name: &str,
    hmm_acc: &str,
    seq_name: &str,
    seq_acc: &str,
    ad: &AliDisplay,
    cs_line: Option<&str>,
    linewidth: usize,
    show_accessions: bool,
) {
    let n = ad.model.chars().count();
    if n == 0 {
        return;
    }
    // Mirror C p7_alidisplay.c:1176-1180: with --acc, prefer the accession
    // when one is present; otherwise fall back to the name. namewidth is then
    // derived from whichever string is actually shown.
    let hmm_name = if show_accessions && !hmm_acc.is_empty() {
        hmm_acc
    } else {
        hmm_name
    };
    let seq_name = if show_accessions && !seq_acc.is_empty() {
        seq_acc
    } else {
        seq_name
    };
    let namewidth = hmm_name.len().max(seq_name.len());
    let coordwidth = [ad.hmmfrom, ad.hmmto, ad.sqfrom, ad.sqto]
        .iter()
        .map(|v| v.to_string().len())
        .max()
        .unwrap_or(1);
    let mut aliwidth = if linewidth > 0 {
        linewidth.saturating_sub(namewidth + 2 * coordwidth + 5)
    } else {
        n
    };
    if aliwidth < n && aliwidth < 40 {
        aliwidth = 40;
    }
    if aliwidth == 0 {
        aliwidth = n;
    }

    let model_chars: Vec<char> = ad.model.chars().collect();
    let mline_chars: Vec<char> = ad.mline.chars().collect();
    let aseq_chars: Vec<char> = ad.aseq.chars().collect();
    let pp_chars: Vec<char> = ad.ppline.chars().collect();
    let rf_chars: Vec<char> = ad.rfline.chars().collect();
    let cs_chars: Option<Vec<char>> = cs_line.map(|s| s.chars().collect());

    let reverse = ad.sqfrom > ad.sqto;
    let mut k1 = ad.hmmfrom as i64;
    let mut i1 = ad.sqfrom as i64;

    let indent_w = namewidth + coordwidth + 1;
    let mut first_block = true;
    let mut pos = 0;
    while pos < n {
        if !first_block {
            writeln!(out).unwrap();
        }
        first_block = false;
        let end = (pos + aliwidth).min(n);
        let mut nk = 0i64;
        let mut ni = 0i64;
        for z in pos..end {
            if model_chars[z] != '.' {
                nk += 1;
            }
            if aseq_chars[z] != '-' {
                ni += 1;
            }
        }
        let k2 = k1 + nk - 1;
        let i2 = if reverse { i1 - ni + 1 } else { i1 + ni - 1 };

        if let Some(ref cs) = cs_chars {
            let chunk: String = cs[pos..end.min(cs.len())].iter().collect();
            writeln!(out, "  {:>indent_w$} {} CS", "", chunk, indent_w = indent_w).unwrap();
        }
        if !rf_chars.is_empty() {
            let chunk: String = rf_chars[pos..end.min(rf_chars.len())].iter().collect();
            writeln!(out, "  {:>indent_w$} {} RF", "", chunk, indent_w = indent_w).unwrap();
        }
        {
            let chunk: String = model_chars[pos..end].iter().collect();
            writeln!(
                out,
                "  {:>nw$} {:>cw$} {} {:<cw$}",
                hmm_name,
                k1,
                chunk,
                k2,
                nw = namewidth,
                cw = coordwidth
            )
            .unwrap();
        }
        {
            let chunk: String = mline_chars[pos..end.min(mline_chars.len())]
                .iter()
                .collect();
            writeln!(out, "  {:>indent_w$} {}", "", chunk, indent_w = indent_w).unwrap();
        }
        if ni > 0 {
            let chunk: String = aseq_chars[pos..end].iter().collect();
            writeln!(
                out,
                "  {:>nw$} {:>cw$} {} {:<cw$}",
                seq_name,
                i1,
                chunk,
                i2,
                nw = namewidth,
                cw = coordwidth
            )
            .unwrap();
        } else {
            let chunk: String = aseq_chars[pos..end].iter().collect();
            writeln!(
                out,
                "  {:>nw$} {:>cw$} {} {:>cw$}",
                seq_name,
                "-",
                chunk,
                "-",
                nw = namewidth,
                cw = coordwidth
            )
            .unwrap();
        }
        if !pp_chars.is_empty() {
            let chunk: String = pp_chars[pos..end.min(pp_chars.len())].iter().collect();
            writeln!(out, "  {:>indent_w$} {} PP", "", chunk, indent_w = indent_w).unwrap();
        }

        k1 += nk;
        if reverse {
            i1 -= ni;
        } else {
            i1 += ni;
        }
        pos = end;
    }
}
