use super::*;

/// Read all Stockholm-format alignments from an open reader.
///
/// Scans for `# STOCKHOLM` block headers and dispatches each block (terminated
/// by `//`) to `parse_stockholm_block`. Concatenated multi-MSA files are
/// supported. Lightweight Rust port of the Stockholm subset of
/// `esl_msafile_stockholm.c`.
pub fn read_stockholm_from_reader<R: Read>(reader: BufReader<R>) -> HmmerResult<Vec<Msa>> {
    Ok(read_stockholm_preserved_from_reader(reader)?
        .into_iter()
        .map(|stockholm| stockholm.msa)
        .collect())
}

/// Read all Stockholm-format alignments from an open reader, preserving body
/// lines for writers that need to round-trip annotations.
pub fn read_stockholm_preserved_from_reader<R: Read>(
    mut reader: BufReader<R>,
) -> HmmerResult<Vec<StockholmMsa>> {
    let mut msas = Vec::new();
    let text = read_text_msa_to_string(&mut reader)?;
    let lines: Vec<String> = text.lines().map(str::to_owned).collect();

    let mut i = 0;
    while i < lines.len() {
        if lines[i].starts_with("# STOCKHOLM") {
            let start = i;
            // Find end
            while i < lines.len() && lines[i].trim() != "//" {
                i += 1;
            }
            if i == lines.len() {
                return Err(HmmerError::Format(
                    "missing // terminator after MSA".to_string(),
                ));
            }
            let end = i;
            if let Some(stockholm) = parse_stockholm_block(&lines[start..=end])? {
                msas.push(stockholm);
            }
        } else {
            let trimmed = lines[i].trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                return Err(HmmerError::Format(format!(
                    "Expected Stockholm header, got: {}",
                    trimmed
                )));
            }
        }
        i += 1;
    }

    Ok(msas)
}
pub(super) fn parse_stockholm_cutoffs(tag: &str, value: &str) -> HmmerResult<[f32; 2]> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    let Some(first) = parts.first() else {
        return Err(HmmerError::Format(format!(
            "Missing Stockholm #=GF {tag} cutoff value"
        )));
    };
    let first = parse_stockholm_cutoff_value(tag, first)?;
    let second = match parts.get(1) {
        Some(value) => parse_stockholm_cutoff_value(tag, value)?,
        None => first,
    };
    Ok([first, second])
}

fn parse_stockholm_cutoff_value(tag: &str, value: &str) -> HmmerResult<f32> {
    value
        .trim_end_matches(';')
        .parse()
        .map_err(|_| HmmerError::Format(format!("Bad Stockholm #=GF {tag} cutoff value: {value}")))
}

pub(super) fn validate_pp_annotation(line: &[u8], label: &str) -> HmmerResult<()> {
    if let Some(&bad) = line
        .iter()
        .find(|&&ch| !(ch == b'.' || ch == b'*' || ch.is_ascii_digit()))
    {
        return Err(HmmerError::Format(format!(
            "Stockholm {} annotation contains invalid PP character '{}'",
            label, bad as char
        )));
    }
    Ok(())
}
