use super::*;

pub(super) fn parse_interleaved_phylip_alignment(
    text: &str,
    alignment_name: String,
) -> HmmerResult<Msa> {
    let (nseq, alen, records) = parse_phylip_lines(text)?;
    let mut sqname = Vec::with_capacity(nseq);
    let mut sqdesc = Vec::with_capacity(nseq);
    let mut aseq = vec![Vec::<u8>::new(); nseq];

    for (row, (line_idx, line)) in records.into_iter().enumerate() {
        if row < nseq {
            let (name, chunk) = parse_named_phylip_row(line, line_idx)?;
            sqname.push(name.to_string());
            sqdesc.push(String::new());
            aseq[row].extend_from_slice(chunk.as_bytes());
        } else {
            let seq_idx = row % nseq;
            let chunk = parse_interleaved_phylip_continuation(line, &sqname[seq_idx]);
            if !chunk.is_empty() {
                aseq[seq_idx].extend_from_slice(chunk.as_bytes());
            }
        }
    }

    if sqname.len() != nseq {
        return Err(HmmerError::Format(format!(
            "PHYLIP alignment declares {nseq} sequences but contains {}",
            sqname.len()
        )));
    }
    msa_from_rows_with_alen("PHYLIP", alignment_name, sqname, sqdesc, aseq, alen)
}

pub(super) fn parse_sequential_phylip_alignment(
    text: &str,
    alignment_name: String,
) -> HmmerResult<Msa> {
    let (nseq, alen, records) = parse_phylip_lines(text)?;
    let mut iter = records.into_iter();
    let mut sqname = Vec::with_capacity(nseq);
    let mut sqdesc = Vec::with_capacity(nseq);
    let mut aseq = Vec::with_capacity(nseq);

    for seq_idx in 0..nseq {
        let Some((line_idx, line)) = iter.next() else {
            return Err(HmmerError::Format(format!(
                "PHYLIPS alignment declares {nseq} sequences but contains {seq_idx}"
            )));
        };
        let (name, first_chunk) = parse_named_phylip_row(line, line_idx)?;
        let mut seq = first_chunk.as_bytes().to_vec();
        while seq.len() < alen {
            let Some((_cont_idx, continuation)) = iter.next() else {
                return Err(HmmerError::Format(format!(
                    "PHYLIPS sequence {name} has aligned length {}, expected {alen}",
                    seq.len()
                )));
            };
            seq.extend_from_slice(strip_phylip_sequence_spaces(continuation).as_bytes());
        }
        sqname.push(name.to_string());
        sqdesc.push(String::new());
        aseq.push(seq);
    }

    if let Some((line_idx, _)) = iter.next() {
        return Err(HmmerError::Format(format!(
            "PHYLIPS alignment has extra data after {nseq} sequences at line {}",
            line_idx + 1
        )));
    }
    msa_from_rows_with_alen("PHYLIPS", alignment_name, sqname, sqdesc, aseq, alen)
}

pub(super) type PhylipRows<'a> = Vec<(usize, &'a str)>;

fn parse_phylip_lines(text: &str) -> HmmerResult<(usize, usize, PhylipRows<'_>)> {
    let mut lines = text.lines().enumerate().filter_map(|(idx, line)| {
        let trimmed = line.trim();
        (!trimmed.is_empty()).then_some((idx, trimmed))
    });
    let Some((header_idx, header)) = lines.next() else {
        return Err(HmmerError::Format(
            "PHYLIP alignment contains no records".to_string(),
        ));
    };
    let mut fields = header.split_whitespace();
    let nseq = fields
        .next()
        .ok_or_else(|| {
            HmmerError::Format(format!(
                "PHYLIP header at line {} is missing nseq",
                header_idx + 1
            ))
        })?
        .parse::<usize>()
        .map_err(|_| {
            HmmerError::Format(format!(
                "PHYLIP header at line {} has invalid nseq",
                header_idx + 1
            ))
        })?;
    let alen = fields
        .next()
        .ok_or_else(|| {
            HmmerError::Format(format!(
                "PHYLIP header at line {} is missing alen",
                header_idx + 1
            ))
        })?
        .parse::<usize>()
        .map_err(|_| {
            HmmerError::Format(format!(
                "PHYLIP header at line {} has invalid alen",
                header_idx + 1
            ))
        })?;
    if nseq == 0 || alen == 0 {
        return Err(HmmerError::Format(
            "PHYLIP header declares zero sequences or columns".to_string(),
        ));
    }
    Ok((nseq, alen, lines.collect()))
}

fn parse_named_phylip_row(line: &str, line_idx: usize) -> HmmerResult<(&str, String)> {
    let mut fields = line.split_whitespace();
    let name = fields.next().ok_or_else(|| {
        HmmerError::Format(format!("PHYLIP row at line {} has no name", line_idx + 1))
    })?;
    let chunk: String = fields.collect();
    if chunk.is_empty() {
        return Err(HmmerError::Format(format!(
            "PHYLIP row at line {} is missing sequence data",
            line_idx + 1
        )));
    }
    Ok((name, chunk))
}

fn parse_interleaved_phylip_continuation(line: &str, expected_name: &str) -> String {
    let mut fields = line.split_whitespace();
    if fields.next() == Some(expected_name) {
        fields.collect()
    } else {
        strip_phylip_sequence_spaces(line)
    }
}

fn strip_phylip_sequence_spaces(line: &str) -> String {
    line.split_whitespace().collect()
}
