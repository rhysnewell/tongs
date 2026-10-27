use super::*;

pub(super) fn parse_afa_alignment(text: &str, alignment_name: String) -> HmmerResult<Msa> {
    let mut sqname = Vec::new();
    let mut sqdesc = Vec::new();
    let mut aseq: Vec<Vec<u8>> = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_desc = String::new();
    let mut current_seq = Vec::new();

    for (line_idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(';') {
            continue;
        }
        if let Some(header) = trimmed.strip_prefix('>') {
            if header.trim().is_empty() {
                return Err(HmmerError::Format(format!(
                    "aligned FASTA record at line {} has an empty name",
                    line_idx + 1
                )));
            }
            flush_afa_record(
                &mut current_name,
                &mut current_desc,
                &mut current_seq,
                &mut sqname,
                &mut sqdesc,
                &mut aseq,
            )?;
            let mut fields = header.trim().splitn(2, char::is_whitespace);
            current_name = fields.next().map(str::to_string);
            current_desc = fields.next().unwrap_or("").trim().to_string();
            continue;
        }
        if current_name.is_none() {
            return Err(HmmerError::Format(format!(
                "aligned FASTA sequence data before first header at line {}",
                line_idx + 1
            )));
        }
        for raw in trimmed.bytes() {
            if !raw.is_ascii_whitespace() {
                current_seq.push(raw);
            }
        }
    }

    flush_afa_record(
        &mut current_name,
        &mut current_desc,
        &mut current_seq,
        &mut sqname,
        &mut sqdesc,
        &mut aseq,
    )?;
    if sqname.is_empty() {
        return Err(HmmerError::Format(
            "aligned FASTA alignment contains no records".to_string(),
        ));
    }
    let alen = aseq[0].len();
    for (name, seq) in sqname.iter().zip(&aseq) {
        if seq.len() != alen {
            return Err(HmmerError::Format(format!(
                "aligned FASTA sequence {name} has aligned length {}, expected {alen}",
                seq.len()
            )));
        }
    }
    let nseq = sqname.len();
    Ok(Msa {
        name: alignment_name,
        acc: None,
        desc: None,
        author: None,
        sqname,
        sqacc: vec![String::new(); nseq],
        sqdesc,
        weights: None,
        aseq,
        pp: vec![None; nseq],
        nseq,
        alen,
        rf: None,
        mm: None,
        ss_cons: None,
        sa_cons: None,
        pp_cons: None,
    })
}

pub(super) struct A2mRecord {
    pub(super) name: String,
    pub(super) desc: String,
    pub(super) inserts: Vec<Vec<u8>>,
    pub(super) consensus: Vec<u8>,
}

pub(super) fn parse_a2m_alignment(text: &str, alignment_name: String) -> HmmerResult<Msa> {
    let mut records = Vec::<A2mRecord>::new();
    let mut current_name: Option<String> = None;
    let mut current_desc = String::new();
    let mut current_raw = Vec::new();

    for (line_idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with(';') {
            continue;
        }
        if let Some(header) = trimmed.strip_prefix('>') {
            if header.trim().is_empty() {
                return Err(HmmerError::Format(format!(
                    "A2M record at line {} has an empty name",
                    line_idx + 1
                )));
            }
            flush_a2m_record(
                &mut current_name,
                &mut current_desc,
                &mut current_raw,
                &mut records,
            )?;
            let mut fields = header.trim().splitn(2, char::is_whitespace);
            current_name = fields.next().map(str::to_string);
            current_desc = fields.next().unwrap_or("").trim().to_string();
            continue;
        }
        if current_name.is_none() {
            return Err(HmmerError::Format(format!(
                "A2M sequence data before first header at line {}",
                line_idx + 1
            )));
        }
        for raw in trimmed.bytes() {
            if !raw.is_ascii_whitespace() {
                current_raw.push(raw);
            }
        }
    }

    flush_a2m_record(
        &mut current_name,
        &mut current_desc,
        &mut current_raw,
        &mut records,
    )?;
    if records.is_empty() {
        return Err(HmmerError::Format(
            "A2M alignment contains no records".to_string(),
        ));
    }

    let ncons = records[0].consensus.len();
    if ncons == 0 {
        return Err(HmmerError::Format(
            "A2M alignment contains no consensus columns".to_string(),
        ));
    }
    for record in &records {
        if record.consensus.len() != ncons {
            return Err(HmmerError::Format(format!(
                "A2M sequence {} has {} consensus columns, expected {ncons}",
                record.name,
                record.consensus.len()
            )));
        }
    }

    let mut insert_widths = vec![0usize; ncons + 1];
    for record in &records {
        for (idx, insert) in record.inserts.iter().enumerate() {
            insert_widths[idx] = insert_widths[idx].max(insert.len());
        }
    }

    let alen = ncons + insert_widths.iter().sum::<usize>();
    let mut rf = Vec::with_capacity(alen);
    for (idx, &width) in insert_widths.iter().enumerate().take(ncons + 1) {
        rf.extend(std::iter::repeat_n(b'.', width));
        if idx < ncons {
            rf.push(b'x');
        }
    }

    let mut sqname = Vec::with_capacity(records.len());
    let mut sqdesc = Vec::with_capacity(records.len());
    let mut aseq = Vec::with_capacity(records.len());
    for record in records {
        let mut row = Vec::with_capacity(alen);
        for (idx, &width) in insert_widths.iter().enumerate().take(ncons + 1) {
            let insert = &record.inserts[idx];
            row.extend_from_slice(insert);
            // Insert (non-consensus) gap columns use '.' in A2M; C
            // `esl_msafile_a2m.c` emits '.' here (consensus gaps use '-').
            row.extend(std::iter::repeat_n(b'.', width - insert.len()));
            if idx < ncons {
                row.push(record.consensus[idx]);
            }
        }
        sqname.push(record.name);
        sqdesc.push(record.desc);
        aseq.push(row);
    }
    let nseq = sqname.len();
    Ok(Msa {
        name: alignment_name,
        acc: None,
        desc: None,
        author: None,
        sqname,
        sqacc: vec![String::new(); nseq],
        sqdesc,
        weights: None,
        aseq,
        pp: vec![None; nseq],
        nseq,
        alen,
        rf: Some(rf),
        mm: None,
        ss_cons: None,
        sa_cons: None,
        pp_cons: None,
    })
}

pub(super) fn parse_psiblast_alignment(text: &str, alignment_name: String) -> HmmerResult<Msa> {
    let mut sqname = Vec::new();
    let mut sqdesc = Vec::new();
    let mut aseq = Vec::new();

    for (line_idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut fields = trimmed.split_whitespace();
        let Some(name) = fields.next() else {
            continue;
        };
        let Some(seq) = fields.next() else {
            return Err(HmmerError::Format(format!(
                "PSIBLAST alignment row at line {} is missing sequence data",
                line_idx + 1
            )));
        };
        if fields.next().is_some() {
            return Err(HmmerError::Format(format!(
                "PSIBLAST alignment row at line {} has unexpected trailing fields",
                line_idx + 1
            )));
        }
        sqname.push(name.to_string());
        sqdesc.push(String::new());
        aseq.push(seq.as_bytes().to_vec());
    }

    msa_from_rows("PSIBLAST", alignment_name, sqname, sqdesc, aseq)
}

pub(super) fn parse_clustal_alignment(text: &str, alignment_name: String) -> HmmerResult<Msa> {
    let mut lines = text.lines();
    let Some(header) = lines.find(|line| !line.trim().is_empty()) else {
        return Err(HmmerError::Format(
            "CLUSTAL alignment contains no records".to_string(),
        ));
    };
    let header = header.trim_start();
    if !header.starts_with("CLUSTAL") && !header.starts_with("MUSCLE") {
        return Err(HmmerError::Format(
            "CLUSTAL alignment does not start with a CLUSTAL/MUSCLE header".to_string(),
        ));
    }

    let mut order = Vec::<String>::new();
    let mut rows = HashMap::<String, Vec<u8>>::new();
    for (line_idx, line) in lines.enumerate() {
        if line.trim().is_empty() || line.starts_with(char::is_whitespace) {
            continue;
        }
        let mut fields = line.split_whitespace();
        let Some(name) = fields.next() else {
            continue;
        };
        let Some(seq) = fields.next() else {
            return Err(HmmerError::Format(format!(
                "CLUSTAL alignment row at line {} is missing sequence data",
                line_idx + 2
            )));
        };
        if !rows.contains_key(name) {
            order.push(name.to_string());
        }
        rows.entry(name.to_string())
            .or_default()
            .extend_from_slice(seq.as_bytes());
    }

    let mut sqname = Vec::with_capacity(order.len());
    let mut sqdesc = Vec::with_capacity(order.len());
    let mut aseq = Vec::with_capacity(order.len());
    for name in order {
        let seq = rows
            .remove(&name)
            .ok_or_else(|| HmmerError::Format(format!("missing CLUSTAL row for {name}")))?;
        sqname.push(name);
        sqdesc.push(String::new());
        aseq.push(seq);
    }
    msa_from_rows("CLUSTAL", alignment_name, sqname, sqdesc, aseq)
}

pub(super) fn parse_selex_alignment(text: &str, alignment_name: String) -> HmmerResult<Msa> {
    let mut order = Vec::<String>::new();
    let mut rows = HashMap::<String, Vec<u8>>::new();

    for (line_idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with('#')
            || trimmed.starts_with('%')
            || trimmed.starts_with("//")
        {
            continue;
        }
        let mut fields = trimmed.split_whitespace();
        let Some(name) = fields.next() else {
            continue;
        };
        let Some(seq) = fields.next() else {
            return Err(HmmerError::Format(format!(
                "SELEX alignment row at line {} is missing sequence data",
                line_idx + 1
            )));
        };
        let mut chunk = seq.as_bytes().to_vec();
        for field in fields {
            chunk.extend_from_slice(field.as_bytes());
        }
        if !rows.contains_key(name) {
            order.push(name.to_string());
        }
        rows.entry(name.to_string()).or_default().extend(chunk);
    }

    let mut sqname = Vec::with_capacity(order.len());
    let mut sqdesc = Vec::with_capacity(order.len());
    let mut aseq = Vec::with_capacity(order.len());
    for name in order {
        let seq = rows
            .remove(&name)
            .ok_or_else(|| HmmerError::Format(format!("missing SELEX row for {name}")))?;
        sqname.push(name);
        sqdesc.push(String::new());
        aseq.push(seq);
    }
    msa_from_rows("SELEX", alignment_name, sqname, sqdesc, aseq)
}
