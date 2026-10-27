use super::*;

pub(super) fn msa_from_rows_with_alen(
    format_name: &str,
    alignment_name: String,
    sqname: Vec<String>,
    sqdesc: Vec<String>,
    aseq: Vec<Vec<u8>>,
    declared_alen: usize,
) -> HmmerResult<Msa> {
    let msa = msa_from_rows(format_name, alignment_name, sqname, sqdesc, aseq)?;
    if msa.alen != declared_alen {
        return Err(HmmerError::Format(format!(
            "{format_name} alignment declares aligned length {declared_alen} but parsed {}",
            msa.alen
        )));
    }
    Ok(msa)
}

pub(super) fn msa_from_rows(
    format_name: &str,
    alignment_name: String,
    sqname: Vec<String>,
    sqdesc: Vec<String>,
    aseq: Vec<Vec<u8>>,
) -> HmmerResult<Msa> {
    if sqname.is_empty() {
        return Err(HmmerError::Format(format!(
            "{format_name} alignment contains no records"
        )));
    }
    let alen = aseq[0].len();
    for (name, seq) in sqname.iter().zip(&aseq) {
        if seq.len() != alen {
            return Err(HmmerError::Format(format!(
                "{format_name} sequence {name} has aligned length {}, expected {alen}",
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

pub(super) fn flush_a2m_record(
    current_name: &mut Option<String>,
    current_desc: &mut String,
    current_raw: &mut Vec<u8>,
    records: &mut Vec<A2mRecord>,
) -> HmmerResult<()> {
    let Some(name) = current_name.take() else {
        return Ok(());
    };
    if current_raw.is_empty() {
        return Err(HmmerError::Format(format!(
            "A2M record {name} has no sequence data"
        )));
    }

    let mut inserts = vec![Vec::<u8>::new()];
    let mut consensus = Vec::<u8>::new();
    let mut current_insert = 0usize;
    for &raw in current_raw.iter() {
        if raw == b'.' {
            continue;
        }
        if raw.is_ascii_lowercase() {
            inserts[current_insert].push(raw);
        } else {
            consensus.push(raw);
            inserts.push(Vec::new());
            current_insert += 1;
        }
    }
    records.push(A2mRecord {
        name,
        desc: std::mem::take(current_desc),
        inserts,
        consensus,
    });
    current_raw.clear();
    Ok(())
}

pub(super) fn flush_afa_record(
    current_name: &mut Option<String>,
    current_desc: &mut String,
    current_seq: &mut Vec<u8>,
    sqname: &mut Vec<String>,
    sqdesc: &mut Vec<String>,
    aseq: &mut Vec<Vec<u8>>,
) -> HmmerResult<()> {
    let Some(name) = current_name.take() else {
        return Ok(());
    };
    if current_seq.is_empty() {
        return Err(HmmerError::Format(format!(
            "aligned FASTA record {name} has no sequence data"
        )));
    }
    sqname.push(name);
    sqdesc.push(std::mem::take(current_desc));
    aseq.push(std::mem::take(current_seq));
    Ok(())
}

pub(super) fn alignment_name_from_path(path: &Path) -> String {
    path.file_stem()
        .or_else(|| path.file_name())
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty() && *name != "-")
        .unwrap_or("alignment")
        .to_string()
}
