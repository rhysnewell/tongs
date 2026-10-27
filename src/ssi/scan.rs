use super::*;

pub(super) struct HmmRecord {
    pub(super) name: String,
    pub(super) acc: Option<String>,
    pub(super) offset: u64,
}

pub(super) fn scan_hmm_records(path: &Path) -> HmmerResult<Vec<HmmRecord>> {
    if hmmfile_binary::looks_like_binary_hmm_file(path)? {
        return scan_binary_hmm_records(path);
    }
    scan_ascii_hmm_records(path)
}

fn scan_ascii_hmm_records(path: &Path) -> HmmerResult<Vec<HmmRecord>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    let mut records = Vec::new();
    let mut record_offset: Option<u64> = None;
    let mut record_text = String::new();

    loop {
        let offset = reader.stream_position().map_err(HmmerError::Io)?;
        let Some(line) =
            hmmfile::read_capped_text_line(&mut reader, hmmfile::MAX_ASCII_HMM_LINE_LEN)?
        else {
            break;
        };

        let trimmed = line.trim();
        if trimmed.starts_with("HMMER3/") {
            if record_offset.is_some() {
                return Err(HmmerError::Format(
                    "Missing // terminator before next HMM record".to_string(),
                ));
            }
            record_offset = Some(offset);
            record_text.clear();
        }
        if record_offset.is_some() {
            record_text.push_str(&line);
        }
        if trimmed == "//" {
            finish_ascii_record(&mut records, record_offset.take(), &record_text)?;
            record_text.clear();
        }
    }

    if record_offset.is_some() {
        return Err(HmmerError::Format(
            "Missing // terminator at end of HMM file".to_string(),
        ));
    }
    Ok(records)
}

fn scan_binary_hmm_records(path: &Path) -> HmmerResult<Vec<HmmRecord>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    let mut reader = BufReader::new(file);
    let mut records = Vec::new();

    loop {
        let offset = reader.stream_position().map_err(HmmerError::Io)?;
        let Some(hmm) = hmmfile_binary::read_binary_hmm(&mut reader)? else {
            break;
        };
        records.push(HmmRecord {
            name: hmm.name,
            acc: hmm.acc,
            offset,
        });
    }

    Ok(records)
}

fn finish_ascii_record(
    records: &mut Vec<HmmRecord>,
    record_offset: Option<u64>,
    record_text: &str,
) -> HmmerResult<()> {
    let Some(offset) = record_offset else {
        return Ok(());
    };
    let mut hmms =
        hmmfile::read_hmms(BufReader::new(Cursor::new(record_text.as_bytes().to_vec())))?;
    if hmms.len() != 1 {
        return Err(HmmerError::Format(format!(
            "Expected one HMM record at offset {offset}, found {}",
            hmms.len()
        )));
    }
    let hmm = hmms.remove(0);
    let name = hmm.name;
    if name.is_empty() {
        return Err(HmmerError::Format(
            "Every HMM must have a name to be indexed".to_string(),
        ));
    }
    let acc = hmm.acc;
    records.push(HmmRecord { name, acc, offset });
    Ok(())
}
