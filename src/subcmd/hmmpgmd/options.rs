use super::*;

pub(super) fn master_init_body(state: &ServerState) -> Vec<u8> {
    let mut body = vec![0u8; HMMD_INIT_BODY_SIZE];
    let mut data = Vec::new();

    match state {
        ServerState::Seq(db) => {
            let path = db.path.to_string_lossy();
            let sid = db.cache_id.as_bytes();
            let copy_len = sid.len().min(63);
            body[..copy_len].copy_from_slice(&sid[..copy_len]);
            write_u32_ne(&mut body, 64, 0);
            write_u32_ne(&mut body, 72, 1);
            write_u32_ne(
                &mut body,
                76,
                db.sequences.len().min(u32::MAX as usize) as u32,
            );
            data.extend_from_slice(path.as_bytes());
            data.push(0);
        }
        ServerState::Hmm(db) => {
            let path = db.path.to_string_lossy();
            write_u32_ne(&mut body, 68, 0);
            write_u32_ne(&mut body, 80, 1);
            write_u32_ne(&mut body, 84, db.hmms.len().min(u32::MAX as usize) as u32);
            data.extend_from_slice(path.as_bytes());
            data.push(0);
        }
    }

    body.extend_from_slice(&data);
    body
}

pub(super) fn load_worker_init_state(body: &[u8]) -> Result<Option<ServerState>, String> {
    if body.len() < HMMD_INIT_BODY_SIZE {
        return Err("short HMMD_CMD_INIT body".to_string());
    }

    let seq_cnt = read_native_u32(body, 72)? as usize;
    let hmm_cnt = read_native_u32(body, 80)? as usize;
    if seq_cnt != 0 {
        let seqdb_off = read_native_u32(body, 64)? as usize;
        let path = init_body_path(body, seqdb_off, "sequence database")?;
        return Ok(Some(load_seqdb(&PathBuf::from(path))));
    }
    if hmm_cnt != 0 {
        let hmmdb_off = read_native_u32(body, 68)? as usize;
        let path = init_body_path(body, hmmdb_off, "HMM database")?;
        return Ok(Some(load_hmmdb(&PathBuf::from(path))));
    }
    Ok(None)
}

fn init_body_path(body: &[u8], offset: usize, label: &str) -> Result<String, String> {
    let start = HMMD_INIT_BODY_SIZE
        .checked_add(offset)
        .ok_or_else(|| format!("HMMD_CMD_INIT {label} path offset overflow"))?;
    let (path, _) = take_hmmd_query_string(body, start, label)?;
    if path.is_empty() {
        return Err(format!("HMMD_CMD_INIT {label} path is empty"));
    }
    Ok(path)
}

pub(super) fn parse_hmmpgmd_search_options(options: &str) -> Result<SearchOptions, String> {
    let words = options
        .split_whitespace()
        .map(|word| word.strip_prefix('@').unwrap_or(word))
        .collect::<Vec<_>>();
    let mut parsed = SearchOptions::default();
    let mut i = 0usize;
    while i < words.len() {
        let word = words[i];
        match word {
            "--hmmdb" | "--seqdb" | "--seqdb_ranges" | "--cpu" => i += 2,
            "--cut_ga" => {
                parsed.bit_cutoff = BitCutoff::GA;
                i += 1;
            }
            "--cut_tc" => {
                parsed.bit_cutoff = BitCutoff::TC;
                i += 1;
            }
            "--cut_nc" => {
                parsed.bit_cutoff = BitCutoff::NC;
                i += 1;
            }
            "--max" => {
                parsed.max = true;
                i += 1;
            }
            "--nobias" => {
                parsed.nobias = true;
                i += 1;
            }
            "--nonull2" => {
                parsed.nonull2 = true;
                i += 1;
            }
            "-E" | "--E" | "-T" | "--T" | "--domE" | "--domT" | "--incE" | "--incT"
            | "--incdomE" | "--incdomT" | "--F1" | "--F2" | "--F3" => {
                let (value, next) = hmmpgmd_option_value(&words, i, word)?;
                apply_hmmpgmd_option(&mut parsed, word, value)?;
                i = next;
            }
            _ if word.starts_with("--hmmdb=")
                || word.starts_with("--seqdb=")
                || word.starts_with("--seqdb_ranges=") =>
            {
                i += 1;
            }
            _ => {
                if let Some((option, value)) = hmmpgmd_split_attached_option(word) {
                    apply_hmmpgmd_option(&mut parsed, option, value)?;
                }
                i += 1;
            }
        }
    }
    Ok(parsed)
}

fn hmmpgmd_split_attached_option(word: &str) -> Option<(&'static str, &str)> {
    for option in [
        "--incdomE",
        "--incdomT",
        "--domE",
        "--domT",
        "--incE",
        "--incT",
        "--F1",
        "--F2",
        "--F3",
        "--E",
        "--T",
    ] {
        if let Some(value) = word.strip_prefix(&format!("{option}=")) {
            return Some((option, value));
        }
    }
    if let Some(value) = word.strip_prefix("-E") {
        if !value.is_empty() {
            return Some(("-E", value));
        }
    }
    if let Some(value) = word.strip_prefix("-T") {
        if !value.is_empty() {
            return Some(("-T", value));
        }
    }
    None
}

fn hmmpgmd_option_value<'a>(
    words: &'a [&str],
    index: usize,
    option: &str,
) -> Result<(&'a str, usize), String> {
    words
        .get(index + 1)
        .copied()
        .map(|value| (value, index + 2))
        .ok_or_else(|| format!("{option} requires a value"))
}

fn apply_hmmpgmd_option(
    parsed: &mut SearchOptions,
    option: &str,
    value: &str,
) -> Result<(), String> {
    match option {
        "-E" | "--E" => parsed.e_value_threshold = parse_hmmpgmd_f64_option(option, value)?,
        "-T" | "--T" => parsed.t = Some(parse_hmmpgmd_f64_option(option, value)?),
        "--domE" => parsed.dom_e_value_threshold = parse_hmmpgmd_f64_option(option, value)?,
        "--domT" => parsed.dom_t = Some(parse_hmmpgmd_f64_option(option, value)?),
        "--incE" => parsed.inc_e = parse_hmmpgmd_f64_option(option, value)?,
        "--incT" => parsed.inc_t = Some(parse_hmmpgmd_f64_option(option, value)?),
        "--incdomE" => parsed.inc_dome = parse_hmmpgmd_f64_option(option, value)?,
        "--incdomT" => parsed.inc_dom_t = Some(parse_hmmpgmd_f64_option(option, value)?),
        "--F1" => parsed.f1 = parse_hmmpgmd_f64_option(option, value)?,
        "--F2" => parsed.f2 = parse_hmmpgmd_f64_option(option, value)?,
        "--F3" => parsed.f3 = parse_hmmpgmd_f64_option(option, value)?,
        _ => {}
    }
    Ok(())
}

fn parse_hmmpgmd_f64_option(option: &str, value: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|_| format!("{option} requires a numeric value"))
}

pub(super) fn parse_seqdb_ranges_option(options: &str) -> Result<Option<Vec<DbRange>>, String> {
    let mut words = options.split_whitespace();
    while let Some(word) = words.next() {
        if word == "--seqdb_ranges" {
            let value = words
                .next()
                .ok_or_else(|| "--seqdb_ranges requires coords <from>..<to>".to_string())?;
            return parse_db_ranges(value).map(Some);
        }
        if let Some(value) = word.strip_prefix("--seqdb_ranges=") {
            return parse_db_ranges(value).map(Some);
        }
    }
    Ok(None)
}

fn parse_db_ranges(value: &str) -> Result<Vec<DbRange>, String> {
    let mut ranges = Vec::new();
    for part in value.split(',') {
        let (start, end) = part.split_once("..").ok_or_else(|| {
            format!("--seqdb_ranges takes coords <from>..<to>; {part} not recognized")
        })?;
        let start = start.parse::<usize>().map_err(|_| {
            format!("--seqdb_ranges takes coords <from>..<to>; {part} not recognized")
        })?;
        let end = end.parse::<usize>().map_err(|_| {
            format!("--seqdb_ranges takes coords <from>..<to>; {part} not recognized")
        })?;
        if start == 0 || end < start {
            return Err(format!(
                "--seqdb_ranges takes coords <from>..<to>; {part} not recognized"
            ));
        }
        ranges.push(DbRange { start, end });
    }
    if ranges.is_empty() {
        return Err("--seqdb_ranges requires coords <from>..<to>".to_string());
    }
    Ok(ranges)
}

pub(super) fn parse_client_query(query: &str, db_mode: DbMode) -> Result<ClientQuery, String> {
    let query = query.trim_start();
    if query.starts_with('>') {
        return parse_fasta_query(query).map(|(name, desc, seq)| ClientQuery::Sequence {
            name,
            desc,
            seq,
        });
    }
    if query.starts_with("HMM") {
        if db_mode == DbMode::Hmm {
            return Err("A HMM cannot be used to search a hmm database".to_string());
        }
        return parse_hmm_query(query).map(ClientQuery::Hmm);
    }
    Err("Unknown query sequence/hmm format".to_string())
}

fn parse_fasta_query(fasta: &str) -> Result<(String, String, String), String> {
    let mut lines = fasta.lines();
    let header = lines
        .next()
        .ok_or_else(|| "Missing search sequence/hmm".to_string())?;
    if !header.starts_with('>') {
        return Err("Unknown query sequence/hmm format".to_string());
    }
    let mut header_parts = header[1..].splitn(2, char::is_whitespace);
    let name = header_parts
        .next()
        .ok_or_else(|| "Error parsing FASTA sequence".to_string())?;
    if name.is_empty() {
        return Err("Error parsing FASTA sequence".to_string());
    }
    let desc = header_parts.next().unwrap_or("").trim().to_string();
    let seq: String = lines
        .take_while(|line| line.trim_end() != "//")
        .map(str::trim)
        .collect();
    if seq.is_empty() {
        return Err("Error zero length FASTA sequence".to_string());
    }
    Ok((name.to_string(), desc, seq))
}

fn parse_hmm_query(query: &str) -> Result<Hmm, String> {
    let mut record = String::new();
    let mut saw_end = false;
    for line in query.lines() {
        record.push_str(line);
        record.push('\n');
        if line.trim_end() == "//" {
            saw_end = true;
            break;
        }
    }
    if !saw_end {
        return Err("Error reading query hmm: missing // terminator".to_string());
    }

    let mut hmms = hmmfile::read_hmms(BufReader::new(Cursor::new(record.into_bytes())))
        .map_err(|e| format!("Error reading query hmm: {e}"))?;
    if hmms.len() != 1 {
        return Err(format!(
            "Error reading query hmm: expected one HMM, found {}",
            hmms.len()
        ));
    }
    Ok(hmms.remove(0))
}
