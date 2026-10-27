use super::*;

/// Parse one HMMER3 ASCII HMM record from a line iterator (port of `read_asc30hmm`).
///
/// Handles every header key/value pair (NAME, LENG, ALPH, RF, MM, CONS, CS, MAP,
/// DATE, COM, NSEQ, EFFN, CKSUM, STATS, GA, TC, NC), then reads the alphabet
/// header, an optional COMPO line, node-0 insert/transitions, and the M
/// node blocks of match/insert/transition lines, terminated by `//`.
/// File values are stored as `-ln(p)` and are exponentiated back into
/// probabilities on the fly. Returns `Ok(None)` on clean EOF.
pub(super) fn read_one_hmm_with_format<B: BufRead>(
    lines: &mut CappedLineReader<B>,
) -> HmmerResult<Option<(Hmm, String)>> {
    // Find the format header line
    let header = loop {
        match lines.next_line()? {
            None => return Ok(None),
            Some(line) => {
                let trimmed = line.trim();
                if trimmed.starts_with("HMMER3/") {
                    break trimmed.to_string();
                }
                if trimmed.starts_with("HMMER2.") || trimmed.starts_with("HMMER2/") {
                    return Err(HmmerError::Format(
                        "HMMER2 ASCII input is intentionally unsupported".to_string(),
                    ));
                }
                // Skip blank lines between HMMs
                if !trimmed.is_empty() && !trimmed.starts_with('#') {
                    return Err(HmmerError::Format(format!(
                        "Expected HMMER3/ header, got: {}",
                        trimmed
                    )));
                }
            }
        }
    };

    let format_version = parse_hmmer3_ascii_magic(&header)?;
    // Parse header key-value pairs
    let mut name: Option<String> = None;
    let mut acc: Option<String> = None;
    let mut desc: Option<String> = None;
    let mut m: usize = 0;
    let mut max_length: i32 = -1;
    let mut abc_type = AlphabetType::Unknown;
    let mut rf_flag = false;
    let mut mm_flag = false;
    let mut cons_flag = false;
    let mut cs_flag = false;
    let mut map_flag = false;
    let mut nseq: i32 = -1;
    let mut eff_nseq: f32 = -1.0;
    let mut ctime: Option<String> = None;
    let mut comlog: Option<String> = None;
    let mut checksum: u32 = 0;
    // checksum presence is tracked via P7H_CHKSUM flag
    let mut evparam = [EVPARAM_UNSET; NEVPARAM];
    let mut cutoff = [CUTOFF_UNSET; NCUTOFFS];
    let mut flags: u32 = 0;
    let mut stat_msv = false;
    let mut stat_viterbi = false;
    let mut stat_forward = false;

    // Read header lines until "HMM" line
    loop {
        let line = lines
            .next_line()?
            .ok_or_else(|| HmmerError::Format("Unexpected EOF in HMM header".to_string()))?;
        let trimmed = line.trim();

        if trimmed.starts_with("HMM ") || trimmed == "HMM" {
            validate_alphabet_header(trimmed, abc_type)?;
            break;
        }

        let (key, value) = match trimmed.split_once(char::is_whitespace) {
            Some((k, v)) => (k, v.trim()),
            None => continue,
        };

        match key {
            "NAME" => name = Some(value.to_string()),
            "ACC" => {
                acc = Some(value.to_string());
                flags |= P7H_ACC;
            }
            "DESC" => {
                desc = Some(value.to_string());
                flags |= P7H_DESC;
            }
            "LENG" => {
                m = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad LENG".to_string()))?;
                if m == 0 || m > MAX_HMM_MODEL_LENGTH {
                    return Err(HmmerError::Format(format!(
                        "Invalid model length {value} on LENG line"
                    )));
                }
            }
            "MAXL" => {
                max_length = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad MAXL".to_string()))?;
                if max_length == 0 {
                    return Err(HmmerError::Format(format!(
                        "Invalid max length {value} on MAXL line"
                    )));
                }
            }
            "ALPH" => {
                abc_type = if value.eq_ignore_ascii_case("amino") {
                    AlphabetType::Amino
                } else if value.eq_ignore_ascii_case("DNA") {
                    AlphabetType::Dna
                } else if value.eq_ignore_ascii_case("RNA") {
                    AlphabetType::Rna
                } else {
                    return Err(HmmerError::Format(format!("Unknown alphabet: {}", value)));
                };
            }
            "RF" => rf_flag = parse_hmm_yes_no("RF", value)?,
            "MM" => mm_flag = parse_hmm_yes_no("MM", value)? && format_version == "3f",
            "CONS" => cons_flag = parse_hmm_yes_no("CONS", value)?,
            "CS" => cs_flag = parse_hmm_yes_no("CS", value)?,
            "MAP" => map_flag = parse_hmm_yes_no("MAP", value)?,
            "DATE" => ctime = Some(value.to_string()),
            "COM" => {
                // COM lines may be numbered: "COM   [1] hmmbuild ..."
                let cmd = if value.starts_with('[') {
                    value
                        .split_once(']')
                        .map(|(_, r)| r.trim())
                        .unwrap_or(value)
                } else {
                    value
                };
                match &mut comlog {
                    Some(existing) => {
                        existing.push('\n');
                        existing.push_str(cmd);
                    }
                    None => comlog = Some(cmd.to_string()),
                }
            }
            "NSEQ" => {
                nseq = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad NSEQ".to_string()))?;
                if nseq == 0 {
                    return Err(HmmerError::Format(format!(
                        "Invalid nseq on NSEQ line: should be integer, not {value}"
                    )));
                }
            }
            "EFFN" => {
                eff_nseq = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad EFFN".to_string()))?;
                if eff_nseq <= 0.0 {
                    return Err(HmmerError::Format(format!(
                        "Invalid eff_nseq on EFFN line: should be a real number, not {value}"
                    )));
                }
            }
            "CKSUM" => {
                checksum = value
                    .parse()
                    .map_err(|_| HmmerError::Format("Bad CKSUM".to_string()))?;
                flags |= P7H_CHKSUM;
            }
            "STATS" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if format_version == "3a" {
                    // HMMER3/a (reverse compatibility): 3-token form
                    //   "LOCAL VLAMBDA <v>", "LOCAL VMU <v>", "LOCAL FTAU <v>"
                    if parts.len() < 3 {
                        return Err(HmmerError::Format(
                            "Too few fields on STATS line".to_string(),
                        ));
                    }
                    if !parts[0].eq_ignore_ascii_case("LOCAL") {
                        return Err(HmmerError::Format(format!(
                            "Failed to parse STATS, {} unrecognized as field 2",
                            parts[0]
                        )));
                    }
                    let v: f32 = parts[2]
                        .parse()
                        .map_err(|_| HmmerError::Format("Bad STATS value".to_string()))?;
                    if parts[1].eq_ignore_ascii_case("VLAMBDA") {
                        evparam[P7_MLAMBDA] = v;
                        evparam[P7_VLAMBDA] = v;
                        evparam[P7_FLAMBDA] = v;
                        stat_msv = true;
                    } else if parts[1].eq_ignore_ascii_case("VMU") {
                        evparam[P7_MMU] = v;
                        evparam[P7_VMU] = v;
                        stat_viterbi = true;
                    } else if parts[1].eq_ignore_ascii_case("FTAU") {
                        evparam[P7_FTAU] = v;
                        stat_forward = true;
                    } else {
                        return Err(HmmerError::Format(format!(
                            "Failed to parse STATS, {} unrecognized as field 3",
                            parts[1]
                        )));
                    }
                } else {
                    // HMMER3/b+ : 4-token form "LOCAL MSV -6.4582 0.72049"
                    if parts.len() < 4 {
                        return Err(HmmerError::Format(
                            "Too few fields on STATS line".to_string(),
                        ));
                    }
                    if !parts[0].eq_ignore_ascii_case("LOCAL") {
                        return Err(HmmerError::Format(format!(
                            "Failed to parse STATS, {} unrecognized as field 2",
                            parts[0]
                        )));
                    }
                    let v1: f32 = parts[2]
                        .parse()
                        .map_err(|_| HmmerError::Format("Bad STATS value".to_string()))?;
                    let v2: f32 = parts[3]
                        .parse()
                        .map_err(|_| HmmerError::Format("Bad STATS value".to_string()))?;
                    if parts[1].eq_ignore_ascii_case("MSV") {
                        evparam[P7_MMU] = v1;
                        evparam[P7_MLAMBDA] = v2;
                        stat_msv = true;
                    } else if parts[1].eq_ignore_ascii_case("VITERBI") {
                        evparam[P7_VMU] = v1;
                        evparam[P7_VLAMBDA] = v2;
                        stat_viterbi = true;
                    } else if parts[1].eq_ignore_ascii_case("FORWARD") {
                        evparam[P7_FTAU] = v1;
                        evparam[P7_FLAMBDA] = v2;
                        stat_forward = true;
                    } else {
                        return Err(HmmerError::Format(format!(
                            "Failed to parse STATS, {} unrecognized as field 3",
                            parts[1]
                        )));
                    }
                }
            }
            "GA" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if let Some(v) = parts.first() {
                    cutoff[P7_GA1] = parse_cutoff_value("GA", v)?;
                } else {
                    return Err(HmmerError::Format("Missing GA cutoff value".to_string()));
                }
                if let Some(v) = parts.get(1) {
                    cutoff[P7_GA2] = parse_cutoff_value("GA", v)?;
                } else {
                    cutoff[P7_GA2] = cutoff[P7_GA1];
                }
                flags |= P7H_GA;
            }
            "TC" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if let Some(v) = parts.first() {
                    cutoff[P7_TC1] = parse_cutoff_value("TC", v)?;
                } else {
                    return Err(HmmerError::Format("Missing TC cutoff value".to_string()));
                }
                if let Some(v) = parts.get(1) {
                    cutoff[P7_TC2] = parse_cutoff_value("TC", v)?;
                } else {
                    cutoff[P7_TC2] = cutoff[P7_TC1];
                }
                flags |= P7H_TC;
            }
            "NC" => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if let Some(v) = parts.first() {
                    cutoff[P7_NC1] = parse_cutoff_value("NC", v)?;
                } else {
                    return Err(HmmerError::Format("Missing NC cutoff value".to_string()));
                }
                if let Some(v) = parts.get(1) {
                    cutoff[P7_NC2] = parse_cutoff_value("NC", v)?;
                } else {
                    cutoff[P7_NC2] = cutoff[P7_NC1];
                }
                flags |= P7H_NC;
            }
            _ => {} // Ignore unknown header fields
        }
    }

    let name = name.ok_or_else(|| HmmerError::Format("Missing NAME in HMM".to_string()))?;
    if m == 0 {
        return Err(HmmerError::Format("Missing or zero LENG".to_string()));
    }
    if abc_type == AlphabetType::Unknown {
        return Err(HmmerError::Format("Missing ALPH in HMM".to_string()));
    }
    if stat_msv || stat_viterbi || stat_forward {
        if !(stat_msv && stat_viterbi && stat_forward) {
            return Err(HmmerError::Format(
                "Incomplete STATS block in HMM header".to_string(),
            ));
        }
        flags |= P7H_STATS;
    }

    let abc = crate::alphabet::Alphabet::new(abc_type);
    let k = abc.k;

    // Skip the transition label line ("m->m m->i ...")
    // Skip the transition label line ("m->m m->i ...")
    lines
        .next_line()?
        .ok_or_else(|| HmmerError::Format("Missing transition label line".to_string()))?;

    // Create the HMM
    let mut hmm = Hmm::new(m, abc_type, k);
    hmm.name = name;
    hmm.acc = acc;
    hmm.desc = desc;
    hmm.nseq = nseq;
    hmm.eff_nseq = eff_nseq;
    hmm.max_length = max_length;
    hmm.ctime = ctime;
    hmm.comlog = comlog;
    hmm.checksum = checksum;
    hmm.evparam = evparam;
    hmm.cutoff = cutoff;
    hmm.flags = flags;

    // Initialize optional annotation arrays
    if rf_flag {
        let mut rf = vec![b' '; m + 2];
        rf[0] = b' ';
        hmm.rf = Some(rf);
        hmm.flags |= P7H_RF;
    }
    if mm_flag {
        let mut mm = vec![b' '; m + 2];
        mm[0] = b' ';
        hmm.mm = Some(mm);
        hmm.flags |= P7H_MMASK;
    }
    if cons_flag {
        let mut cons = vec![b' '; m + 2];
        cons[0] = b' ';
        hmm.consensus = Some(cons);
        hmm.flags |= P7H_CONS;
    }
    if cs_flag {
        let mut cs = vec![b' '; m + 2];
        cs[0] = b' ';
        hmm.cs = Some(cs);
        hmm.flags |= P7H_CS;
    }
    if map_flag {
        hmm.map = Some(vec![0i32; m + 1]);
        hmm.flags |= P7H_MAP;
    }

    // Read COMPO line (optional, node 0 match emissions)
    let compo_line = lines
        .next_line()?
        .ok_or_else(|| HmmerError::Format("Missing COMPO/insert line".to_string()))?;

    let compo_trimmed = compo_line.trim();
    if compo_trimmed.starts_with("COMPO") {
        // Parse composition values
        let parts: Vec<&str> = compo_trimmed.split_whitespace().collect();
        if parts.len() < k + 1 {
            return Err(HmmerError::Format(format!(
                "COMPO line has too few fields: expected {k}, got {}",
                parts.len().saturating_sub(1)
            )));
        }
        for i in 0..k.min(MAXABET) {
            hmm.compo[i] = parse_hmm_value(parts[i + 1])?;
        }
        hmm.flags |= P7H_COMPO;

        // Read node 0 insert emissions
        let ins_line = lines
            .next_line()?
            .ok_or_else(|| HmmerError::Format("Missing node 0 insert line".to_string()))?;
        parse_emission_line(&ins_line, k, &mut hmm.ins[0])?;
    } else {
        // No COMPO line — this line IS the node 0 insert emissions
        parse_emission_line(compo_trimmed, k, &mut hmm.ins[0])?;
    }

    // Read node 0 transitions
    let trans_line = lines
        .next_line()?
        .ok_or_else(|| HmmerError::Format("Missing node 0 transition line".to_string()))?;
    parse_transition_line(&trans_line, &mut hmm.t[0])?;

    // Read nodes 1..M
    for node in 1..=m {
        // Match emission line: "  k  <K values> <map> <cons> <rf> <mm/cs>"
        let match_line = lines
            .next_line()?
            .ok_or_else(|| HmmerError::Format(format!("Missing node {} match line", node)))?;
        let parts: Vec<&str> = match_line.split_whitespace().collect();

        // Parse node number (first field)
        let node_num: usize = parts
            .first()
            .ok_or_else(|| HmmerError::Format("Empty match line".to_string()))?
            .parse()
            .map_err(|_| HmmerError::Format("Bad node number".to_string()))?;
        if node_num != node {
            return Err(HmmerError::Format(format!(
                "Expected node {}, got {}",
                node, node_num
            )));
        }

        // Parse K emission values
        if parts.len() < k + 1 {
            return Err(HmmerError::Format(format!(
                "Node {} match emission line has too few fields",
                node
            )));
        }
        for i in 0..k {
            hmm.mat[node][i] = parse_hmm_value(parts[i + 1])?;
        }

        let mut annot_idx = k + 1;
        let map_val = parts.get(annot_idx).ok_or_else(|| {
            HmmerError::Format(format!("Node {node} match line missing MAP column"))
        })?;
        annot_idx += 1;
        if map_flag {
            if let Some(map) = &mut hmm.map {
                map[node] = map_val
                    .parse()
                    .map_err(|_| HmmerError::Format(format!("Bad MAP value: {map_val}")))?;
            }
        }

        let has_cons_column = matches!(format_version.as_str(), "3e" | "3f");
        if has_cons_column {
            let cons_val = parts.get(annot_idx).ok_or_else(|| {
                HmmerError::Format(format!("Node {node} match line missing CONS column"))
            })?;
            annot_idx += 1;
            if cons_flag {
                if let Some(cons) = &mut hmm.consensus {
                    cons[node] = annotation_byte("CONS", cons_val)?;
                }
            }
        }

        let rf_val = parts.get(annot_idx).ok_or_else(|| {
            HmmerError::Format(format!("Node {node} match line missing RF column"))
        })?;
        annot_idx += 1;
        if rf_flag {
            if let Some(rf) = &mut hmm.rf {
                rf[node] = annotation_byte("RF", rf_val)?;
            }
        }

        if format_version == "3f" {
            let mm_val = parts.get(annot_idx).ok_or_else(|| {
                HmmerError::Format(format!("Node {node} match line missing MM column"))
            })?;
            annot_idx += 1;
            if mm_flag {
                if let Some(mm) = &mut hmm.mm {
                    mm[node] = annotation_byte("MM", mm_val)?;
                }
            }
        }

        let cs_val = parts.get(annot_idx).ok_or_else(|| {
            HmmerError::Format(format!("Node {node} match line missing CS column"))
        })?;
        if cs_flag {
            if let Some(cs) = &mut hmm.cs {
                cs[node] = annotation_byte("CS", cs_val)?;
            }
        }

        // Insert emission line
        let ins_line = lines
            .next_line()?
            .ok_or_else(|| HmmerError::Format(format!("Missing node {} insert line", node)))?;
        parse_emission_line(&ins_line, k, &mut hmm.ins[node])?;

        // Transition line
        let trans_line = lines
            .next_line()?
            .ok_or_else(|| HmmerError::Format(format!("Missing node {} transition line", node)))?;
        parse_transition_line(&trans_line, &mut hmm.t[node])?;
    }

    // Read end-of-record marker "//"
    let mut saw_terminator = false;
    loop {
        match lines.next_line()? {
            None => break,
            Some(line) => {
                let trimmed = line.trim();
                if trimmed == "//" {
                    saw_terminator = true;
                    break;
                } else if !trimmed.is_empty() {
                    return Err(HmmerError::Format(format!(
                        "Expected end-of-record marker //, got: {trimmed}"
                    )));
                }
            }
        }
    }
    if !saw_terminator {
        return Err(HmmerError::Format(
            "Unexpected EOF before HMM end-of-record marker //".to_string(),
        ));
    }

    // Convert from -ln(prob) to probability
    // In the file, values are stored as -ln(p). Convert to p.
    for node in 0..=m {
        for i in 0..k {
            hmm.mat[node][i] = c_expf_to_f32(-hmm.mat[node][i]);
            hmm.ins[node][i] = c_expf_to_f32(-hmm.ins[node][i]);
        }
        for i in 0..NTRANSITIONS {
            if hmm.t[node][i] != f32::INFINITY {
                hmm.t[node][i] = c_expf_to_f32(-hmm.t[node][i]);
            } else {
                hmm.t[node][i] = 0.0;
            }
        }
    }

    // Convert compo from -ln(prob) to prob
    for i in 0..k.min(MAXABET) {
        if hmm.compo[i] != COMPO_UNSET {
            hmm.compo[i] = c_expf_to_f32(-hmm.compo[i]);
        }
    }

    Ok(Some((hmm, format_version)))
}
