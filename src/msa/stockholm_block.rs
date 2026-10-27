use super::*;

/// Parse one Stockholm block (between `# STOCKHOLM` and `//`) into an [`Msa`].
///
/// Recognises common GF/GS/GR/GC metadata plus bare `name sequence` rows.
/// Returns `Ok(None)` if the block contained no sequences.
pub(super) fn parse_stockholm_block(lines: &[String]) -> HmmerResult<Option<StockholmMsa>> {
    let mut name = String::new();
    let mut acc: Option<String> = None;
    let mut desc: Option<String> = None;
    let mut author: Option<String> = None;
    let mut cutoffs = StockholmCutoffs::default();
    let mut seq_order: Vec<String> = Vec::new();
    let mut seq_data: HashMap<String, Vec<u8>> = HashMap::new();
    let mut sqacc: HashMap<String, String> = HashMap::new();
    let mut sqdesc: HashMap<String, String> = HashMap::new();
    let mut weights: HashMap<String, f64> = HashMap::new();
    let mut pp: HashMap<String, Vec<u8>> = HashMap::new();
    let mut rf: Option<Vec<u8>> = None;
    let mut mm: Option<Vec<u8>> = None;
    let mut ss_cons: Option<Vec<u8>> = None;
    let mut sa_cons: Option<Vec<u8>> = None;
    let mut pp_cons: Option<Vec<u8>> = None;
    let mut body_lines = Vec::new();
    let mut expected_block_order: Option<Vec<BlockLineKey>> = None;
    let mut current_block_order: Vec<BlockLineKey> = Vec::new();
    let mut current_block_names: HashSet<String> = HashSet::new();

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum BlockLineKey {
        Sequence(String),
        Gr { seq: String, tag: String },
        Gc(String),
    }

    impl BlockLineKey {
        fn label(&self) -> String {
            match self {
                Self::Sequence(name) => format!("sequence {name}"),
                Self::Gr { seq, tag } => format!("#=GR {seq} {tag}"),
                Self::Gc(tag) => format!("#=GC {tag}"),
            }
        }
    }

    fn finish_sequence_block(
        expected_block_order: &mut Option<Vec<BlockLineKey>>,
        current_block_order: &mut Vec<BlockLineKey>,
        current_block_names: &mut HashSet<String>,
    ) -> HmmerResult<()> {
        if current_block_order.is_empty() {
            return Ok(());
        }
        if let Some(expected) = expected_block_order.as_ref() {
            if current_block_order.len() != expected.len() {
                return Err(HmmerError::Format(format!(
                    "Stockholm sequence block has {} ordered lines, expected {}",
                    current_block_order.len(),
                    expected.len()
                )));
            }
        } else {
            *expected_block_order = Some(current_block_order.clone());
        }
        current_block_order.clear();
        current_block_names.clear();
        Ok(())
    }

    fn record_block_line(
        expected_block_order: &Option<Vec<BlockLineKey>>,
        current_block_order: &mut Vec<BlockLineKey>,
        key: BlockLineKey,
    ) -> HmmerResult<()> {
        if let Some(expected) = expected_block_order {
            let position = current_block_order.len();
            if expected.get(position) != Some(&key) {
                let expected_label = expected
                    .get(position)
                    .map(BlockLineKey::label)
                    .unwrap_or_else(|| "end of block".to_string());
                return Err(HmmerError::Format(format!(
                    "Stockholm block line {} is out of order in a later block: got {}, expected {}",
                    position + 1,
                    key.label(),
                    expected_label
                )));
            }
        }
        current_block_order.push(key);
        Ok(())
    }

    for line in lines {
        let trimmed = line.trim();
        if trimmed == "//" || trimmed.starts_with("# STOCKHOLM") {
            continue;
        }
        body_lines.push(line.clone());
        if trimmed.is_empty() {
            finish_sequence_block(
                &mut expected_block_order,
                &mut current_block_order,
                &mut current_block_names,
            )?;
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("#=GF ID") {
            let fields: Vec<&str> = rest.split_whitespace().collect();
            if fields.len() != 1 {
                return Err(HmmerError::Format(format!(
                    "Stockholm #=GF ID annotation must contain exactly one name token, got '{}'",
                    rest.trim()
                )));
            }
            name = fields[0].to_string();
        } else if let Some(rest) = trimmed.strip_prefix("#=GF AC") {
            acc = Some(rest.trim().to_string());
        } else if let Some(rest) = trimmed.strip_prefix("#=GF DE") {
            // C `stockholm_parse_gf` calls `esl_msa_SetDesc`, which frees the
            // existing value and overwrites it: the LAST `#=GF DE` line wins.
            // (esl_msa.c esl_msa_SetDesc.)
            desc = Some(rest.trim().to_string());
        } else if let Some(rest) = trimmed.strip_prefix("#=GF AU") {
            // Likewise C `esl_msa_SetAuthor` overwrites; the LAST `#=GF AU` wins.
            author = Some(rest.trim().to_string());
        } else if let Some(rest) = trimmed.strip_prefix("#=GF GA") {
            cutoffs.ga = Some(parse_stockholm_cutoffs("GA", rest.trim())?);
        } else if let Some(rest) = trimmed.strip_prefix("#=GF TC") {
            cutoffs.tc = Some(parse_stockholm_cutoffs("TC", rest.trim())?);
        } else if let Some(rest) = trimmed.strip_prefix("#=GF NC") {
            cutoffs.nc = Some(parse_stockholm_cutoffs("NC", rest.trim())?);
        } else if let Some(rest) = trimmed.strip_prefix("#=GC RF") {
            record_block_line(
                &expected_block_order,
                &mut current_block_order,
                BlockLineKey::Gc("RF".to_string()),
            )?;
            let rf_str = rest.trim();
            match &mut rf {
                Some(existing) => existing.extend_from_slice(rf_str.as_bytes()),
                None => rf = Some(rf_str.as_bytes().to_vec()),
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GC MM") {
            record_block_line(
                &expected_block_order,
                &mut current_block_order,
                BlockLineKey::Gc("MM".to_string()),
            )?;
            let mm_str = rest.trim();
            match &mut mm {
                Some(existing) => existing.extend_from_slice(mm_str.as_bytes()),
                None => mm = Some(mm_str.as_bytes().to_vec()),
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GC SS_cons") {
            record_block_line(
                &expected_block_order,
                &mut current_block_order,
                BlockLineKey::Gc("SS_cons".to_string()),
            )?;
            let ss_str = rest.trim();
            match &mut ss_cons {
                Some(existing) => existing.extend_from_slice(ss_str.as_bytes()),
                None => ss_cons = Some(ss_str.as_bytes().to_vec()),
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GC SA_cons") {
            record_block_line(
                &expected_block_order,
                &mut current_block_order,
                BlockLineKey::Gc("SA_cons".to_string()),
            )?;
            let sa_str = rest.trim();
            match &mut sa_cons {
                Some(existing) => existing.extend_from_slice(sa_str.as_bytes()),
                None => sa_cons = Some(sa_str.as_bytes().to_vec()),
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GC PP_cons") {
            record_block_line(
                &expected_block_order,
                &mut current_block_order,
                BlockLineKey::Gc("PP_cons".to_string()),
            )?;
            let pp_str = rest.trim();
            match &mut pp_cons {
                Some(existing) => existing.extend_from_slice(pp_str.as_bytes()),
                None => pp_cons = Some(pp_str.as_bytes().to_vec()),
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GS ") {
            let mut name_split = rest.splitn(2, char::is_whitespace);
            let seq_name = name_split.next().unwrap_or_default();
            let annotation = name_split.next().unwrap_or_default().trim_start();
            let mut tag_split = annotation.splitn(2, char::is_whitespace);
            let tag = tag_split.next().unwrap_or_default();
            let value = tag_split.next().unwrap_or_default().trim();
            if !seq_name.is_empty() && !tag.is_empty() && !value.is_empty() {
                if tag == "AC" {
                    if sqacc
                        .insert(seq_name.to_string(), value.to_string())
                        .is_some()
                    {
                        return Err(HmmerError::Format(format!(
                            "Stockholm sequence {} has more than one AC annotation",
                            seq_name
                        )));
                    }
                } else if tag == "DE" {
                    if sqdesc
                        .insert(seq_name.to_string(), value.to_string())
                        .is_some()
                    {
                        return Err(HmmerError::Format(format!(
                            "Stockholm sequence {} has more than one DE annotation",
                            seq_name
                        )));
                    }
                } else if tag == "WT" {
                    let weight = value.parse::<f64>().map_err(|e| {
                        HmmerError::Format(format!(
                            "Stockholm #=GS {} WT annotation is not a valid weight: {}",
                            seq_name, e
                        ))
                    })?;
                    if !weight.is_finite() {
                        return Err(HmmerError::Format(format!(
                            "Stockholm #=GS {} WT annotation must be a finite weight",
                            seq_name
                        )));
                    }
                    if weights.insert(seq_name.to_string(), weight).is_some() {
                        return Err(HmmerError::Format(format!(
                            "Stockholm sequence {} has more than one WT annotation",
                            seq_name
                        )));
                    }
                }
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GR ") {
            let fields: Vec<&str> = rest.splitn(3, char::is_whitespace).collect();
            if fields.len() == 3 {
                record_block_line(
                    &expected_block_order,
                    &mut current_block_order,
                    BlockLineKey::Gr {
                        seq: fields[0].to_string(),
                        tag: fields[1].to_string(),
                    },
                )?;
            }
            if fields.len() == 3 && fields[1] == "PP" {
                pp.entry(fields[0].to_string())
                    .and_modify(|line| line.extend_from_slice(fields[2].trim().as_bytes()))
                    .or_insert_with(|| fields[2].trim().as_bytes().to_vec());
            }
        } else if let Some(rest) = trimmed.strip_prefix("#=GC ") {
            if let Some(tag) = rest.split_whitespace().next() {
                record_block_line(
                    &expected_block_order,
                    &mut current_block_order,
                    BlockLineKey::Gc(tag.to_string()),
                )?;
            }
        } else if trimmed.starts_with('#') {
            // Other annotation — skip
            continue;
        } else {
            // Sequence line: name  sequence
            let parts: Vec<&str> = trimmed.splitn(2, char::is_whitespace).collect();
            if parts.len() == 2 {
                let sqname = parts[0].to_string();
                let sqdata = parts[1].trim().as_bytes();

                if !current_block_names.insert(sqname.clone()) {
                    return Err(HmmerError::Format(format!(
                        "Stockholm sequence {} occurs more than once in the same block",
                        sqname
                    )));
                }
                record_block_line(
                    &expected_block_order,
                    &mut current_block_order,
                    BlockLineKey::Sequence(sqname.clone()),
                )?;

                match seq_data.entry(sqname.clone()) {
                    Entry::Vacant(entry) => {
                        seq_order.push(sqname);
                        entry.insert(sqdata.to_vec());
                    }
                    Entry::Occupied(mut entry) => {
                        entry.get_mut().extend_from_slice(sqdata);
                    }
                }
            }
        }
    }
    finish_sequence_block(
        &mut expected_block_order,
        &mut current_block_order,
        &mut current_block_names,
    )?;

    if seq_order.is_empty() {
        return Ok(None);
    }

    let alen = seq_data
        .get(&seq_order[0])
        .map(|v| v.len())
        .unwrap_or_default();
    for name in &seq_order {
        let len = seq_data.get(name).map(|v| v.len()).unwrap_or_default();
        if len != alen {
            return Err(HmmerError::Format(format!(
                "Stockholm sequence {} has aligned length {}, expected {}",
                name, len, alen
            )));
        }
    }
    if let Some(ref rf) = rf {
        if rf.len() != alen {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GC RF annotation has length {}, expected {}",
                rf.len(),
                alen
            )));
        }
    }
    if let Some(ref mm) = mm {
        if mm.len() != alen {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GC MM annotation has length {}, expected {}",
                mm.len(),
                alen
            )));
        }
    }
    if let Some(ref ss_cons) = ss_cons {
        if ss_cons.len() != alen {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GC SS_cons annotation has length {}, expected {}",
                ss_cons.len(),
                alen
            )));
        }
    }
    if let Some(ref sa_cons) = sa_cons {
        if sa_cons.len() != alen {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GC SA_cons annotation has length {}, expected {}",
                sa_cons.len(),
                alen
            )));
        }
    }
    if let Some(ref pp_cons) = pp_cons {
        if pp_cons.len() != alen {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GC PP_cons annotation has length {}, expected {}",
                pp_cons.len(),
                alen
            )));
        }
    }
    for (name, line) in &pp {
        if !seq_order.iter().any(|seq_name| seq_name == name) {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GR {} PP annotation refers to unknown sequence",
                name
            )));
        }
        if line.len() != alen {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GR {} PP annotation has length {}, expected {}",
                name,
                line.len(),
                alen
            )));
        }
        validate_pp_annotation(line, &format!("#=GR {} PP", name))?;
    }
    for name in sqdesc.keys() {
        if !seq_order.iter().any(|seq_name| seq_name == name) {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GS {} DE annotation refers to unknown sequence",
                name
            )));
        }
    }
    for name in sqacc.keys() {
        if !seq_order.iter().any(|seq_name| seq_name == name) {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GS {} AC annotation refers to unknown sequence",
                name
            )));
        }
    }
    for name in weights.keys() {
        if !seq_order.iter().any(|seq_name| seq_name == name) {
            return Err(HmmerError::Format(format!(
                "Stockholm #=GS {} WT annotation refers to unknown sequence",
                name
            )));
        }
    }
    if !weights.is_empty() && weights.len() != seq_order.len() {
        if let Some(name) = seq_order.iter().find(|name| !weights.contains_key(*name)) {
            return Err(HmmerError::Format(format!(
                "Stockholm record ended without a weight for {}",
                name
            )));
        }
    }
    if let Some(ref pp_cons) = pp_cons {
        validate_pp_annotation(pp_cons, "#=GC PP_cons")?;
    }
    let nseq = seq_order.len();

    let aseq: Vec<Vec<u8>> = seq_order
        .iter()
        .map(|name| seq_data.remove(name).unwrap_or_default())
        .collect();
    let sqdesc_vec: Vec<String> = seq_order
        .iter()
        .map(|name| sqdesc.remove(name).unwrap_or_default())
        .collect();
    let sqacc_vec: Vec<String> = seq_order
        .iter()
        .map(|name| sqacc.remove(name).unwrap_or_default())
        .collect();
    let weight_vec = if weights.is_empty() {
        None
    } else {
        Some(
            seq_order
                .iter()
                .map(|name| weights.remove(name).unwrap())
                .collect(),
        )
    };
    let pp_vec: Vec<Option<Vec<u8>>> = seq_order.iter().map(|name| pp.remove(name)).collect();

    Ok(Some(StockholmMsa {
        msa: Msa {
            name,
            acc,
            desc,
            author,
            sqname: seq_order,
            sqacc: sqacc_vec,
            sqdesc: sqdesc_vec,
            weights: weight_vec,
            pp: pp_vec,
            aseq,
            nseq,
            alen,
            rf,
            mm,
            ss_cons,
            sa_cons,
            pp_cons,
        },
        cutoffs,
        body_lines,
    }))
}
