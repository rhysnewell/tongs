use super::*;

pub(super) fn sharded_hmmdb_scan_requests(
    request: &str,
    command: &ClientCommand,
    worker_count: usize,
    total_nmodels: u64,
) -> Vec<WorkerRequest> {
    let mut remaining = total_nmodels as usize;
    let mut start = 0usize;
    let mut remaining_workers = worker_count;
    let mut requests = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let count = remaining / remaining_workers;
        requests.push(binary_worker_request(
            request,
            command,
            HMMD_CMD_SCAN,
            start,
            count,
        ));
        start = start.saturating_add(count);
        remaining = remaining.saturating_sub(count);
        remaining_workers -= 1;
    }
    requests
}

pub(super) fn binary_worker_request(
    request: &str,
    command: &ClientCommand,
    worker_command: u32,
    start: usize,
    count: usize,
) -> WorkerRequest {
    let (options, _query_text) = split_c_request(request);
    let opts = options.as_bytes();
    let query_type = match &command.query {
        ClientQuery::Sequence { .. } => HMMD_SEQUENCE,
        ClientQuery::Hmm(_) => HMMD_HMM,
    };
    let query = match &command.query {
        ClientQuery::Sequence { name, desc, seq } => serialize_hmmd_sequence_query(name, desc, seq),
        ClientQuery::Hmm(hmm) => serialize_hmmd_hmm_query(hmm),
    };
    let query_length = match &command.query {
        ClientQuery::Sequence { seq, .. } => seq.len().saturating_add(2),
        ClientQuery::Hmm(hmm) => hmm.m,
    };
    let mut body = Vec::with_capacity(
        HMMD_SEARCH_CMD_FIXED_SIZE
            + opts.len()
            + 1
            + query.len()
            + C_HMMD_COMMAND_SEARCH_PADDING_SIZE,
    );
    for value in [
        0u32,
        0u32,
        start.min(u32::MAX as usize) as u32,
        count.min(u32::MAX as usize) as u32,
        query_type,
        query_length.min(u32::MAX as usize) as u32,
        (opts.len() + 1).min(u32::MAX as usize) as u32,
    ] {
        body.extend_from_slice(&value.to_ne_bytes());
    }
    body.extend_from_slice(opts);
    body.push(0);
    body.extend_from_slice(&query);
    body.extend(std::iter::repeat_n(0, C_HMMD_COMMAND_SEARCH_PADDING_SIZE));
    WorkerRequest {
        command: worker_command,
        body,
    }
}

fn serialize_hmmd_sequence_query(name: &str, desc: &str, seq: &str) -> Vec<u8> {
    let abc = Alphabet::amino();
    let dsq = abc.digitize(seq.as_bytes());
    let mut query = Vec::with_capacity(name.len() + desc.len() + dsq.len() + 2);
    query.extend_from_slice(name.as_bytes());
    query.push(0);
    query.extend_from_slice(desc.as_bytes());
    query.push(0);
    query.extend_from_slice(&dsq);
    query
}

fn serialize_hmmd_hmm_query(hmm: &Hmm) -> Vec<u8> {
    let mut query = vec![0u8; C_P7_HMM_SHELL_SIZE];
    let mut pointer_fixups = Vec::new();
    write_i32_ne(&mut query, 0, hmm.m.min(i32::MAX as usize) as i32);
    write_i32_ne(&mut query, 104, hmm.nseq);
    write_f32_ne(&mut query, 108, hmm.eff_nseq);
    write_i32_ne(&mut query, 112, hmm.max_length);
    write_u32_ne(&mut query, 136, hmm.checksum);
    for (idx, value) in hmm.evparam.iter().enumerate() {
        write_f32_ne(&mut query, 140 + idx * 4, *value);
    }
    for (idx, value) in hmm.cutoff.iter().enumerate() {
        write_f32_ne(&mut query, 164 + idx * 4, *value);
    }
    for (idx, value) in hmm.compo.iter().enumerate() {
        write_f32_ne(&mut query, 188 + idx * 4, *value);
    }
    record_hmmd_hmm_pointer(&mut pointer_fixups, 280, true, 0);
    write_i32_ne(&mut query, 288, hmm.flags as i32);

    let transition_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 8, !hmm.t.is_empty(), transition_offset);
    for row in &hmm.t {
        for value in row {
            query.extend_from_slice(&value.to_ne_bytes());
        }
    }
    let match_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 16, !hmm.mat.is_empty(), match_offset);
    for (node, row) in hmm.mat.iter().enumerate() {
        for (sym, value) in row.iter().take(hmm.abc_k).enumerate() {
            let value = if node == 0 {
                if sym == 0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                *value
            };
            query.extend_from_slice(&value.to_ne_bytes());
        }
    }
    let insert_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 24, !hmm.ins.is_empty(), insert_offset);
    for row in &hmm.ins {
        for value in row.iter().take(hmm.abc_k) {
            query.extend_from_slice(&value.to_ne_bytes());
        }
    }

    let name_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 32, !hmm.name.is_empty(), name_offset);
    append_c_string(&mut query, Some(&hmm.name));
    let acc_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 40, hmm.acc.is_some(), acc_offset);
    append_c_string(&mut query, hmm.acc.as_ref());
    let desc_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 48, hmm.desc.is_some(), desc_offset);
    append_c_string(&mut query, hmm.desc.as_ref());
    let rf_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 56, hmm.rf.is_some(), rf_offset);
    append_hmm_annotation(&mut query, hmm.rf.as_deref(), hmm.m);
    let mm_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 64, hmm.mm.is_some(), mm_offset);
    append_hmm_annotation(&mut query, hmm.mm.as_deref(), hmm.m);
    let consensus_offset = query.len();
    record_hmmd_hmm_pointer(
        &mut pointer_fixups,
        72,
        hmm.consensus.is_some(),
        consensus_offset,
    );
    append_hmm_annotation(&mut query, hmm.consensus.as_deref(), hmm.m);
    let cs_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 80, hmm.cs.is_some(), cs_offset);
    append_hmm_annotation(&mut query, hmm.cs.as_deref(), hmm.m);
    let ca_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 88, hmm.ca.is_some(), ca_offset);
    append_hmm_annotation(&mut query, hmm.ca.as_deref(), hmm.m);
    let map_offset = query.len();
    record_hmmd_hmm_pointer(&mut pointer_fixups, 128, hmm.map.is_some(), map_offset);
    if let Some(map) = &hmm.map {
        for idx in 0..=hmm.m {
            let value = map.get(idx).copied().unwrap_or_default();
            query.extend_from_slice(&value.to_ne_bytes());
        }
    }
    record_hmmd_hmm_pointer(&mut pointer_fixups, 96, hmm.comlog.is_some(), 0);
    record_hmmd_hmm_pointer(&mut pointer_fixups, 120, hmm.ctime.is_some(), 0);
    patch_hmmd_hmm_pointers(&mut query, &pointer_fixups);
    query
}

fn record_hmmd_hmm_pointer(
    pointer_fixups: &mut Vec<(usize, usize)>,
    shell_offset: usize,
    present: bool,
    data_offset: usize,
) {
    if present {
        pointer_fixups.push((shell_offset, data_offset));
    }
}

fn patch_hmmd_hmm_pointers(query: &mut [u8], pointer_fixups: &[(usize, usize)]) {
    let base = query.as_ptr() as usize;
    for &(shell_offset, data_offset) in pointer_fixups {
        // C serializes raw process-local pointers in this shell. Matching the
        // native address value is impossible across processes, but preserving
        // non-null native-width pointers keeps the wire shape and decoder
        // semantics byte-compatible after pointer-slot normalization.
        write_ptr_value(query, shell_offset, base.wrapping_add(data_offset));
    }
}

fn append_c_string(out: &mut Vec<u8>, value: Option<&String>) {
    if let Some(value) = value {
        out.extend_from_slice(value.as_bytes());
        out.push(0);
    }
}

fn append_hmm_annotation(out: &mut Vec<u8>, value: Option<&[u8]>, m: usize) {
    if let Some(value) = value {
        for idx in 0..m + 2 {
            let byte = if idx == m + 1 {
                0
            } else {
                value.get(idx).copied().unwrap_or(b' ')
            };
            out.push(byte);
        }
    }
}

fn split_c_request(request: &str) -> (&str, &str) {
    if let Some(first_newline) = request.find('\n') {
        let (options, rest) = request.split_at(first_newline);
        (options, rest.trim_start_matches('\n'))
    } else {
        (request, "")
    }
}

pub(super) fn request_with_seqdb_range(request: &str, start: usize, end: usize) -> String {
    request_with_seqdb_ranges(request, &[DbRange { start, end }])
}

pub(super) fn request_with_seqdb_ranges(request: &str, ranges: &[DbRange]) -> String {
    let ranges = ranges
        .iter()
        .map(|range| format!("{}..{}", range.start, range.end))
        .collect::<Vec<_>>()
        .join(",");
    let Some(first_newline) = request.find('\n') else {
        let options = options_with_seqdb_ranges(request, &ranges);
        return options;
    };
    let (options, rest) = request.split_at(first_newline);
    format!("{}{rest}", options_with_seqdb_ranges(options, &ranges))
}

fn options_with_seqdb_ranges(options: &str, ranges: &str) -> String {
    let words = options.split_whitespace().collect::<Vec<_>>();
    let mut filtered = Vec::with_capacity(words.len() + 2);
    let mut i = 0;
    while i < words.len() {
        let word = words[i];
        if word == "--seqdb_ranges" {
            i += 2;
            continue;
        }
        if word.starts_with("--seqdb_ranges=") {
            i += 1;
            continue;
        }
        filtered.push(word);
        i += 1;
    }
    filtered.push("--seqdb_ranges");
    filtered.push(ranges);
    filtered.join(" ")
}
