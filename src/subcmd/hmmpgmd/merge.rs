use super::*;

pub(super) fn run_worker_connection_search(
    stream: &mut TcpStream,
    request: &WorkerRequest,
) -> std::io::Result<Vec<u8>> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    write_header(
        stream,
        HmmdHeader {
            length: checked_u32_frame_len(request.body.len(), "worker request body")?,
            command: request.command,
            status: 0,
        },
    )?;
    stream.write_all(&request.body)?;
    stream.flush()?;

    let mut status = [0; 12];
    stream.read_exact(&mut status)?;
    let code = u32::from_be_bytes(status[0..4].try_into().unwrap());
    let msg_size = u64::from_be_bytes(status[4..12].try_into().unwrap());
    let payload = read_limited_u64_body(
        stream,
        msg_size,
        MAX_HMMPGMD_SEARCH_RESPONSE_BODY,
        "worker status payload",
    )?;
    if code == 0 {
        Ok(payload)
    } else {
        Err(std::io::Error::other(
            String::from_utf8_lossy(&payload).into_owned(),
        ))
    }
}

pub(super) fn shutdown_worker_pool(worker_pool: &WorkerPool) -> std::io::Result<()> {
    let mut workers = worker_pool
        .lock()
        .map_err(|_| std::io::Error::other("worker pool lock poisoned"))?;
    while !workers.is_empty() {
        match shutdown_worker_connection(&mut workers[0].stream) {
            Ok(()) => {
                workers.swap_remove(0);
            }
            Err(e) => {
                eprintln!("Dropping failed worker connection during shutdown: {}", e);
                workers.swap_remove(0);
            }
        }
    }
    Ok(())
}

fn shutdown_worker_connection(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    write_header(
        stream,
        HmmdHeader {
            length: 0,
            command: HMMD_CMD_SHUTDOWN,
            status: 0,
        },
    )?;
    stream.flush()?;

    let header = read_header(stream)?;
    if header.command != HMMD_CMD_SHUTDOWN || header.status != 0 {
        return Err(std::io::Error::other("worker rejected SHUTDOWN"));
    }
    if header.length > 0 {
        let _ = read_limited_header_body(
            stream,
            header,
            MAX_HMMPGMD_SHUTDOWN_BODY,
            "worker shutdown response body",
        )?;
    }
    Ok(())
}

// Master-side merge of per-worker serialized search payloads.
//
// Mirrors C `forward_results()` (hmmer/src/hmmdmstr.c): gather every worker's
// P7_HIT byte records, sort them, and re-emit one HMMD_SEARCH_STATS block with
// a fresh hit_offsets array followed by the concatenated hit payloads.
//
// C re-runs `p7_tophits_Threshold(&th, pli)` over the merged list to recompute
// nreported/nincluded against the global Z. Here each worker is dispatched with
// the *global* nmodels/nseqs (see `searched_counts` / `serialize_c_search_payload`,
// which derive Z from those totals) and only searches a *non-overlapping* DB
// shard, so a hit appears in exactly one worker. Per-hit reporting/inclusion
// thresholds are independent given a fixed Z, hence summing the per-shard
// nreported/nincluded is identical to a single global threshold pass — we keep
// the C-compatible totals without needing access to the pipeline state here.
//
// DISTRIBUTED-COORDINATION GAP: the global stats (Z, domZ, n_past_*) are taken
// from the master's totals plus summed worker filter counts; this file does not
// reconstruct a P7_PIPELINE on the master to re-run Threshold, because the
// pipeline thresholds object is per-query and not part of the serialized wire
// payload. Per-hit reporting/inclusion thresholds are independent given a fixed
// global Z, so summing per-shard counts equals a single global Threshold pass.
//
// NO OVERLAPPING-SHARD DEDUP IS NEEDED: both C hmmpgmd master variants partition
// the database into DISJOINT pieces — by monotonic physical index range
// (`hmmdmstr.c` `inx += srch_cnt`) or by modulo shard (`hmmdmstr_shard.c`
// `my_shard`/`num_shards`). Every target is scored by exactly one worker, so C's
// `forward_results()` only sorts + Thresholds and never calls
// `p7_tophits_RemoveDuplicates` (that is solely an nhmmer/nhmmscan within-process
// concern for overlapping nucleotide windows). The Rust master likewise produces
// disjoint shards: master-generated ranges advance monotonically, and any
// user-supplied `--seqdb_ranges` are coalesced into disjoint ranges by
// `coalesce_db_ranges` before partitioning. Hence the concatenated worker hit
// lists contain each target once and this merge stays exact without dedup.
pub(super) fn merge_worker_payloads(
    partials: &[Vec<u8>],
    total_nmodels: u64,
    total_nseqs: u64,
) -> std::io::Result<Vec<u8>> {
    let mut hits = Vec::new();
    let mut stats = WorkerSearchStats::default();

    for payload in partials {
        let partial = parse_worker_payload(payload)?;
        stats.add(&partial.stats);
        hits.extend(partial.hits);
    }

    hits.sort_by(compare_serialized_hits);

    let mut hit_payload = Vec::new();
    let mut hit_offsets = Vec::with_capacity(hits.len());
    for hit in hits {
        hit_offsets.push(hit_payload.len() as u64);
        hit_payload.extend_from_slice(&hit.bytes);
    }

    let mut payload = Vec::with_capacity(
        HMMD_SEARCH_STATS_SIZE + hit_offsets.len().saturating_mul(8) + hit_payload.len(),
    );
    write_c_stats_payload(
        &mut payload,
        total_nmodels,
        total_nseqs,
        hit_offsets.len() as u64,
        &stats,
        &hit_offsets,
    );
    payload.extend_from_slice(&hit_payload);
    Ok(payload)
}

pub(super) struct ParsedWorkerPayload {
    pub(super) stats: WorkerSearchStats,
    pub(super) hits: Vec<SerializedHit>,
}

#[derive(Clone, Copy, Default)]
pub(super) struct WorkerSearchStats {
    pub(super) n_past_msv: u64,
    pub(super) n_past_bias: u64,
    pub(super) n_past_vit: u64,
    pub(super) n_past_fwd: u64,
    pub(super) nreported: u64,
    pub(super) nincluded: u64,
}

impl WorkerSearchStats {
    fn add(&mut self, other: &Self) {
        self.n_past_msv = self.n_past_msv.saturating_add(other.n_past_msv);
        self.n_past_bias = self.n_past_bias.saturating_add(other.n_past_bias);
        self.n_past_vit = self.n_past_vit.saturating_add(other.n_past_vit);
        self.n_past_fwd = self.n_past_fwd.saturating_add(other.n_past_fwd);
        self.nreported = self.nreported.saturating_add(other.nreported);
        self.nincluded = self.nincluded.saturating_add(other.nincluded);
    }
}

pub(super) struct SerializedHit {
    pub(super) sortkey: f64,
    pub(super) name: String,
    pub(super) first_domain_iali: i64,
    pub(super) first_domain_jali: i64,
    pub(super) bytes: Vec<u8>,
}

pub(super) fn compare_serialized_hits(a: &SerializedHit, b: &SerializedHit) -> std::cmp::Ordering {
    a.sortkey
        .partial_cmp(&b.sortkey)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.name.cmp(&b.name))
        .then_with(|| {
            let a_dir = if a.first_domain_iali < a.first_domain_jali {
                1
            } else {
                -1
            };
            let b_dir = if b.first_domain_iali < b.first_domain_jali {
                1
            } else {
                -1
            };
            if a_dir != b_dir {
                b_dir.cmp(&a_dir)
            } else {
                a.first_domain_iali.cmp(&b.first_domain_iali)
            }
        })
}
