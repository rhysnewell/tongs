use super::*;

pub(super) fn run_worker_pool_search(
    worker_pool: &WorkerPool,
    request: &str,
    command: &ClientCommand,
    total_nmodels: u64,
    total_nseqs: u64,
) -> std::io::Result<Option<Vec<u8>>> {
    let active_workers = {
        let mut workers = worker_pool
            .lock()
            .map_err(|_| std::io::Error::other("worker pool lock poisoned"))?;
        if workers.is_empty() {
            return Ok(None);
        }
        std::mem::take(&mut *workers)
    };

    let sharded_requests = worker_sharded_requests(
        request,
        command,
        active_workers.len(),
        total_nmodels,
        total_nseqs,
    );
    let (partials, mut surviving_workers) =
        run_worker_connections_search(active_workers, &sharded_requests);
    let complete = partials.len() == sharded_requests.len();

    if !surviving_workers.is_empty() {
        let mut workers = worker_pool
            .lock()
            .map_err(|_| std::io::Error::other("worker pool lock poisoned"))?;
        workers.append(&mut surviving_workers);
    }

    if complete {
        return merge_worker_payloads(&partials, total_nmodels, total_nseqs).map(Some);
    }

    if !partials.is_empty() {
        eprintln!(
            "Discarding incomplete worker search result: received {} of {} shards",
            partials.len(),
            sharded_requests.len()
        );
    }

    {
        let active_workers = {
            let mut workers = worker_pool
                .lock()
                .map_err(|_| std::io::Error::other("worker pool lock poisoned"))?;
            if workers.is_empty() {
                return Ok(None);
            }
            std::mem::take(&mut *workers)
        };

        let sharded_requests = worker_sharded_requests(
            request,
            command,
            active_workers.len(),
            total_nmodels,
            total_nseqs,
        );
        let (partials, mut surviving_workers) =
            run_worker_connections_search(active_workers, &sharded_requests);
        let complete = partials.len() == sharded_requests.len();
        if !surviving_workers.is_empty() {
            let mut workers = worker_pool
                .lock()
                .map_err(|_| std::io::Error::other("worker pool lock poisoned"))?;
            workers.append(&mut surviving_workers);
        }

        if complete {
            return merge_worker_payloads(&partials, total_nmodels, total_nseqs).map(Some);
        }
        if !partials.is_empty() {
            eprintln!(
                "Discarding incomplete worker retry result: received {} of {} shards",
                partials.len(),
                sharded_requests.len()
            );
        }
    }

    Ok(None)
}

fn run_worker_connections_search(
    workers: Vec<WorkerConnection>,
    request_bodies: &[WorkerRequest],
) -> (Vec<Vec<u8>>, Vec<WorkerConnection>) {
    let mut partials = Vec::new();
    let mut surviving_workers = Vec::new();
    for (idx, mut worker) in workers.into_iter().enumerate() {
        let Some(request_body) = request_bodies.get(idx) else {
            surviving_workers.push(worker);
            continue;
        };
        match run_worker_connection_search(&mut worker.stream, request_body) {
            Ok(payload) => {
                if let Err(e) = parse_worker_payload(&payload) {
                    eprintln!("Dropping failed worker connection: {}", e);
                } else {
                    partials.push(payload);
                    surviving_workers.push(worker);
                }
            }
            Err(e) => {
                eprintln!("Dropping failed worker connection: {}", e);
            }
        }
    }
    (partials, surviving_workers)
}

pub(super) fn worker_sharded_requests(
    request: &str,
    command: &ClientCommand,
    worker_count: usize,
    total_nmodels: u64,
    total_nseqs: u64,
) -> Vec<WorkerRequest> {
    if worker_count == 0 {
        return Vec::new();
    }

    if command.db_mode == DbMode::Hmm && command.db_slice.is_none() {
        return sharded_hmmdb_scan_requests(request, command, worker_count, total_nmodels);
    }

    if command.db_mode != DbMode::Seq {
        return (0..worker_count)
            .map(|_| WorkerRequest::ascii(HMMD_CMD_SEARCH, request))
            .collect();
    }

    if worker_count == 1 && command.seqdb_ranges.is_some() {
        return vec![WorkerRequest::ascii(HMMD_CMD_SEARCH, request)];
    }

    if worker_count == 1 {
        return if matches!(command.query, ClientQuery::Hmm(_)) {
            vec![binary_worker_request(
                request,
                command,
                HMMD_CMD_SEARCH,
                0,
                total_nseqs.min(u32::MAX as u64) as usize,
            )]
        } else {
            vec![WorkerRequest::ascii(HMMD_CMD_SEARCH, request)]
        };
    }

    let requested_ranges =
        coalesce_db_ranges(
            &command
                .seqdb_ranges
                .clone()
                .unwrap_or_else(|| match usize::try_from(total_nseqs) {
                    Ok(0) | Err(_) => Vec::new(),
                    Ok(total) => vec![DbRange {
                        start: 1,
                        end: total,
                    }],
                }),
        );
    if matches!(command.query, ClientQuery::Hmm(_)) && !db_ranges_are_contiguous(&requested_ranges)
    {
        return vec![WorkerRequest::ascii(HMMD_CMD_SEARCH, request)];
    }
    let total_targets = count_db_ranges(&requested_ranges);
    let empty_index = total_nseqs.min(usize::MAX as u64 - 1).saturating_add(1) as usize;

    let mut range_cursor = DbRangeCursor::new(&requested_ranges);
    let mut remaining = total_targets;
    let mut remaining_workers = worker_count;
    let mut requests = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let count = remaining / remaining_workers;
        if count == 0 {
            let ranged = request_with_seqdb_range(request, empty_index, empty_index);
            if matches!(command.query, ClientQuery::Hmm(_)) {
                requests.push(binary_worker_request(
                    &ranged,
                    command,
                    HMMD_CMD_SEARCH,
                    0,
                    count,
                ));
            } else {
                requests.push(WorkerRequest::ascii(HMMD_CMD_SEARCH, &ranged));
            }
        } else {
            let ranges = range_cursor.take(count);
            if ranges.is_empty() {
                let ranged = request_with_seqdb_range(request, empty_index, empty_index);
                if matches!(command.query, ClientQuery::Hmm(_)) {
                    requests.push(binary_worker_request(
                        &ranged,
                        command,
                        HMMD_CMD_SEARCH,
                        0,
                        count,
                    ));
                } else {
                    requests.push(WorkerRequest::ascii(HMMD_CMD_SEARCH, &ranged));
                }
            } else {
                let ranged = request_with_seqdb_ranges(request, &ranges);
                if matches!(command.query, ClientQuery::Hmm(_)) {
                    let (slice_start, slice_count) = covering_binary_slice(&ranges);
                    requests.push(binary_worker_request(
                        &ranged,
                        command,
                        HMMD_CMD_SEARCH,
                        slice_start,
                        slice_count,
                    ));
                } else {
                    requests.push(WorkerRequest::ascii(HMMD_CMD_SEARCH, &ranged));
                }
            }
            remaining -= count;
        }
        remaining_workers -= 1;
    }
    requests
}

fn covering_binary_slice(ranges: &[DbRange]) -> (usize, usize) {
    let Some(first) = ranges.first() else {
        return (0, 0);
    };
    let start = first.start.saturating_sub(1);
    let end = ranges
        .iter()
        .map(|range| range.end)
        .max()
        .unwrap_or(first.end);
    (start, end.saturating_sub(first.start).saturating_add(1))
}

fn db_ranges_are_contiguous(ranges: &[DbRange]) -> bool {
    ranges
        .windows(2)
        .all(|window| window[0].end.saturating_add(1) == window[1].start)
}

// Coalesce a list of (possibly overlapping or unsorted) `--seqdb_ranges` into a
// sorted set of disjoint ranges that together cover exactly the distinct target
// indices the user asked for.
//
// FAITHFULNESS TO C: the C hmmpgmd master (`hmmdmstr.c` lines 346-385) never
// assigns the same target to two workers. It walks the database by *physical
// position* with a monotonically advancing `inx`, testing each position once via
// `hmmpgmd_IsWithinRanges`, and hands each position to exactly one worker
// (`inx += worker->srch_cnt`). The sharded master (`hmmdmstr_shard.c`) likewise
// gives each worker a disjoint modulo-`num_shards` partition. In both variants
// every target is scored exactly once, so `forward_results()` only sorts and runs
// `p7_tophits_Threshold` — it does NOT run `p7_tophits_RemoveDuplicates` (that
// primitive is exclusive to nhmmer/nhmmscan's overlapping-window pipeline). There
// is therefore no "overlapping shard" hit-dedup scheme in C hmmpgmd to reconstruct.
//
// The only way the Rust master could produce overlapping shards is by slicing a
// user-supplied overlapping range string (e.g. `1..5,3..8`) directly. Coalescing
// here reproduces C's "visit each index once" semantics up front, so the existing
// exact, byte-identical worker-payload merge stays correct: each surviving target
// is emitted by exactly one worker and the concatenated hit lists need no dedup.
pub(super) fn coalesce_db_ranges(ranges: &[DbRange]) -> Vec<DbRange> {
    let mut sorted: Vec<DbRange> = ranges
        .iter()
        .copied()
        .filter(|r| r.end >= r.start)
        .collect();
    sorted.sort_by(|a, b| a.start.cmp(&b.start).then(a.end.cmp(&b.end)));
    let mut out: Vec<DbRange> = Vec::with_capacity(sorted.len());
    for range in sorted {
        match out.last_mut() {
            // Merge when the next range overlaps or is directly adjacent to the
            // current accumulated range (adjacent so 1..3,4..6 collapses to 1..6,
            // matching one contiguous monotonic walk in C).
            Some(last) if range.start <= last.end.saturating_add(1) => {
                last.end = last.end.max(range.end);
            }
            _ => out.push(range),
        }
    }
    out
}

pub(super) fn count_db_ranges(ranges: &[DbRange]) -> usize {
    ranges
        .iter()
        .map(|range| range.end.saturating_sub(range.start).saturating_add(1))
        .sum()
}

pub(super) struct DbRangeCursor<'a> {
    pub(super) ranges: &'a [DbRange],
    pub(super) range_index: usize,
    pub(super) next: usize,
}

impl<'a> DbRangeCursor<'a> {
    pub(super) fn new(ranges: &'a [DbRange]) -> Self {
        Self {
            ranges,
            range_index: 0,
            next: ranges.first().map(|range| range.start).unwrap_or(0),
        }
    }

    pub(super) fn take(&mut self, mut count: usize) -> Vec<DbRange> {
        let mut out = Vec::new();
        while count > 0 && self.range_index < self.ranges.len() {
            let range = self.ranges[self.range_index];
            if self.next > range.end {
                self.range_index += 1;
                self.next = self
                    .ranges
                    .get(self.range_index)
                    .map(|range| range.start)
                    .unwrap_or(0);
                continue;
            }

            let take_count = count.min(range.end - self.next + 1);
            let end = self.next + take_count - 1;
            out.push(DbRange {
                start: self.next,
                end,
            });
            count -= take_count;
            self.next = end.saturating_add(1);
        }
        out
    }
}

#[derive(Clone)]
pub(super) struct WorkerRequest {
    pub(super) command: u32,
    pub(super) body: Vec<u8>,
}

impl WorkerRequest {
    fn ascii(command: u32, request: &str) -> Self {
        Self {
            command,
            body: request.as_bytes().to_vec(),
        }
    }
}
