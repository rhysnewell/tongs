use super::*;
use clap::Parser;

#[test]
fn hmmpgmd_frame_length_guards_reject_oversized_bodies_before_allocation() {
    assert!(checked_u32_frame_len((u32::MAX as usize) + 1, "test frame").is_err());

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let join = std::thread::spawn(move || TcpStream::connect(addr).unwrap());
    let (mut server, _) = listener.accept().unwrap();
    let mut client = join.join().unwrap();

    let err = read_limited_u64_body(&mut server, u64::MAX, 1, "oversized payload").unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(err.to_string().contains("oversized payload is too large"));

    client.write_all(&[1, 2, 3, 4]).unwrap();
    let body = read_limited_u64_body(&mut server, 4, 4, "small payload").unwrap();
    assert_eq!(body, [1, 2, 3, 4]);
}

#[test]
fn master_with_both_databases_parses() {
    // Regression for audit H1: C `hmmpgmd.c` declares `--seqdb`/`--hmmdb`
    // with `incomp = "--worker"` only (NOT each other). The canonical
    // master invocation `--master --seqdb X --hmmdb Y` must parse without a
    // clap conflict error.
    let args = Args::try_parse_from(["hmmpgmd", "--master", "--seqdb", "x.fa", "--hmmdb", "y.hmm"])
        .expect("master may cache both --seqdb and --hmmdb");
    assert!(args.master);
    assert_eq!(args.seqdb.as_deref(), Some(std::path::Path::new("x.fa")));
    assert_eq!(args.hmmdb.as_deref(), Some(std::path::Path::new("y.hmm")));
    assert!(args.worker.is_none());
}

#[test]
fn served_databases_are_not_mutually_exclusive() {
    // Audit H1: the only forbidden DB combination C enforces via the option
    // table is master-vs-worker, NOT seqdb-vs-hmmdb. Confirm `--seqdb`
    // alone, `--hmmdb` alone, and both together all parse.
    for argv in [
        vec!["hmmpgmd", "--master", "--seqdb", "x.fa"],
        vec!["hmmpgmd", "--master", "--hmmdb", "y.hmm"],
        vec!["hmmpgmd", "--master", "--seqdb", "x.fa", "--hmmdb", "y.hmm"],
    ] {
        assert!(
            Args::try_parse_from(&argv).is_ok(),
            "should parse: {argv:?}"
        );
    }
}

#[test]
fn master_conflicts_with_worker() {
    // C `hmmpgmd.c`: `--master` is `incomp "--worker"` and vice-versa.
    let res = Args::try_parse_from(["hmmpgmd", "--master", "--worker", "127.0.0.1"]);
    assert!(res.is_err(), "--master and --worker are mutually exclusive");
}

#[test]
fn zero_hit_worker_payload_rejects_trailing_bytes() {
    let mut payload = Vec::new();
    write_c_stats_payload(&mut payload, 1, 2, 0, &WorkerSearchStats::default(), &[]);
    assert_eq!(payload.len(), HMMD_SEARCH_STATS_SIZE);
    assert!(parse_worker_payload(&payload).is_ok());

    payload.extend_from_slice(b"garbage");
    assert!(parse_worker_payload(&payload).is_err());
}

#[test]
fn worker_payload_rejects_huge_hit_count_before_allocation() {
    let mut payload = Vec::new();
    write_c_stats_payload(&mut payload, 1, 2, 0, &WorkerSearchStats::default(), &[]);
    payload[90..98].copy_from_slice(&u64::MAX.to_be_bytes());

    let err = match parse_worker_payload(&payload) {
        Ok(_) => panic!("huge worker hit count was accepted"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("hit offset table overflow")
            || err.to_string().contains("hit count too large")
    );
}

#[test]
fn hmmd_hmm_query_rejects_oversized_length_before_allocation() {
    let query = vec![0u8; C_P7_HMM_SHELL_SIZE];
    let err = parse_hmmd_hmm_query(&query, MAX_HMMD_HMM_QUERY_LENGTH + 1).unwrap_err();
    assert!(err.contains("query length out of range"));
}

fn sample_search_hit_with_domains() -> SearchHit {
    let n = 5usize;
    let ad = AliDisplay {
        model: "MKLVE".to_string(),
        mline: "MK+VE".to_string(),
        aseq: "MKIVE".to_string(),
        ppline: "*****".to_string(),
        rfline: "xxxxx".to_string(),
        hmmfrom: 1,
        hmmto: n,
        sqfrom: 10,
        sqto: 14,
    };
    let domain = Domain {
        iali: 10,
        jali: 14,
        ienv: 9,
        jenv: 15,
        bitscore: 42.5,
        lnp: -30.0,
        dombias: 0.5,
        oasc: 4.5,
        envsc: 50.0,
        domcorrection: 1.0,
        is_reported: true,
        is_included: true,
        ad: Some(ad),
    };
    SearchHit {
        name: "target1".to_string(),
        acc: "ACC1".to_string(),
        desc: "a description".to_string(),
        window_length: 0,
        sortkey: 42.5,
        score: 42.5,
        pre_score: 43.0,
        sum_score: 42.5,
        lnp: -30.0,
        pre_lnp: -29.0,
        sum_lnp: -30.0,
        nexpected: 1.0,
        nregions: 1,
        nclustered: 0,
        noverlaps: 0,
        nenvelopes: 1,
        flags: 0,
        nreported: 1,
        nincluded: 1,
        best_domain: 0,
        seqidx: 0,
        subseq_start: 0,
        hmm_name: "modelA".to_string(),
        hmm_acc: "PF00001".to_string(),
        hmm_desc: "model description".to_string(),
        seq_name: "target1".to_string(),
        seq_acc: "ACC1".to_string(),
        seq_desc: "a description".to_string(),
        model_length: n as i32,
        sequence_length: 100,
        domains: vec![domain],
    }
}

#[test]
fn full_hit_payload_passes_full_validation() {
    // Serialize a hit carrying a full P7_DOMAIN + P7_ALIDISPLAY payload and
    // confirm it round-trips through full (not shell-only) validation:
    // domain record + alidisplay record with all lines/strings present.
    let hit = sample_search_hit_with_domains();
    let mut out = Vec::new();
    serialize_p7_hit(&mut out, &hit);

    // Hit base + name/acc/desc, then exactly one domain (size 92) and one
    // alidisplay record must consume the whole buffer.
    let end = serialized_hit_payload_end(&out, 0).expect("hit payload should validate");
    assert_eq!(end, out.len(), "full hit payload must be exactly consumed");
    validate_serialized_hit_payload(&out).expect("full hit payload must validate");

    // The single domain must be exactly the C base size (no scores_per_pos).
    let hit_size = u32::from_be_bytes(out[0..4].try_into().unwrap()) as usize;
    let domain_size = u32::from_be_bytes(out[hit_size..hit_size + 4].try_into().unwrap()) as usize;
    assert_eq!(domain_size, P7_DOMAIN_BASE_SIZE);

    let (iali, jali) = serialized_hit_first_domain_alipos(&out).unwrap();
    assert_eq!((iali, jali), (10, 14));
    assert_eq!(serialized_hit_name(&out).unwrap(), "target1");
}

#[test]
fn full_hit_payload_round_trips_through_search_payload_and_merge() {
    // A complete search payload with a real domain/alidisplay hit must parse
    // and merge (master-side gather) without falling back to shell records.
    let mut results = vec![sample_search_hit_with_domains()];
    let payload = serialize_c_search_payload(1, 2, &mut results);

    let parsed = parse_worker_payload(&payload).expect("search payload must parse");
    assert_eq!(parsed.hits.len(), 1);
    assert_eq!(parsed.hits[0].name, "target1");

    let merged = merge_worker_payloads(&[payload], 1, 2).expect("merge must succeed");
    let reparsed = parse_worker_payload(&merged).expect("merged payload must parse");
    assert_eq!(reparsed.hits.len(), 1);
    // Merged hit bytes are byte-identical to the original serialized hit.
    assert_eq!(reparsed.hits[0].bytes, parsed.hits[0].bytes);
}

#[test]
fn coalesce_db_ranges_merges_overlapping_and_adjacent() {
    // Overlapping and adjacent user ranges collapse to the disjoint set of
    // distinct indices C visits once. 1..5 & 3..8 overlap -> 1..8; 10..12 &
    // 13..15 are adjacent -> 10..15; unsorted input is normalized.
    let coalesced = coalesce_db_ranges(&[
        DbRange { start: 13, end: 15 },
        DbRange { start: 1, end: 5 },
        DbRange { start: 3, end: 8 },
        DbRange { start: 10, end: 12 },
    ]);
    assert_eq!(
        coalesced,
        vec![DbRange { start: 1, end: 8 }, DbRange { start: 10, end: 15 }]
    );
    // Coalesced length is the distinct-index count (8 + 6), not the naive
    // sum of the raw ranges (5 + 6 + 3 + 3 = 17) which would double-count.
    assert_eq!(count_db_ranges(&coalesced), 14);
}

#[test]
fn overlapping_seqdb_ranges_produce_disjoint_worker_shards() {
    // A SEQUENCE query (ASCII-sharded) with overlapping user ranges must be
    // split across workers so that each target index is searched by exactly
    // one worker, matching C's "visit each physical index once" master.
    let request = "@--seqdb 1 --seqdb_ranges 1..5,3..8\n>q\nACDEF\n//\n";
    let command = ClientCommand {
        db_mode: DbMode::Seq,
        query: ClientQuery::Sequence {
            name: "q".to_string(),
            desc: String::new(),
            seq: "ACDEF".to_string(),
        },
        options: SearchOptions::default(),
        seqdb_ranges: Some(vec![
            DbRange { start: 1, end: 5 },
            DbRange { start: 3, end: 8 },
        ]),
        db_slice: None,
    };

    // Two workers split the coalesced 1..8 (8 distinct targets) into 4 + 4.
    let requests = worker_sharded_requests(request, &command, 2, 1, 8);
    assert_eq!(requests.len(), 2);

    // Collect the indices each worker is told to search and assert they form
    // a disjoint cover of exactly 1..=8 (no index appears in two shards).
    let mut all_indices = Vec::new();
    for req in &requests {
        let body = std::str::from_utf8(&req.body).unwrap();
        let ranges_str = body
            .split("--seqdb_ranges")
            .nth(1)
            .expect("worker request carries --seqdb_ranges")
            .split_whitespace()
            .next()
            .unwrap();
        for part in ranges_str.split(',') {
            let (s, e) = part.split_once("..").unwrap();
            for idx in s.parse::<usize>().unwrap()..=e.parse::<usize>().unwrap() {
                all_indices.push(idx);
            }
        }
    }
    all_indices.sort_unstable();
    let deduped: Vec<usize> = {
        let mut d = all_indices.clone();
        d.dedup();
        d
    };
    assert_eq!(
        all_indices, deduped,
        "no target index may be assigned to more than one worker"
    );
    assert_eq!(all_indices, (1..=8).collect::<Vec<_>>());
}

#[test]
fn merge_concatenates_disjoint_shard_hit_lists_without_duplication() {
    // Two workers each return a hit for a distinct target (disjoint shards,
    // as the master guarantees). The master merge must contain both hits,
    // each exactly once, sorted — no dedup pass drops or double-counts them.
    // sortkey is lnP-style (lower = more significant = sorted first), matching
    // Pipeline::hit_sortkey and the merge's ascending compare_serialized_hits.
    let mut a = sample_search_hit_with_domains();
    a.name = "targetA".to_string();
    a.seq_name = "targetA".to_string();
    a.sortkey = -50.0;
    let mut b = sample_search_hit_with_domains();
    b.name = "targetB".to_string();
    b.seq_name = "targetB".to_string();
    b.sortkey = -40.0;

    let payload_a = serialize_c_search_payload(1, 8, &mut vec![a]);
    let payload_b = serialize_c_search_payload(1, 8, &mut vec![b]);

    let merged = merge_worker_payloads(&[payload_a, payload_b], 1, 8).expect("merge must succeed");
    let reparsed = parse_worker_payload(&merged).expect("merged payload must parse");
    assert_eq!(reparsed.hits.len(), 2);
    let names: Vec<&str> = reparsed.hits.iter().map(|h| h.name.as_str()).collect();
    // Ascending sortkey: targetA (-50) is more significant, comes first.
    assert_eq!(names, vec!["targetA", "targetB"]);
}

#[test]
fn noncontiguous_seqdb_hmm_ranges_are_not_binary_sharded() {
    let request = "@--seqdb 1 --seqdb_ranges 1..1,3..3,5..5\nHMMER3/f\n//\n";
    let abc = Alphabet::amino();
    let command = ClientCommand {
        db_mode: DbMode::Seq,
        query: ClientQuery::Hmm(Hmm::new(1, AlphabetType::Amino, abc.k)),
        options: SearchOptions::default(),
        seqdb_ranges: Some(vec![
            DbRange { start: 1, end: 1 },
            DbRange { start: 3, end: 3 },
            DbRange { start: 5, end: 5 },
        ]),
        db_slice: None,
    };

    let requests = worker_sharded_requests(request, &command, 2, 0, 5);

    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].command, HMMD_CMD_SEARCH);
    assert!(requests[0].body.starts_with(b"@"));
    let body = std::str::from_utf8(&requests[0].body).unwrap();
    assert!(body.contains("--seqdb_ranges 1..1,3..3,5..5"));
}
