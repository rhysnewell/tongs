use super::*;

pub(super) fn run_master(
    state: ServerState,
    cport: u16,
    wport: u16,
    ccncts: usize,
    wcncts: usize,
) -> ExitCode {
    let worker_listener = bind_listener("0.0.0.0", wport, "worker", wcncts);
    let worker_pool = Arc::new(Mutex::new(Vec::new()));
    let worker_init_body = Arc::new(master_init_body(&state));
    thread::spawn({
        let worker_pool = Arc::clone(&worker_pool);
        let worker_init_body = Arc::clone(&worker_init_body);
        move || accept_workers(worker_listener, worker_pool, worker_init_body)
    });

    let listener = bind_listener("0.0.0.0", cport, "client", ccncts);

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => match handle_client(&state, &worker_pool, &mut stream) {
                Ok(true) => {}
                Ok(false) => break,
                Err(e) => eprintln!("Connection error: {}", e),
            },
            Err(e) => eprintln!("Connection error: {}", e),
        }
    }

    ExitCode::SUCCESS
}

fn handle_client(
    state: &ServerState,
    worker_pool: &WorkerPool,
    stream: &mut TcpStream,
) -> std::io::Result<bool> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let request = read_client_request_from_reader(&mut reader)?;
    if request.trim().is_empty() || request.trim() == "QUIT" {
        writeln!(stream, "BYE")?;
        stream.flush()?;
        return Ok(true);
    }

    if request.trim_start().starts_with('!') {
        return handle_control_client(worker_pool, stream, reader, request);
    }

    if request.trim_start().starts_with('@') {
        match parse_c_client_command(&request) {
            Ok(command) => {
                let db_ok = matches!(
                    (state, command.db_mode),
                    (ServerState::Hmm(_), DbMode::Hmm) | (ServerState::Seq(_), DbMode::Seq)
                );
                if !db_ok {
                    write_c_error(stream, "Requested database is not loaded")?;
                    stream.flush()?;
                    return Ok(true);
                }
                let (nmodels, nseqs) = searched_counts(state, &command.query);
                let payload =
                    match run_worker_pool_search(worker_pool, &request, &command, nmodels, nseqs) {
                        Ok(Some(payload)) => payload,
                        Ok(None) => {
                            let mut results = run_query(state, &command, SearchRuntime::default())?;
                            serialize_c_search_payload(nmodels, nseqs, &mut results)
                        }
                        Err(e) => {
                            write_c_error(stream, &e.to_string())?;
                            stream.flush()?;
                            return Ok(true);
                        }
                    };
                write_c_status(stream, 0, payload.len() as u64)?;
                stream.write_all(&payload)?;
            }
            Err(e) => write_c_error(stream, &e)?,
        }
    } else {
        let query = ClientQuery::Sequence {
            name: "query".to_string(),
            desc: String::new(),
            seq: request.trim().to_string(),
        };
        let command = ClientCommand {
            db_mode: match state {
                ServerState::Hmm(_) => DbMode::Hmm,
                ServerState::Seq(_) => DbMode::Seq,
            },
            query,
            options: SearchOptions::default(),
            seqdb_ranges: None,
            db_slice: None,
        };
        let mut results = run_query(state, &command, SearchRuntime::default())?;
        write_hit_response(stream, &mut results);
    }

    stream.flush()?;
    Ok(true)
}

fn handle_control_client(
    worker_pool: &WorkerPool,
    stream: &mut TcpStream,
    mut reader: BufReader<TcpStream>,
    first_request: String,
) -> std::io::Result<bool> {
    let mut request = first_request;
    loop {
        if request.trim_start().starts_with("!shutdown") {
            shutdown_worker_pool(worker_pool)?;
            write_c_status(stream, 0, HMMD_SEARCH_STATS_SIZE as u64)?;
            write_c_stats(stream, 0, 0, 0)?;
            stream.flush()?;
            return Ok(false);
        }

        write_c_error(stream, "Unknown server command")?;
        stream.flush()?;

        let previous_timeout = stream.read_timeout()?;
        stream.set_read_timeout(Some(Duration::from_millis(50)))?;
        let next_request = match read_client_request_from_reader(&mut reader) {
            Ok(next_request) => next_request,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                stream.set_read_timeout(previous_timeout)?;
                return Ok(true);
            }
            Err(e) => {
                stream.set_read_timeout(previous_timeout)?;
                return Err(e);
            }
        };
        stream.set_read_timeout(previous_timeout)?;

        if next_request.trim().is_empty() {
            return Ok(true);
        }
        if !next_request.trim_start().starts_with('!') {
            write_c_error(stream, "Expected server command after control command")?;
            stream.flush()?;
            return Ok(true);
        }
        request = next_request;
    }
}
fn bind_listener(host: &str, port: u16, role: &str, backlog: usize) -> TcpListener {
    let addr = format!("{}:{}", host, port);
    let ip = host.parse::<std::net::Ipv4Addr>().unwrap_or_else(|e| {
        eprintln!("Cannot parse bind address {}: {}", host, e);
        std::process::exit(1);
    });
    let backlog = backlog.min(i32::MAX as usize) as i32;
    let socket =
        Socket::new(SocketDomain::IPV4, Type::STREAM, Some(Protocol::TCP)).unwrap_or_else(|e| {
            eprintln!("Cannot create socket for {}: {}", addr, e);
            std::process::exit(1);
        });
    if let Err(e) = socket.set_reuse_address(true) {
        eprintln!("Cannot set SO_REUSEADDR on {}: {}", addr, e);
        std::process::exit(1);
    }
    let sockaddr = SockAddr::from(SocketAddrV4::new(ip, port));
    socket.bind(&sockaddr).unwrap_or_else(|e| {
        eprintln!("Cannot bind to {}: {}", addr, e);
        std::process::exit(1);
    });
    socket.listen(backlog).unwrap_or_else(|e| {
        eprintln!("Cannot listen on {}: {}", addr, e);
        std::process::exit(1);
    });
    let listener = TcpListener::from(socket);
    eprintln!("Listening for {role} connections on {addr} (backlog {backlog})");
    listener
}

fn read_client_request_from_reader(reader: &mut impl BufRead) -> std::io::Result<String> {
    let mut first = String::new();
    reader.read_line(&mut first)?;
    if first.trim_start().starts_with('@') || first.trim_start().starts_with('!') {
        let mut request = first;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            let done = line.trim_end() == "//";
            request.push_str(&line);
            if done {
                break;
            }
        }
        Ok(request)
    } else {
        Ok(first.trim().to_string())
    }
}

fn write_hit_response(stream: &mut std::net::TcpStream, results: &mut [SearchHit]) {
    sort_search_hits(results);
    let _ = writeln!(stream, "HITS {}", results.len());
    for hit in results.iter() {
        let _ = writeln!(
            stream,
            "{}\t{}",
            hit.name,
            tongs::output::fmt_score(hit.score)
        );
    }
    let _ = writeln!(stream, "//");
    let _ = stream.flush();
}

pub(super) fn sort_search_hits(results: &mut [SearchHit]) {
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}
fn accept_workers(listener: TcpListener, worker_pool: WorkerPool, init_body: Arc<Vec<u8>>) {
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                let worker_pool = Arc::clone(&worker_pool);
                let init_body = Arc::clone(&init_body);
                thread::spawn(move || {
                    if let Err(e) = initialize_worker(&mut stream, &init_body) {
                        eprintln!("Worker connection error: {}", e);
                        return;
                    }
                    eprintln!("Worker initialized");
                    match worker_pool.lock() {
                        Ok(mut workers) => workers.push(WorkerConnection { stream }),
                        Err(_) => eprintln!("Worker pool lock poisoned"),
                    }
                });
            }
            Err(e) => eprintln!("Worker accept error: {}", e),
        }
    }
}

fn initialize_worker(stream: &mut TcpStream, init_body: &[u8]) -> std::io::Result<()> {
    write_header(
        stream,
        HmmdHeader {
            length: checked_u32_frame_len(init_body.len(), "worker init body")?,
            command: HMMD_CMD_INIT,
            status: 0,
        },
    )?;
    stream.write_all(init_body)?;
    stream.flush()?;

    let header = read_header(stream)?;
    if header.command != HMMD_CMD_INIT || header.status != 0 {
        return Err(std::io::Error::other("worker rejected INIT"));
    }
    let _ = read_limited_header_body(
        stream,
        header,
        MAX_HMMPGMD_INIT_BODY,
        "worker init response body",
    )?;
    Ok(())
}
