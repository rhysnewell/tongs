use super::*;

pub(super) fn run_worker(
    host: &str,
    wport: u16,
    cpu: usize,
    mut state: Option<ServerState>,
) -> ExitCode {
    eprintln!("Worker search CPU threads: {cpu}");
    let thread_pool = if cpu > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(cpu)
                .build()
                .unwrap_or_else(|e| {
                    eprintln!("Cannot initialize worker search thread pool: {e}");
                    std::process::exit(1);
                }),
        )
    } else {
        None
    };
    let addr = format!("{}:{}", host, wport);
    let mut stream = TcpStream::connect(&addr).unwrap_or_else(|e| {
        eprintln!("Cannot connect to master worker port {}: {}", addr, e);
        std::process::exit(1);
    });

    loop {
        let header = match read_header(&mut stream) {
            Ok(header) => header,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return ExitCode::SUCCESS;
            }
            Err(e) => {
                eprintln!("Worker read failed: {}", e);
                return ExitCode::FAILURE;
            }
        };
        let max_len = worker_command_body_limit(header.command);
        let body =
            match read_limited_header_body(&mut stream, header, max_len, "worker command body") {
                Ok(body) => body,
                Err(e) => {
                    eprintln!("Worker command body read failed: {}", e);
                    return ExitCode::FAILURE;
                }
            };

        match header.command {
            HMMD_CMD_INIT => {
                if state.is_none() {
                    match load_worker_init_state(&body) {
                        Ok(Some(init_state)) => state = Some(init_state),
                        Ok(None) => {}
                        Err(message) => {
                            eprintln!("Worker INIT failed: {}", message);
                            if let Err(e) = write_header(
                                &mut stream,
                                HmmdHeader {
                                    length: 0,
                                    command: HMMD_CMD_INIT,
                                    status: 1,
                                },
                            )
                            .and_then(|_| stream.flush())
                            {
                                eprintln!("Worker INIT error response failed: {}", e);
                            }
                            return ExitCode::FAILURE;
                        }
                    }
                }
                if let Err(e) = write_header(
                    &mut stream,
                    HmmdHeader {
                        length: HMMD_INIT_RESPONSE_SIZE as u32,
                        command: HMMD_CMD_INIT,
                        status: 0,
                    },
                )
                .and_then(|_| stream.write_all(&[0; HMMD_INIT_RESPONSE_SIZE]))
                .and_then(|_| stream.flush())
                {
                    eprintln!("Worker INIT response failed: {}", e);
                    return ExitCode::FAILURE;
                }
            }
            HMMD_CMD_SEARCH | HMMD_CMD_SCAN => {
                let runtime = SearchRuntime {
                    thread_pool: thread_pool.as_ref(),
                };
                let result = match state.as_ref() {
                    Some(state) => run_worker_search(state, &body, runtime),
                    None => Err("worker has no loaded --hmmdb or --seqdb".to_string()),
                };
                let write_result = match result {
                    Ok(payload) => write_c_status(&mut stream, 0, payload.len() as u64)
                        .and_then(|_| stream.write_all(&payload))
                        .and_then(|_| stream.flush()),
                    Err(message) => write_c_status(&mut stream, 1, message.len() as u64)
                        .and_then(|_| stream.write_all(message.as_bytes()))
                        .and_then(|_| stream.flush()),
                };
                if let Err(e) = write_result {
                    eprintln!("Worker search response failed: {}", e);
                    return ExitCode::FAILURE;
                }
            }
            HMMD_CMD_SHUTDOWN => {
                let _ = write_header(
                    &mut stream,
                    HmmdHeader {
                        length: 0,
                        command: HMMD_CMD_SHUTDOWN,
                        status: 0,
                    },
                );
                let _ = stream.flush();
                return ExitCode::SUCCESS;
            }
            _ => {
                eprintln!("Unknown worker command {}", header.command);
                return ExitCode::FAILURE;
            }
        }
    }
}

fn run_worker_search(
    state: &ServerState,
    body: &[u8],
    runtime: SearchRuntime<'_>,
) -> Result<Vec<u8>, String> {
    let command = parse_worker_command_body(body)?;
    let db_ok = matches!(
        (state, command.db_mode),
        (ServerState::Hmm(_), DbMode::Hmm) | (ServerState::Seq(_), DbMode::Seq)
    );
    if !db_ok {
        return Err("Requested database is not loaded".to_string());
    }
    let mut results = run_query(state, &command, runtime).map_err(|e| e.to_string())?;
    let (nmodels, nseqs) = searched_counts(state, &command.query);
    Ok(serialize_c_search_payload(nmodels, nseqs, &mut results))
}

pub(super) fn read_header(stream: &mut TcpStream) -> std::io::Result<HmmdHeader> {
    let mut buf = [0; HMMD_HEADER_SIZE];
    stream.read_exact(&mut buf)?;
    Ok(HmmdHeader {
        length: u32::from_ne_bytes(buf[0..4].try_into().unwrap()),
        command: u32::from_ne_bytes(buf[4..8].try_into().unwrap()),
        status: u32::from_ne_bytes(buf[8..12].try_into().unwrap()),
    })
}

fn worker_command_body_limit(command: u32) -> usize {
    match command {
        HMMD_CMD_INIT => MAX_HMMPGMD_INIT_BODY,
        HMMD_CMD_SEARCH | HMMD_CMD_SCAN => MAX_HMMPGMD_SEARCH_BODY,
        HMMD_CMD_SHUTDOWN => MAX_HMMPGMD_SHUTDOWN_BODY,
        _ => 0,
    }
}

pub(super) fn checked_u32_frame_len(len: usize, label: &str) -> std::io::Result<u32> {
    u32::try_from(len).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{label} is too large for an HMMD frame: {len} bytes"),
        )
    })
}

pub(super) fn read_limited_header_body(
    stream: &mut TcpStream,
    header: HmmdHeader,
    max_len: usize,
    label: &str,
) -> std::io::Result<Vec<u8>> {
    read_limited_u64_body(stream, u64::from(header.length), max_len, label)
}

pub(super) fn read_limited_u64_body(
    stream: &mut TcpStream,
    len: u64,
    max_len: usize,
    label: &str,
) -> std::io::Result<Vec<u8>> {
    let max_len_u64 = u64::try_from(max_len).unwrap_or(u64::MAX);
    if len > max_len_u64 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{label} is too large: {len} bytes > {max_len}"),
        ));
    }
    let len = usize::try_from(len).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{label} is too large for this platform: {len} bytes"),
        )
    })?;
    let mut body = vec![0; len];
    stream.read_exact(&mut body)?;
    Ok(body)
}

pub(super) fn write_header(stream: &mut TcpStream, header: HmmdHeader) -> std::io::Result<()> {
    stream.write_all(&header.length.to_ne_bytes())?;
    stream.write_all(&header.command.to_ne_bytes())?;
    stream.write_all(&header.status.to_ne_bytes())
}
