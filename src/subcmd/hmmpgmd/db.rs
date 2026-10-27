use super::*;

pub(super) fn load_hmmdb(hmmdb: &Path) -> ServerState {
    logsum::p7_flogsuminit();

    eprintln!("Loading HMM database: {}", hmmdb.display());
    let hmms = hmmfile::read_hmm_file_auto(hmmdb).unwrap_or_else(|e| {
        eprintln!("Error loading HMMs: {}", e);
        std::process::exit(1);
    });
    eprintln!("Loaded {} HMMs", hmms.len());

    let abc = Alphabet::amino();
    let bg = Bg::new(&abc);

    // Pre-build profiles
    let profiles: Vec<(Profile, OProfile)> = hmms
        .iter()
        .map(|hmm| {
            let mut gm = Profile::new(hmm.m, &abc);
            profile::profile_config(hmm, &bg, &mut gm, 400, P7_LOCAL);
            let om = OProfile::convert(&gm);
            (gm, om)
        })
        .collect();
    eprintln!("Profiles built");

    ServerState::Hmm(HmmDbState {
        path: hmmdb.to_path_buf(),
        hmms,
        abc,
        bg,
        profiles,
    })
}

pub(super) fn load_seqdb(seqdb: &PathBuf) -> ServerState {
    logsum::p7_flogsuminit();

    let abc = Alphabet::amino();
    let bg = Bg::new(&abc);

    eprintln!("Loading sequence database: {}", seqdb.display());
    let mut sqf = sequence::SeqFile::new(
        File::open(seqdb).unwrap_or_else(|e| {
            eprintln!("Error opening sequence database: {}", e);
            std::process::exit(1);
        }),
        abc.clone(),
    )
    .with_fasta_only();
    let mut sequences = Vec::new();
    let mut sq = Sequence::new();
    while sqf.read(&mut sq).unwrap_or_else(|e| {
        eprintln!("Error reading sequence database: {}", e);
        std::process::exit(1);
    }) {
        sequences.push(sq.clone());
        sq.reuse();
    }
    if sequences.is_empty() {
        eprintln!("Error: no sequences found in {}", seqdb.display());
        std::process::exit(1);
    }
    eprintln!("Loaded {} sequences", sequences.len());

    ServerState::Seq(SeqDbState {
        path: seqdb.clone(),
        cache_id: seqdb_cache_id(seqdb),
        sequences,
        abc,
        bg,
    })
}

fn seqdb_cache_id(seqdb: &PathBuf) -> String {
    let Ok(file) = File::open(seqdb) else {
        return String::new();
    };
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    if reader
        .read_line(&mut line)
        .ok()
        .filter(|n| *n > 0)
        .is_none()
    {
        return String::new();
    }
    let Some(rest) = line.strip_prefix('#') else {
        return String::new();
    };

    let mut fields = rest.split_whitespace();
    let _res_count = fields.next();
    let _seq_count = fields.next();
    let db_count = fields
        .next()
        .and_then(|field| field.parse::<usize>().ok())
        .unwrap_or(0);
    for _ in 0..db_count.saturating_mul(2) {
        let _ = fields.next();
    }
    fields.collect::<Vec<_>>().join(" ")
}
