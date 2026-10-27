use super::*;

pub(super) fn validate_args(args: &Args) {
    if let Some(ref format) = args.tformat {
        if format.eq_ignore_ascii_case("fasta") || is_fmindex_target_format(format) {
            // FASTA is already accepted by the sequence reader's auto-detection.
        } else {
            eprintln!(
                "nhmmer --tformat={} is not implemented: supported target format assertions are fasta and fmindex",
            format
        );
            std::process::exit(1);
        }
    }
    if let Some(ref format) = args.qformat {
        if is_stockholm_query_format(format) || is_text_msa_query_format(format) {
            // Supported below by the query loader.
        } else if query_sequence_format(format).is_some() {
            // Supported below by the single-sequence query loader.
        } else {
            eprintln!(
                "nhmmer --qformat={} is not implemented: supported query format assertions are fasta, uniprot, genbank, embl, ddbj, afa, a2m, psiblast, clustal, clustallike, selex, phylip, phylips, stockholm, and pfam",
                format
            );
            std::process::exit(1);
        }
    }
    if args.dna && args.rna {
        eprintln!("Error: options --dna and --rna are mutually exclusive");
        std::process::exit(1);
    }
    if args.qsingle_seqs
        && args
            .qformat
            .as_deref()
            .is_some_and(|f| f.eq_ignore_ascii_case("hmm"))
    {
        eprintln!("--qsingle_seqs flag is incompatible with an hmm-formatted query file");
        std::process::exit(1);
    }
    if matches!(args.w_length, Some(0..=3)) {
        eprintln!("Invalid window length value");
        std::process::exit(1);
    }
}

pub(super) fn query_alphabet(args: &Args, hmm: &Hmm) -> Alphabet {
    match (args.dna, args.rna, hmm.abc_type) {
        (true, _, tongs::alphabet::AlphabetType::Dna) => Alphabet::dna(),
        (_, true, tongs::alphabet::AlphabetType::Rna) => Alphabet::rna(),
        (true, _, _) => {
            eprintln!(
                "Error reading hmm from file {}: expected DNA query HMM",
                args.hmmfile.display()
            );
            std::process::exit(1);
        }
        (_, true, _) => {
            eprintln!(
                "Error reading hmm from file {}: expected RNA query HMM",
                args.hmmfile.display()
            );
            std::process::exit(1);
        }
        (_, _, tongs::alphabet::AlphabetType::Dna) => Alphabet::dna(),
        (_, _, tongs::alphabet::AlphabetType::Rna) => Alphabet::rna(),
        _ => {
            eprintln!("Error: Invalid alphabet type in query for nhmmer. Expect DNA or RNA.");
            std::process::exit(1);
        }
    }
}

pub(super) fn read_hmms(path: &std::path::Path) -> tongs::errors::HmmerResult<Vec<tongs::Hmm>> {
    if path == std::path::Path::new("-") {
        hmmfile::read_hmms_auto(BufReader::new(std::io::stdin().lock()))
    } else {
        hmmfile::read_hmm_file_auto(path)
    }
}

pub(super) fn read_query_hmms(args: &Args) -> Result<Vec<tongs::Hmm>, String> {
    if args.qsingle_seqs {
        return match read_query_sequence_hmms(
            args,
            args.qformat.as_deref().and_then(query_sequence_format),
            true,
        ) {
            Ok(hmms) => Ok(hmms),
            Err(seq_err) => match read_hmms(&args.hmmfile) {
                Ok(_) => Err(
                    "--qsingle_seqs flag is incompatible with an hmm-formatted query file"
                        .to_string(),
                ),
                Err(_) => Err(seq_err),
            },
        };
    }

    match args.qformat.as_deref() {
        Some(format) if is_stockholm_query_format(format) => read_query_msa_hmms(args),
        Some(format) if is_text_msa_query_format(format) => read_query_msa_hmms(args),
        Some(format) if query_sequence_format(format).is_some() => {
            read_query_sequence_hmms(args, query_sequence_format(format), true)
        }
        Some(format) => Err(format!(
            "nhmmer --qformat={} is not implemented: supported query format assertions are fasta, uniprot, genbank, embl, ddbj, afa, a2m, psiblast, clustal, clustallike, selex, phylip, phylips, stockholm, and pfam",
            format
        )),
        None => match read_hmms(&args.hmmfile) {
            Ok(hmms) => Ok(hmms),
            Err(hmm_err) => {
                if args.hmmfile == std::path::Path::new("-") {
                    Err(
                        "Must specify query file format (--qformat) to read <query file> from stdin ('-')"
                            .to_string(),
                    )
                } else {
                    match read_query_msa_hmms(args) {
                        Ok(hmms) => Ok(hmms),
                        Err(msa_err) => read_query_sequence_hmms(args, None, false).map_err(
                            |seq_err| {
                                format!("Error reading query file: {hmm_err}; {msa_err}; {seq_err}")
                            },
                        ),
                    }
                }
            }
        },
    }
}

fn read_query_sequence_hmms(
    args: &Args,
    format: Option<sequence::SequenceFormat>,
    explicit_format: bool,
) -> Result<Vec<Hmm>, String> {
    let bytes = read_query_file_bytes(&args.hmmfile)
        .map_err(|e| format!("Error reading query sequence file: {e}"))?;
    let abc_type = if args.dna {
        AlphabetType::Dna
    } else if args.rna {
        AlphabetType::Rna
    } else if format == Some(sequence::SequenceFormat::Fasta) || format.is_none() {
        guess_fasta_alphabet(&bytes)?
    } else {
        return Err(
            "Unable to guess alphabet for query sequence file; please specify --dna or --rna"
                .to_string(),
        );
    };
    if abc_type != AlphabetType::Dna && abc_type != AlphabetType::Rna {
        return Err(
            "Error: Invalid alphabet type in query for nhmmer. Expect DNA or RNA.".to_string(),
        );
    }

    let abc = Alphabet::new(abc_type);
    let mut sqf = sequence::SeqFile::new(std::io::Cursor::new(bytes), abc.clone());
    if let Some(format) = format {
        sqf = sqf.with_format(format);
    } else {
        sqf = sqf.with_fasta_only();
    }
    let format_name = format.map_or("FASTA", |format| match format {
        sequence::SequenceFormat::Fasta => "FASTA",
        sequence::SequenceFormat::UniProt => "UniProt",
        sequence::SequenceFormat::GenBank => "GenBank",
        sequence::SequenceFormat::Embl => "EMBL",
        sequence::SequenceFormat::Ddbj => "DDBJ",
        sequence::SequenceFormat::Stockholm => "Stockholm",
    });
    let mut sequences = Vec::new();
    let mut sq = Sequence::new();
    while sqf
        .read(&mut sq)
        .map_err(|e| format!("Error reading {format_name} query file: {e}"))?
    {
        sequences.push(sq.clone());
        sq.reuse();
    }
    if sequences.is_empty() {
        return Err(format!(
            "Error reading {format_name} query file: no sequences found in {}",
            args.hmmfile.display()
        ));
    }
    if !explicit_format && fasta_queries_are_alignment_ambiguous(&sequences) {
        return Err(
            "Query file type could be either aligned or unaligned; please specify (--qformat [afa|fasta])"
                .to_string(),
        );
    }

    let bg = nhmmer_query_bg(args, &abc)?;
    let builder = nhmmer_builder(args, &abc, &bg)?;
    sequences
        .iter()
        .map(|seq| {
            // p7_SingleBuilder(), nhmmer.c:905.
            let hmm = builder
                .single_builder(&seq.name, &seq.dsq, seq.n, &abc, &bg)
                .map_err(|e| format!("Error: build failed: {e}"))?;
            // Note: unlike hmmbuild.c:966, nhmmer never sets eff_nseq after
            // p7_SingleBuilder, so the written model carries no EFFN line.
            Ok(hmm)
        })
        .collect()
}

pub(super) fn read_query_file_bytes(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    if path == std::path::Path::new("-") {
        std::io::stdin().lock().read_to_end(&mut bytes)?;
    } else {
        std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    }
    Ok(bytes)
}

fn fasta_queries_are_alignment_ambiguous(sequences: &[Sequence]) -> bool {
    sequences.len() > 1
        && sequences
            .first()
            .is_some_and(|first| sequences.iter().all(|seq| seq.n == first.n))
}

fn guess_fasta_alphabet(bytes: &[u8]) -> Result<AlphabetType, String> {
    let mut counts = [0usize; 256];
    let mut total = 0usize;
    let mut in_header = false;
    let mut at_line_start = true;

    for &raw in bytes {
        if at_line_start && raw == b'>' {
            in_header = true;
            at_line_start = false;
            continue;
        }
        if raw == b'\n' || raw == b'\r' {
            in_header = false;
            at_line_start = true;
            continue;
        }
        if in_header {
            at_line_start = false;
            continue;
        }
        at_line_start = false;
        if raw.is_ascii_whitespace() {
            continue;
        }
        if raw == b'-' || raw == b'.' || raw == b'_' || raw == b'~' {
            continue;
        }
        if raw.is_ascii_alphabetic() {
            let ch = raw.to_ascii_uppercase();
            counts[ch as usize] += 1;
            total += 1;
        }
    }

    if total == 0 {
        return Err("Unable to guess alphabet for empty query sequence file".to_string());
    }

    let idx = |ch: u8| ch as usize;
    let a = counts[idx(b'A')];
    let c = counts[idx(b'C')];
    let g = counts[idx(b'G')];
    let t = counts[idx(b'T')];
    let u = counts[idx(b'U')];
    let n = counts[idx(b'N')] + counts[idx(b'X')];
    let dna_iupac = b"RYMKSWHBVD"
        .iter()
        .map(|&ch| counts[idx(ch)])
        .sum::<usize>();
    let dna_core = a + c + g + t + n + dna_iupac;
    let rna_core = a + c + g + u + n + dna_iupac;
    let frac = |count: usize| count as f64 / total as f64;

    if frac(dna_core) >= 0.98 && u == 0 {
        return Ok(AlphabetType::Dna);
    }
    if frac(rna_core) >= 0.98 && t == 0 {
        return Ok(AlphabetType::Rna);
    }
    Err(
        "Unable to guess alphabet for query sequence file; please specify --dna or --rna"
            .to_string(),
    )
}

pub(super) fn is_stockholm_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("stockholm")
        || format.eq_ignore_ascii_case("sto")
        || format.eq_ignore_ascii_case("pfam")
}

pub(super) fn query_sequence_format(format: &str) -> Option<sequence::SequenceFormat> {
    sequence::SequenceFormat::from_name(format)
}
