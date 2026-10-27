use super::*;

pub(super) fn read_query_msa_hmms(args: &Args) -> Result<Vec<tongs::Hmm>, String> {
    let mut msas = match args.qformat.as_deref() {
        Some(format) if is_text_msa_query_format(format) => {
            read_text_msa_query(format, &args.hmmfile)
                .map_err(|e| format!("Error reading MSA query file: {e}"))?
        }
        _ => read_stockholm_query(&args.hmmfile)
            .map_err(|e| format!("Error reading MSA query file: {e}"))?,
    };
    if msas.is_empty() {
        return Err(format!(
            "Error reading MSA query file: no alignments found in {}",
            args.hmmfile.display()
        ));
    }

    let msa_count = msas.len();
    let mut unnamed = 0usize;
    for (idx, alignment) in msas.iter_mut().enumerate() {
        if alignment.name.is_empty() {
            if msa_count > 1 {
                return Err(format!(
                    "Name annotation is required for each alignment in a multi MSA query file; failed on #{}",
                    idx + 1
                ));
            }
            alignment.name = query_name_from_path(&args.hmmfile);
            unnamed += 1;
        }
    }
    if unnamed > 1 {
        return Err("Error assigning names to query alignments".to_string());
    }

    let first_abc_type = if args.dna {
        AlphabetType::Dna
    } else if args.rna {
        AlphabetType::Rna
    } else {
        guess_msa_alphabet(&msas[0])?
    };
    if first_abc_type != AlphabetType::Dna && first_abc_type != AlphabetType::Rna {
        return Err(
            "Error: Invalid alphabet type in query for nhmmer. Expect DNA or RNA.".to_string(),
        );
    }
    let abc = Alphabet::new(first_abc_type);
    let bg = nhmmer_query_bg(args, &abc)?;
    // nhmmer.c:887-894 sets the score system whenever a single sequence may
    // need it: --qsingle_seqs, or an MSA of one sequence under --singlemx.
    let single_builder = if args.singlemx || args.mxfile.is_some() {
        Some(nhmmer_builder(args, &abc, &bg)?)
    } else {
        None
    };

    let mut hmms = Vec::with_capacity(msas.len());
    for alignment in &msas {
        if args.dna && guess_msa_alphabet(alignment)? != AlphabetType::Dna {
            return Err(format!(
                "Error reading alignment from file {}: expected DNA query alignment",
                args.hmmfile.display()
            ));
        }
        if args.rna && guess_msa_alphabet(alignment)? != AlphabetType::Rna {
            return Err(format!(
                "Error reading alignment from file {}: expected RNA query alignment",
                args.hmmfile.display()
            ));
        }
        let hmm = if args.singlemx && alignment.nseq == 1 {
            build_nhmmer_singlemx_msa_hmm(alignment, &abc, &bg, single_builder.as_ref().unwrap())?
        } else {
            builder::build_hmm_from_msa_with_prior(
                alignment,
                &abc,
                &bg,
                0.5,
                0.5,
                // nhmmer has no --hand: p7_Builder always uses the fast
                // (--symfrac) architecture, even for A2M queries whose reader
                // supplies an implied RF line.
                false,
                builder::RelativeWeighting::PositionBased,
                builder::EffectiveSeqNumber::Entropy {
                    target_re: None,
                    target_sigma: Some(45.0),
                },
                PriorStrategy::Default,
                CalibrationConfig::default(),
                args.seed,
            )
            // nhmmer.c:933: p7_Fail("build failed: %s", builder->errbuf).
            .map_err(|e| format!("Error: build failed: {e}"))?
        };
        hmms.push(hmm);
    }
    Ok(hmms)
}

/// The background model the query builder must use.
///
/// C's nhmmer reads `--bgfile` into `bg_manual` (nhmmer.c:817-821), clones it
/// into `info[i].bg` (nhmmer.c:837-840), and hands *that* to both
/// `p7_builder_LoadScoreSystem()` (nhmmer.c:893) and `p7_SingleBuilder()`
/// (nhmmer.c:905). So a custom background changes the query model's match
/// emissions and insert emissions, not just the search null model.
pub(super) fn nhmmer_query_bg(args: &Args, abc: &Alphabet) -> Result<Bg, String> {
    let mut bg = Bg::new(abc);
    if let Some(path) = &args.bgfile {
        bg.read_file(abc, path).map_err(|e| format!("Error: {e}"))?;
    }
    Ok(bg)
}

/// Port of nhmmer.c:877-894: create the builder, give it the window settings,
/// and set its score system from `--mxfile` or the `--mx` built-in.
///
/// Note the builder is `p7_builder_Create(NULL, abc)` (nhmmer.c:878), i.e. `go`
/// is NULL, so it keeps the default seed 42 and the default EmL..Eft
/// calibration config. nhmmer's own `--seed` is *not* threaded into the query
/// model builder in C, and must not be here either.
pub(super) fn nhmmer_builder(args: &Args, abc: &Alphabet, bg: &Bg) -> Result<Builder, String> {
    let mut builder = Builder::new(abc.abc_type).with_window(args.w_length, args.w_beta);
    let result = if args.mxfile.is_some() {
        builder.set_score_system(args.mxfile.as_deref(), args.popen, args.pextend, bg, abc)
    } else {
        builder.load_score_system(&args.matrix, args.popen, args.pextend, bg, abc)
    };
    result.map_err(|e| format!("\nError: Failed to set single query seq score system:\n{e}\n"))?;
    Ok(builder)
}

fn build_nhmmer_singlemx_msa_hmm(
    alignment: &msa::Msa,
    abc: &Alphabet,
    bg: &Bg,
    builder: &Builder,
) -> Result<Hmm, String> {
    let mut dsq = vec![DSQ_SENTINEL];
    for &sym in &alignment.aseq[0] {
        if matches!(sym, b'-' | b'.' | b'_' | b'~') || sym.is_ascii_whitespace() {
            continue;
        }
        let code = abc.digitize_symbol(sym);
        if !abc.is_residue(code) {
            return Err(format!(
                "Error: nhmmer --singlemx sequence contains non-residue '{}'",
                sym as char
            ));
        }
        dsq.push(code);
    }
    if dsq.len() == 1 {
        return Err("Error: nhmmer --singlemx requires at least one residue".to_string());
    }
    dsq.push(DSQ_SENTINEL);
    let name = if !alignment.name.is_empty() {
        alignment.name.as_str()
    } else {
        alignment.sqname[0].as_str()
    };
    // p7_SingleBuilder(), nhmmer.c:931.
    let mut hmm = builder
        .single_builder(name, &dsq, dsq.len() - 2, abc, bg)
        .map_err(|e| format!("Error: build failed: {e}"))?;
    if let Some(ref acc) = alignment.acc {
        hmm.acc = Some(acc.clone());
        hmm.flags |= p7hmm::P7H_ACC;
    }
    if let Some(ref desc) = alignment.desc {
        hmm.desc = Some(desc.clone());
        hmm.flags |= p7hmm::P7H_DESC;
    }
    Ok(hmm)
}

fn is_afa_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("afa")
        || format.eq_ignore_ascii_case("afasta")
        || format.eq_ignore_ascii_case("alignedfasta")
        || format.eq_ignore_ascii_case("aligned-fasta")
}

fn is_a2m_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("a2m")
}

fn is_psiblast_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("psiblast")
}

fn is_clustal_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("clustal") || format.eq_ignore_ascii_case("clustallike")
}

fn is_selex_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("selex")
}

fn is_phylip_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("phylip")
}

fn is_phylips_query_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("phylips")
}

pub(super) fn is_text_msa_query_format(format: &str) -> bool {
    is_afa_query_format(format)
        || is_a2m_query_format(format)
        || is_psiblast_query_format(format)
        || is_clustal_query_format(format)
        || is_selex_query_format(format)
        || is_phylip_query_format(format)
        || is_phylips_query_format(format)
}

fn read_stockholm_query(path: &std::path::Path) -> tongs::errors::HmmerResult<Vec<msa::Msa>> {
    if path == std::path::Path::new("-") {
        msa::read_stockholm_from_reader(BufReader::new(std::io::stdin().lock()))
    } else {
        msa::read_stockholm(path)
    }
}

fn read_text_msa_query(format: &str, path: &std::path::Path) -> Result<Vec<msa::Msa>, String> {
    let bytes = read_query_file_bytes(path)
        .map_err(|e| format!("Error reading {} file: {e}", format.to_ascii_uppercase()))?;
    let mut reader = std::io::Cursor::new(bytes);
    let name = query_name_from_path(path);
    let result = if is_a2m_query_format(format) {
        msa::read_a2m_from_reader(&mut reader, name)
    } else if is_afa_query_format(format) {
        msa::read_afa_from_reader(&mut reader, name)
    } else if is_psiblast_query_format(format) {
        msa::read_psiblast_from_reader(&mut reader, name)
    } else if is_clustal_query_format(format) {
        msa::read_clustal_from_reader(&mut reader, name)
    } else if is_selex_query_format(format) {
        msa::read_selex_from_reader(&mut reader, name)
    } else if is_phylip_query_format(format) {
        msa::read_phylip_from_reader(&mut reader, name)
    } else if is_phylips_query_format(format) {
        msa::read_phylips_from_reader(&mut reader, name)
    } else {
        unreachable!("unsupported text MSA query format {format}")
    };
    result.map_err(|e| e.to_string())
}

fn query_name_from_path(path: &std::path::Path) -> String {
    if path == std::path::Path::new("-") {
        return "Query".to_string();
    }
    path.file_stem()
        .or_else(|| path.file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "Query".to_string())
}

fn guess_msa_alphabet(msa: &msa::Msa) -> Result<AlphabetType, String> {
    let mut counts = [0usize; 256];
    let mut total = 0usize;
    for row in &msa.aseq {
        for &raw in row {
            if matches!(raw, b'-' | b'.' | b'_' | b'~') || raw.is_ascii_whitespace() {
                continue;
            }
            let ch = raw.to_ascii_uppercase();
            counts[ch as usize] += 1;
            total += 1;
        }
    }
    if total == 0 {
        return Err("Unable to guess alphabet for empty query alignment".to_string());
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
    Err("Unable to guess alphabet for query alignment; please specify --dna or --rna".to_string())
}
