use super::*;

pub(super) fn read_build_alignments_maybe_stdin(
    path: &std::path::Path,
    format: Option<&str>,
) -> tongs::errors::HmmerResult<Vec<BuildAlignment>> {
    if format.is_some_and(is_a2m_msa_format) {
        let msas = if path == std::path::Path::new("-") {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            msa::read_a2m_from_reader(&mut reader, "alignment".to_string())
        } else {
            msa::read_a2m(path)
        }?;
        Ok(msas
            .into_iter()
            .map(|msa| BuildAlignment {
                msa,
                cutoffs: msa::StockholmCutoffs::default(),
                // Easel's A2M reader sets an implied RF line (x for consensus
                // columns, . for inserts; esl_msafile_a2m.c:522-528), but C
                // hmmbuild only consults RF under --hand. Without it the
                // architecture comes from --symfrac like any other format,
                // so a lowercase (insert) A2M column can still become a match
                // state. The RF line is carried into the model's RF annotation
                // either way.
                force_hand_arch: false,
                original_stockholm: None,
            })
            .collect())
    } else if format.is_some_and(is_afa_msa_format) {
        let msas = if path == std::path::Path::new("-") {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            msa::read_afa_from_reader(&mut reader, "alignment".to_string())
        } else {
            msa::read_afa(path)
        }?;
        Ok(msas
            .into_iter()
            .map(|msa| BuildAlignment {
                msa,
                cutoffs: msa::StockholmCutoffs::default(),
                force_hand_arch: false,
                original_stockholm: None,
            })
            .collect())
    } else if format.is_some_and(is_psiblast_msa_format) {
        let msas = if path == std::path::Path::new("-") {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            msa::read_psiblast_from_reader(&mut reader, "alignment".to_string())
        } else {
            msa::read_psiblast(path)
        }?;
        Ok(msas
            .into_iter()
            .map(|msa| BuildAlignment {
                msa,
                cutoffs: msa::StockholmCutoffs::default(),
                force_hand_arch: false,
                original_stockholm: None,
            })
            .collect())
    } else if format.is_some_and(is_clustal_msa_format) {
        let msas = if path == std::path::Path::new("-") {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            msa::read_clustal_from_reader(&mut reader, "alignment".to_string())
        } else {
            msa::read_clustal(path)
        }?;
        Ok(msas
            .into_iter()
            .map(|msa| BuildAlignment {
                msa,
                cutoffs: msa::StockholmCutoffs::default(),
                force_hand_arch: false,
                original_stockholm: None,
            })
            .collect())
    } else if format.is_some_and(is_selex_msa_format) {
        let msas = if path == std::path::Path::new("-") {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            msa::read_selex_from_reader(&mut reader, "alignment".to_string())
        } else {
            msa::read_selex(path)
        }?;
        Ok(msas
            .into_iter()
            .map(|msa| BuildAlignment {
                msa,
                cutoffs: msa::StockholmCutoffs::default(),
                force_hand_arch: false,
                original_stockholm: None,
            })
            .collect())
    } else if format.is_some_and(is_phylip_msa_format) {
        let msas = if path == std::path::Path::new("-") {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            msa::read_phylip_from_reader(&mut reader, "alignment".to_string())
        } else {
            msa::read_phylip(path)
        }?;
        Ok(msas
            .into_iter()
            .map(|msa| BuildAlignment {
                msa,
                cutoffs: msa::StockholmCutoffs::default(),
                force_hand_arch: false,
                original_stockholm: None,
            })
            .collect())
    } else if format.is_some_and(is_phylips_msa_format) {
        let msas = if path == std::path::Path::new("-") {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            msa::read_phylips_from_reader(&mut reader, "alignment".to_string())
        } else {
            msa::read_phylips(path)
        }?;
        Ok(msas
            .into_iter()
            .map(|msa| BuildAlignment {
                msa,
                cutoffs: msa::StockholmCutoffs::default(),
                force_hand_arch: false,
                original_stockholm: None,
            })
            .collect())
    } else if path == std::path::Path::new("-") {
        let stdin = std::io::stdin();
        let mut bytes = Vec::new();
        stdin.lock().read_to_end(&mut bytes)?;
        let preserved = msa::read_stockholm_preserved_from_reader(BufReader::new(&bytes[..]))?;
        let full = msa::read_stockholm_full_from_reader(BufReader::new(&bytes[..]))?;
        Ok(preserved
            .into_iter()
            .zip(full)
            .map(|(record, original_stockholm)| BuildAlignment {
                msa: record.msa,
                cutoffs: record.cutoffs,
                force_hand_arch: false,
                original_stockholm: Some(original_stockholm),
            })
            .collect())
    } else {
        let full = msa::read_stockholm_full(path)?;
        Ok(msa::read_stockholm_preserved(path)?
            .into_iter()
            .zip(full)
            .map(|(record, original_stockholm)| BuildAlignment {
                msa: record.msa,
                cutoffs: record.cutoffs,
                force_hand_arch: false,
                original_stockholm: Some(original_stockholm),
            })
            .collect())
    }
}

pub(super) fn is_supported_msa_format(format: &str) -> bool {
    is_stockholm_msa_format(format)
        || is_a2m_msa_format(format)
        || is_afa_msa_format(format)
        || is_psiblast_msa_format(format)
        || is_clustal_msa_format(format)
        || is_selex_msa_format(format)
        || is_phylip_msa_format(format)
        || is_phylips_msa_format(format)
}

fn is_stockholm_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("stockholm")
        || format.eq_ignore_ascii_case("sto")
        || format.eq_ignore_ascii_case("pfam")
}

fn is_a2m_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("a2m")
}

fn is_afa_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("afa")
        || format.eq_ignore_ascii_case("afasta")
        || format.eq_ignore_ascii_case("alignedfasta")
        || format.eq_ignore_ascii_case("aligned-fasta")
}

fn is_psiblast_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("psiblast")
}

fn is_clustal_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("clustal") || format.eq_ignore_ascii_case("clustallike")
}

fn is_selex_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("selex")
}

fn is_phylip_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("phylip")
}

fn is_phylips_msa_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("phylips")
}
pub(super) fn guess_msa_alphabet(msa: &msa::Msa) -> Result<AlphabetType, String> {
    let mut counts = [0usize; 26];
    for row in &msa.aseq {
        for &ch in row {
            if ch.is_ascii_alphabetic() {
                counts[(ch.to_ascii_uppercase() - b'A') as usize] += 1;
            }
        }
    }
    let n: usize = counts.iter().sum();
    if n <= 10 {
        return Err("could not determine alignment alphabet from <=10 residues".to_string());
    }

    let idx = |ch: u8| (ch - b'A') as usize;
    let amino_only = b"EFIJLOPQZ"
        .iter()
        .map(|&ch| counts[idx(ch)])
        .sum::<usize>();
    if amino_only > 0 {
        return Ok(AlphabetType::Amino);
    }

    let dna_core = b"ACGTN".iter().map(|&ch| counts[idx(ch)]).sum::<usize>();
    let rna_core = b"ACGUN".iter().map(|&ch| counts[idx(ch)]).sum::<usize>();
    let frac = |x: usize| x as f64 / n as f64;
    let t = counts[idx(b'T')];
    let u = counts[idx(b'U')];
    if frac(dna_core) >= 0.98 && u == 0 {
        return Ok(AlphabetType::Dna);
    }
    if frac(rna_core) >= 0.98 && t == 0 {
        return Ok(AlphabetType::Rna);
    }

    let distinct = counts.iter().filter(|&&c| c > 0).count();
    if frac(dna_core.max(rna_core)) < 0.98 && distinct >= 15 {
        return Ok(AlphabetType::Amino);
    }

    Err("could not determine alignment alphabet".to_string())
}
