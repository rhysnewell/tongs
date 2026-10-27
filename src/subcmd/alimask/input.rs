use super::*;

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

pub(super) fn mark_fragments_old_full(
    msa: &mut msa::FullStockholm,
    abc: &Alphabet,
    fragthresh: f64,
) {
    for row in &mut msa.aseq {
        let rlen = row
            .iter()
            .filter(|&&ch| {
                let code = abc.digitize_symbol(ch);
                code != tongs::alphabet::DSQ_ILLEGAL
                    && code != tongs::alphabet::DSQ_IGNORED
                    && abc.is_residue(code)
            })
            .count();
        if (rlen as f64) > fragthresh * (msa.alen as f64) {
            continue;
        }

        for ch in row.iter_mut() {
            let code = abc.digitize_symbol(*ch);
            if code != tongs::alphabet::DSQ_ILLEGAL
                && code != tongs::alphabet::DSQ_IGNORED
                && abc.is_residue(code)
            {
                break;
            }
            *ch = b'~';
        }
        for ch in row.iter_mut().rev() {
            let code = abc.digitize_symbol(*ch);
            if code != tongs::alphabet::DSQ_ILLEGAL
                && code != tongs::alphabet::DSQ_IGNORED
                && abc.is_residue(code)
            {
                break;
            }
            *ch = b'~';
        }
    }
}
/// Compute the relative weights that C's `alimask` would write as `#=GS ... WT`
/// in `--modelrange` mode. Returns `Some` only for the weighting schemes that
/// raise `eslMSA_HASWGTS` in C (`--wpb`/`--wgsc`/`--wblosum`); `--wnone` and
/// `--wgiven` return `None` (no WT lines written), matching C.
pub(super) fn compute_output_weights(
    alignment: &msa::Msa,
    abc: &Alphabet,
    hand: bool,
    weighting_strategy: RelativeWeighting,
) -> Option<Vec<f64>> {
    match weighting_strategy {
        RelativeWeighting::PositionBased => Some(builder::pb_weights(alignment, abc, !hand)),
        RelativeWeighting::Gsc => Some(builder::gsc_weights(alignment, abc)),
        RelativeWeighting::Blosum { identity_cutoff } => {
            Some(builder::blosum_weights(alignment, abc, identity_cutoff))
        }
        RelativeWeighting::None | RelativeWeighting::Given => None,
    }
}

pub(super) fn is_rf_consensus(sym: u8) -> bool {
    sym != b'.' && sym != b'-' && sym != b'_' && sym != b'~'
}

/// Read the input alignment(s) twice: once into the simplified `StockholmMsa`
/// model (used for the coordinate/weighting/masking math) and once into the
/// full-fidelity `FullStockholm` model (used to re-serialize the masked MSA
/// byte-for-byte like C's `esl_msafile_Write`). When the input is stdin, the
/// bytes are buffered so both parses see identical content.
pub(super) fn read_stockholm_maybe_stdin(
    path: &std::path::Path,
) -> tongs::errors::HmmerResult<(Vec<msa::StockholmMsa>, Vec<msa::FullStockholm>)> {
    if path == std::path::Path::new("-") {
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut std::io::stdin().lock(), &mut bytes)
            .map_err(tongs::errors::HmmerError::Io)?;
        let preserved = msa::read_stockholm_preserved_from_reader(BufReader::new(&bytes[..]))?;
        let full = msa::read_stockholm_full_from_reader(BufReader::new(&bytes[..]))?;
        Ok((preserved, full))
    } else {
        let preserved = msa::read_stockholm_preserved(path)?;
        let full = msa::read_stockholm_full(path)?;
        Ok((preserved, full))
    }
}

pub(super) fn is_stockholm_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("stockholm")
        || format.eq_ignore_ascii_case("sto")
        || format.eq_ignore_ascii_case("pfam")
}

pub(super) fn is_pfam_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("pfam")
}
