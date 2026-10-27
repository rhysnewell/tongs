use super::*;

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
// C writes WT lines only when eslMSA_HASWGTS is set, which --wpb, --wgsc and --wblosum do.
// --wnone and --wgiven leave it unset, so they return None.
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

// Parsed twice because only FullStockholm re-serializes byte-for-byte like esl_msafile_Write.
// Stdin is buffered so both parses see the same bytes.
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
