use super::*;

pub(super) fn nhmmscan_max_length(
    hmm: &tongs::hmm::Hmm,
    _w_length: Option<i32>,
    _w_beta: Option<f64>,
) -> i32 {
    // C nhmmscan parses and reports --w_beta/--w_length, but the scan path
    // searches pressed profiles with their serialized MAXL unchanged.
    if hmm.max_length > 0 {
        hmm.max_length
    } else {
        builder::max_length_from_beta(hmm, DEFAULT_WINDOW_BETA)
    }
}

pub(super) fn nhmmscan_long_target_ln_adjustment(
    seq_len: usize,
    max_length: i32,
    search_space_ln: f64,
) -> f64 {
    let nw_ratio = (seq_len as f32) / (max_length as f32);
    tongs::util::cmath::c_log_f64(nw_ratio as f64) + search_space_ln
}
pub(super) fn nucleotide_alphabet_from_hmms(hmms: &[tongs::Hmm]) -> Result<Alphabet, String> {
    let hmm = hmms
        .first()
        .ok_or_else(|| "pressed HMM database contains no models".to_string())?;
    match hmm.abc_type {
        tongs::alphabet::AlphabetType::Dna => Ok(Alphabet::dna()),
        tongs::alphabet::AlphabetType::Rna => Ok(Alphabet::rna()),
        other => Err(format!(
            "nhmmscan requires a DNA or RNA HMM database; found {:?}",
            other
        )),
    }
}
pub(super) fn convert_crick_hits_to_forward(hits: &mut [tongs::tophits::Hit], n: usize) {
    for hit in hits {
        for dom in &mut hit.dcl {
            let ali_hi = n as i64 - dom.iali + 1;
            let ali_lo = n as i64 - dom.jali + 1;
            dom.iali = ali_hi;
            dom.jali = ali_lo;
            let env_hi = n as i64 - dom.ienv + 1;
            let env_lo = n as i64 - dom.jenv + 1;
            dom.ienv = env_hi;
            dom.jenv = env_lo;
            if let Some(ref mut ad) = dom.ad {
                let ad_hi = n.saturating_sub(ad.sqfrom) + 1;
                let ad_lo = n.saturating_sub(ad.sqto) + 1;
                ad.sqfrom = ad_hi;
                ad.sqto = ad_lo;
            }
        }
    }
}
