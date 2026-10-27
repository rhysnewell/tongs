use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn build_single_sequence_hmm(
    msa: &msa::Msa,
    abc: &Alphabet,
    bg: &Bg,
    matrix: &seqmodel::ScoreMatrix,
    popen: f64,
    pextend: f64,
    seed: u32,
    calibration: tongs::calibrate::CalibrationConfig,
    w_length: Option<usize>,
    w_beta: Option<f64>,
) -> tongs::Hmm {
    let mut dsq = vec![tongs::alphabet::DSQ_SENTINEL];
    for &sym in &msa.aseq[0] {
        if matches!(sym, b'-' | b'.' | b'_' | b'~') || sym.is_ascii_whitespace() {
            continue;
        }
        let code = abc.digitize_symbol(sym);
        if code == tongs::alphabet::DSQ_IGNORED || !abc.is_residue(code) {
            eprintln!(
                "Error: hmmbuild --singlemx sequence contains non-residue '{}'",
                sym as char
            );
            std::process::exit(1);
        }
        dsq.push(code);
    }
    if dsq.len() == 1 {
        eprintln!("Error: hmmbuild --singlemx requires at least one residue");
        std::process::exit(1);
    }
    dsq.push(tongs::alphabet::DSQ_SENTINEL);
    let name = msa.sqname[0].as_str();
    let mut builder = Builder::new(abc.abc_type)
        .with_seed(seed)
        .with_calibration(calibration)
        .with_window(w_length.map(|w| w.min(i32::MAX as usize) as i32), w_beta);
    builder
        .set_score_system_from_matrix(matrix.clone(), popen, pextend, bg, abc)
        .unwrap_or_else(|e| {
            eprintln!("\nError: Failed to set single query seq score system:\n{e}\n");
            std::process::exit(1);
        });
    // p7_SingleBuilder(), hmmbuild.c:963.
    let mut hmm = builder
        .single_builder(name, &dsq, dsq.len() - 2, abc, bg)
        .unwrap_or_else(|e| {
            eprintln!("\nError: build failed: {e}");
            std::process::exit(1);
        });
    hmm.eff_nseq = 1.0;
    if let Some(ref acc) = msa.acc {
        hmm.acc = Some(acc.clone());
        hmm.flags |= tongs::hmm::P7H_ACC;
    }
    if let Some(ref desc) = msa.desc {
        hmm.desc = Some(desc.clone());
        hmm.flags |= tongs::hmm::P7H_DESC;
    }
    hmm
}

pub(super) fn apply_window_length_options(
    hmm: &mut tongs::Hmm,
    alphabet_type: AlphabetType,
    w_length: Option<usize>,
    w_beta: Option<f64>,
) {
    if !matches!(alphabet_type, AlphabetType::Dna | AlphabetType::Rna) {
        return;
    }
    if let Some(w_length) = w_length {
        hmm.max_length = w_length.min(i32::MAX as usize) as i32;
    } else if let Some(w_beta) = w_beta {
        if w_beta == 0.0 {
            hmm.max_length = (hmm.m.saturating_mul(4)).min(i32::MAX as usize) as i32;
        } else {
            builder::set_max_length_from_beta(hmm, w_beta);
        }
    }
}

pub(super) fn apply_fixed_gap_params(
    hmm: &mut tongs::Hmm,
    popen: Option<f64>,
    pextend: Option<f64>,
) {
    use tongs::hmm::{DD, DM, II, IM, MD, MI, MM};

    for node in 0..=hmm.m {
        if let Some(popen) = popen {
            hmm.t[node][MM] = (1.0 - 2.0 * popen) as f32;
            hmm.t[node][MI] = popen as f32;
            hmm.t[node][MD] = popen as f32;
        }
        if let Some(pextend) = pextend {
            hmm.t[node][IM] = (1.0 - pextend) as f32;
            hmm.t[node][II] = pextend as f32;
            hmm.t[node][DM] = (1.0 - pextend) as f32;
            hmm.t[node][DD] = pextend as f32;
        }
    }

    if let Some(popen) = popen {
        hmm.t[hmm.m][MM] = (1.0 - popen) as f32;
    }
    hmm.t[hmm.m][MD] = 0.0;
    hmm.t[hmm.m][DM] = 1.0;
    hmm.t[hmm.m][DD] = 0.0;
}

pub(super) fn write_processed_msa(
    out: &mut dyn Write,
    alignment: &BuildAlignment,
    model_mask: &[u8],
    abc: &Alphabet,
    fragthresh: f32,
    hand_arch: bool,
    weighting_strategy: builder::RelativeWeighting,
) {
    let msa = &alignment.msa;
    let (rf, aseq) = builder::tracealigned_post_msa_rows(msa, abc, model_mask, fragthresh);
    let wgt = match weighting_strategy {
        builder::RelativeWeighting::None => None,
        builder::RelativeWeighting::PositionBased => {
            Some(builder::pb_weights(msa, abc, !hand_arch))
        }
        builder::RelativeWeighting::Gsc => Some(builder::gsc_weights(msa, abc)),
        builder::RelativeWeighting::Blosum { identity_cutoff } => {
            Some(builder::blosum_weights(msa, abc, identity_cutoff))
        }
        builder::RelativeWeighting::Given => msa.weights.clone(),
    };
    let (name, acc, desc, sqname) = if let Some(original) = alignment.original_stockholm.as_ref() {
        (
            original.name.clone(),
            original.acc.clone(),
            original.desc.clone(),
            original.sqname.clone(),
        )
    } else {
        (
            (!msa.name.is_empty()).then(|| msa.name.clone()),
            msa.acc.clone(),
            msa.desc.clone(),
            msa.sqname.clone(),
        )
    };
    let alen = rf.len();
    let nseq = sqname.len();
    let full = msa::FullStockholm {
        name,
        acc,
        desc,
        sqname,
        aseq,
        has_wgts: wgt.is_some(),
        wgt: wgt.unwrap_or_else(|| vec![-1.0; nseq]),
        sqacc: vec![None; nseq],
        sqdesc: vec![None; nseq],
        ss: vec![None; nseq],
        sa: vec![None; nseq],
        pp: vec![None; nseq],
        rf: Some(rf),
        alen,
        ..Default::default()
    };
    msa::write_stockholm_full(out, &full, 200, None).unwrap();
}
pub(super) fn mean_match_relative_entropy(hmm: &tongs::Hmm, bg: &Bg) -> f32 {
    let mut sum = 0.0_f32;
    for node in 1..=hmm.m {
        for x in 0..hmm.abc_k {
            let p = hmm.mat[node][x];
            if p > 0.0 && bg.f[x] > 0.0 {
                sum += p * c_log_f64((p / bg.f[x]) as f64) as f32 * (ESL_CONST_LOG2R as f32);
            }
        }
    }
    sum / hmm.m as f32
}
