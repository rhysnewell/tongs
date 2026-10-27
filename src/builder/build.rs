use super::*;

/// Derive the builder's alignment-column architecture assignment.
///
/// Returns one byte per alignment column: `x` for match columns and `.` for
/// insert columns. This is the same fragment-marking, weighting, and `--hand`
/// logic used by `build_hmm_from_msa_with_prior()`, exposed so `hmmbuild -O`
/// can write the post-processed MSA that corresponds to the built model.
pub fn model_mask_from_msa(
    msa: &Msa,
    abc: &Alphabet,
    symfrac: f32,
    fragthresh: f32,
    hand_arch: bool,
    weighting_strategy: RelativeWeighting,
) -> Vec<u8> {
    let mut ax = msa.digitize(abc);
    let weights = match weighting_strategy {
        RelativeWeighting::PositionBased => pb_weights(msa, abc, !hand_arch),
        RelativeWeighting::Gsc => gsc_weights(msa, abc),
        RelativeWeighting::Blosum { identity_cutoff } => blosum_weights(msa, abc, identity_cutoff),
        RelativeWeighting::Given => given_weights(msa),
        RelativeWeighting::None => vec![1.0; msa.nseq],
    };
    mark_fragments_old(&mut ax, abc, msa.alen, fragthresh);
    model_mask_from_digitized(msa, abc, &ax, &weights, symfrac, hand_arch)
}
/// Build a profile HMM from a multiple sequence alignment.
///
/// Pipeline: PB weights -> mark fragments -> assign match columns
/// (fast architecture, or `--hand` from RF) -> faux trace counts ->
/// effective Neff estimation -> Dirichlet priors ->
/// composition/consensus annotation -> E-value calibration.
/// Counterpart to C's `p7_Builder()` (with `build.c`'s `build_model()`).
#[allow(clippy::too_many_arguments)]
pub fn build_hmm_from_msa(
    msa: &Msa,
    abc: &Alphabet,
    bg: &Bg,
    symfrac: f32,
    fragthresh: f32,
    hand_arch: bool,
    weighting_strategy: RelativeWeighting,
    effn_strategy: EffectiveSeqNumber,
    seed: u32,
) -> Result<Hmm, String> {
    build_hmm_from_msa_with_prior(
        msa,
        abc,
        bg,
        symfrac,
        fragthresh,
        hand_arch,
        weighting_strategy,
        effn_strategy,
        PriorStrategy::Default,
        CalibrationConfig::default(),
        seed,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build_hmm_from_msa_with_prior(
    msa: &Msa,
    abc: &Alphabet,
    bg: &Bg,
    symfrac: f32,
    fragthresh: f32,
    hand_arch: bool,
    weighting_strategy: RelativeWeighting,
    effn_strategy: EffectiveSeqNumber,
    prior_strategy: PriorStrategy,
    calibration_config: CalibrationConfig,
    seed: u32,
) -> Result<Hmm, String> {
    build_hmm_from_msa_with_prior_and_max_insert(
        msa,
        abc,
        bg,
        symfrac,
        fragthresh,
        hand_arch,
        weighting_strategy,
        effn_strategy,
        prior_strategy,
        calibration_config,
        seed,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build_hmm_from_msa_with_prior_and_max_insert(
    msa: &Msa,
    abc: &Alphabet,
    bg: &Bg,
    symfrac: f32,
    fragthresh: f32,
    hand_arch: bool,
    weighting_strategy: RelativeWeighting,
    effn_strategy: EffectiveSeqNumber,
    prior_strategy: PriorStrategy,
    calibration_config: CalibrationConfig,
    seed: u32,
    max_insert_len: Option<usize>,
) -> Result<Hmm, String> {
    let k = abc.k;
    let nseq = msa.nseq;

    // Digitize the alignment, then mark fragment flanks as missing data
    // before architecture assignment and faux-trace counting, matching
    // esl_msa_MarkFragments_old() in the C builder flow.
    let mut ax = msa.digitize(abc);
    let weights = match weighting_strategy {
        RelativeWeighting::PositionBased => pb_weights(msa, abc, !hand_arch),
        RelativeWeighting::Gsc => gsc_weights(msa, abc),
        RelativeWeighting::Blosum { identity_cutoff } => blosum_weights(msa, abc, identity_cutoff),
        RelativeWeighting::Given => given_weights(msa),
        RelativeWeighting::None => vec![1.0; nseq],
    };
    mark_fragments_old(&mut ax, abc, msa.alen, fragthresh);

    let matassign: Vec<bool> =
        model_mask_from_digitized(msa, abc, &ax, &weights, symfrac, hand_arch)
            .into_iter()
            .map(|sym| sym == b'x')
            .collect();

    // Apply the #=GC MM model mask (do_modelmask, build.c:222): rewrite
    // non-gap residues in 'm'-marked columns to the degenerate "any" symbol
    // (Kp-3) before counting, so those columns get degenerate emission counts.
    // Architecture and weighting above used the unmasked residues, matching C
    // (do_modelmask runs at the top of matassign2hmm, after modelmaking).
    do_modelmask(&mut ax, abc, msa);

    let m = matassign.iter().filter(|&&x| x).count();
    if m == 0 {
        // No match columns — return a trivial HMM
        let mut hmm = Hmm::new(1, abc.abc_type, k);
        hmm.name = msa.name.clone();
        if let Some(ref acc) = msa.acc {
            hmm.acc = Some(acc.clone());
            hmm.flags |= P7H_ACC;
        }
        if let Some(ref desc) = msa.desc {
            hmm.desc = Some(desc.clone());
            hmm.flags |= P7H_DESC;
        }
        return Ok(hmm);
    }

    let mut hmm = Hmm::new(m, abc.abc_type, k);
    hmm.name = msa.name.clone();
    if let Some(ref acc) = msa.acc {
        hmm.acc = Some(acc.clone());
        hmm.flags |= P7H_ACC;
    }
    if let Some(ref desc) = msa.desc {
        hmm.desc = Some(desc.clone());
        hmm.flags |= P7H_DESC;
    }
    hmm.nseq = nseq as i32;
    hmm.eff_nseq = nseq as f32;

    let eff_nseq: f32 = weights.iter().sum::<f64>() as f32;
    hmm.eff_nseq = eff_nseq;

    // Step 2: Count weighted residues and transitions
    for node in 0..=m {
        for x in 0..k {
            hmm.mat[node][x] = 0.0;
            hmm.ins[node][x] = 0.0;
        }
        hmm.t[node] = [0.0; NTRANSITIONS];
    }

    let traces = faux_trace_from_msa(&ax, &matassign, abc);

    // Count observed residues and transitions from doctored faux traces,
    // matching C's build.c -> p7_trace_FauxFromMSA() -> p7_trace_Doctor()
    // -> p7_trace_Count() flow.
    for seq in 0..nseq {
        let w = weights[seq] as f32;
        if traces[seq].n == 0 {
            continue;
        }
        count_trace(&mut hmm, &ax[seq], w, &traces[seq], abc);
    }

    if let Some(max_insert_len) = max_insert_len {
        clamp_insert_self_transition_counts(&mut hmm, max_insert_len);
    }

    // Transfer rf/mm/cs/ca annotation from the MSA (annotate_model, build.c:338):
    // for each annotation line present, copy the original character at every
    // match column into a 1-based model array hmm->X[1..M], with index 0 = ' '.
    annotate_model(&mut hmm, &matassign, msa);

    // Store alignment column map for hmmalign --mapali.
    let mut map = vec![0i32; m + 1];
    let mut node = 0usize;
    for (col, &is_match) in matassign.iter().enumerate().take(msa.alen) {
        if is_match {
            node += 1;
            map[node] = (col + 1) as i32;
        }
    }
    hmm.map = Some(map);
    hmm.flags |= P7H_MAP;

    // Store the training alignment checksum for hmmalign --mapali verification.
    hmm.checksum = msa::checksum(msa, abc);
    hmm.flags |= P7H_CHKSUM;

    // Effective sequence number estimation acts on count HMMs, then the
    // resulting ratio rescales counts before final parameterization.
    let uniform_neff = match effn_strategy {
        EffectiveSeqNumber::Entropy {
            target_re,
            target_sigma,
        } => Some(crate::eweight::entropy_weight(
            &mut hmm,
            bg,
            prior_strategy,
            target_re,
            target_sigma,
        )),
        EffectiveSeqNumber::EntropyExp {
            target_re,
            target_sigma,
        } => {
            crate::eweight::entropy_weight_exp(
                &mut hmm,
                bg,
                prior_strategy,
                target_re,
                target_sigma,
            );
            None
        }
        EffectiveSeqNumber::Cluster { identity_cutoff } => {
            let nclusters = single_linkage_cluster_count(msa, abc, identity_cutoff) as f32;
            hmm.eff_nseq = nclusters;
            Some(nclusters)
        }
        EffectiveSeqNumber::None => {
            hmm.eff_nseq = hmm.nseq as f32;
            Some(hmm.nseq as f32)
        }
        EffectiveSeqNumber::Set(value) => {
            hmm.eff_nseq = value;
            Some(value)
        }
    };
    if let Some(neff) = uniform_neff {
        let scale = if hmm.nseq > 0 {
            neff as f64 / hmm.nseq as f64
        } else {
            1.0
        };
        crate::eweight::scale_counts(&mut hmm, scale);
    }

    // Parameterize emission/transition counts with the selected prior strategy.
    crate::prior::apply_priors_with_strategy(&mut hmm, prior_strategy);

    set_hmm_composition(&mut hmm);
    set_hmm_consensus(&mut hmm, abc);

    // E-value calibration by simulation (p7_builder.c:1010 -> p7_Calibrate).
    // A failed fit is fatal in C ("build failed: failed to determine fwd tau").
    crate::calibrate::calibrate_with_config(&mut hmm, abc, bg, seed, calibration_config)?;
    if abc.abc_type != AlphabetType::Amino {
        set_max_length_from_beta(&mut hmm, DEFAULT_WINDOW_BETA);
    }

    Ok(hmm)
}

pub fn copy_stockholm_cutoffs_to_hmm(cutoffs: msa::StockholmCutoffs, hmm: &mut Hmm) {
    if let Some([seq_cutoff, dom_cutoff]) = cutoffs.ga {
        hmm.cutoff[P7_GA1] = seq_cutoff;
        hmm.cutoff[P7_GA2] = dom_cutoff;
        hmm.flags |= P7H_GA;
    }
    if let Some([seq_cutoff, dom_cutoff]) = cutoffs.tc {
        hmm.cutoff[P7_TC1] = seq_cutoff;
        hmm.cutoff[P7_TC2] = dom_cutoff;
        hmm.flags |= P7H_TC;
    }
    if let Some([seq_cutoff, dom_cutoff]) = cutoffs.nc {
        hmm.cutoff[P7_NC1] = seq_cutoff;
        hmm.cutoff[P7_NC2] = dom_cutoff;
        hmm.flags |= P7H_NC;
    }
}

fn clamp_insert_self_transition_counts(hmm: &mut Hmm, max_insert_len: usize) {
    let max_insert_len = max_insert_len as f32;
    for node in 1..hmm.m {
        let max_ii = max_insert_len * hmm.t[node][MI];
        if hmm.t[node][II] > max_ii {
            hmm.t[node][II] = max_ii;
        }
    }
}
