use super::*;

pub(super) fn validate_sequence_format(label: &str, format: Option<&str>) {
    if let Some(format) = format {
        if SequenceFormat::from_name(format).is_none() {
            eprintln!("{label}={format} is not a recognized input sequence file format");
            std::process::exit(1);
        }
    }
}

pub(super) fn open_query_seq_file(
    path: &std::path::Path,
    abc: &Alphabet,
    format: Option<&str>,
) -> tongs::errors::HmmerResult<sequence::SeqFile<Box<dyn std::io::Read>>> {
    if let Some(format) = format {
        sequence::open_seq_file_with_format(path, abc, SequenceFormat::from_name(format).unwrap())
    } else {
        sequence::open_seq_file(path, abc)
    }
}
pub(super) fn score_hmmscan_model(
    hmm: &tongs::Hmm,
    gm: &mut Profile,
    oprofile: &mut OProfile,
    bg: &Bg,
    sq: &Sequence,
    args: &Args,
    bit_cutoff: Option<BitCutoff>,
    model_idx: usize,
) -> (Option<tongs::tophits::Hit>, PipelineStats) {
    let mut local_bg = bg.clone();
    local_bg.set_filter(hmm.m, &hmm.compo);
    local_bg.set_length(sq.n);

    let mut pli = Pipeline::new();
    pli.new_model(gm);
    configure_thresholds(&mut pli, args);
    configure_acceleration(&mut pli, args);
    if let Some(z) = args.z_value {
        pli.z = z;
        pli.z_setby = tongs::pipeline::ZSetBy::Option;
    }
    if let Some(domz) = args.domz_value {
        pli.domz = domz;
        pli.domz_setby = tongs::pipeline::ZSetBy::Option;
    }
    if let Some(cutoff) = bit_cutoff {
        pli.use_bit_cutoffs = cutoff;
        if pli.new_model_thresholds(&hmm.cutoff).is_err() {
            return (None, PipelineStats::default());
        }
    }
    pli.do_alignment = true;
    pli.do_alignment_display = !args.noali;
    pli.n_targets = model_idx as u64;

    let mut th = TopHits::new();
    let hit = if pli.run(gm, oprofile, &local_bg, hmm, sq, &mut th) {
        th.hits.into_iter().next().map(|mut hit| {
            hit.name = hmm.name.clone();
            hit.acc = hmm.acc.clone().unwrap_or_default();
            hit.desc = hmm.desc.clone().unwrap_or_default();
            hit.n = hmm.m;
            hit
        })
    } else {
        None
    };
    (
        hit,
        PipelineStats {
            n_past_msv: pli.n_past_msv,
            n_past_bias: pli.n_past_bias,
            n_past_vit: pli.n_past_vit,
            n_past_fwd: pli.n_past_fwd,
        },
    )
}
pub(super) fn alphabet_from_hmms(hmms: &[tongs::Hmm]) -> Result<Alphabet, String> {
    let hmm = hmms
        .first()
        .ok_or_else(|| "pressed HMM database contains no models".to_string())?;
    Ok(Alphabet::new(hmm.abc_type))
}
