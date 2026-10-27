use super::*;

pub(super) fn run_query(
    state: &ServerState,
    command: &ClientCommand,
    runtime: SearchRuntime<'_>,
) -> std::io::Result<Vec<SearchHit>> {
    match (state, &command.query) {
        (ServerState::Hmm(db), ClientQuery::Sequence { name, desc, seq }) => {
            run_hmmdb_sequence_query(
                db,
                name,
                desc,
                seq,
                &command.options,
                command.db_slice,
                runtime,
            )
        }
        (ServerState::Hmm(_), ClientQuery::Hmm(_)) => {
            Err("A HMM cannot be used to search a hmm database".to_string())
        }
        (ServerState::Seq(db), ClientQuery::Sequence { name, seq, .. }) => {
            run_seqdb_sequence_query(
                db,
                name,
                seq,
                &command.options,
                command.seqdb_ranges.as_deref(),
                runtime,
            )
        }
        (ServerState::Seq(db), ClientQuery::Hmm(hmm)) => run_seqdb_hmm_query(
            db,
            hmm,
            &command.options,
            command.seqdb_ranges.as_deref(),
            runtime,
        ),
    }
    .map_err(std::io::Error::other)
}

fn run_hmmdb_sequence_query(
    db: &HmmDbState,
    query_name: &str,
    query_desc: &str,
    seq_text: &str,
    options: &SearchOptions,
    db_slice: Option<DbSlice>,
    runtime: SearchRuntime<'_>,
) -> Result<Vec<SearchHit>, String> {
    let dsq = db
        .abc
        .digitize_checked(seq_text.as_bytes())
        .map_err(|e| e.to_string())?;
    let l = dsq.len() - 2;

    if l == 0 {
        return Err("empty sequence".to_string());
    }

    if runtime.is_parallel() {
        use rayon::prelude::*;
        let partials = runtime.thread_pool.unwrap().install(|| {
            db.profiles
                .par_iter()
                .enumerate()
                .map(|(i, (gm, om))| {
                    if !db_index_is_in_slice(i, db_slice) {
                        return Ok(Vec::new());
                    }
                    score_hmmdb_sequence_model(
                        db, i, gm, om, query_name, query_desc, &dsq, l, options,
                    )
                })
                .collect::<Vec<_>>()
        });
        return flatten_search_results(partials);
    }

    let mut results = Vec::new();
    for (i, (gm, om)) in db.profiles.iter().enumerate() {
        if !db_index_is_in_slice(i, db_slice) {
            continue;
        }
        results.extend(score_hmmdb_sequence_model(
            db, i, gm, om, query_name, query_desc, &dsq, l, options,
        )?);
    }

    Ok(results)
}

#[allow(clippy::too_many_arguments)]
fn score_hmmdb_sequence_model(
    db: &HmmDbState,
    index: usize,
    gm: &Profile,
    om: &OProfile,
    query_name: &str,
    query_desc: &str,
    dsq: &[u8],
    l: usize,
    options: &SearchOptions,
) -> Result<Vec<SearchHit>, String> {
    let mut local_bg = db.bg.clone();
    local_bg.set_length(l);
    let mut local_gm = gm.clone();
    let mut local_om = om.clone();
    let sq = Sequence {
        name: query_name.to_string(),
        acc: String::new(),
        desc: query_desc.to_string(),
        dsq: dsq.to_vec(),
        n: l,
        l,
        taxid: -1,
    };

    let mut pli = Pipeline::new();
    configure_hmmpgmd_pipeline(&mut pli, options, &db.hmms[index])?;
    pli.new_model(&local_gm);
    let mut th = TopHits::new();
    if !pli.run(
        &mut local_gm,
        &mut local_om,
        &local_bg,
        &db.hmms[index],
        &sq,
        &mut th,
    ) {
        return Ok(Vec::new());
    }

    Ok(th
        .hits
        .iter()
        .map(|hit| {
            SearchHit::from_pipeline_hit(
                hit,
                db.hmms[index].name.clone(),
                &db.hmms[index],
                &format!("{:09}", index + 1),
                query_name,
                "",
                query_desc,
                l,
            )
        })
        .collect())
}

fn db_index_is_in_slice(index: usize, db_slice: Option<DbSlice>) -> bool {
    db_slice
        .map(|slice| index >= slice.start && index < slice.start.saturating_add(slice.count))
        .unwrap_or(true)
}

fn run_seqdb_sequence_query(
    db: &SeqDbState,
    query_name: &str,
    seq_text: &str,
    options: &SearchOptions,
    seqdb_ranges: Option<&[DbRange]>,
    runtime: SearchRuntime<'_>,
) -> Result<Vec<SearchHit>, String> {
    let dsq = db
        .abc
        .digitize_checked(seq_text.as_bytes())
        .map_err(|e| e.to_string())?;
    let l = dsq.len() - 2;
    if l == 0 {
        return Err("empty sequence".to_string());
    }

    let hmm = seqmodel::build_single_seq_hmm(query_name, &dsq, l, &db.abc, &db.bg, 0.02, 0.4);
    let mut model_bg = db.bg.clone();
    model_bg.set_filter(hmm.m, &hmm.compo);

    let mut gm = Profile::new(hmm.m, &db.abc);
    profile::profile_config(&hmm, &model_bg, &mut gm, 400, P7_LOCAL);
    let om = OProfile::convert(&gm);

    if runtime.is_parallel() {
        use rayon::prelude::*;
        let partials = runtime.thread_pool.unwrap().install(|| {
            db.sequences
                .par_iter()
                .enumerate()
                .map(|(target_idx, target)| {
                    if !db_index_is_in_ranges(target_idx + 1, seqdb_ranges) {
                        return Ok(Vec::new());
                    }
                    score_seqdb_target(db, &hmm, &model_bg, &gm, &om, target, options)
                })
                .collect::<Vec<_>>()
        });
        return flatten_search_results(partials);
    }

    let mut results = Vec::new();
    for (target_idx, target) in db.sequences.iter().enumerate() {
        if !db_index_is_in_ranges(target_idx + 1, seqdb_ranges) {
            continue;
        }
        results.extend(score_seqdb_target(
            db, &hmm, &model_bg, &gm, &om, target, options,
        )?);
    }

    Ok(results)
}

fn run_seqdb_hmm_query(
    db: &SeqDbState,
    hmm: &Hmm,
    options: &SearchOptions,
    seqdb_ranges: Option<&[DbRange]>,
    runtime: SearchRuntime<'_>,
) -> Result<Vec<SearchHit>, String> {
    if hmm.abc_type != AlphabetType::Amino {
        return Err("Only amino HMM queries can search a protein sequence database".to_string());
    }

    let mut model_bg = db.bg.clone();
    model_bg.set_filter(hmm.m, &hmm.compo);

    let mut gm = Profile::new(hmm.m, &db.abc);
    profile::profile_config(hmm, &model_bg, &mut gm, 400, P7_LOCAL);
    let om = OProfile::convert(&gm);

    if runtime.is_parallel() {
        use rayon::prelude::*;
        let partials = runtime.thread_pool.unwrap().install(|| {
            db.sequences
                .par_iter()
                .enumerate()
                .map(|(target_idx, target)| {
                    if !db_index_is_in_ranges(target_idx + 1, seqdb_ranges) {
                        return Ok(Vec::new());
                    }
                    score_seqdb_target(db, hmm, &model_bg, &gm, &om, target, options)
                })
                .collect::<Vec<_>>()
        });
        return flatten_search_results(partials);
    }

    let mut results = Vec::new();
    for (target_idx, target) in db.sequences.iter().enumerate() {
        if !db_index_is_in_ranges(target_idx + 1, seqdb_ranges) {
            continue;
        }
        results.extend(score_seqdb_target(
            db, hmm, &model_bg, &gm, &om, target, options,
        )?);
    }

    Ok(results)
}

fn score_seqdb_target(
    _db: &SeqDbState,
    hmm: &Hmm,
    model_bg: &Bg,
    gm: &Profile,
    om: &OProfile,
    target: &Sequence,
    options: &SearchOptions,
) -> Result<Vec<SearchHit>, String> {
    let mut local_bg = model_bg.clone();
    let mut local_gm = gm.clone();
    let mut local_om = om.clone();
    let mut pli = Pipeline::new();
    configure_hmmpgmd_pipeline(&mut pli, options, hmm)?;
    pli.new_model(&local_gm);
    local_bg.set_length(target.n);

    let mut th = TopHits::new();
    if !pli.run(
        &mut local_gm,
        &mut local_om,
        &local_bg,
        hmm,
        target,
        &mut th,
    ) {
        return Ok(Vec::new());
    }

    Ok(th
        .hits
        .iter()
        .map(|hit| {
            SearchHit::from_pipeline_hit(
                hit,
                target.name.clone(),
                hmm,
                &hmm.name,
                &target.name,
                &target.acc,
                &target.desc,
                target.n,
            )
        })
        .collect())
}

fn flatten_search_results(
    partials: Vec<Result<Vec<SearchHit>, String>>,
) -> Result<Vec<SearchHit>, String> {
    let mut results = Vec::new();
    for partial in partials {
        results.extend(partial?);
    }
    Ok(results)
}

fn configure_hmmpgmd_pipeline(
    pli: &mut Pipeline,
    options: &SearchOptions,
    hmm: &Hmm,
) -> Result<(), String> {
    pli.e_value_threshold = options.e_value_threshold;
    pli.dom_e_value_threshold = options.dom_e_value_threshold;
    pli.inc_e = options.inc_e;
    pli.inc_dome = options.inc_dome;
    pli.f1 = if options.max { 1.0 } else { options.f1 };
    pli.f2 = if options.max { 1.0 } else { options.f2 };
    pli.f3 = if options.max { 1.0 } else { options.f3 };
    pli.do_max = options.max;
    pli.do_biasfilter = !(options.max || options.nobias);
    pli.do_null2 = !options.nonull2;

    if let Some(t) = options.t {
        pli.t = Some(t);
        pli.by_e = false;
    }
    if let Some(t) = options.dom_t {
        pli.dom_t = Some(t);
        pli.dom_by_e = false;
    }
    if let Some(t) = options.inc_t {
        pli.inc_t = Some(t);
        pli.inc_by_e = false;
    }
    if let Some(t) = options.inc_dom_t {
        pli.inc_dom_t = Some(t);
        pli.incdom_by_e = false;
    }

    pli.use_bit_cutoffs = options.bit_cutoff;
    if pli.use_bit_cutoffs != BitCutoff::None {
        pli.new_model_thresholds(&hmm.cutoff)?;
    }
    Ok(())
}

fn db_index_is_in_ranges(index: usize, ranges: Option<&[DbRange]>) -> bool {
    ranges
        .map(|ranges| {
            ranges
                .iter()
                .any(|range| index >= range.start && index <= range.end)
        })
        .unwrap_or(true)
}

pub(super) fn searched_counts(state: &ServerState, query: &ClientQuery) -> (u64, u64) {
    match (state, query) {
        (ServerState::Hmm(db), ClientQuery::Sequence { .. }) => (db.hmms.len() as u64, 1),
        (ServerState::Hmm(db), ClientQuery::Hmm(_)) => (db.hmms.len() as u64, 0),
        (ServerState::Seq(db), ClientQuery::Sequence { .. })
        | (ServerState::Seq(db), ClientQuery::Hmm(_)) => (1, db.sequences.len() as u64),
    }
}
