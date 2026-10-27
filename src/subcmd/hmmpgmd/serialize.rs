use super::*;

pub(super) fn write_c_status(
    stream: &mut TcpStream,
    status: u32,
    msg_size: u64,
) -> std::io::Result<()> {
    stream.write_all(&status.to_be_bytes())?;
    stream.write_all(&msg_size.to_be_bytes())
}

pub(super) fn write_c_error(stream: &mut TcpStream, message: &str) -> std::io::Result<()> {
    write_c_status(stream, 1, message.len() as u64)?;
    stream.write_all(message.as_bytes())
}

pub(super) fn write_c_stats(
    stream: &mut TcpStream,
    nmodels: u64,
    nseqs: u64,
    included: u64,
) -> std::io::Result<()> {
    let searched = nseqs.max(nmodels);
    for value in [0.0_f64, 0.0, 0.0, searched as f64, searched as f64] {
        stream.write_all(&value.to_bits().to_be_bytes())?;
    }
    stream.write_all(&[0, 0])?;
    for value in [nmodels, nseqs, 0, 0, 0, 0, 0, 0, included, u64::MAX] {
        stream.write_all(&value.to_be_bytes())?;
    }
    Ok(())
}

pub(super) fn serialize_c_search_payload(
    nmodels: u64,
    nseqs: u64,
    results: &mut Vec<SearchHit>,
) -> Vec<u8> {
    sort_search_hits(results);

    let mut hit_payload = Vec::new();
    let mut hit_offsets = Vec::with_capacity(results.len());
    for hit in &mut *results {
        hit_offsets.push(hit_payload.len() as u64);
        serialize_p7_hit(&mut hit_payload, hit);
    }

    let included = results.iter().filter(|hit| hit.nincluded > 0).count() as u64;
    let mut payload = Vec::with_capacity(
        HMMD_SEARCH_STATS_SIZE + hit_offsets.len().saturating_mul(8) + hit_payload.len(),
    );
    write_c_stats_payload(
        &mut payload,
        nmodels,
        nseqs,
        results.len() as u64,
        &WorkerSearchStats {
            nreported: results.len() as u64,
            nincluded: included,
            ..WorkerSearchStats::default()
        },
        &hit_offsets,
    );
    payload.extend_from_slice(&hit_payload);
    payload
}

pub(super) fn write_c_stats_payload(
    out: &mut Vec<u8>,
    nmodels: u64,
    nseqs: u64,
    nhits: u64,
    stats: &WorkerSearchStats,
    hit_offsets: &[u64],
) {
    let searched = nseqs.max(nmodels);
    for value in [0.0_f64, 0.0, 0.0, searched as f64, searched as f64] {
        out.extend_from_slice(&value.to_bits().to_be_bytes());
    }
    out.extend_from_slice(&[0, 0]);
    for value in [
        nmodels,
        nseqs,
        stats.n_past_msv,
        stats.n_past_bias,
        stats.n_past_vit,
        stats.n_past_fwd,
        nhits,
        stats.nreported,
        stats.nincluded,
    ] {
        out.extend_from_slice(&value.to_be_bytes());
    }
    if hit_offsets.is_empty() {
        out.extend_from_slice(&u64::MAX.to_be_bytes());
    } else {
        for offset in hit_offsets {
            out.extend_from_slice(&offset.to_be_bytes());
        }
    }
}

pub(super) fn serialize_p7_hit(out: &mut Vec<u8>, hit: &SearchHit) {
    let mut presence = 0;
    if !hit.acc.is_empty() {
        presence |= P7_HIT_ACC_PRESENT;
    }
    if !hit.desc.is_empty() {
        presence |= P7_HIT_DESC_PRESENT;
    }
    let name = hit.name.as_bytes();
    let acc = hit.acc.as_bytes();
    let desc = hit.desc.as_bytes();
    let serialized_size = P7_HIT_BASE_SIZE
        + name.len()
        + 1
        + if presence & P7_HIT_ACC_PRESENT != 0 {
            acc.len() + 1
        } else {
            0
        }
        + if presence & P7_HIT_DESC_PRESENT != 0 {
            desc.len() + 1
        } else {
            0
        };

    write_u32(out, serialized_size as u32);
    write_i32(out, hit.window_length);
    write_f64(out, hit.sortkey);
    write_f32(out, hit.score);
    write_f32(out, hit.pre_score);
    write_f32(out, hit.sum_score);
    write_f64(out, hit.lnp);
    write_f64(out, hit.pre_lnp);
    write_f64(out, hit.sum_lnp);
    write_f32(out, hit.nexpected);
    write_i32(out, hit.nregions);
    write_i32(out, hit.nclustered);
    write_i32(out, hit.noverlaps);
    write_i32(out, hit.nenvelopes);
    write_i32(out, hit.domains.len().min(i32::MAX as usize) as i32);
    write_u32(out, hit.flags);
    write_i32(out, hit.nreported);
    write_i32(out, hit.nincluded);
    write_i32(out, hit.best_domain);
    write_i64(out, hit.seqidx);
    write_i64(out, hit.subseq_start);
    out.push(presence);
    out.extend_from_slice(name);
    out.push(0);
    if presence & P7_HIT_ACC_PRESENT != 0 {
        out.extend_from_slice(acc);
        out.push(0);
    }
    if presence & P7_HIT_DESC_PRESENT != 0 {
        out.extend_from_slice(desc);
        out.push(0);
    }

    for domain in &hit.domains {
        serialize_p7_domain(out, hit, domain);
    }
}

fn serialize_p7_domain(out: &mut Vec<u8>, hit: &SearchHit, domain: &Domain) {
    write_u32(out, P7_DOMAIN_BASE_SIZE as u32);
    write_i64(out, domain.ienv);
    write_i64(out, domain.jenv);
    write_i64(out, domain.iali);
    write_i64(out, domain.jali);
    write_i64(out, 0);
    write_i64(out, 0);
    write_f32(out, domain.envsc);
    write_f32(out, domain.domcorrection);
    write_f32(out, domain.dombias);
    write_f32(out, domain.oasc);
    write_f32(out, domain.bitscore);
    write_f64(out, domain.lnp);
    write_i32(out, i32::from(domain.is_reported));
    write_i32(out, i32::from(domain.is_included));
    write_i32(out, 0);
    serialize_p7_alidisplay(out, hit, domain.ad.as_ref());
}

fn serialize_p7_alidisplay(out: &mut Vec<u8>, hit: &SearchHit, ad: Option<&AliDisplay>) {
    let n = ad.map(|ad| ad.model.len()).unwrap_or(0);
    let empty = "";
    let model = ad.map(|ad| ad.model.as_str()).unwrap_or(empty);
    let mline = ad.map(|ad| ad.mline.as_str()).unwrap_or(empty);
    let aseq = ad.map(|ad| ad.aseq.as_str()).unwrap_or(empty);
    let ppline = ad.map(|ad| ad.ppline.as_str()).unwrap_or(empty);
    let rfline = ad
        .and_then(|ad| (!ad.rfline.is_empty()).then_some(ad.rfline.as_str()))
        .unwrap_or(empty);
    let hmmfrom = ad.map(|ad| ad.hmmfrom as i32).unwrap_or(-1);
    let hmmto = ad.map(|ad| ad.hmmto as i32).unwrap_or(-1);
    let sqfrom = ad.map(|ad| ad.sqfrom as i64).unwrap_or(0);
    let sqto = ad.map(|ad| ad.sqto as i64).unwrap_or(0);

    let mut presence = P7_ALIDISPLAY_ASEQ_PRESENT;
    if !rfline.is_empty() {
        presence |= P7_ALIDISPLAY_RFLINE_PRESENT;
    }
    if !ppline.is_empty() {
        presence |= P7_ALIDISPLAY_PPLINE_PRESENT;
    }

    let mut serialized_size = P7_ALIDISPLAY_BASE_SIZE + model.len() + 1 + mline.len() + 1;
    if presence & P7_ALIDISPLAY_RFLINE_PRESENT != 0 {
        serialized_size += rfline.len() + 1;
    }
    if presence & P7_ALIDISPLAY_ASEQ_PRESENT != 0 {
        serialized_size += aseq.len() + 1;
    }
    if presence & P7_ALIDISPLAY_PPLINE_PRESENT != 0 {
        serialized_size += ppline.len() + 1;
    }
    for value in [
        &hit.hmm_name,
        &hit.hmm_acc,
        &hit.hmm_desc,
        &hit.seq_name,
        &hit.seq_acc,
        &hit.seq_desc,
    ] {
        serialized_size += value.len() + 1;
    }

    write_u32(out, serialized_size as u32);
    write_i32(out, n.min(i32::MAX as usize) as i32);
    write_i32(out, hmmfrom);
    write_i32(out, hmmto);
    write_i32(out, hit.model_length);
    write_i64(out, sqfrom);
    write_i64(out, sqto);
    write_i64(out, hit.sequence_length);
    out.push(presence);
    if presence & P7_ALIDISPLAY_RFLINE_PRESENT != 0 {
        write_c_string(out, rfline);
    }
    write_c_string(out, model);
    write_c_string(out, mline);
    if presence & P7_ALIDISPLAY_ASEQ_PRESENT != 0 {
        write_c_string(out, aseq);
    }
    if presence & P7_ALIDISPLAY_PPLINE_PRESENT != 0 {
        write_c_string(out, ppline);
    }
    write_c_string(out, &hit.hmm_name);
    write_c_string(out, &hit.hmm_acc);
    write_c_string(out, &hit.hmm_desc);
    write_c_string(out, &hit.seq_name);
    write_c_string(out, &hit.seq_acc);
    write_c_string(out, &hit.seq_desc);
}

fn write_c_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(value.as_bytes());
    out.push(0);
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn write_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn write_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn write_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_bits().to_be_bytes());
}

fn write_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_bits().to_be_bytes());
}

impl SearchHit {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_pipeline_hit(
        hit: &Hit,
        name: String,
        hmm: &Hmm,
        alidisplay_hmm_name: &str,
        seq_name: &str,
        seq_acc: &str,
        seq_desc: &str,
        sequence_length: usize,
    ) -> Self {
        let best_domain = hit
            .dcl
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.bitscore
                    .partial_cmp(&b.bitscore)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx as i32)
            .unwrap_or(-1);
        SearchHit {
            name,
            acc: hit.acc.clone(),
            desc: hit.desc.clone(),
            window_length: hit.n.min(i32::MAX as usize) as i32,
            sortkey: hit.sortkey,
            score: hit.score,
            pre_score: hit.pre_score,
            sum_score: hit.sum_score,
            lnp: hit.lnp,
            pre_lnp: hit.pre_lnp,
            sum_lnp: hit.sum_lnp,
            nexpected: hit.nexpected,
            nregions: hit.nregions.min(i32::MAX as usize) as i32,
            nclustered: hit.nclustered.min(i32::MAX as usize) as i32,
            noverlaps: hit.noverlaps.min(i32::MAX as usize) as i32,
            nenvelopes: hit.nenvelopes.min(i32::MAX as usize) as i32,
            flags: hit.flags,
            nreported: hit.nreported.min(i32::MAX as usize) as i32,
            nincluded: hit.nincluded.min(i32::MAX as usize) as i32,
            best_domain,
            seqidx: hit.seqidx,
            subseq_start: hit.subseq_start,
            hmm_name: alidisplay_hmm_name.to_string(),
            hmm_acc: hmm.acc.clone().unwrap_or_default(),
            hmm_desc: hmm.desc.clone().unwrap_or_default(),
            seq_name: seq_name.to_string(),
            seq_acc: seq_acc.to_string(),
            seq_desc: seq_desc.to_string(),
            model_length: hmm.m.min(i32::MAX as usize) as i32,
            sequence_length: sequence_length.min(i64::MAX as usize) as i64,
            domains: hit.dcl.clone(),
        }
    }
}
