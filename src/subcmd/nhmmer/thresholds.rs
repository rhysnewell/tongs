use super::*;

#[derive(Clone, Copy)]
pub(crate) struct NhmmerThresholdConfig {
    pub(super) use_bit_cutoffs: tongs::pipeline::BitCutoff,
    pub(super) by_e: bool,
    pub(super) dom_by_e: bool,
    pub(super) inc_by_e: bool,
    pub(super) incdom_by_e: bool,
    pub(super) e_value_threshold: f64,
    pub(super) dom_e_value_threshold: f64,
    pub(super) inc_e: f64,
    pub(super) inc_dome: f64,
    pub(super) t: Option<f64>,
    pub(super) dom_t: Option<f64>,
    pub(super) inc_t: Option<f64>,
    pub(super) inc_dom_t: Option<f64>,
}

impl NhmmerThresholdConfig {
    pub(crate) fn from_pipeline(pli: &Pipeline) -> Self {
        Self {
            use_bit_cutoffs: pli.use_bit_cutoffs,
            by_e: pli.by_e,
            dom_by_e: pli.dom_by_e,
            inc_by_e: pli.inc_by_e,
            incdom_by_e: pli.incdom_by_e,
            e_value_threshold: pli.e_value_threshold,
            dom_e_value_threshold: pli.dom_e_value_threshold,
            inc_e: pli.inc_e,
            inc_dome: pli.inc_dome,
            t: pli.t,
            dom_t: pli.dom_t,
            inc_t: pli.inc_t,
            inc_dom_t: pli.inc_dom_t,
        }
    }

    pub(super) fn apply_to(self, pli: &mut Pipeline) {
        pli.use_bit_cutoffs = self.use_bit_cutoffs;
        pli.by_e = self.by_e;
        pli.dom_by_e = self.dom_by_e;
        pli.inc_by_e = self.inc_by_e;
        pli.incdom_by_e = self.incdom_by_e;
        pli.e_value_threshold = self.e_value_threshold;
        pli.dom_e_value_threshold = self.dom_e_value_threshold;
        pli.inc_e = self.inc_e;
        pli.inc_dome = self.inc_dome;
        pli.t = self.t;
        pli.dom_t = self.dom_t;
        pli.inc_t = self.inc_t;
        pli.inc_dom_t = self.inc_dom_t;
    }
}

pub(super) fn longtarget_domain_hit(
    sq: &Sequence,
    src_hit: &tongs::tophits::Hit,
    dom: tongs::tophits::Domain,
    threshold_config: NhmmerThresholdConfig,
) -> tongs::tophits::Hit {
    let score = dom.bitscore;
    // C prints the raw (occasionally negative) bias correction in
    // tblout/domtblout and the human tables (p7_tophits.c:1664, unclamped);
    // do not clamp to 0.0. Matches the `tongs::pipeline` / `tongs::domaindef` unclamping.
    let bias = dom.dombias;
    let mut flags = src_hit.flags;
    let mut nreported = 1;
    let mut nincluded = 1;
    if threshold_config.use_bit_cutoffs != tongs::pipeline::BitCutoff::None {
        flags &= !(tongs::tophits::P7_IS_REPORTED
            | tongs::tophits::P7_IS_INCLUDED
            | tongs::tophits::P7_IS_DROPPED);
        let target_reported = threshold_config
            .t
            .map(|threshold| score as f64 >= threshold)
            .unwrap_or(src_hit.flags & tongs::tophits::P7_IS_REPORTED != 0);
        let target_included = threshold_config
            .inc_t
            .map(|threshold| score as f64 >= threshold)
            .unwrap_or(src_hit.flags & tongs::tophits::P7_IS_INCLUDED != 0);
        let reported = dom.is_reported && target_reported;
        let included = dom.is_included && reported && target_included;
        nreported = usize::from(reported);
        nincluded = usize::from(included);
        if reported {
            flags |= tongs::tophits::P7_IS_REPORTED;
        } else {
            flags |= tongs::tophits::P7_IS_DROPPED;
        }
        if included {
            flags |= tongs::tophits::P7_IS_INCLUDED;
        }
    }

    tongs::tophits::Hit {
        name: sq.name.clone(),
        acc: sq.acc.clone(),
        desc: sq.desc.clone(),
        n: sq.n,
        score,
        bias,
        pre_score: score + bias,
        sum_score: score,
        lnp: dom.lnp,
        pre_lnp: dom.lnp,
        sum_lnp: dom.lnp,
        sortkey: dom.lnp,
        nexpected: src_hit.nexpected,
        nregions: src_hit.nregions,
        nclustered: src_hit.nclustered,
        noverlaps: src_hit.noverlaps,
        nenvelopes: src_hit.nenvelopes,
        ndom: 1,
        nreported,
        nincluded,
        flags,
        seqidx: src_hit.seqidx,
        subseq_start: src_hit.subseq_start,
        dcl: vec![dom],
    }
}
