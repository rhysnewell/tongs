use crate::alphabet::{Alphabet, Dsq};
use crate::spensemble::{self, ClusterParams, SegmentPair};
use crate::trace::{State, Trace};
use crate::util::cmath::c_logf_to_f32;
use crate::util::random::MersenneTwister;

use super::null2::null2_ratio_for_code;
use super::DomainDefinitionStats;

const RT1: f32 = 0.25;
const RT2: f32 = 0.10;
const RT3: f32 = 0.20;
const NSAMPLES: usize = 200;

#[derive(Clone, Copy)]
pub(super) struct Job {
    pub ienv: usize,
    pub jenv: usize,
    pub null2_is_done: bool,
}

pub(super) trait Sampler {
    fn sample(&mut self, rng: &mut MersenneTwister, tr: &mut Trace);
    fn null2(&self, tr: &Trace, zstart: usize, zend: usize) -> Vec<f32>;
}

pub(super) fn find_domain_regions(
    btot: &[f32],
    etot: &[f32],
    mocc: &[f32],
    l: usize,
) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut i: i64 = -1;
    let mut triggered = false;

    for j in 1..=l {
        if !triggered {
            if mocc[j] - (btot[j] - btot[j - 1]) < RT2 || i == -1 {
                i = j as i64;
            }
            if mocc[j] >= RT1 {
                triggered = true;
            }
        } else if mocc[j] - (etot[j] - etot[j - 1]) < RT2 {
            if i >= 1 {
                regions.push((i as usize, j));
            }
            i = -1;
            triggered = false;
        }
    }
    regions
}

pub(super) fn is_multidomain_region(btot: &[f32], etot: &[f32], i: usize, j: usize) -> bool {
    let mut max = -1.0_f32;
    for z in i..=j {
        let expected_n = (etot[z] - etot[i - 1]).min(btot[j] - btot[z - 1]);
        max = max.max(expected_n);
    }
    max >= RT3
}

#[allow(clippy::too_many_arguments)]
pub(super) fn cluster_region(
    dsq: &[Dsq],
    l: usize,
    (ri, rj): (usize, usize),
    abc: &Alphabet,
    seed: u32,
    sampler: &mut impl Sampler,
    tr: &mut Trace,
    n2sc: &mut [f32],
    stats: &mut DomainDefinitionStats,
    jobs: &mut Vec<Job>,
) {
    let region_len = rj - ri + 1;
    let mut rng = MersenneTwister::new(seed);
    let mut segments = Vec::new();
    let mut trace_domains = Vec::new();
    n2sc[ri..=rj].fill(0.0);

    for trace_idx in 0..NSAMPLES {
        sampler.sample(&mut rng, tr);
        trace_domains.clear();
        let mut in_domain = false;
        let (mut seg_i, mut seg_k, mut seg_j, mut seg_m, mut zstart) = (0, 0, 0, 0, 0);
        for z in 0..tr.n {
            match tr.st[z] {
                State::B => {
                    in_domain = true;
                    seg_i = 0;
                    seg_k = 0;
                    zstart = z;
                }
                State::M if in_domain => {
                    if seg_i == 0 {
                        seg_i = tr.i[z];
                        seg_k = tr.k[z];
                    }
                    seg_j = tr.i[z];
                    seg_m = tr.k[z];
                }
                State::E if in_domain => {
                    if seg_i > 0 && seg_j > 0 {
                        segments.push(SegmentPair {
                            i: seg_i + ri - 1,
                            j: seg_j + ri - 1,
                            k: seg_k,
                            m: seg_m,
                            trace_idx,
                        });
                        trace_domains.push((seg_i, seg_j, zstart, z));
                    }
                    in_domain = false;
                }
                _ => {}
            }
        }

        let mut pos = 1usize;
        for &(sqfrom, sqto, tfrom, tto) in &trace_domains {
            let null2 = sampler.null2(tr, tfrom, tto);
            while pos <= sqfrom && pos <= region_len {
                n2sc[ri + pos - 1] += 1.0;
                pos += 1;
            }
            while pos <= sqto && pos <= region_len {
                let x = dsq[ri + pos - 1] as usize;
                n2sc[ri + pos - 1] += null2_ratio_for_code(&null2, abc, x);
                pos += 1;
            }
        }
        while pos <= region_len {
            n2sc[ri + pos - 1] += 1.0;
            pos += 1;
        }
    }

    for value in &mut n2sc[ri..=rj] {
        *value = c_logf_to_f32(*value / NSAMPLES as f32);
    }

    if segments.is_empty() {
        stats.nenvelopes += 1;
        jobs.push(Job {
            ienv: ri,
            jenv: rj,
            null2_is_done: false,
        });
        return;
    }
    let mut last_jenv = 0usize;
    for env in spensemble::cluster(&segments, NSAMPLES, &ClusterParams::default()) {
        let ienv = env.ienv.clamp(1, l);
        let jenv = env.jenv.clamp(1, l);
        if jenv < ienv {
            continue;
        }
        if ienv <= last_jenv {
            stats.noverlaps += 1;
        }
        stats.nenvelopes += 1;
        jobs.push(Job {
            ienv,
            jenv,
            null2_is_done: true,
        });
        last_jenv = jenv;
    }
}
