use super::*;

/// Build the printable alignment-display strings (model, mline, aseq, ppline,
/// rfline) for the first domain in `tr`. Equivalent to a minimal
/// `p7_alidisplay_Create()` call. Use `alignment_display_with_pp` for real PPs.
pub fn alignment_display(
    tr: &Trace,
    dsq: &[Dsq],
    hmm: &crate::hmm::Hmm,
    abc: &Alphabet,
) -> Option<AliDisplay> {
    alignment_display_with_pp(tr, dsq, hmm, abc, None)
}

/// Like `alignment_display`, but uses the supplied posterior-probability
/// matrix to emit a real ppline. Port of `p7_alidisplay_Create()` with PPs.
pub fn alignment_display_with_pp(
    tr: &Trace,
    dsq: &[Dsq],
    hmm: &crate::hmm::Hmm,
    abc: &Alphabet,
    pp: Option<&crate::dp::gmx::Gmx>,
) -> Option<AliDisplay> {
    alignment_display_with_pp_bg(tr, dsq, hmm, abc, pp, None)
}

/// Like `alignment_display_with_pp` but with an override for the per-residue
/// background frequencies. Used in long_target mode where the model is
/// reparameterized: C's p7_alidisplay_Create compares `FGetEmission > 1.0`
/// (i.e. emission / reparameterized_bg > 1), so Rust's mline `+` threshold
/// needs the same reparameterized bg to be byte-identical.
pub fn alignment_display_with_pp_bg(
    tr: &Trace,
    dsq: &[Dsq],
    hmm: &crate::hmm::Hmm,
    abc: &Alphabet,
    pp: Option<&crate::dp::gmx::Gmx>,
    bg_override: Option<&[f32]>,
) -> Option<AliDisplay> {
    alignment_display_with_pp_emission_odds(tr, dsq, hmm, abc, pp, bg_override, None)
}

/// Build an alignment display, optionally using optimized-profile emission
/// odds for C-style match-line `+` decisions.
pub fn alignment_display_with_pp_emission_odds(
    tr: &Trace,
    dsq: &[Dsq],
    hmm: &crate::hmm::Hmm,
    abc: &Alphabet,
    pp: Option<&crate::dp::gmx::Gmx>,
    bg_override: Option<&[f32]>,
    emission_odds: Option<&dyn Fn(usize, usize) -> Option<f32>>,
) -> Option<AliDisplay> {
    // C anchors the display window on M states ONLY (p7_alidisplay.c:108-119):
    // z1 = first p7T_M state, z2 = last p7T_M state. Interior D/I between them
    // are still emitted, but leading/trailing D and I (e.g. a `...M D D E`
    // delete-trailer) are excluded so they don't leak into the display or
    // over-advance hmmto/sqto.
    let mut z1 = None;
    let mut z2 = None;
    for z in 0..tr.n {
        if tr.st[z] == State::M {
            if z1.is_none() {
                z1 = Some(z);
            }
            z2 = Some(z);
        }
    }

    let z1 = z1?;
    let z2 = z2?;

    let mut model = String::new();
    let mut mline = String::new();
    let mut aseq = String::new();
    let mut ppline = String::new();
    let mut rfline = String::new();
    let has_rf = hmm
        .rf
        .as_ref()
        .is_some_and(|rf| !rf.is_empty() && rf[0] != 0);
    let mut hmmfrom = 0;
    let mut hmmto = 0;
    let mut sqfrom = 0;
    let mut sqto = 0;

    for z in z1..=z2 {
        match tr.st[z] {
            State::M => {
                let k = tr.k[z];
                let i = tr.i[z];
                if hmmfrom == 0 {
                    hmmfrom = k;
                    sqfrom = i;
                }
                hmmto = k;
                sqto = i;
                if has_rf {
                    let rf = hmm.rf.as_ref().unwrap();
                    rfline.push(if k < rf.len() { rf[k] as char } else { '.' });
                }

                // Model consensus
                let cons_ch = if let Some(ref cons) = hmm.consensus {
                    if k < cons.len() {
                        cons[k] as char
                    } else {
                        'x'
                    }
                } else {
                    'x'
                };
                model.push(cons_ch);

                // Target sequence
                let seq_ch = if i > 0 && (dsq[i] as usize) < abc.kp {
                    abc.sym[dsq[i] as usize] as char
                } else {
                    '?'
                };
                aseq.push(seq_ch);

                // Match line: C HMMER (p7_alidisplay.c:220-224) pushes the
                // model character on identity, `+` when emission-odds ratio
                // exceeds 1.0 (positive log-odds), space otherwise.
                if cons_ch.eq_ignore_ascii_case(&seq_ch) {
                    mline.push(cons_ch);
                } else {
                    let x = dsq[i] as usize;
                    let sc = hmm
                        .mat
                        .get(k)
                        .and_then(|row| row.get(x))
                        .copied()
                        .unwrap_or(0.0);
                    let bg = if let Some(bg_f) = bg_override {
                        bg_f.get(x).copied().unwrap_or(0.25)
                    } else if hmm.abc_k == 4 {
                        0.25_f32
                    } else {
                        crate::bg::AMINO_FREQUENCIES.get(x).copied().unwrap_or(0.05)
                    };
                    let positive_emission =
                        if let Some(get_odds) = emission_odds.filter(|_| hmm.abc_k > 4) {
                            get_odds(k, x)
                                .map(|odds| odds > 1.0)
                                .unwrap_or(x < hmm.abc_k && sc > bg)
                        } else {
                            x < hmm.abc_k && sc > bg
                        };
                    if positive_emission {
                        mline.push('+');
                    } else {
                        mline.push(' ');
                    }
                }
                // PP for match state
                if let Some(pp_mx) = pp {
                    let pp_val = pp_mx.mmx(i, k);
                    ppline.push(crate::dp::generic_optacc::pp_to_char(pp_val.min(1.0)));
                } else {
                    ppline.push('*');
                }
            }
            State::I => {
                let i = tr.i[z];
                let k = tr.k[z];
                if has_rf {
                    rfline.push('.');
                }
                model.push('.');
                let seq_ch = if i > 0 && (dsq[i] as usize) < abc.kp {
                    (abc.sym[dsq[i] as usize] as char).to_ascii_lowercase()
                } else {
                    'x'
                };
                aseq.push(seq_ch);
                mline.push(' ');
                // PP for insert state: C HMMER uses the I-state posterior at
                // (i, k). We look at pp.imx(i, k) when available, else '.'.
                if let Some(pp_mx) = pp {
                    let pp_val = pp_mx.imx(i, k);
                    ppline.push(crate::dp::generic_optacc::pp_to_char(pp_val.min(1.0)));
                } else {
                    ppline.push('.');
                }
                if sqto == 0 && sqfrom == 0 {
                    sqfrom = i;
                }
                sqto = i;
            }
            State::D => {
                let k = tr.k[z];
                if hmmfrom == 0 {
                    hmmfrom = k;
                }
                hmmto = k;
                if has_rf {
                    let rf = hmm.rf.as_ref().unwrap();
                    rfline.push(if k < rf.len() { rf[k] as char } else { '.' });
                }

                let cons_ch = if let Some(ref cons) = hmm.consensus {
                    if k < cons.len() {
                        cons[k] as char
                    } else {
                        'x'
                    }
                } else {
                    'x'
                };
                model.push(cons_ch);
                aseq.push('-');
                mline.push(' ');
                ppline.push('.');
            }
            _ => {}
        }
    }

    Some(AliDisplay {
        model,
        mline,
        aseq,
        ppline,
        rfline,
        hmmfrom,
        hmmto,
        sqfrom,
        sqto,
    })
}

/// Compute the same coordinate span as `alignment_display_with_pp` without
/// constructing printable alignment strings.
pub fn alignment_coords(tr: &Trace) -> Option<(usize, usize, usize, usize)> {
    // M-only anchoring, matching C's p7_alidisplay_Create (z1 = first M,
    // z2 = last M). Leading/trailing D and I are excluded.
    let mut z1 = None;
    let mut z2 = None;
    for z in 0..tr.n {
        if tr.st[z] == State::M {
            if z1.is_none() {
                z1 = Some(z);
            }
            z2 = Some(z);
        }
    }

    let z1 = z1?;
    let z2 = z2?;
    let mut hmmfrom = 0;
    let mut hmmto = 0;
    let mut sqfrom = 0;
    let mut sqto = 0;

    for z in z1..=z2 {
        match tr.st[z] {
            State::M => {
                let k = tr.k[z];
                let i = tr.i[z];
                if hmmfrom == 0 {
                    hmmfrom = k;
                    sqfrom = i;
                }
                hmmto = k;
                sqto = i;
            }
            State::I => {
                let i = tr.i[z];
                if sqto == 0 && sqfrom == 0 {
                    sqfrom = i;
                }
                sqto = i;
            }
            State::D => {
                let k = tr.k[z];
                if hmmfrom == 0 {
                    hmmfrom = k;
                }
                hmmto = k;
            }
            _ => {}
        }
    }

    Some((hmmfrom, hmmto, sqfrom, sqto))
}
