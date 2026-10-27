use super::*;

/// A full-fidelity Stockholm annotation model, mirroring the subset of Easel's
/// `ESL_MSA` that the text-mode Stockholm writer (`stockholm_write`) consumes.
///
/// This is a parallel, additive model to [`Msa`]/[`StockholmMsa`]: it captures
/// every annotation type and ordering needed to re-serialize a Stockholm file
/// byte-for-byte the way C does (used by `alimask`). It is read in *text mode*:
/// residue / GR / GC characters pass through verbatim (no gap-character
/// normalization), matching `alimask`, which opens the alignment with a NULL
/// alphabet.
#[derive(Debug, Clone, Default)]
pub struct FullStockholm {
    /// Free-text comment lines (`#<comment>`), with the leading `#` and
    /// following whitespace stripped (as in `stockholm_parse_comment`).
    pub comments: Vec<String>,
    /// `#=GF ID` alignment name.
    pub name: Option<String>,
    /// `#=GF AC` accession.
    pub acc: Option<String>,
    /// `#=GF DE` description (last line wins, like `esl_msa_SetDesc`).
    pub desc: Option<String>,
    /// `#=GF AU` author (last line wins, like `esl_msa_SetAuthor`).
    pub au: Option<String>,
    /// `#=GF GA` gathering cutoffs (one or two values).
    pub ga: Option<(f32, Option<f32>)>,
    /// `#=GF NC` noise cutoffs.
    pub nc: Option<(f32, Option<f32>)>,
    /// `#=GF TC` trusted cutoffs.
    pub tc: Option<(f32, Option<f32>)>,
    /// Remaining `#=GF` tags, in input order: `(tag, value)`.
    pub gf: Vec<(String, String)>,

    /// Sequence names, in first-seen order.
    pub sqname: Vec<String>,
    /// Aligned sequence rows (text, verbatim), parallel to `sqname`.
    pub aseq: Vec<Vec<u8>>,
    /// Whether any `#=GS <seq> WT` weight was seen (`eslMSA_HASWGTS`).
    pub has_wgts: bool,
    /// Per-sequence weights (parallel to `sqname`); valid only if `has_wgts`.
    pub wgt: Vec<f64>,
    /// Per-sequence `#=GS <seq> AC` accession (None if unset).
    pub sqacc: Vec<Option<String>>,
    /// Per-sequence `#=GS <seq> DE` description (None if unset).
    pub sqdesc: Vec<Option<String>>,
    /// Remaining `#=GS` tags, in first-seen order. Each entry is
    /// `(tag, per-seq values)` where the per-seq vector is parallel to
    /// `sqname`; each value is `\n`-joined when a sequence has multiple lines
    /// of the same tag (e.g. `DR PDB;`), matching `esl_msa_AddGS`.
    pub gs: Vec<(String, Vec<Option<String>>)>,

    /// Per-sequence `#=GR <seq> SS` (parallel to `sqname`).
    pub ss: Vec<Option<Vec<u8>>>,
    /// Per-sequence `#=GR <seq> SA`.
    pub sa: Vec<Option<Vec<u8>>>,
    /// Per-sequence `#=GR <seq> PP`.
    pub pp: Vec<Option<Vec<u8>>>,
    /// Remaining `#=GR` tags, first-seen order: `(tag, per-seq values)`.
    pub gr: Vec<(String, Vec<Option<Vec<u8>>>)>,

    /// `#=GC SS_cons`.
    pub ss_cons: Option<Vec<u8>>,
    /// `#=GC SA_cons`.
    pub sa_cons: Option<Vec<u8>>,
    /// `#=GC PP_cons`.
    pub pp_cons: Option<Vec<u8>>,
    /// `#=GC RF`.
    pub rf: Option<Vec<u8>>,
    /// `#=GC MM`.
    pub mm: Option<Vec<u8>>,
    /// Remaining `#=GC` tags, first-seen order: `(tag, value)`.
    pub gc: Vec<(String, Vec<u8>)>,

    /// Alignment length (columns).
    pub alen: usize,
}

impl FullStockholm {
    fn seqidx(&mut self, name: &str) -> usize {
        if let Some(i) = self.sqname.iter().position(|n| n == name) {
            return i;
        }
        self.sqname.push(name.to_string());
        self.aseq.push(Vec::new());
        self.wgt.push(-1.0);
        self.sqacc.push(None);
        self.sqdesc.push(None);
        self.ss.push(None);
        self.sa.push(None);
        self.pp.push(None);
        for (_, vals) in &mut self.gs {
            vals.push(None);
        }
        for (_, vals) in &mut self.gr {
            vals.push(None);
        }
        self.sqname.len() - 1
    }

    fn gs_tagidx(&mut self, tag: &str) -> usize {
        if let Some(i) = self.gs.iter().position(|(t, _)| t == tag) {
            return i;
        }
        self.gs
            .push((tag.to_string(), vec![None; self.sqname.len()]));
        self.gs.len() - 1
    }

    fn gr_tagidx(&mut self, tag: &str) -> usize {
        if let Some(i) = self.gr.iter().position(|(t, _)| t == tag) {
            return i;
        }
        self.gr
            .push((tag.to_string(), vec![None; self.sqname.len()]));
        self.gr.len() - 1
    }

    fn gc_tagidx(&mut self, tag: &str) -> usize {
        if let Some(i) = self.gc.iter().position(|(t, _)| t == tag) {
            return i;
        }
        self.gc.push((tag.to_string(), Vec::new()));
        self.gc.len() - 1
    }
}

/// Read every Stockholm alignment in `path` into the full-fidelity
/// [`FullStockholm`] model (text mode).
pub fn read_stockholm_full(path: &Path) -> HmmerResult<Vec<FullStockholm>> {
    let file = std::fs::File::open(path).map_err(HmmerError::Io)?;
    read_stockholm_full_from_reader(BufReader::new(file))
}

/// Read every Stockholm alignment from a reader into [`FullStockholm`] models.
///
/// Faithful text-mode port of the bucketing performed by the Easel Stockholm
/// reader (`stockholm_parse_{gf,gs,gc,gr,sq,comment}`). One `FullStockholm` is
/// returned per `# STOCKHOLM`/`//` record.
pub fn read_stockholm_full_from_reader<R: Read>(
    reader: BufReader<R>,
) -> HmmerResult<Vec<FullStockholm>> {
    let mut out = Vec::new();
    let mut cur: Option<FullStockholm> = None;

    for line in reader.lines() {
        let line = line.map_err(HmmerError::Io)?;
        // Strip leading whitespace, as the C reader does before dispatch.
        let p = line.trim_start_matches([' ', '\t']);

        if cur.is_none() {
            // Skip leading blank/comment lines until a Stockholm header.
            if p.is_empty() {
                continue;
            }
            if p.starts_with("# STOCKHOLM") {
                cur = Some(FullStockholm::default());
                continue;
            }
            if p.starts_with('#') {
                continue;
            }
            return Err(HmmerError::Format(format!(
                "Expected Stockholm header, got: {p}"
            )));
        }

        if p == "//" {
            let mut msa = cur.take().unwrap();
            finalize_full(&mut msa)?;
            out.push(msa);
            continue;
        }
        if p.starts_with("# STOCKHOLM") {
            continue;
        }
        let msa = cur.as_mut().unwrap();
        if p.is_empty() {
            continue;
        }
        if let Some(rest) = p.strip_prefix('#') {
            if let Some(body) = rest.strip_prefix("=GF") {
                parse_full_gf(msa, body)?;
            } else if let Some(body) = rest.strip_prefix("=GS") {
                parse_full_gs(msa, body)?;
            } else if let Some(body) = rest.strip_prefix("=GC") {
                parse_full_gc(msa, body)?;
            } else if let Some(body) = rest.strip_prefix("=GR") {
                parse_full_gr(msa, body)?;
            } else {
                // Free-text comment: strip leading whitespace after '#'.
                msa.comments
                    .push(rest.trim_start_matches([' ', '\t']).to_string());
            }
        } else {
            // Sequence line: name aseq
            let mut it = p.splitn(2, [' ', '\t']);
            let name = it.next().unwrap_or("");
            let seq = it.next().unwrap_or("").trim_start_matches([' ', '\t']);
            if name.is_empty() {
                continue;
            }
            let idx = msa.seqidx(name);
            msa.aseq[idx].extend_from_slice(seq.as_bytes());
        }
    }

    if cur.is_some() {
        return Err(HmmerError::Format(
            "missing // terminator after MSA".to_string(),
        ));
    }
    Ok(out)
}

/// Tokenize Stockholm markup into (first whitespace-delimited token, remainder
/// with leading whitespace stripped, trailing whitespace preserved), matching
/// `esl_memtok` (which skips the trailing delimiter run after the token).
fn split_tok(s: &str) -> (&str, &str) {
    let s = s.trim_start_matches([' ', '\t']);
    match s.find([' ', '\t']) {
        Some(i) => (&s[..i], s[i..].trim_start_matches([' ', '\t'])),
        None => (s, ""),
    }
}

fn parse_cutoffs(value: &str) -> HmmerResult<(f32, Option<f32>)> {
    // C `esl_memtof` parses a leading real number and ignores trailing junk
    // such as the `;` Rfam/Pfam append (e.g. "27.00 27.00;").
    fn parse_real(tok: &str) -> HmmerResult<f32> {
        let trimmed = tok.trim_end_matches(';');
        trimmed
            .parse::<f32>()
            .map_err(|_| HmmerError::Format(format!("bad #=GF cutoff value: {tok}")))
    }
    let mut it = value.split([' ', '\t']).filter(|t| !t.is_empty());
    let first = it
        .next()
        .ok_or_else(|| HmmerError::Format("missing #=GF cutoff value".to_string()))?;
    let first = parse_real(first)?;
    let second = match it.next() {
        Some(v) => Some(parse_real(v)?),
        None => None,
    };
    Ok((first, second))
}

fn parse_full_gf(msa: &mut FullStockholm, body: &str) -> HmmerResult<()> {
    let (tag, value) = split_tok(body);
    match tag {
        "ID" => msa.name = Some(split_tok(value).0.to_string()),
        "AC" => msa.acc = Some(split_tok(value).0.to_string()),
        "DE" => msa.desc = Some(value.to_string()),
        "AU" => msa.au = Some(value.to_string()),
        "GA" => msa.ga = Some(parse_cutoffs(value)?),
        "NC" => msa.nc = Some(parse_cutoffs(value)?),
        "TC" => msa.tc = Some(parse_cutoffs(value)?),
        _ => msa.gf.push((tag.to_string(), value.to_string())),
    }
    Ok(())
}

fn parse_full_gs(msa: &mut FullStockholm, body: &str) -> HmmerResult<()> {
    let (seqname, rest) = split_tok(body);
    let (tag, value) = split_tok(rest);
    if seqname.is_empty() || tag.is_empty() {
        return Ok(());
    }
    let idx = msa.seqidx(seqname);
    match tag {
        "WT" => {
            let w: f64 = split_tok(value)
                .0
                .parse()
                .map_err(|_| HmmerError::Format("bad #=GS WT value".to_string()))?;
            msa.wgt[idx] = w;
            msa.has_wgts = true;
        }
        "AC" => msa.sqacc[idx] = Some(split_tok(value).0.to_string()),
        "DE" => msa.sqdesc[idx] = Some(value.to_string()),
        _ => {
            let t = msa.gs_tagidx(tag);
            let slot = &mut msa.gs[t].1[idx];
            match slot {
                Some(existing) => {
                    existing.push('\n');
                    existing.push_str(value);
                }
                None => *slot = Some(value.to_string()),
            }
        }
    }
    Ok(())
}

fn parse_full_gc(msa: &mut FullStockholm, body: &str) -> HmmerResult<()> {
    let (tag, value) = split_tok(body);
    if tag.is_empty() {
        return Ok(());
    }
    let bytes = value.as_bytes();
    let append = |slot: &mut Option<Vec<u8>>| match slot {
        Some(v) => v.extend_from_slice(bytes),
        None => *slot = Some(bytes.to_vec()),
    };
    match tag {
        "SS_cons" => append(&mut msa.ss_cons),
        "SA_cons" => append(&mut msa.sa_cons),
        "PP_cons" => append(&mut msa.pp_cons),
        "RF" => append(&mut msa.rf),
        "MM" => append(&mut msa.mm),
        _ => {
            let t = msa.gc_tagidx(tag);
            msa.gc[t].1.extend_from_slice(bytes);
        }
    }
    Ok(())
}

fn parse_full_gr(msa: &mut FullStockholm, body: &str) -> HmmerResult<()> {
    let (seqname, rest) = split_tok(body);
    let (tag, value) = split_tok(rest);
    if seqname.is_empty() || tag.is_empty() {
        return Ok(());
    }
    let idx = msa.seqidx(seqname);
    let bytes = value.as_bytes();
    let append = |slot: &mut Option<Vec<u8>>| match slot {
        Some(v) => v.extend_from_slice(bytes),
        None => *slot = Some(bytes.to_vec()),
    };
    match tag {
        "SS" => append(&mut msa.ss[idx]),
        "SA" => append(&mut msa.sa[idx]),
        "PP" => append(&mut msa.pp[idx]),
        _ => {
            let t = msa.gr_tagidx(tag);
            append(&mut msa.gr[t].1[idx]);
        }
    }
    Ok(())
}

fn finalize_full(msa: &mut FullStockholm) -> HmmerResult<()> {
    msa.alen = msa.aseq.first().map(|s| s.len()).unwrap_or(0);
    Ok(())
}

/// Maximum byte-length of a set of strings (Easel `esl_str_GetMaxWidth`).
pub(super) fn max_width<I: IntoIterator<Item = usize>>(lens: I) -> usize {
    lens.into_iter().max().unwrap_or(0)
}
