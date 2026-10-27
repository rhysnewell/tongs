use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn build_c_stream_payload(
    fwd_only: bool,
    fm_alphabet: FmAlphabet,
    sa_freq: usize,
    bin_length: usize,
    char_count: usize,
    seq_data: &[SequenceMetadata],
    blocks: &[BlockRecord],
    ambig_ranges: &[AmbiguityRange],
    all_text: &[u8],
) -> std::io::Result<Vec<u8>> {
    let compact_starts = c_block_compact_sequence_starts(seq_data, blocks);
    let compact_ambig_ranges = compact_ambiguity_ranges(seq_data, &compact_starts, ambig_ranges);
    let mut payload = Vec::new();
    write_c_metadata_payload(
        &mut payload,
        fwd_only,
        fm_alphabet,
        sa_freq,
        bin_length,
        char_count,
        seq_data,
        blocks,
        &compact_starts,
        &compact_ambig_ranges,
    )?;

    // Single RNG for the whole build, seeded exactly as C
    // (`esl_randomness_Create(42)`). Degenerate-residue replacement draws from
    // it in block order via `c_block_text`, so the BWT/SA of degenerate blocks
    // match a C build bit-for-bit. (C also draws once for the reverse-strand
    // pass? No: C builds each block's text T once, replacing degenerates while
    // reading the block, then builds BOTH the forward and reverse FM records
    // from that same already-substituted T. So we likewise substitute once per
    // block here and reuse `c_text` for both strands.)
    let mut rng = Mt19937::new(MAKEHMMERDB_RNG_SEED);

    for block in blocks {
        let c_text = c_block_text(fm_alphabet, seq_data, block, all_text, &mut rng)?;
        write_c_fm_record(
            &mut payload,
            fm_alphabet,
            &c_text,
            block,
            sa_freq,
            bin_length,
            true,
            true,
        )?;
        if !fwd_only {
            write_c_fm_record(
                &mut payload,
                fm_alphabet,
                &c_text,
                block,
                sa_freq,
                bin_length,
                false,
                false,
            )?;
        }
    }

    Ok(payload)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_c_metadata_payload<W: Write + ?Sized>(
    out: &mut W,
    fwd_only: bool,
    fm_alphabet: FmAlphabet,
    sa_freq: usize,
    bin_length: usize,
    char_count: usize,
    seq_data: &[SequenceMetadata],
    blocks: &[BlockRecord],
    compact_starts: &[usize],
    compact_ambig_ranges: &[AmbiguityRange],
) -> std::io::Result<()> {
    out.write_all(&[fwd_only as u8])?;
    out.write_all(&[fm_alphabet.alph_type])?;
    out.write_all(&[fm_alphabet.alph_size])?;
    out.write_all(&[fm_alphabet.char_bits])?;
    out.write_all(&checked_u32(sa_freq, "sa_freq")?.to_le_bytes())?;
    out.write_all(&FM_FREQ_CNT_SB.to_le_bytes())?;
    out.write_all(&checked_u32(bin_length, "bin_length")?.to_le_bytes())?;
    out.write_all(&checked_u16(blocks.len(), "block_count")?.to_le_bytes())?;
    out.write_all(&checked_u32(seq_data.len(), "seq_count")?.to_le_bytes())?;
    out.write_all(&checked_u32(compact_ambig_ranges.len(), "ambig_count")?.to_le_bytes())?;
    out.write_all(&checked_u64(char_count, "char_count")?.to_le_bytes())?;

    for (seq, &compact_start) in seq_data.iter().zip(compact_starts.iter()) {
        out.write_all(&checked_u32(seq.target_id, "target_id")?.to_le_bytes())?;
        out.write_all(&checked_u64(seq.target_start, "target_start")?.to_le_bytes())?;
        out.write_all(&checked_u32(compact_start, "fm_start")?.to_le_bytes())?;
        out.write_all(&checked_u32(seq.length, "length")?.to_le_bytes())?;
        // The per-sequence `source` field depends on C's read path. makehmmerdb
        // reads DNA/RNA via the long_target *windowed* path
        // (`esl_sqio_ReadBlock(.., long_target = alphatype != eslAMINO)`),
        // whose `sqascii_ReadWindow` sets `sq->source = sq->name`
        // (esl_sqio_ascii.c:1341, 2012). The amino path is the plain
        // `sqascii_Read`, which never calls SetSource, so `sq->source` keeps the
        // empty string from `esl_sq_CreateDigital`. Hence: source == name for
        // DNA/RNA, source == "" for amino.
        // (For DNA/RNA sequences split across blocks, C additionally reformats
        // `name` to "name/start-end" while `source` keeps the bare name;
        // reproducing that window-name suffix is a separate, pre-existing
        // metadata delta and is not handled here — `name` remains the bare name
        // in that case.)
        let source = if fm_alphabet.amino {
            ""
        } else {
            seq.name.as_str()
        };
        reject_embedded_nul("sequence name", &seq.name)?;
        reject_embedded_nul("sequence accession", &seq.acc)?;
        reject_embedded_nul("sequence source", source)?;
        reject_embedded_nul("sequence description", &seq.desc)?;
        out.write_all(&checked_u16(seq.name.len(), "name_length")?.to_le_bytes())?;
        out.write_all(&checked_u16(seq.acc.len(), "acc_length")?.to_le_bytes())?;
        out.write_all(&checked_u16(source.len(), "source_length")?.to_le_bytes())?;
        out.write_all(&checked_u16(seq.desc.len(), "desc_length")?.to_le_bytes())?;
        write_c_string(out, &seq.name)?;
        write_c_string(out, &seq.acc)?;
        write_c_string(out, source)?;
        write_c_string(out, &seq.desc)?;
    }

    for range in compact_ambig_ranges {
        out.write_all(&checked_i32(range.lower, "ambiguity lower")?.to_le_bytes())?;
        out.write_all(&checked_i32(range.upper, "ambiguity upper")?.to_le_bytes())?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_c_fm_record<W: Write + ?Sized>(
    out: &mut W,
    fm_alphabet: FmAlphabet,
    c_text: &[u8],
    block: &BlockRecord,
    sa_freq: usize,
    bin_length: usize,
    reverse_text_for_bwt: bool,
    include_text_and_sa: bool,
) -> std::io::Result<()> {
    let record = build_c_fm_record(
        fm_alphabet,
        c_text,
        sa_freq,
        bin_length,
        reverse_text_for_bwt,
    )?;

    out.write_all(&checked_u64(record.n, "FM record length")?.to_le_bytes())?;
    out.write_all(&checked_u32(record.term_loc, "terminal location")?.to_le_bytes())?;
    out.write_all(&checked_u32(block.seq_offset, "seq_offset")?.to_le_bytes())?;
    out.write_all(&checked_u32(block.ambig_offset, "ambig_offset")?.to_le_bytes())?;
    // C writes the block's overlap (`block->list[0].C`) ONLY for the forward
    // (reversed-text) FM record (makehmmerdb.c:694); the reverse-strand record
    // is emitted with overlap=0 (makehmmerdb.c:700). The forward record is the
    // one built from reversed text, i.e. `reverse_text_for_bwt == true`.
    let overlap = if reverse_text_for_bwt {
        block.overlap_bases
    } else {
        0
    };
    out.write_all(&checked_u32(overlap, "overlap")?.to_le_bytes())?;
    out.write_all(&checked_u32(block.seq_count, "seq_count")?.to_le_bytes())?;
    out.write_all(&checked_u32(block.ambig_count, "ambig_count")?.to_le_bytes())?;

    if include_text_and_sa {
        out.write_all(&fm_alphabet.pack(&record.forward_text))?;
    }
    out.write_all(&fm_alphabet.pack(&record.bwt))?;
    if include_text_and_sa {
        for sample in &record.sa_samples {
            out.write_all(&sample.to_le_bytes())?;
        }
    }
    for count in &record.occ_cnts_b {
        out.write_all(&count.to_le_bytes())?;
    }
    for count in &record.occ_cnts_sb {
        out.write_all(&count.to_le_bytes())?;
    }

    Ok(())
}

pub(super) struct CfmRecord {
    pub(super) n: usize,
    pub(super) term_loc: usize,
    pub(super) forward_text: Vec<u8>,
    pub(super) bwt: Vec<u8>,
    pub(super) sa_samples: Vec<u32>,
    pub(super) occ_cnts_b: Vec<u16>,
    pub(super) occ_cnts_sb: Vec<u32>,
}

fn build_c_fm_record(
    fm_alphabet: FmAlphabet,
    c_text: &[u8],
    sa_freq: usize,
    bin_length: usize,
    reverse_text_for_bwt: bool,
) -> std::io::Result<CfmRecord> {
    let mut forward_text = c_text.to_vec();
    forward_text.push(0);

    let mut sortable_text: Vec<u8> = c_text.iter().map(|&base| base + 1).collect();
    if reverse_text_for_bwt {
        sortable_text.reverse();
    }
    sortable_text.push(0);

    let n = sortable_text.len();
    let mut sa = vec![0i32; n];
    sort_in_place(&sortable_text, &mut sa);

    let mut bwt = Vec::with_capacity(n);
    let mut term_loc = 0usize;
    let num_sa_samples = 1 + n / sa_freq;
    let mut sa_samples = vec![0u32; num_sa_samples];

    for (j, &suffix_start) in sa.iter().enumerate() {
        let suffix_start = suffix_start as usize;
        if suffix_start == 0 {
            term_loc = j;
            bwt.push(0);
        } else {
            bwt.push(sortable_text[suffix_start - 1].saturating_sub(1));
        }

        if j != 0 && j % sa_freq == 0 {
            sa_samples[j / sa_freq] = if suffix_start == n - 1 {
                u32::MAX
            } else {
                checked_u32(suffix_start, "suffix array sample")?
            };
        }
    }

    let (occ_cnts_b, occ_cnts_sb) =
        build_c_occ_tables(fm_alphabet, &bwt, bin_length, FM_FREQ_CNT_SB as usize)?;

    Ok(CfmRecord {
        n,
        term_loc,
        forward_text,
        bwt,
        sa_samples,
        occ_cnts_b,
        occ_cnts_sb,
    })
}

fn build_c_occ_tables(
    fm_alphabet: FmAlphabet,
    bwt: &[u8],
    freq_cnt_b: usize,
    freq_cnt_sb: usize,
) -> std::io::Result<(Vec<u16>, Vec<u32>)> {
    let n = bwt.len();
    let num_freq_cnts_b = 1 + n.div_ceil(freq_cnt_b);
    let num_freq_cnts_sb = 1 + n.div_ceil(freq_cnt_sb);
    let alph_size = fm_alphabet.alph_size as usize;
    let mut occ_cnts_b = vec![0u16; num_freq_cnts_b * alph_size];
    let mut occ_cnts_sb = vec![0u32; num_freq_cnts_sb * alph_size];
    let mut cnts_b = vec![0usize; alph_size];
    let mut cnts_sb = vec![0usize; alph_size];

    for (j, &base) in bwt.iter().enumerate() {
        let idx = base as usize;
        if idx >= alph_size {
            return Err(invalid_data(format!(
                "FM symbol code {idx} is outside alphabet size {alph_size}"
            )));
        }
        cnts_b[idx] += 1;
        cnts_sb[idx] += 1;

        let joffset = j + 1;
        if joffset % freq_cnt_b == 0 {
            store_occ_b(&mut occ_cnts_b, alph_size, joffset / freq_cnt_b, &cnts_b)?;
            if joffset % freq_cnt_sb == 0 {
                store_occ_sb(&mut occ_cnts_sb, alph_size, joffset / freq_cnt_sb, &cnts_sb)?;
                cnts_b.fill(0);
            }
        }
    }

    store_occ_b(&mut occ_cnts_b, alph_size, num_freq_cnts_b - 1, &cnts_b)?;
    store_occ_sb(&mut occ_cnts_sb, alph_size, num_freq_cnts_sb - 1, &cnts_sb)?;

    Ok((occ_cnts_b, occ_cnts_sb))
}

fn store_occ_b(
    out: &mut [u16],
    alph_size: usize,
    row: usize,
    counts: &[usize],
) -> std::io::Result<()> {
    let offset = row * alph_size;
    for (slot, &count) in out[offset..offset + alph_size].iter_mut().zip(counts) {
        *slot = checked_u16(count, "FM occurrence block count")?;
    }
    Ok(())
}

fn store_occ_sb(
    out: &mut [u32],
    alph_size: usize,
    row: usize,
    counts: &[usize],
) -> std::io::Result<()> {
    let offset = row * alph_size;
    for (slot, &count) in out[offset..offset + alph_size].iter_mut().zip(counts) {
        *slot = checked_u32(count, "FM occurrence superblock count")?;
    }
    Ok(())
}

pub(super) fn pack_dna_quads(values: &[u8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(4));
    for chunk in values.chunks(4) {
        let mut byte = 0u8;
        for (i, &value) in chunk.iter().enumerate() {
            byte |= value << (6 - 2 * i);
        }
        packed.push(byte);
    }
    packed
}

/// Encode a block's DNA text to FM codes (A=0,C=1,G=2,T=3), substituting any
/// degenerate residue with an `esl_random()`-drawn canonical base.
///
/// This mirrors makehmmerdb.c:656-674: for each residue, if its FM `inv_alph`
/// code is -1 (i.e. not one of the four canonical DNA bases — every IUPAC
/// degeneracy code, including N), C replaces it with
/// `meta->alph[(int)(esl_random(r)*4)]`. The RNG `r` is a single
/// `esl_randomness_Create(42)` object consumed across the whole build, in block
/// order, with overlap regions of split sequences re-drawn (because they are
/// re-read into the next block's text). Because this function is called once per
/// block in block order from `build_c_stream_payload`, and a block's text walks
/// its windows (including the leading-overlap window) in `fm_start` order, the
/// `rng` draws here occur in exactly the same order as C's.
pub(super) fn c_dna_text_without_separators(
    text: &[u8],
    rng: &mut Mt19937,
) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len());
    for &base in text {
        if base == b'$' {
            continue;
        }
        let code = match base {
            b'A' | b'a' => 0,
            b'C' | b'c' => 1,
            b'G' | b'g' => 2,
            b'T' | b't' | b'U' | b'u' => 3,
            _ if is_dna_degenerate(base) => rng.dna_replacement_code(),
            _ => {
                return Err(invalid_data(format!(
                    "base '{}' cannot be encoded in C DNA FM stream",
                    base as char
                )))
            }
        };
        out.push(code);
    }
    Ok(out)
}

/// IUPAC DNA/RNA degeneracy codes (everything that maps to FM `inv_alph` == -1
/// in C, i.e. is a valid residue but not one of the four canonical bases).
fn is_dna_degenerate(base: u8) -> bool {
    matches!(
        base.to_ascii_uppercase(),
        b'R' | b'Y' | b'S' | b'W' | b'K' | b'M' | b'B' | b'D' | b'H' | b'V' | b'N'
    )
}

pub(super) fn c_amino_text_without_separators(text: &[u8]) -> std::io::Result<Vec<u8>> {
    text.iter()
        .filter(|&&residue| residue != b'$')
        .map(|&residue| {
            let residue = residue.to_ascii_uppercase();
            FM_AMINO_ALPHABET
                .iter()
                .position(|&symbol| symbol == residue)
                .map(|idx| idx as u8)
                .ok_or_else(|| {
                    invalid_data(format!(
                        "residue '{}' cannot be encoded in C amino FM stream",
                        residue as char
                    ))
                })
        })
        .collect()
}

fn c_block_text(
    fm_alphabet: FmAlphabet,
    seq_data: &[SequenceMetadata],
    block: &BlockRecord,
    all_text: &[u8],
    rng: &mut Mt19937,
) -> std::io::Result<Vec<u8>> {
    let mut text = Vec::with_capacity(
        seq_data[block.seq_offset..block.seq_offset + block.seq_count]
            .iter()
            .map(|seq| seq.length)
            .sum(),
    );
    for seq in &seq_data[block.seq_offset..block.seq_offset + block.seq_count] {
        let end = seq.fm_start + seq.length;
        text.extend(fm_alphabet.encode_text_without_separators(&all_text[seq.fm_start..end], rng)?);
    }
    Ok(text)
}

pub(super) fn c_block_compact_sequence_starts(
    seq_data: &[SequenceMetadata],
    blocks: &[BlockRecord],
) -> Vec<usize> {
    let mut starts = vec![0; seq_data.len()];
    for block in blocks {
        let mut start = 0usize;
        for (idx, seq) in seq_data.iter().enumerate() {
            if seq.block_id == block.id {
                starts[idx] = start;
                start += seq.length;
            }
        }
    }
    starts
}

pub(super) fn compact_ambiguity_ranges(
    seq_data: &[SequenceMetadata],
    compact_starts: &[usize],
    ambig_ranges: &[AmbiguityRange],
) -> Vec<AmbiguityRange> {
    // Emit in C's order: BLOCK-major (sequence-window order), then position
    // order within a window. C records each range in `block_length`
    // (block-LOCAL) coordinates while scanning the block's concatenated text
    // (makehmmerdb.c:644,664-667 — `block_length` is the running offset within
    // the block, which `compact_start` reproduces here as the window's start
    // within its block). Iterating windows (outer) then ranges (inner) matches
    // both that ordering and that coordinate basis, and re-emits overlap-region
    // ranges once per block exactly as C re-reads the overlap context. See the
    // matching loop nesting in `assign_ambiguities_to_blocks`.
    let mut compact_ranges = Vec::new();
    for (seq, &compact_start) in seq_data.iter().zip(compact_starts) {
        for range in ambig_ranges {
            if let Some(intersection) = intersect_ambiguity_with_sequence(range, seq) {
                compact_ranges.push(AmbiguityRange {
                    lower: compact_start + (intersection.lower - seq.fm_start),
                    upper: compact_start + (intersection.upper - seq.fm_start),
                });
            }
        }
    }
    compact_ranges
}

pub(super) fn intersect_ambiguity_with_sequence(
    range: &AmbiguityRange,
    seq: &SequenceMetadata,
) -> Option<AmbiguityRange> {
    let seq_lower = seq.fm_start;
    let seq_upper = seq.fm_start + seq.length - 1;
    let lower = range.lower.max(seq_lower);
    let upper = range.upper.min(seq_upper);
    (lower <= upper).then_some(AmbiguityRange { lower, upper })
}
