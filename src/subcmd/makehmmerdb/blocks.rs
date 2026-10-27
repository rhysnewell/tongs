use super::*;

pub(super) fn read_sequences<R: Read>(
    mut sqf: sequence::SeqFile<R>,
    abc: &Alphabet,
    replace_degenerate_residues: bool,
    all_text: &mut Vec<u8>,
    seq_data: &mut Vec<SequenceMetadata>,
    ambig_ranges: &mut Vec<AmbiguityRange>,
) {
    let mut sq = Sequence::new();
    loop {
        let has_seq = sqf.read(&mut sq).unwrap_or_else(|e| {
            eprintln!("Error reading sequence file: {}", e);
            std::process::exit(1);
        });
        if !has_seq {
            break;
        }
        let fm_start = all_text.len();
        seq_data.push(SequenceMetadata {
            target_id: seq_data.len(),
            target_start: 1,
            fm_start,
            length: sq.n,
            block_id: 0,
            block_offset: 0,
            overlap_bases: 0,
            name: sq.name.clone(),
            acc: sq.acc.clone(),
            desc: sq.desc.clone(),
        });
        append_sequence_text(
            abc,
            &sq,
            replace_degenerate_residues,
            all_text,
            ambig_ranges,
        );
        all_text.push(b'$');
        sq.reuse();
    }
}

pub(super) fn append_sequence_text(
    abc: &Alphabet,
    sq: &Sequence,
    replace_degenerate_residues: bool,
    all_text: &mut Vec<u8>,
    ambig_ranges: &mut Vec<AmbiguityRange>,
) {
    // We store the ORIGINAL alphabet symbol here (canonical letters such as
    // A/C/G/T, or degenerate IUPAC codes such as N/R/Y for DNA). Unlike C, we
    // do NOT substitute degenerate residues during this initial read: C draws
    // its replacement bases from `esl_random()` while building each FM *block's*
    // text (makehmmerdb.c:656-684), in block order and re-drawing for residues
    // that fall in inter-block overlap regions. To reproduce that RNG draw order
    // bit-for-bit, substitution is deferred to `c_dna_text_without_separators()`
    // at block-build time. Here we only record the ambiguity ranges (positions),
    // which are independent of the substituted values.
    let mut in_ambig_run = false;
    for i in 1..=sq.n {
        let dsq = sq.dsq[i];
        let ch = if abc.is_canonical(dsq) || (abc.is_residue(dsq) && !replace_degenerate_residues) {
            in_ambig_run = false;
            abc.sym[dsq as usize]
        } else if abc.is_residue(dsq) {
            let pos = all_text.len();
            if in_ambig_run {
                ambig_ranges.last_mut().unwrap().upper = pos;
            } else {
                ambig_ranges.push(AmbiguityRange {
                    lower: pos,
                    upper: pos,
                });
                in_ambig_run = true;
            }
            abc.sym[dsq as usize]
        } else {
            eprintln!(
                "Error: non-residue symbol '{}' is not supported by makehmmerdb",
                abc.sym[dsq as usize] as char
            );
            std::process::exit(1);
        };
        all_text.push(ch);
    }
}

/// Partition the input sequences into FM-index blocks, mirroring the
/// per-sequence windowing of C `esl_sqio_ReadBlock()` (long_target mode) as
/// driven by makehmmerdb.c's main loop.
///
/// C model (esl_sqio_ascii.c `sqascii_ReadBlock`, makehmmerdb.c:563-704):
///   * `block_size_bases` (= `--block_size` * 1e6) bounds the NEW residues per
///     block; `size` counts `n - C` (window length minus retained context).
///   * Each whole sequence shorter than the remaining budget is read in one
///     window and added intact (an EOD is "burned off"). A block closes once
///     `size >= block_size`.
///   * A *single* sequence longer than a window is split across blocks; the
///     continuation window re-reads `FM_BLOCK_OVERLAP` (`overlap_bases`) bases
///     of the same sequence as leading context (`->C`). `block->list[0].C`
///     becomes the block's `overlap` field.
///   * Non-init windows request `max(block_size - size, block_size * 0.05)`
///     new residues, keeping blocks near `block_size`.
///
/// Overlap is therefore retained ONLY when continuing the same long sequence
/// across a block boundary — a boundary that falls between two distinct
/// sequences carries no overlap, unlike the previous global-text chopper.
pub(super) fn build_readblock_like_records(
    input_seqs: &[SequenceMetadata],
    block_size_bases: usize,
    overlap_bases: usize,
) -> (Vec<SequenceMetadata>, Vec<BlockRecord>) {
    let compact_len: usize = input_seqs.iter().map(|seq| seq.length).sum();
    if compact_len == 0 {
        return (Vec::new(), Vec::new());
    }

    let block_size = block_size_bases.max(1);
    let max_overlap = overlap_bases.min(block_size.saturating_sub(1));
    let slop = ((block_size as f64) * 0.05).ceil() as usize;

    let mut seq_data = Vec::new();
    let mut blocks = Vec::new();

    // Currently-open block accumulator.
    let mut block_id = 0usize;
    let mut seq_offset = 0usize;
    let mut size = 0usize; // NEW residues in the open block (matches C's `size`)
    let mut block_overlap = 0usize; // C's block->list[0].C for this block
    let mut block_started = false;

    // Helper to finalize the currently-open block into `blocks`.
    let finish_block = |blocks: &mut Vec<BlockRecord>,
                        seq_data: &[SequenceMetadata],
                        block_id: usize,
                        seq_offset: usize,
                        block_overlap: usize| {
        let seq_count = seq_data.len() - seq_offset;
        let (fm_start, length) = if seq_count == 0 {
            (0, 0)
        } else {
            let first = &seq_data[seq_offset];
            let last = &seq_data[seq_data.len() - 1];
            (first.fm_start, last.fm_start + last.length - first.fm_start)
        };
        blocks.push(BlockRecord {
            id: block_id,
            fm_start,
            length,
            seq_offset,
            seq_count,
            ambig_offset: 0,
            ambig_count: 0,
            overlap_bases: block_overlap,
        });
    };

    for seq in input_seqs {
        // Residues of this sequence already consumed by a window.
        let mut consumed = 0usize;
        // Whether the currently-open window of THIS sequence is a continuation
        // across a block boundary (i.e. it carries leading overlap context).
        let mut continuation = false;

        while consumed < seq.length {
            if !block_started {
                block_started = true;
                block_overlap = if continuation {
                    // Continuation of a split sequence: re-read up to
                    // FM_BLOCK_OVERLAP bases of context. C does this by setting
                    // block->list->C, which becomes the block's overlap.
                    max_overlap.min(consumed)
                } else {
                    0
                };
                seq_offset = seq_data.len();
            }

            // C's non-init request size: max(block_size - size, block_size*0.05).
            let request_size = block_size.saturating_sub(size).max(slop).max(1);
            let remaining = seq.length - consumed;
            let new_residues = request_size.min(remaining);

            // The window includes leading overlap context (only when this is
            // the first window of a continued sequence in a fresh block).
            let leading_overlap = if continuation && consumed > 0 && size == 0 {
                max_overlap.min(consumed)
            } else {
                0
            };
            let window_offset = consumed - leading_overlap;
            let window_len = leading_overlap + new_residues;

            seq_data.push(SequenceMetadata {
                target_id: seq.target_id,
                target_start: seq.target_start + window_offset,
                fm_start: seq.fm_start + window_offset,
                length: window_len,
                block_id,
                block_offset: 0,
                overlap_bases: leading_overlap,
                name: seq.name.clone(),
                acc: seq.acc.clone(),
                desc: seq.desc.clone(),
            });

            consumed += new_residues;
            size += new_residues;

            // Did this fill the block? (size has reached block_size and the
            // sequence is not yet exhausted, OR exactly hit the limit).
            let seq_exhausted = consumed >= seq.length;
            if size >= block_size {
                finish_block(&mut blocks, &seq_data, block_id, seq_offset, block_overlap);
                block_id += 1;
                size = 0;
                block_started = false;
                // If the sequence isn't done, the next window of it is a
                // continuation that should carry overlap context.
                continuation = !seq_exhausted;
            } else {
                // Sequence finished without filling the block; keep the block
                // open for the next input sequence (no overlap across the
                // boundary between distinct sequences).
                continuation = false;
            }
        }
    }

    // Flush any partially-filled trailing block.
    if block_started {
        finish_block(&mut blocks, &seq_data, block_id, seq_offset, block_overlap);
    }

    // Recompute per-window block_offset (position within the block's text).
    let mut current_block = usize::MAX;
    let mut block_offset = 0usize;
    for window in &mut seq_data {
        if window.block_id != current_block {
            current_block = window.block_id;
            block_offset = 0;
        }
        window.block_offset = block_offset;
        block_offset += window.length;
    }

    (seq_data, blocks)
}

pub(super) fn assign_ambiguities_to_blocks(
    seq_data: &[SequenceMetadata],
    ambig_ranges: &[AmbiguityRange],
    blocks: &mut [BlockRecord],
) {
    for block in &mut *blocks {
        block.ambig_offset = 0;
        block.ambig_count = 0;
    }

    // Match C's emission order (makehmmerdb.c:656-668): C records ambiguity
    // ranges while walking each FM *block's* text, sequence-window by
    // sequence-window, residue by residue. So the flat `ambig_list` is
    // BLOCK-major (in block order), then within a block in the order the
    // window's residues are scanned (i.e. global position order). Because
    // `seq_data` already lists windows in block order, iterating windows
    // (outer) then ambiguity ranges in their recorded position order (inner)
    // reproduces C's exact ordering — including re-emitting a degenerate run
    // that falls in an inter-block overlap region once per block, since the
    // continuation window re-reads (and thus re-records) those residues.
    let mut compact_idx = 0usize;
    for seq in seq_data {
        for range in ambig_ranges {
            if intersect_ambiguity_with_sequence(range, seq).is_some() {
                if let Some(block) = blocks.get_mut(seq.block_id) {
                    if block.ambig_count == 0 {
                        block.ambig_offset = compact_idx;
                    }
                    block.ambig_count += 1;
                }
                compact_idx += 1;
            }
        }
    }

    for block in blocks.iter_mut() {
        if block.ambig_count == 0 {
            block.ambig_offset = compact_idx;
        }
    }
}

pub(super) fn build_index_records(
    all_text: &[u8],
    blocks: &[BlockRecord],
    include_reverse_strand: bool,
) -> Vec<IndexRecord> {
    let mut records = Vec::with_capacity(blocks.len() * if include_reverse_strand { 2 } else { 1 });
    for block in blocks {
        let block_text = &all_text[block.fm_start..block.fm_start + block.length];
        let reversed_text: Vec<u8> = block_text.iter().rev().copied().collect();
        records.push(IndexRecord {
            block_id: block.id,
            kind: INDEX_KIND_FORWARD_STRAND,
            text_start: block.fm_start,
            text_len: block.length,
            seq_offset: block.seq_offset,
            seq_count: block.seq_count,
            ambig_offset: block.ambig_offset,
            ambig_count: block.ambig_count,
            overlap_bases: block.overlap_bases,
            fm: FmIndex::build(&reversed_text),
        });

        if include_reverse_strand {
            records.push(IndexRecord {
                block_id: block.id,
                kind: INDEX_KIND_REVERSE_STRAND,
                text_start: block.fm_start,
                text_len: block.length,
                seq_offset: block.seq_offset,
                seq_count: block.seq_count,
                ambig_offset: block.ambig_offset,
                ambig_count: block.ambig_count,
                overlap_bases: block.overlap_bases,
                fm: FmIndex::build(block_text),
            });
        }
    }
    records
}
