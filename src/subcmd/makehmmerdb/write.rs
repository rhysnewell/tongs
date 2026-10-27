use super::*;

pub(super) fn write_index_extension<W: Write + ?Sized>(
    out: &mut W,
    fwd_only: bool,
    records: &[IndexRecord],
) -> std::io::Result<()> {
    out.write_all(INDEX_MAGIC)?;
    out.write_all(&INDEX_VERSION.to_le_bytes())?;
    out.write_all(&(fwd_only as u32).to_le_bytes())?;
    out.write_all(&(records.len() as u64).to_le_bytes())?;

    for record in records {
        for value in [
            record.block_id,
            record.text_start,
            record.text_len,
            record.seq_offset,
            record.seq_count,
            record.ambig_offset,
            record.ambig_count,
            record.overlap_bases,
        ] {
            out.write_all(&(value as u64).to_le_bytes())?;
        }
        out.write_all(&record.kind.to_le_bytes())?;
        out.write_all(&(record.fm.bwt.len() as u64).to_le_bytes())?;
        out.write_all(&(record.fm.sa.len() as u64).to_le_bytes())?;
        out.write_all(&(record.fm.c.len() as u64).to_le_bytes())?;
        out.write_all(&record.fm.bwt)?;
        for &sa_val in &record.fm.sa {
            out.write_all(&sa_val.to_le_bytes())?;
        }
        for &c_val in &record.fm.c {
            out.write_all(&(c_val as u64).to_le_bytes())?;
        }
    }

    Ok(())
}

pub(super) fn write_metadata_extension<W: Write + ?Sized>(
    out: &mut W,
    block_size_mb: usize,
    overlap_bases: usize,
    seq_data: &[SequenceMetadata],
    blocks: &[BlockRecord],
    ambig_ranges: &[AmbiguityRange],
) -> std::io::Result<()> {
    out.write_all(META_MAGIC)?;
    out.write_all(&META_VERSION.to_le_bytes())?;
    out.write_all(&(block_size_mb as u64).to_le_bytes())?;
    out.write_all(&(overlap_bases as u64).to_le_bytes())?;
    out.write_all(&(seq_data.len() as u64).to_le_bytes())?;
    out.write_all(&(blocks.len() as u64).to_le_bytes())?;
    out.write_all(&(ambig_ranges.len() as u64).to_le_bytes())?;

    for seq in seq_data {
        for value in [
            seq.target_id,
            seq.target_start,
            seq.fm_start,
            seq.length,
            seq.block_id,
            seq.block_offset,
            seq.overlap_bases,
        ] {
            out.write_all(&(value as u64).to_le_bytes())?;
        }
        write_string(out, &seq.name)?;
        write_string(out, &seq.acc)?;
        write_string(out, &seq.desc)?;
    }

    for block in blocks {
        for value in [
            block.id,
            block.fm_start,
            block.length,
            block.seq_offset,
            block.seq_count,
            block.ambig_offset,
            block.ambig_count,
            block.overlap_bases,
        ] {
            out.write_all(&(value as u64).to_le_bytes())?;
        }
    }

    for range in ambig_ranges {
        out.write_all(&(range.lower as u64).to_le_bytes())?;
        out.write_all(&(range.upper as u64).to_le_bytes())?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_c_metadata_extension<W: Write + ?Sized>(
    out: &mut W,
    fwd_only: bool,
    fm_alphabet: FmAlphabet,
    sa_freq: usize,
    bin_length: usize,
    char_count: usize,
    seq_data: &[SequenceMetadata],
    blocks: &[BlockRecord],
    ambig_ranges: &[AmbiguityRange],
) -> std::io::Result<()> {
    let compact_starts = c_block_compact_sequence_starts(seq_data, blocks);
    let compact_ambig_ranges = compact_ambiguity_ranges(seq_data, &compact_starts, ambig_ranges);

    out.write_all(C_META_MAGIC)?;
    out.write_all(&C_META_VERSION.to_le_bytes())?;
    write_c_metadata_payload(
        out,
        fwd_only,
        fm_alphabet,
        sa_freq,
        bin_length,
        char_count,
        seq_data,
        blocks,
        &compact_starts,
        &compact_ambig_ranges,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_c_stream_extension<W: Write + ?Sized>(
    out: &mut W,
    fwd_only: bool,
    fm_alphabet: FmAlphabet,
    sa_freq: usize,
    bin_length: usize,
    char_count: usize,
    seq_data: &[SequenceMetadata],
    blocks: &[BlockRecord],
    ambig_ranges: &[AmbiguityRange],
    all_text: &[u8],
) -> std::io::Result<()> {
    let payload = build_c_stream_payload(
        fwd_only,
        fm_alphabet,
        sa_freq,
        bin_length,
        char_count,
        seq_data,
        blocks,
        ambig_ranges,
        all_text,
    )?;

    out.write_all(C_STREAM_MAGIC)?;
    out.write_all(&C_STREAM_VERSION.to_le_bytes())?;
    out.write_all(&checked_u64(payload.len(), "C stream payload length")?.to_le_bytes())?;
    out.write_all(&payload)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn write_native_c_stream<W: Write + ?Sized>(
    out: &mut W,
    fwd_only: bool,
    fm_alphabet: FmAlphabet,
    sa_freq: usize,
    bin_length: usize,
    char_count: usize,
    seq_data: &[SequenceMetadata],
    blocks: &[BlockRecord],
    ambig_ranges: &[AmbiguityRange],
    all_text: &[u8],
) -> std::io::Result<()> {
    let payload = build_c_stream_payload(
        fwd_only,
        fm_alphabet,
        sa_freq,
        bin_length,
        char_count,
        seq_data,
        blocks,
        ambig_ranges,
        all_text,
    )?;
    out.write_all(&payload)
}
