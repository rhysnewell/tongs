use super::*;

#[derive(Debug, Clone)]
pub(super) struct MappedPackedBytes {
    pub(super) bytes: Arc<MmapBytes>,
    pub(super) offset: usize,
    pub(super) len: usize,
}

impl MappedPackedBytes {
    pub(super) fn byte_at(&self, idx: usize) -> Option<u8> {
        if idx >= self.len {
            return None;
        }
        self.bytes
            .as_slice()
            .get(self.offset.checked_add(idx)?)
            .copied()
    }
}

#[derive(Debug)]
pub(super) struct NhmmerFmIndexRecord {
    pub(super) block_id: usize,
    pub(super) kind: u32,
    pub(super) text_start: usize,
    pub(super) text_len: usize,
    pub(super) text: Arc<[u8]>,
    pub(super) packed_text: Arc<[u8]>,
    pub(super) mapped_packed_text: Option<MappedPackedBytes>,
    pub(super) seq_offset: usize,
    pub(super) seq_count: usize,
    pub(super) ambig_offset: usize,
    pub(super) ambig_count: usize,
    pub(super) overlap_bases: usize,
    pub(super) fm: FmIndex,
}

impl NhmmerFmIndexRecord {
    pub(super) fn has_text(&self) -> bool {
        !self.text.is_empty() || !self.packed_text.is_empty() || self.mapped_packed_text.is_some()
    }
}

#[derive(Debug)]
pub(super) struct NhmmerFmOccSource {
    pub(super) block_id: usize,
    pub(super) kind: u32,
    pub(super) term_loc: usize,
    pub(super) freq_cnt_b: usize,
    pub(super) occ_b_offset: usize,
    pub(super) occ_b_count: usize,
    pub(super) occ_sb_offset: usize,
    pub(super) occ_sb_count: usize,
}

impl NhmmerFmOccSource {
    pub(super) fn from_record(block_id: usize, kind: u32, record: &NhmmerFmRecord) -> Self {
        Self {
            block_id,
            kind,
            term_loc: record.term_loc,
            freq_cnt_b: record.freq_cnt_b,
            occ_b_offset: record.occ_b_offset,
            occ_b_count: record.occ_b_count,
            occ_sb_offset: record.occ_sb_offset,
            occ_sb_count: record.occ_sb_count,
        }
    }
}

#[cfg(test)]
pub(super) fn parse_makehmmerdb_index_extension(
    bytes: &[u8],
) -> Result<Vec<NhmmerFmIndexRecord>, String> {
    parse_makehmmerdb_index_extension_mapped(bytes, None, 0, &[])
}

pub(super) fn parse_makehmmerdb_index_extension_mapped(
    bytes: &[u8],
    mapped_bytes: Option<Arc<MmapBytes>>,
    search_start: usize,
    c_stream_records: &[NhmmerFmOccSource],
) -> Result<Vec<NhmmerFmIndexRecord>, String> {
    let search_start = search_start.min(bytes.len());
    let index_start = if bytes[search_start..].starts_with(MAKEHMMERDB_INDEX_MAGIC) {
        Some(search_start)
    } else {
        find_bytes(&bytes[search_start..], MAKEHMMERDB_INDEX_MAGIC)
            .map(|offset| search_start + offset)
    };
    let Some(index_start) = index_start else {
        return Ok(Vec::new());
    };
    let mut cursor = Cursor::new(&bytes[index_start + MAKEHMMERDB_INDEX_MAGIC.len()..]);
    let version = read_u32(&mut cursor)?;
    if version != 1 {
        return Err(format!(
            "unsupported makehmmerdb FM-index extension version {version}"
        ));
    }
    let _fwd_only = read_u32(&mut cursor)? != 0;
    let record_count = u64_to_usize(
        read_u64(&mut cursor)?,
        "makehmmerdb FM-index record table count",
    )?;
    ensure_makehmmerdb_remaining(
        &cursor,
        record_count,
        MAKEHMMERDB_INDEX_MIN_RECORD_BYTES,
        "makehmmerdb FM-index record table",
    )?;
    let mut records = Vec::with_capacity(record_count);
    for _ in 0..record_count {
        let block_id = u64_to_usize(read_u64(&mut cursor)?, "makehmmerdb FM-index block id")?;
        let text_start = u64_to_usize(read_u64(&mut cursor)?, "makehmmerdb FM-index text start")?;
        let text_len = u64_to_usize(read_u64(&mut cursor)?, "makehmmerdb FM-index text length")?;
        let seq_offset = u64_to_usize(
            read_u64(&mut cursor)?,
            "makehmmerdb FM-index sequence offset",
        )?;
        let seq_count = u64_to_usize(
            read_u64(&mut cursor)?,
            "makehmmerdb FM-index sequence count",
        )?;
        let ambig_offset = u64_to_usize(
            read_u64(&mut cursor)?,
            "makehmmerdb FM-index ambiguity offset",
        )?;
        let ambig_count = u64_to_usize(
            read_u64(&mut cursor)?,
            "makehmmerdb FM-index ambiguity count",
        )?;
        let overlap_bases =
            u64_to_usize(read_u64(&mut cursor)?, "makehmmerdb FM-index overlap bases")?;
        let kind = read_u32(&mut cursor)?;
        let bwt_len = u64_to_usize(read_u64(&mut cursor)?, "makehmmerdb FM-index BWT length")?;
        let sa_len = u64_to_usize(
            read_u64(&mut cursor)?,
            "makehmmerdb FM-index suffix array length",
        )?;
        let c_len = u64_to_usize(
            read_u64(&mut cursor)?,
            "makehmmerdb FM-index C table length",
        )?;
        if c_len != 256 {
            return Err(format!(
                "makehmmerdb FM-index C table has {c_len} entries, expected 256"
            ));
        }
        let bwt_bytes = bwt_len;
        let sa_bytes = checked_count_bytes(sa_len, 4, "makehmmerdb FM-index suffix array")?;
        let c_bytes = checked_count_bytes(c_len, 8, "makehmmerdb FM-index C table")?;
        let remaining_record_bytes = bwt_bytes
            .checked_add(sa_bytes)
            .and_then(|value| value.checked_add(c_bytes))
            .ok_or_else(|| "makehmmerdb FM-index record byte span overflows usize".to_string())?;
        if makehmmerdb_remaining(&cursor) < remaining_record_bytes {
            return Err("makehmmerdb FM-index record payload is truncated".to_string());
        }

        let payload_base = index_start + MAKEHMMERDB_INDEX_MAGIC.len();
        let bwt_offset = payload_base
            .checked_add(cursor.position() as usize)
            .ok_or_else(|| "makehmmerdb FM-index BWT offset overflows usize".to_string())?;
        bwt_offset
            .checked_add(bwt_len)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| "makehmmerdb FM-index BWT is truncated".to_string())?;
        cursor.set_position((cursor.position() as usize + bwt_len) as u64);

        let sa_offset = payload_base
            .checked_add(cursor.position() as usize)
            .ok_or_else(|| {
                "makehmmerdb FM-index suffix-array offset overflows usize".to_string()
            })?;
        let sa_end = sa_offset
            .checked_add(sa_bytes)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| "makehmmerdb FM-index suffix array is truncated".to_string())?;
        let sa = if mapped_bytes.is_none() {
            let mut sa_cursor = Cursor::new(&bytes[sa_offset..sa_end]);
            let mut sa = Vec::with_capacity(sa_len);
            for _ in 0..sa_len {
                sa.push(read_i32(&mut sa_cursor)?);
            }
            Some(sa)
        } else {
            None
        };
        cursor.set_position((cursor.position() as usize + sa_bytes) as u64);

        let mut c = [0usize; 256];
        for value in &mut c {
            *value = u64_to_usize(read_u64(&mut cursor)?, "makehmmerdb FM-index C table value")?;
        }

        let fm = if let Some(mapped) = mapped_bytes.clone() {
            if let Some(occ_source) = c_stream_records
                .iter()
                .find(|source| source.block_id == block_id && source.kind == kind)
            {
                FmIndex::from_mapped_file_parts_with_makehmmerdb_occ(
                    mapped,
                    bwt_offset,
                    bwt_len,
                    sa_offset,
                    sa_len,
                    c,
                    text_len,
                    occ_source.term_loc,
                    occ_source.freq_cnt_b,
                    occ_source.occ_b_offset,
                    occ_source.occ_b_count,
                    occ_source.occ_sb_offset,
                    occ_source.occ_sb_count,
                )?
            } else {
                FmIndex::from_mapped_file_parts(
                    mapped, bwt_offset, bwt_len, sa_offset, sa_len, c, text_len,
                )?
            }
        } else {
            let bwt = bytes[bwt_offset..bwt_offset + bwt_len].to_vec();
            FmIndex::from_parts_sampled(bwt, sa.unwrap(), c, text_len)?
        };
        records.push(NhmmerFmIndexRecord {
            block_id,
            kind,
            text_start,
            text_len,
            text: Arc::from([]),
            packed_text: Arc::from([]),
            mapped_packed_text: None,
            seq_offset,
            seq_count,
            ambig_offset,
            ambig_count,
            overlap_bases,
            fm,
        });
    }
    Ok(records)
}

pub(super) fn validate_makehmmerdb_fm_records(
    records_by_block: &[NhmmerFmRecord],
    ambiguity_count: usize,
) -> Result<(), String> {
    for record in records_by_block {
        if record.seq_offset.checked_add(record.seq_count).is_none() {
            return Err("makehmmerdb C stream sequence span overflows usize".to_string());
        }
        let Some(ambig_end) = record.ambig_offset.checked_add(record.ambig_count) else {
            return Err("makehmmerdb C stream ambiguity span overflows usize".to_string());
        };
        if ambig_end > ambiguity_count {
            return Err(
                "makehmmerdb C stream ambiguity span is outside metadata table".to_string(),
            );
        }
        if record.overlap_bases > record.text_bases_len {
            return Err("makehmmerdb C stream overlap exceeds block text length".to_string());
        }
    }
    Ok(())
}

pub(super) fn validate_makehmmerdb_index_records(
    records: &[NhmmerFmIndexRecord],
    records_by_block: &[NhmmerFmRecord],
    fwd_only: bool,
    ambiguity_count: usize,
) -> Result<(), String> {
    let block_count = records_by_block.len();
    let mut expected_starts = Vec::with_capacity(block_count);
    let mut text_start = 0usize;
    for block in records_by_block {
        expected_starts.push(text_start);
        text_start = text_start
            .checked_add(block.text_bases_len)
            .ok_or_else(|| "makehmmerdb FM-index text span overflows usize".to_string())?;
    }
    let mut seen = vec![[false; 2]; block_count];
    for record in records {
        if record.block_id >= block_count {
            return Err(format!(
                "makehmmerdb FM-index record references block {} but only {block_count} block(s) exist",
                record.block_id
            ));
        }
        if record.kind > 1 {
            return Err(format!(
                "makehmmerdb FM-index record has unsupported strand kind {}",
                record.kind
            ));
        }
        if record.fm.n != record.text_len {
            return Err("makehmmerdb FM-index text length metadata is inconsistent".to_string());
        }
        if record.text_start.checked_add(record.text_len).is_none() {
            return Err("makehmmerdb FM-index text span overflows usize".to_string());
        }
        if record.seq_offset.checked_add(record.seq_count).is_none() {
            return Err("makehmmerdb FM-index sequence span overflows usize".to_string());
        }
        if record
            .ambig_offset
            .checked_add(record.ambig_count)
            .is_none()
        {
            return Err("makehmmerdb FM-index ambiguity span overflows usize".to_string());
        }
        if record.overlap_bases > record.text_len {
            return Err("makehmmerdb FM-index overlap exceeds block text length".to_string());
        }
        let block = &records_by_block[record.block_id];
        if record.text_start != expected_starts[record.block_id]
            || record.text_len != block.text_bases_len
            || record.seq_offset != block.seq_offset
            || record.seq_count != block.seq_count
            || record.ambig_offset != block.ambig_offset
            || record.ambig_count != block.ambig_count
            || record.overlap_bases != block.overlap_bases
        {
            return Err(
                "makehmmerdb FM-index record metadata does not match C stream block".to_string(),
            );
        }
        let Some(ambig_end) = record.ambig_offset.checked_add(record.ambig_count) else {
            return Err("makehmmerdb FM-index ambiguity span overflows usize".to_string());
        };
        if ambig_end > ambiguity_count {
            return Err(
                "makehmmerdb FM-index ambiguity span is outside metadata table".to_string(),
            );
        }
        let kind = record.kind as usize;
        if seen[record.block_id][kind] {
            return Err("makehmmerdb FM-index contains duplicate block/strand record".to_string());
        }
        seen[record.block_id][kind] = true;
    }
    for (block_id, block_seen) in seen.iter().enumerate() {
        if !block_seen[0] {
            return Err(format!(
                "makehmmerdb FM-index is missing forward-strand record for block {block_id}"
            ));
        }
        if !fwd_only && !block_seen[1] {
            return Err(format!(
                "makehmmerdb FM-index is missing reverse-strand record for block {block_id}"
            ));
        }
    }
    Ok(())
}

pub(super) type FmIndexRecords = (Vec<NhmmerFmIndexRecord>, Vec<Arc<[u8]>>);

pub(super) fn build_raw_c_stream_fm_index_records(
    records_by_block: &mut [NhmmerFmRecord],
    reverse_records_by_block: &mut [Option<NhmmerFmRecord>],
    fwd_only: bool,
    mapped_bytes: Option<Arc<MmapBytes>>,
) -> Result<FmIndexRecords, String> {
    let record_passes = if fwd_only { 1 } else { 2 };
    let mut records = Vec::with_capacity(records_by_block.len() * record_passes);
    let mut block_texts = Vec::with_capacity(records_by_block.len());
    let mut text_start = 0usize;

    for (block_id, block) in records_by_block.iter_mut().enumerate() {
        let text_len = block.text_bases_len;
        if text_len == 0 {
            block_texts.push(Arc::from([]));
            text_start = text_start
                .checked_add(text_len)
                .ok_or_else(|| "raw makehmmerdb C-stream text span overflows usize".to_string())?;
            continue;
        }

        let block_text = Arc::from([]);
        let block_mapped_packed_text = mapped_bytes.as_ref().map(|bytes| MappedPackedBytes {
            bytes: Arc::clone(bytes),
            offset: block.packed_text_offset,
            len: block.packed_text_bytes,
        });
        let block_packed_text = if block_mapped_packed_text.is_some() {
            Arc::from([])
        } else {
            Arc::<[u8]>::from(std::mem::take(&mut block.packed_text))
        };
        block_texts.push(Arc::clone(&block_text));
        let fm = if let Some(mapped_bytes) = mapped_bytes.clone() {
            FmIndex::from_makehmmerdb_mapped_packed_dna_parts_mapped_sa(
                Arc::clone(&mapped_bytes),
                block.packed_bwt_offset,
                block.packed_bwt_bytes,
                block.bwt_len,
                mapped_bytes,
                block.raw_sa_offset,
                block.raw_sa_count,
                block.term_loc,
                block.freq_sa,
                block.freq_cnt_b,
                std::mem::take(&mut block.occ_b_codes),
                std::mem::take(&mut block.occ_sb_codes),
            )?
        } else {
            FmIndex::from_makehmmerdb_packed_dna_parts(
                std::mem::take(&mut block.packed_bwt),
                block.bwt_len,
                std::mem::take(&mut block.raw_sa_samples),
                block.term_loc,
                block.freq_sa,
                block.freq_cnt_b,
                std::mem::take(&mut block.occ_b_codes),
                std::mem::take(&mut block.occ_sb_codes),
            )?
        };
        records.push(NhmmerFmIndexRecord {
            block_id,
            kind: 0,
            text_start,
            text_len,
            text: Arc::clone(&block_text),
            packed_text: Arc::clone(&block_packed_text),
            mapped_packed_text: block_mapped_packed_text.clone(),
            seq_offset: block.seq_offset,
            seq_count: block.seq_count,
            ambig_offset: block.ambig_offset,
            ambig_count: block.ambig_count,
            overlap_bases: block.overlap_bases,
            fm,
        });

        if !fwd_only {
            let reverse = reverse_records_by_block
                .get_mut(block_id)
                .and_then(|record| record.as_mut())
                .ok_or_else(|| {
                    "raw makehmmerdb C-stream is missing reverse FM record".to_string()
                })?;
            records.push(NhmmerFmIndexRecord {
                block_id,
                kind: 1,
                text_start,
                text_len,
                text: Arc::clone(&block_text),
                packed_text: Arc::clone(&block_packed_text),
                mapped_packed_text: block_mapped_packed_text.clone(),
                seq_offset: block.seq_offset,
                seq_count: block.seq_count,
                ambig_offset: block.ambig_offset,
                ambig_count: block.ambig_count,
                overlap_bases: block.overlap_bases,
                fm: if let Some(mapped_bytes) = mapped_bytes.clone() {
                    FmIndex::from_makehmmerdb_mapped_packed_dna_parts_mapped_sa(
                        Arc::clone(&mapped_bytes),
                        reverse.packed_bwt_offset,
                        reverse.packed_bwt_bytes,
                        reverse.bwt_len,
                        mapped_bytes,
                        reverse.raw_sa_offset,
                        reverse.raw_sa_count,
                        reverse.term_loc,
                        reverse.freq_sa,
                        reverse.freq_cnt_b,
                        std::mem::take(&mut reverse.occ_b_codes),
                        std::mem::take(&mut reverse.occ_sb_codes),
                    )?
                } else {
                    FmIndex::from_makehmmerdb_packed_dna_parts(
                        std::mem::take(&mut reverse.packed_bwt),
                        reverse.bwt_len,
                        std::mem::take(&mut reverse.raw_sa_samples),
                        reverse.term_loc,
                        reverse.freq_sa,
                        reverse.freq_cnt_b,
                        std::mem::take(&mut reverse.occ_b_codes),
                        std::mem::take(&mut reverse.occ_sb_codes),
                    )?
                },
            });
        }

        text_start = text_start
            .checked_add(text_len)
            .ok_or_else(|| "raw makehmmerdb C-stream text span overflows usize".to_string())?;
    }

    Ok((records, block_texts))
}
