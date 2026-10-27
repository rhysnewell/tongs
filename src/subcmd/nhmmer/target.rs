use super::*;

fn open_target_seq_file(
    path: &Path,
    abc: &Alphabet,
    tformat: Option<&str>,
) -> tongs::errors::HmmerResult<sequence::SeqFile<Box<dyn std::io::Read>>> {
    if tformat.is_some_and(|format| format.eq_ignore_ascii_case("fasta")) {
        sequence::open_fasta_seq_file(path, abc)
    } else {
        sequence::open_seq_file(path, abc)
    }
}

pub(super) struct NhmmerTargetDb {
    pub(super) sequences: Vec<Sequence>,
    pub(super) fm_sequence_meta: Vec<NhmmerFmSequenceMeta>,
    pub(super) fm_ambiguities: Vec<(usize, usize)>,
    pub(super) fm_index_records: Vec<NhmmerFmIndexRecord>,
}

#[derive(Clone, Copy)]
pub(crate) struct NhmmerSearchFmContext<'a> {
    pub(super) record: &'a NhmmerFmIndexRecord,
    pub(super) seq_meta: &'a [NhmmerFmSequenceMeta],
    pub(super) ambiguities: &'a [(usize, usize)],
    pub(super) seq_idx: usize,
}

pub(super) fn read_target_database(args: &Args, abc: &Alphabet) -> Result<NhmmerTargetDb, String> {
    let path = &args.seqdb;
    let tformat = args.tformat.as_deref();
    if tformat.is_some_and(|format| format.eq_ignore_ascii_case("fasta"))
        && path != Path::new("-")
        && path_looks_like_makehmmerdb_target(path)
    {
        return Err(format!(
            "Unable to guess alphabet for target sequence database file {}",
            path.display()
        ));
    }
    if tformat.is_some_and(is_fmindex_target_format) {
        if args.restrictdb_stkey.is_some() || args.restrictdb_n.is_some() || args.ssifile.is_some()
        {
            return Err("restricted database options require FASTA targets".to_string());
        }
        reject_fmindex_max(args.max)?;
        return read_makehmmerdb_target_database(path, abc);
    }

    let restrict_requested =
        args.restrictdb_stkey.is_some() || args.restrictdb_n.is_some() || args.ssifile.is_some();
    if !restrict_requested && tformat.is_none() && path != Path::new("-") {
        let mut magic = [0u8; 8];
        if std::fs::File::open(path)
            .and_then(|mut file| file.read_exact(&mut magic))
            .is_ok()
            && magic == *b"HMMERDB\0"
        {
            reject_fmindex_max(args.max)?;
            return read_makehmmerdb_target_database(path, abc);
        }

        if is_likely_raw_makehmmerdb_c_stream(&magic) {
            reject_fmindex_max(args.max)?;
            return read_makehmmerdb_target_database(path, abc);
        }
    }

    let mut sqf = if let Some(stkey) = args.restrictdb_stkey.as_deref() {
        crate::subcmd::hmmsearch::open_restricted_target_seq_file(
            path,
            abc,
            tformat,
            stkey,
            args.ssifile.as_deref(),
        )?
    } else {
        open_target_seq_file(path, abc, tformat).map_err(|e| e.to_string())?
    };
    let mut sequences = Vec::new();
    let mut sq = Sequence::new();
    let mut restrict_seen = 0usize;
    while sqf.read(&mut sq).map_err(|e| e.to_string())? {
        if args
            .restrictdb_n
            .is_some_and(|limit| restrict_seen >= limit)
        {
            break;
        }
        restrict_seen += 1;
        sequences.push(sq.clone());
        sq.reuse();
    }
    Ok(NhmmerTargetDb {
        sequences,
        fm_sequence_meta: Vec::new(),
        fm_ambiguities: Vec::new(),
        fm_index_records: Vec::new(),
    })
}

fn path_looks_like_makehmmerdb_target(path: &Path) -> bool {
    let mut magic = [0u8; 8];
    std::fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut magic))
        .is_ok()
        && (magic == *b"HMMERDB\0" || is_likely_raw_makehmmerdb_c_stream(&magic))
}

fn reject_fmindex_max(do_max: bool) -> Result<(), String> {
    if do_max {
        Err("--max flag is incompatible with the fmindex target type".to_string())
    } else {
        Ok(())
    }
}

pub(super) fn is_fmindex_target_format(format: &str) -> bool {
    format.eq_ignore_ascii_case("fmindex")
        || format.eq_ignore_ascii_case("fm")
        || format.eq_ignore_ascii_case("hmmerdb")
}

fn is_likely_raw_makehmmerdb_c_stream(prefix: &[u8]) -> bool {
    matches!(prefix.first(), Some(0 | 1))
        && prefix.get(1) == Some(&0)
        && prefix.get(2) == Some(&4)
        && prefix.get(3) == Some(&2)
}

fn read_makehmmerdb_target_database(path: &Path, abc: &Alphabet) -> Result<NhmmerTargetDb, String> {
    if path == Path::new("-") {
        return Err("nhmmer --tformat fmindex does not support stdin targets yet".to_string());
    }
    let mmap = MmapBytes::open(path, MAKEHMMERDB_MAX_IN_MEMORY_BYTES)?;
    parse_makehmmerdb_target_database(mmap.as_slice(), abc, Some(Arc::clone(&mmap)))
}

#[derive(Debug)]
pub(super) struct NhmmerFmSequenceMeta {
    pub(super) name: String,
    pub(super) acc: String,
    pub(super) desc: String,
    pub(super) target_start: usize,
    pub(super) fm_start: usize,
    pub(super) length: usize,
}

fn parse_makehmmerdb_target_database(
    bytes: &[u8],
    abc: &Alphabet,
    mapped_bytes: Option<Arc<MmapBytes>>,
) -> Result<NhmmerTargetDb, String> {
    let is_container = bytes.starts_with(b"HMMERDB\0");
    let (payload_bytes, payload_base_offset, index_start_hint) = if is_container {
        let stream_start = find_bytes(bytes, b"HMMERDB_C_STREAM\0")
            .ok_or_else(|| "makehmmerdb file is missing the C FM stream extension".to_string())?;
        let mut cursor = Cursor::new(&bytes[stream_start + b"HMMERDB_C_STREAM\0".len()..]);
        let version = read_u32(&mut cursor)?;
        if version != 1 {
            return Err(format!(
                "unsupported makehmmerdb C stream version {version}"
            ));
        }
        let payload_len = u64_to_usize(
            read_u64(&mut cursor)?,
            "makehmmerdb C stream payload length",
        )?;
        let payload_start = stream_start + b"HMMERDB_C_STREAM\0".len() + 12;
        let payload_end = payload_start
            .checked_add(payload_len)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| "makehmmerdb C stream payload is truncated".to_string())?;
        (
            &bytes[payload_start..payload_end],
            payload_start,
            payload_end,
        )
    } else {
        (bytes, 0, 0)
    };
    let mut payload = Cursor::new(payload_bytes);

    let (fwd_only, freq_sa, freq_cnt_b, block_count, sequences, ambiguities) =
        read_makehmmerdb_c_metadata_payload(&mut payload)?;
    let mut records_by_block = Vec::with_capacity(block_count);
    let mut reverse_records_by_block = Vec::with_capacity(block_count);
    let mut c_stream_records = Vec::with_capacity(block_count * if fwd_only { 1 } else { 2 });
    for block_id in 0..block_count {
        let record = read_makehmmerdb_c_fm_record(
            &mut payload,
            freq_sa,
            freq_cnt_b,
            true,
            payload_base_offset,
            mapped_bytes.is_none(),
        )?;
        c_stream_records.push(NhmmerFmOccSource::from_record(block_id, 0, &record));
        records_by_block.push(record);
        if !fwd_only {
            let reverse_record = read_makehmmerdb_c_fm_record(
                &mut payload,
                freq_sa,
                freq_cnt_b,
                false,
                payload_base_offset,
                mapped_bytes.is_none(),
            )?;
            c_stream_records.push(NhmmerFmOccSource::from_record(block_id, 1, &reverse_record));
            reverse_records_by_block.push(Some(reverse_record));
        } else {
            reverse_records_by_block.push(None);
        }
    }
    if makehmmerdb_remaining(&payload) != 0 {
        return Err("trailing bytes after makehmmerdb C stream records".to_string());
    }
    if records_by_block.is_empty() && !sequences.is_empty() {
        return Err("makehmmerdb C stream has sequence metadata but no FM records".to_string());
    }
    let container_index_records = if is_container {
        parse_makehmmerdb_index_extension_mapped(
            bytes,
            mapped_bytes.clone(),
            index_start_hint,
            &c_stream_records,
        )?
    } else {
        Vec::new()
    };
    validate_makehmmerdb_fm_records(&records_by_block, ambiguities.len())?;
    if !container_index_records.is_empty() {
        validate_makehmmerdb_index_records(
            &container_index_records,
            &records_by_block,
            fwd_only,
            ambiguities.len(),
        )?;
    }
    let (mut fm_index_records, raw_block_texts) = if !container_index_records.is_empty() {
        (container_index_records, Vec::new())
    } else if !is_container {
        build_raw_c_stream_fm_index_records(
            &mut records_by_block,
            &mut reverse_records_by_block,
            fwd_only,
            mapped_bytes.clone(),
        )?
    } else {
        (Vec::new(), Vec::new())
    };
    if !fm_index_records.is_empty() {
        let block_texts = if raw_block_texts.is_empty() {
            records_by_block
                .iter_mut()
                .map(|record| {
                    let text = std::mem::take(&mut record.text);
                    if text.len() == record.text_bases_len {
                        Arc::<[u8]>::from(text.into_boxed_slice())
                    } else {
                        Arc::<[u8]>::from(text[..record.text_bases_len].to_vec())
                    }
                })
                .collect()
        } else {
            raw_block_texts
        };
        for record in &mut fm_index_records {
            if record.text.is_empty() {
                record.text = Arc::clone(&block_texts[record.block_id]);
            }
            if record.packed_text.is_empty() {
                record.packed_text =
                    Arc::from(records_by_block[record.block_id].packed_text.clone());
            }
            if record.mapped_packed_text.is_none() {
                if let Some(bytes) = mapped_bytes.as_ref() {
                    let block = &records_by_block[record.block_id];
                    if block.packed_text_bytes != 0 {
                        record.mapped_packed_text = Some(MappedPackedBytes {
                            bytes: Arc::clone(bytes),
                            offset: block.packed_text_offset,
                            len: block.packed_text_bytes,
                        });
                    }
                }
            }
        }
    }

    let fm_target_has_text =
        !fm_index_records.is_empty() && fm_index_records.iter().any(|record| record.has_text());
    let mut out = Vec::with_capacity(sequences.len());
    for (seq_idx, seq_meta) in sequences.iter().enumerate() {
        let block = records_by_block
            .iter()
            .find(|record| {
                seq_idx >= record.seq_offset
                    && seq_idx < record.seq_offset + record.seq_count
                    && seq_meta.fm_start + seq_meta.length <= record.text_bases_len
            })
            .ok_or_else(|| {
                format!(
                    "sequence {} is outside all makehmmerdb FM text blocks",
                    seq_meta.name
                )
            })?;
        let mut seq = Sequence::new();
        seq.name = seq_meta.name.clone();
        seq.acc = seq_meta.acc.clone();
        seq.desc = seq_meta.desc.clone();
        if fm_target_has_text {
            seq.dsq = vec![DSQ_SENTINEL, DSQ_SENTINEL];
            seq.n = seq_meta.length;
            seq.l = seq_meta.length;
        } else if !block.packed_text.is_empty() {
            let text =
                unpack_dna_quads_range(&block.packed_text, seq_meta.fm_start, seq_meta.length)?;
            seq.dsq = abc.digitize(&text);
            seq.n = text.len();
            seq.l = text.len();
        } else if let Some(bytes) = mapped_bytes
            .as_ref()
            .filter(|_| block.packed_text_bytes != 0)
        {
            let packed_text = MappedPackedBytes {
                bytes: Arc::clone(bytes),
                offset: block.packed_text_offset,
                len: block.packed_text_bytes,
            };
            let text =
                unpack_mapped_dna_quads_range(&packed_text, seq_meta.fm_start, seq_meta.length)?;
            seq.dsq = abc.digitize(&text);
            seq.n = text.len();
            seq.l = text.len();
        } else {
            let text = &block.text[seq_meta.fm_start..seq_meta.fm_start + seq_meta.length];
            seq.dsq = abc.digitize(text);
            seq.n = text.len();
            seq.l = text.len();
        }
        out.push(seq);
    }
    Ok(NhmmerTargetDb {
        sequences: out,
        fm_sequence_meta: sequences,
        fm_ambiguities: ambiguities,
        fm_index_records,
    })
}
