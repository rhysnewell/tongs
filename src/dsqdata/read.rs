use super::*;

/// Open and read an entire dsqdata database from `<basename>` and its three
/// `.dsq?` companion files.
///
/// Implements a two-thread pipeline mirroring Easel's loader/unpacker design:
/// a background **loader thread** streams raw packed bytes and metadata off
/// disk in chunks of up to [`CHUNK_MAXSEQ`] sequences and sends each
/// [`RawChunk`] over a channel; the main thread receives each chunk and
/// unpacks (decompresses) it while the loader prefetches the next one. This
/// overlaps disk I/O with CPU decompression, faithfully reflecting C's
/// threaded pipeline (which uses more threads but the same two-stage
/// loader→unpacker structure).
///
/// Magic/uniquetag validation is identical to C. The alphabet is determined
/// from the index header's `alphatype` (as C does when the caller passes a
/// NULL alphabet).
///
/// Returns the sequences with their leading/trailing [`DSQ_SENTINEL`] bytes
/// restored. Accession/description are populated from the metadata file.
/// Taxid is read (and format-validated) but not stored; see module docs.
pub fn read_dsqdata(basename: &Path) -> HmmerResult<Vec<Sequence>> {
    let p = basename;

    // ---- Stub file: parse "Easel dsqdata v1 x<tag>" first line. ----
    let stub_file = open(p)?;
    let mut stub_reader = BufReader::new(stub_file);
    let first = crate::hmmfile::read_capped_text_line(&mut stub_reader, MAX_DSQDATA_STUB_LINE_LEN)?
        .ok_or_else(|| HmmerError::Format("dsqdata stub file is empty".into()))?;
    let mut toks = first.split_whitespace();
    if toks.next() != Some("Easel") || toks.next() != Some("dsqdata") {
        return Err(HmmerError::Format("dsqdata stub has bad format".into()));
    }
    let vtok = toks
        .next()
        .ok_or_else(|| HmmerError::Format("dsqdata stub missing version".into()))?;
    if !vtok.starts_with('v') || vtok[1..].parse::<u64>().is_err() {
        return Err(HmmerError::Format("dsqdata stub has bad version".into()));
    }
    let xtok = toks
        .next()
        .ok_or_else(|| HmmerError::Format("dsqdata stub missing tag".into()))?;
    if !xtok.starts_with('x') {
        return Err(HmmerError::Format("dsqdata stub has bad tag".into()));
    }
    let stub_tag: u32 = xtok[1..]
        .parse()
        .map_err(|_| HmmerError::Format("dsqdata stub has non-integer tag".into()))?;

    // ---- Index file header. ----
    let dsqi_path = with_ext(p, "dsqi");
    let dsqm_path = with_ext(p, "dsqm");
    let dsqs_path = with_ext(p, "dsqs");

    let ifile = open(&dsqi_path)?;
    let ifile_len = ifile.metadata().map_err(HmmerError::Io)?.len();
    let mut ifp = BufReader::new(ifile);
    let magic = read_u32(&mut ifp)?;
    let tag = read_u32(&mut ifp)?;
    let alphatype = read_u32(&mut ifp)?;
    let _flags = read_u32(&mut ifp)?;
    let _max_namelen = read_u32(&mut ifp)?;
    let _max_acclen = read_u32(&mut ifp)?;
    let _max_desclen = read_u32(&mut ifp)?;
    let _max_seqlen = read_u64(&mut ifp)?;
    let nseq = read_u64(&mut ifp)?;
    let _nres = read_u64(&mut ifp)?;

    if tag != stub_tag {
        return Err(HmmerError::Format(
            "dsqdata index file has bad tag, doesn't match stub".into(),
        ));
    }
    if magic == MAGIC_V1SWAP {
        return Err(HmmerError::Format(
            "dsqdata: cannot read byte-swapped data (unimplemented, as in C)".into(),
        ));
    }
    if magic != MAGIC_V1 {
        return Err(HmmerError::Format(
            "dsqdata index file has bad magic".into(),
        ));
    }
    let alphabet = match alphatype {
        1 => Alphabet::rna(),
        2 => Alphabet::dna(),
        3 => Alphabet::amino(),
        _ => {
            return Err(HmmerError::Format(format!(
                "dsqdata: invalid/unsupported alphabet type {}",
                alphatype
            )))
        }
    };
    let do_pack5 = alphabet.abc_type == AlphabetType::Amino;

    let nseq_usize = validate_dsqdata_nseq(nseq, ifile_len)?;

    // ---- Read all index records into memory (small: 16 bytes × nseq). ----
    // We read the full index up front so the loader thread can seek freely.
    let mut all_records: Vec<IndexRecord> = Vec::with_capacity(nseq_usize);
    for _ in 0..nseq_usize {
        let metadata_end = read_i64(&mut ifp)?;
        let psq_end = read_i64(&mut ifp)?;
        all_records.push(IndexRecord {
            metadata_end,
            psq_end,
        });
    }
    drop(ifp); // index fully consumed

    // ---- Validate metadata and sequence file headers. ----
    let mfile = open(&dsqm_path)?;
    let mfile_len = mfile.metadata().map_err(HmmerError::Io)?.len();
    let mut mfp = BufReader::new(mfile);
    let m_magic = read_u32(&mut mfp)?;
    let m_tag = read_u32(&mut mfp)?;
    if m_magic != magic {
        return Err(HmmerError::Format(
            "dsqdata metadata file has bad magic".into(),
        ));
    }
    if m_tag != stub_tag {
        return Err(HmmerError::Format(
            "dsqdata metadata file has bad tag, doesn't match stub".into(),
        ));
    }

    let sfile = open(&dsqs_path)?;
    let sfile_len = sfile.metadata().map_err(HmmerError::Io)?.len();
    let mut sfp = BufReader::new(sfile);
    let s_magic = read_u32(&mut sfp)?;
    let s_tag = read_u32(&mut sfp)?;
    if s_magic != magic {
        return Err(HmmerError::Format(
            "dsqdata sequence file has bad magic".into(),
        ));
    }
    if s_tag != stub_tag {
        return Err(HmmerError::Format(
            "dsqdata sequence file has bad tag, doesn't match stub".into(),
        ));
    }

    validate_dsqdata_index_extents(&all_records, mfile_len, sfile_len)?;

    // ---- Spawn the loader thread. ----
    // The loader reads raw bytes in chunks of CHUNK_MAXSEQ sequences and sends
    // RawChunk values over the channel. The main thread receives and unpacks
    // each chunk while the loader prefetches the next one.
    //
    // Channel is bounded to 1 so the loader stays at most one chunk ahead,
    // bounding memory use while still achieving I/O/CPU overlap.
    let (tx, rx) = mpsc::sync_channel::<Result<RawChunk, String>>(1);

    let loader_handle = std::thread::spawn(move || {
        let mut psq_last: i64 = -1;
        let mut meta_last: i64 = -1;
        let mut i = 0usize;
        let nseq_total = all_records.len();

        while i < nseq_total {
            // Take up to CHUNK_MAXSEQ index records for this chunk.
            let chunk_end = (i + CHUNK_MAXSEQ).min(nseq_total);
            let chunk_records = &all_records[i..chunk_end];

            let last_rec = chunk_records[chunk_records.len() - 1];

            // Number of packets in this chunk.
            let n_packets = (last_rec.psq_end - psq_last) as usize;
            // Number of metadata bytes in this chunk.
            let n_meta = (last_rec.metadata_end - meta_last) as usize;

            // Read packed sequence data.
            let mut packed_bytes = vec![0u8; n_packets * 4];
            if let Err(e) = sfp.read_exact(&mut packed_bytes) {
                let _ = tx.send(Err(format!("dsqdata loader: seq read error: {}", e)));
                return;
            }
            let packed: Vec<u32> = packed_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();

            // Read metadata.
            let mut metadata = vec![0u8; n_meta];
            if let Err(e) = mfp.read_exact(&mut metadata) {
                let _ = tx.send(Err(format!("dsqdata loader: meta read error: {}", e)));
                return;
            }

            // Adjust index records to be chunk-relative (subtract the offsets
            // that were consumed before this chunk started).
            let psq_base = psq_last;
            let meta_base = meta_last;
            let records: Vec<IndexRecord> = chunk_records
                .iter()
                .map(|r| IndexRecord {
                    psq_end: r.psq_end - psq_base - 1, // convert to 0-based packet index within chunk
                    metadata_end: r.metadata_end - meta_base - 1, // 0-based byte index within chunk
                })
                .collect();

            psq_last = last_rec.psq_end;
            meta_last = last_rec.metadata_end;
            i = chunk_end;

            if tx
                .send(Ok(RawChunk {
                    packed,
                    metadata,
                    records,
                }))
                .is_err()
            {
                // Receiver dropped (main thread errored out). Stop loading.
                return;
            }
        }
        // Loader done; channel closes automatically when `tx` is dropped here.
    });

    // ---- Main thread: receive and unpack each chunk. ----
    let mut sequences: Vec<Sequence> = Vec::with_capacity(nseq_usize);

    for raw in rx {
        let chunk = raw.map_err(HmmerError::Format)?;
        unpack_chunk(&chunk, do_pack5, &mut sequences)?;
    }

    // Join the loader thread to propagate any panics.
    loader_handle
        .join()
        .map_err(|_| HmmerError::Format("dsqdata loader thread panicked".into()))?;

    Ok(sequences)
}

/// Unpack a [`RawChunk`] into [`Sequence`]s, appending to `out`.
///
/// This is the "unpacker" side of the two-thread pipeline. It mirrors
/// `dsqdata_unpack_chunk()` in the C implementation but operates on
/// chunk-relative offsets pre-computed by the loader thread.
pub(super) fn unpack_chunk(
    chunk: &RawChunk,
    do_pack5: bool,
    out: &mut Vec<Sequence>,
) -> HmmerResult<()> {
    let psq = &chunk.packed;
    let metadata = &chunk.metadata;

    // The chunk's records have chunk-relative cumulative end offsets
    // (0-based, inclusive). Walk them to recover per-sequence slices.
    let mut psq_last: i64 = -1;
    let mut meta_last: i64 = -1;

    for rec in &chunk.records {
        // Packet range for this sequence within the chunk.
        let pstart = (psq_last + 1) as usize;
        let pn = (rec.psq_end - psq_last) as usize;
        if pn == 0 {
            return Err(HmmerError::Format(
                "dsqdata: index record encodes zero packets (every seq needs an EOD packet)".into(),
            ));
        }

        let mut residues = Vec::new();
        let (l, consumed) = if do_pack5 {
            unpack5(psq, pstart, &mut residues)?
        } else {
            unpack2(psq, pstart, &mut residues)?
        };
        if consumed != pn {
            return Err(HmmerError::Format(format!(
                "dsqdata: packet count mismatch (index says {}, unpacked {})",
                pn, consumed
            )));
        }
        psq_last = rec.psq_end;

        // Metadata range for this sequence within the chunk.
        let mstart = (meta_last + 1) as usize;
        let mend = (rec.metadata_end + 1) as usize; // exclusive
        if mend > metadata.len() || mstart > mend {
            return Err(HmmerError::Format(
                "dsqdata: metadata offset out of range".into(),
            ));
        }
        let block = &metadata[mstart..mend];
        // name\0 acc\0 desc\0 taxid(int32)
        let (name, rest) = take_cstr(block)?;
        let (acc, rest) = take_cstr(rest)?;
        let (desc, rest) = take_cstr(rest)?;
        if rest.len() != 4 {
            return Err(HmmerError::Format(
                "dsqdata: metadata record must end with exactly one taxid".into(),
            ));
        }
        // taxid: a per-sequence NCBI taxonomy id (int32; -1 = none), matching
        // C `esl_dsqdata`'s `chu->taxid[i]`.
        let taxid = i32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]);
        meta_last = rec.metadata_end;

        let mut dsq = Vec::with_capacity(l + 2);
        dsq.push(DSQ_SENTINEL);
        dsq.extend_from_slice(&residues);
        dsq.push(DSQ_SENTINEL);

        out.push(Sequence {
            name: String::from_utf8_lossy(name).into_owned(),
            acc: String::from_utf8_lossy(acc).into_owned(),
            desc: String::from_utf8_lossy(desc).into_owned(),
            dsq,
            n: l,
            l,
            taxid,
        });
    }
    Ok(())
}
pub(super) fn validate_dsq(sq: &Sequence) -> HmmerResult<()> {
    for (field, value) in [
        ("name", sq.name.as_bytes()),
        ("accession", sq.acc.as_bytes()),
        ("description", sq.desc.as_bytes()),
    ] {
        if value.contains(&0) {
            return Err(HmmerError::Format(format!(
                "Sequence {} {field} contains an embedded NUL byte",
                sq.name
            )));
        }
    }
    if sq.dsq.len() < sq.n + 2 {
        return Err(HmmerError::Format(format!(
            "Sequence {} digital data is shorter than declared length {}",
            sq.name, sq.n
        )));
    }
    if sq.dsq.first() != Some(&DSQ_SENTINEL) || sq.dsq.get(sq.n + 1) != Some(&DSQ_SENTINEL) {
        return Err(HmmerError::Format(format!(
            "Sequence {} is missing digital sentinels",
            sq.name
        )));
    }
    Ok(())
}

fn validate_dsqdata_nseq(nseq: u64, index_file_len: u64) -> HmmerResult<usize> {
    if index_file_len < DSQDATA_INDEX_HEADER_LEN {
        return Err(HmmerError::Format("dsqdata index file is truncated".into()));
    }
    let index_payload = index_file_len - DSQDATA_INDEX_HEADER_LEN;
    let max_records = index_payload / DSQDATA_INDEX_RECORD_LEN;
    if nseq > max_records {
        return Err(HmmerError::Format(format!(
            "dsqdata index declares {nseq} sequences but only {max_records} index records fit in the file"
        )));
    }
    usize::try_from(nseq).map_err(|_| {
        HmmerError::Format(format!(
            "dsqdata index declares too many sequences for this platform: {nseq}"
        ))
    })
}

pub(super) fn validate_dsqdata_index_extents(
    records: &[IndexRecord],
    metadata_file_len: u64,
    sequence_file_len: u64,
) -> HmmerResult<()> {
    if metadata_file_len < DSQDATA_SIDECAR_HEADER_LEN
        || sequence_file_len < DSQDATA_SIDECAR_HEADER_LEN
    {
        return Err(HmmerError::Format(
            "dsqdata sidecar file is truncated".into(),
        ));
    }
    let metadata_payload_len = metadata_file_len - DSQDATA_SIDECAR_HEADER_LEN;
    let sequence_payload_bytes = sequence_file_len - DSQDATA_SIDECAR_HEADER_LEN;
    if !sequence_payload_bytes.is_multiple_of(4) {
        return Err(HmmerError::Format(
            "dsqdata sequence file has a partial trailing packet".into(),
        ));
    }
    let sequence_packets = sequence_payload_bytes / 4;
    if records.is_empty() {
        if metadata_payload_len != 0 || sequence_packets != 0 {
            return Err(HmmerError::Format(
                "dsqdata sidecar payload has trailing bytes with zero sequences".into(),
            ));
        }
        return Ok(());
    }

    let mut prev_meta = -1i64;
    let mut prev_psq = -1i64;
    for (idx, rec) in records.iter().enumerate() {
        if rec.metadata_end < prev_meta || rec.psq_end <= prev_psq {
            return Err(HmmerError::Format(format!(
                "dsqdata index record {} is not monotonic",
                idx + 1
            )));
        }
        if rec.metadata_end < 0 {
            return Err(HmmerError::Format(format!(
                "dsqdata index record {} has negative metadata extent",
                idx + 1
            )));
        }
        let metadata_end = u64::try_from(rec.metadata_end).map_err(|_| {
            HmmerError::Format(format!(
                "dsqdata index record {} has invalid metadata extent",
                idx + 1
            ))
        })?;
        let psq_end = u64::try_from(rec.psq_end).map_err(|_| {
            HmmerError::Format(format!(
                "dsqdata index record {} has invalid packet extent",
                idx + 1
            ))
        })?;
        if metadata_end >= metadata_payload_len {
            return Err(HmmerError::Format(format!(
                "dsqdata index record {} metadata extent exceeds .dsqm payload",
                idx + 1
            )));
        }
        if psq_end >= sequence_packets {
            return Err(HmmerError::Format(format!(
                "dsqdata index record {} packet extent exceeds .dsqs payload",
                idx + 1
            )));
        }
        prev_meta = rec.metadata_end;
        prev_psq = rec.psq_end;
    }
    let final_metadata_len = u64::try_from(prev_meta + 1)
        .map_err(|_| HmmerError::Format("dsqdata final metadata extent is invalid".into()))?;
    let final_sequence_packets = u64::try_from(prev_psq + 1)
        .map_err(|_| HmmerError::Format("dsqdata final packet extent is invalid".into()))?;
    if final_metadata_len != metadata_payload_len {
        return Err(HmmerError::Format(
            "dsqdata metadata sidecar has trailing bytes after final index record".into(),
        ));
    }
    if final_sequence_packets != sequence_packets {
        return Err(HmmerError::Format(
            "dsqdata sequence sidecar has trailing packets after final index record".into(),
        ));
    }
    Ok(())
}
