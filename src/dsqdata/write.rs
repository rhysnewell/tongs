use super::*;

/// Write a digital sequence database in Easel dsqdata format.
///
/// Creates the four files `<basename>`, `<basename>.dsqi`, `<basename>.dsqm`,
/// and `<basename>.dsqs`. Faithful port of `esl_dsqdata_Write()`. The
/// `alphabet` must be nucleic (DNA/RNA) or amino; nucleic uses 2-bit packing,
/// amino uses 5-bit packing, matching C.
///
/// Each sequence's `taxid` is written from [`Sequence::taxid`] (`-1` = none).
pub fn write_dsqdata(
    basename: &Path,
    sequences: &[Sequence],
    alphabet: &Alphabet,
) -> HmmerResult<()> {
    let alphatype = alphabet.abc_type;
    let do_pack5 = match alphatype {
        AlphabetType::Amino => true,
        AlphabetType::Dna | AlphabetType::Rna => false,
        AlphabetType::Unknown => {
            return Err(HmmerError::Format(
                "dsqdata: alphabet must be protein or nucleic".into(),
            ))
        }
    };

    // First pass: statistics.
    let mut max_namelen = 0u32;
    let mut max_acclen = 0u32;
    let mut max_desclen = 0u32;
    let mut max_seqlen = 0u64;
    let mut nres = 0u64;
    for sq in sequences {
        validate_dsq(sq)?;
        nres += sq.n as u64;
        max_seqlen = max_seqlen.max(sq.n as u64);
        max_namelen = max_namelen.max(sq.name.len() as u32);
        max_acclen = max_acclen.max(sq.acc.len() as u32);
        max_desclen = max_desclen.max(sq.desc.len() as u32);
    }
    let nseq = sequences.len() as u64;

    // A deterministic uniquetag. C uses a random uint32 from its RNG; any value
    // works as long as the four files agree, which they do here.
    let uniquetag: u32 = derive_uniquetag(nseq, nres, max_seqlen);

    let p = basename;
    let mut ifp = BufWriter::new(create(&with_ext(p, "dsqi"))?);
    let mut mfp = BufWriter::new(create(&with_ext(p, "dsqm"))?);
    let mut sfp = BufWriter::new(create(&with_ext(p, "dsqs"))?);
    let mut stubfp = BufWriter::new(create(p)?);

    // Index header: 7 u32 then 3 u64.
    write_u32(&mut ifp, MAGIC_V1)?;
    write_u32(&mut ifp, uniquetag)?;
    write_u32(&mut ifp, alphatype as u32)?;
    write_u32(&mut ifp, 0)?; // flags
    write_u32(&mut ifp, max_namelen)?;
    write_u32(&mut ifp, max_acclen)?;
    write_u32(&mut ifp, max_desclen)?;
    write_u64(&mut ifp, max_seqlen)?;
    write_u64(&mut ifp, nseq)?;
    write_u64(&mut ifp, nres)?;

    // Metadata + sequence headers: magic, uniquetag.
    write_u32(&mut mfp, MAGIC_V1)?;
    write_u32(&mut mfp, uniquetag)?;
    write_u32(&mut sfp, MAGIC_V1)?;
    write_u32(&mut sfp, uniquetag)?;

    // Second pass: per-sequence index/metadata/packed records.
    let mut spos: i64 = 0; // running count of packets written to .dsqs
    let mut mpos: i64 = 0; // running count of bytes written to .dsqm
    for sq in sequences {
        let psq = if do_pack5 {
            pack5(&sq.dsq, sq.n)
        } else {
            pack2(&sq.dsq, sq.n)
        };
        for &v in &psq {
            write_u32(&mut sfp, v)?;
        }
        spos += psq.len() as i64;

        // Metadata: name\0 acc\0 desc\0 taxid(int32; -1 = none)
        mpos += write_cstr(&mut mfp, sq.name.as_bytes())? as i64;
        mpos += write_cstr(&mut mfp, sq.acc.as_bytes())? as i64;
        mpos += write_cstr(&mut mfp, sq.desc.as_bytes())? as i64;
        write_i32(&mut mfp, sq.taxid)?;
        mpos += 4;

        // Index record: inclusive END offsets (hence -1).
        let idx = IndexRecord {
            metadata_end: mpos - 1,
            psq_end: spos - 1,
        };
        write_i64(&mut ifp, idx.metadata_end)?;
        write_i64(&mut ifp, idx.psq_end)?;
    }

    // Stub file (human-readable; first line machine-parsed).
    writeln!(stubfp, "Easel dsqdata v1 x{}", uniquetag).map_err(HmmerError::Io)?;
    writeln!(stubfp).map_err(HmmerError::Io)?;
    writeln!(stubfp, "Original file:   (rust hmmer-rs)").map_err(HmmerError::Io)?;
    writeln!(stubfp, "Original format: unknown").map_err(HmmerError::Io)?;
    writeln!(stubfp, "Type:            {}", decode_type(alphatype)).map_err(HmmerError::Io)?;
    writeln!(stubfp, "Sequences:       {}", nseq).map_err(HmmerError::Io)?;
    writeln!(stubfp, "Residues:        {}", nres).map_err(HmmerError::Io)?;

    ifp.flush().map_err(HmmerError::Io)?;
    mfp.flush().map_err(HmmerError::Io)?;
    sfp.flush().map_err(HmmerError::Io)?;
    stubfp.flush().map_err(HmmerError::Io)?;
    Ok(())
}
