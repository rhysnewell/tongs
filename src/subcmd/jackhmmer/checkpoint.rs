use super::*;

/// Build a degenerate trace for the round-1 single-sequence query that simply
/// walks B -> M1..Mn -> E. Used so the query can be re-included verbatim when
/// constructing the included-hit MSA for the next round.
pub(super) fn exact_match_query_trace(model_len: usize) -> Trace {
    let mut tr = Trace::new();
    tr.append(State::B, 0, 0);
    for i in 1..=model_len {
        tr.append(State::M, i, i);
    }
    tr.append(State::E, 0, 0);
    tr.m = model_len;
    tr.l = model_len;
    tr
}

/// Write the HMM produced at `iteration` to `<prefix>-<iteration>.hmm`.
pub(super) fn write_hmm_checkpoint(prefix: &Path, iteration: usize, hmm: &tongs::hmm::Hmm) {
    let path = checkpoint_path(prefix, iteration, "hmm");
    let mut file = std::fs::File::create(&path).unwrap_or_else(|e| {
        eprintln!("Error creating HMM checkpoint {}: {}", path.display(), e);
        std::process::exit(1);
    });
    let checkpoint_hmm;
    let hmm = if iteration == 1 {
        // C's round-1 single-sequence checkpoint omits EFFN; round 2+ builder
        // checkpoints keep it.
        checkpoint_hmm = {
            let mut h = hmm.clone();
            h.eff_nseq = -1.0;
            h
        };
        &checkpoint_hmm
    } else {
        hmm
    };
    tongs::hmmfile::write_hmm(&mut file, hmm).unwrap_or_else(|e| {
        eprintln!("Error writing HMM checkpoint {}: {}", path.display(), e);
        std::process::exit(1);
    });
}

/// Write the included-hit alignment for `iteration` to `<prefix>-<iteration>.sto`.
pub(super) fn write_msa_checkpoint(prefix: &Path, iteration: usize, msa: &Msa) {
    let path = checkpoint_path(prefix, iteration, "sto");
    let mut file = std::fs::File::create(&path).unwrap_or_else(|e| {
        eprintln!(
            "Error creating alignment checkpoint {}: {}",
            path.display(),
            e
        );
        std::process::exit(1);
    });
    write_stockholm_msa(&mut file, msa);
}

/// Compose a checkpoint filename of the form `<prefix>-<iteration>.<ext>`.
fn checkpoint_path(prefix: &Path, iteration: usize, ext: &str) -> PathBuf {
    let mut os = prefix.as_os_str().to_os_string();
    os.push(format!("-{}.{}", iteration, ext));
    PathBuf::from(os)
}
