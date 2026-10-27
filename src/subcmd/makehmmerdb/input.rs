use super::*;

/// Read a (possibly gzip-compressed) sequence file fully into memory, mirroring
/// the transparent `.gz` handling in `sequence::open_seq_file`.
pub(super) fn read_seqfile_bytes(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    if path.extension().is_some_and(|e| e == "gz") {
        flate2::read::MultiGzDecoder::new(file).read_to_end(&mut buf)?;
    } else {
        std::io::BufReader::new(file).read_to_end(&mut buf)?;
    }
    Ok(buf)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GuessedAlphabet {
    Dna,
    Rna,
    Amino,
}

/// Guess the alphabet from raw FASTA input, faithfully porting the C path
/// `esl_sqfile_GuessAlphabet` -> `sqascii_GuessAlphabet` -> `esl_sq_GuessAlphabet`
/// -> `esl_abc_GuessAlphabet`.
///
/// C inspects only the FIRST sequence, reading at most a 4000-residue leading
/// window (`sqascii_ReadWindow(sqfp, 0, 4000, sq)`), then counts up to 10000 of
/// its residues into a 26-letter histogram (`esl_sq_GuessAlphabet`) and applies
/// the `esl_abc_GuessAlphabet` heuristic. Returns `None` when the alphabet can't
/// be reliably guessed (C's `eslENOALPHABET`).
pub(super) fn guess_alphabet(raw_input: &[u8]) -> Option<GuessedAlphabet> {
    let counts = first_sequence_residue_counts(raw_input);
    esl_abc_guess_alphabet(&counts)
}

/// Histogram of A..Z over the first sequence's leading 4000-residue window,
/// counting at most 10000 residues. Mirrors `esl_sq_GuessAlphabet` operating on
/// the window produced by `sqascii_ReadWindow(.., 0, 4000, ..)`.
fn first_sequence_residue_counts(raw_input: &[u8]) -> [i64; 26] {
    let mut counts = [0i64; 26];
    let mut in_first_seq = false;
    let mut header_done = false;
    let mut window_residues = 0i64; // residues admitted to the 4000-base window
    let mut counted = 0i64; // residues actually tallied (cap 10000)

    for line in raw_input.split(|&b| b == b'\n') {
        let trimmed = trim_ascii(line);
        if trimmed.first() == Some(&b'>') {
            if in_first_seq {
                break; // reached the 2nd sequence; first one is complete
            }
            in_first_seq = true;
            header_done = true;
            continue;
        }
        if !header_done {
            // skip leading blank lines / comments before the first record
            if trimmed.is_empty() {
                continue;
            }
            // Non-FASTA leading content: bail out (C would treat differently;
            // makehmmerdb only supports FASTA here).
            return counts;
        }
        if !in_first_seq {
            continue;
        }
        for &ch in trimmed {
            if ch.is_ascii_whitespace() {
                continue;
            }
            // ReadWindow caps the recorded window at 4000 residues.
            if window_residues >= 4000 {
                return counts;
            }
            window_residues += 1;
            let upper = ch.to_ascii_uppercase();
            if upper.is_ascii_uppercase() {
                counts[(upper - b'A') as usize] += 1;
                counted += 1;
                if counted > 10000 {
                    return counts;
                }
            }
        }
    }
    counts
}

fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let start = bytes.iter().position(|b| !b.is_ascii_whitespace());
    let start = match start {
        Some(s) => s,
        None => return &[],
    };
    let end = bytes
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .unwrap();
    &bytes[start..=end]
}

/// Faithful port of Easel `esl_abc_GuessAlphabet` (esl_alphabet.c:1251). `ct` is
/// a 26-entry A..Z residue histogram.
fn esl_abc_guess_alphabet(ct: &[i64; 26]) -> Option<GuessedAlphabet> {
    let idx = |c: u8| (c - b'A') as usize;
    let aaonly = b"EFIJLOPQZ";
    let allcanon = b"ACG";
    let aacanon = b"DHKMRSVWY";

    let n: i64 = ct.iter().sum();

    let (mut n1, mut x1) = (0i64, 0i32);
    for &c in aaonly {
        let x = ct[idx(c)];
        if x > 0 {
            n1 += x;
            x1 += 1;
        }
    }
    let (mut n2, mut x2) = (0i64, 0i32);
    for &c in allcanon {
        let x = ct[idx(c)];
        if x > 0 {
            n2 += x;
            x2 += 1;
        }
    }
    let (mut n3, mut x3) = (0i64, 0i32);
    for &c in aacanon {
        let x = ct[idx(c)];
        if x > 0 {
            n3 += x;
            x3 += 1;
        }
    }
    let nt = ct[idx(b'T')];
    let xt = if nt != 0 { 1 } else { 0 };
    let nu = ct[idx(b'U')];
    let xu = if nu != 0 { 1 } else { 0 };
    let nx = ct[idx(b'X')];
    let nn = ct[idx(b'N')];
    let xn = if nn != 0 { 1 } else { 0 };

    // Avoid unused-assignment lint mirrors (x1..xn participate in the tests below).
    let _ = (x1, x2, x3, xt, xu, xn);

    if n <= 10 {
        None
    } else if n > 2000 && nn == n {
        Some(GuessedAlphabet::Dna)
    } else if n1 > 0 {
        Some(GuessedAlphabet::Amino)
    } else if (n - (n2 + nt + nn)) as f64 <= 0.02 * n as f64 && x2 + xt == 4 {
        Some(GuessedAlphabet::Dna)
    } else if (n - (n2 + nu + nn)) as f64 <= 0.02 * n as f64 && x2 + xu == 4 {
        Some(GuessedAlphabet::Rna)
    } else if (n - (n1 + n2 + n3 + nn + nt + nx)) as f64 <= 0.02 * n as f64
        && n3 > n2
        && x1 + x2 + x3 + xn + xt >= 15
    {
        Some(GuessedAlphabet::Amino)
    } else {
        None
    }
}
