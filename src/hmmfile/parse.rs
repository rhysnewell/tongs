use super::*;

pub(super) fn parse_hmmer3_ascii_magic(header: &str) -> HmmerResult<String> {
    let tag = header.split_whitespace().next().unwrap_or("");
    let version = match tag {
        "HMMER3/a" => "3a",
        "HMMER3/b" => "3b",
        "HMMER3/c" => "3c",
        "HMMER3/d" => "3d",
        "HMMER3/e" => "3e",
        "HMMER3/f" => "3f",
        _ => {
            return Err(HmmerError::Format(format!(
                "Unsupported HMMER3 ASCII magic tag: {tag}"
            )))
        }
    };
    Ok(version.to_string())
}

pub(super) fn parse_hmm_yes_no(key: &str, value: &str) -> HmmerResult<bool> {
    if value.eq_ignore_ascii_case("yes") {
        Ok(true)
    } else if value.eq_ignore_ascii_case("no") {
        Ok(false)
    } else {
        Err(HmmerError::Format(format!(
            "Bad {key} value in HMM header: expected yes or no, got {value}"
        )))
    }
}

/// Parse one HMM value: a float, or `"*"` meaning zero probability (stored as `+inf` in -ln space).
pub(super) fn parse_hmm_value(s: &str) -> HmmerResult<f32> {
    if s == "*" {
        Ok(f32::INFINITY)
    } else {
        s.parse()
            .map_err(|_| HmmerError::Format(format!("Bad HMM probability value: {s}")))
    }
}

pub(super) fn parse_cutoff_value(key: &str, s: &str) -> HmmerResult<f32> {
    s.trim_end_matches(';')
        .parse()
        .map_err(|_| HmmerError::Format(format!("Bad {key} cutoff value: {s}")))
}

pub(super) fn validate_alphabet_header(line: &str, abc_type: AlphabetType) -> HmmerResult<()> {
    if abc_type == AlphabetType::Unknown {
        return Err(HmmerError::Format(
            "HMM alphabet header appeared before ALPH".to_string(),
        ));
    }
    let abc = Alphabet::new(abc_type);
    let fields: Vec<&str> = line.split_whitespace().skip(1).collect();
    if fields.len() != abc.k {
        return Err(HmmerError::Format(format!(
            "HMM alphabet header has {} symbols, expected {}",
            fields.len(),
            abc.k
        )));
    }
    for (idx, (got, expected)) in fields.iter().zip(abc.sym.iter()).enumerate() {
        if got.len() != 1 || got.as_bytes()[0] != *expected {
            return Err(HmmerError::Format(format!(
                "HMM alphabet header symbol {} is {}, expected {}",
                idx + 1,
                got,
                *expected as char
            )));
        }
    }
    Ok(())
}

pub(super) fn annotation_byte(label: &str, s: &str) -> HmmerResult<u8> {
    s.as_bytes()
        .first()
        .copied()
        .ok_or_else(|| HmmerError::Format(format!("Empty {label} annotation")))
}

/// Parse an emission row of `K` whitespace-separated `-ln(p)` values into `values`.
pub(super) fn parse_emission_line(line: &str, k: usize, values: &mut [f32]) -> HmmerResult<()> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < k {
        return Err(HmmerError::Format(format!(
            "Emission line has too few fields: expected {k}, got {}",
            parts.len()
        )));
    }
    for i in 0..k {
        values[i] = parse_hmm_value(parts[i])?;
    }
    Ok(())
}

/// Parse a transition row of `NTRANSITIONS` whitespace-separated `-ln(p)` values.
pub(super) fn parse_transition_line(
    line: &str,
    values: &mut [f32; NTRANSITIONS],
) -> HmmerResult<()> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < NTRANSITIONS {
        return Err(HmmerError::Format(format!(
            "Transition line has too few fields: expected {NTRANSITIONS}, got {}",
            parts.len()
        )));
    }
    for i in 0..NTRANSITIONS {
        values[i] = parse_hmm_value(parts[i])?;
    }
    Ok(())
}
