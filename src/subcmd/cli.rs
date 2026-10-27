pub(crate) fn command_line_has_option(cmdline: &str, option: &str) -> bool {
    let compact_short = option
        .strip_prefix('-')
        .filter(|rest| !rest.starts_with('-') && rest.chars().count() == 1)
        .map(|_| option);
    cmdline.split_whitespace().any(|token| {
        token == option
            || token.starts_with(&format!("{option}="))
            || compact_short
                .is_some_and(|short| token.starts_with(short) && token.len() > short.len())
    })
}
pub(crate) fn parse_positive_f64(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid positive number: {e}"))?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err("value must be > 0".to_string())
    }
}
pub(crate) fn parse_positive_usize(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid positive integer: {e}"))?;
    if value > 0 {
        Ok(value)
    } else {
        Err("value must be > 0".to_string())
    }
}
pub(crate) fn parse_open_unit_f64(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid probability: {e}"))?;
    if value > 0.0 && value < 1.0 {
        Ok(value)
    } else {
        Err("value must be > 0 and < 1".to_string())
    }
}
pub(crate) fn parse_unit_f64(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid probability: {e}"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("value must be >= 0 and <= 1".to_string())
    }
}
pub(crate) fn parse_textw(s: &str) -> Result<usize, String> {
    let value = s
        .parse::<usize>()
        .map_err(|e| format!("invalid text width: {e}"))?;
    if value < 120 {
        Err("--textw must be >= 120".to_string())
    } else {
        Ok(value)
    }
}
pub(crate) fn parse_popen(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid gap open probability: {e}"))?;
    if (0.0..0.5).contains(&value) {
        Ok(value)
    } else {
        Err("--popen must be >= 0 and < 0.5".to_string())
    }
}
pub(crate) fn parse_pextend(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid gap extend probability: {e}"))?;
    if (0.0..1.0).contains(&value) {
        Ok(value)
    } else {
        Err("--pextend must be >= 0 and < 1".to_string())
    }
}
pub(crate) fn parse_window_length(s: &str) -> Result<i32, String> {
    let value = s
        .parse::<i32>()
        .map_err(|e| format!("invalid window length: {e}"))?;
    if value > 0 {
        Ok(value)
    } else {
        Err("--w_length must be > 0 and fit in a 32-bit signed integer".to_string())
    }
}
pub(crate) fn parse_window_beta(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|e| format!("invalid window-length beta value: {e}"))?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err("Invalid window-length beta value".to_string())
    }
}
// The single binary records its own name and then the subcommand as typed, so tables
// drop the wrapper and print the canonical program name as C does.
pub(crate) fn normalize_table_cmdline(cmdline: &str, names: &[&str]) -> String {
    let mut tokens = cmdline.split_whitespace();
    match (tokens.next(), tokens.next()) {
        (Some(_wrapper), Some(name)) if names.contains(&name) => {
            let rest = tokens.collect::<Vec<_>>();
            if rest.is_empty() {
                names[0].to_string()
            } else {
                format!("{} {}", names[0], rest.join(" "))
            }
        }
        _ => cmdline.to_string(),
    }
}
