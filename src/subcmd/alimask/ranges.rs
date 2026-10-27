use super::*;

pub(super) fn parse_ranges(spec: &str) -> Result<Vec<(usize, usize)>, String> {
    let mut ranges = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        let Some((start, end)) = part.split_once("..").or_else(|| part.split_once('-')) else {
            return Err(format!(
                "Range flags take coords <from>..<to>; {part} not recognized"
            ));
        };
        let start: usize = start
            .parse()
            .map_err(|_| format!("Failed to find <from> or <to> coord in {part}"))?;
        let end: usize = end
            .parse()
            .map_err(|_| format!("Failed to find <from> or <to> coord in {part}"))?;
        if start > end {
            return Err(format!(
                "In range ({part}) <from> can not be larger than <to>"
            ));
        }
        ranges.push((start, end));
    }
    Ok(ranges)
}
pub(super) fn model_ranges_to_alignment_ranges(
    ranges: &[(usize, usize)],
    model_to_ali: &[usize],
) -> Result<Vec<(usize, usize)>, String> {
    if model_to_ali.is_empty() {
        return Err("alimask could not infer any model consensus columns".to_string());
    }

    let mut ali_ranges = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        if start == 0 {
            return Err(
                "Mask ranges can not start before position 1; start 0 is invalid".to_string(),
            );
        }
        if end > model_to_ali.len() {
            return Err(format!(
                "Maximum model mask range {} exceeds model length {}",
                end,
                model_to_ali.len()
            ));
        }
        ali_ranges.push((model_to_ali[start - 1], model_to_ali[end - 1]));
    }
    Ok(ali_ranges)
}

pub(super) fn make_model_to_alignment_map(
    alignment: &msa::Msa,
    abc: Option<&Alphabet>,
    hand: bool,
    symfrac: f64,
    fragthresh: f64,
    weighting_strategy: RelativeWeighting,
) -> Result<Vec<usize>, String> {
    if hand {
        let Some(ref rf) = alignment.rf else {
            return Err("Model file does not contain an RF line, required for --hand.".to_string());
        };
        let map: Vec<usize> = rf
            .iter()
            .enumerate()
            .filter_map(|(idx, &sym)| is_rf_consensus(sym).then_some(idx + 1))
            .collect();
        if map.is_empty() {
            return Err("alimask --hand requires at least one RF consensus column".to_string());
        }
        return Ok(map);
    }

    let abc = abc.expect("alphabet is required for fast model construction");
    let mask = builder::model_mask_from_msa(
        alignment,
        abc,
        symfrac as f32,
        fragthresh as f32,
        false,
        weighting_strategy,
    );
    let map: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(idx, &sym)| (sym == b'x').then_some(idx + 1))
        .collect();
    if map.is_empty() {
        return Err("alimask could not infer any model consensus columns".to_string());
    }
    Ok(map)
}

pub(super) fn write_model_to_alignment_map(
    ranges: &[(usize, usize)],
    model_to_ali: &[usize],
) -> Result<(), String> {
    println!("model coordinates     alignment coordinates");
    for &(start, end) in ranges {
        if start == 0 {
            return Err(
                "Mask ranges can not start before position 1; start 0 is invalid".to_string(),
            );
        }
        if end > model_to_ali.len() {
            return Err(format!(
                "Maximum mask range {} exceeds computed model length {}",
                end,
                model_to_ali.len()
            ));
        }
        println!(
            "{:8}..{:<8} -> {:8}..{:<8}",
            start,
            end,
            model_to_ali[start - 1],
            model_to_ali[end - 1]
        );
    }
    Ok(())
}

pub(super) fn write_alignment_to_model_map(
    ranges: &[(usize, usize)],
    model_to_ali: &[usize],
    alen: usize,
) -> Result<(), String> {
    println!("alignment coordinates     model coordinates");
    for &(start, end) in ranges {
        if start == 0 {
            return Err(
                "Mask ranges can not start before position 1; start 0 is invalid".to_string(),
            );
        }
        if end > alen {
            return Err(format!(
                "Maximum mask range {} exceeds alignment length {}",
                end, alen
            ));
        }
        let first = model_to_ali
            .iter()
            .position(|&apos| apos >= start && apos <= end);
        let Some(first) = first else {
            println!("   {:8}..{:<8} ->       -..-  (no map)", start, end);
            continue;
        };
        let last = model_to_ali
            .iter()
            .rposition(|&apos| apos >= start && apos <= end)
            .unwrap_or(first);
        println!(
            "   {:8}..{:<8} -> {:8}..{:<8}",
            model_to_ali[first],
            model_to_ali[last],
            first + 1,
            last + 1
        );
    }
    Ok(())
}
