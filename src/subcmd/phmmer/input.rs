use super::*;

pub(super) fn count_query_sequences(path: &std::path::Path, abc: &Alphabet) -> usize {
    let mut sqf = sequence::open_seq_file(path, abc).unwrap_or_else(|e| {
        eprintln!("Error opening query file: {}", e);
        std::process::exit(1);
    });
    let mut sq = Sequence::new();
    let mut count = 0usize;
    while sqf.read(&mut sq).unwrap_or_else(|e| {
        eprintln!("Error reading query file: {}", e);
        std::process::exit(1);
    }) {
        count += 1;
        if count > 1 {
            break;
        }
        sq.reuse();
    }
    count
}
pub(super) fn validate_sequence_format(label: &str, format: Option<&str>) {
    if let Some(format) = format {
        if SequenceFormat::from_name(format).is_none() {
            eprintln!("{label}={format} is not a recognized input sequence file format");
            std::process::exit(1);
        }
    }
}

pub(super) fn open_sequence_file(
    path: &std::path::Path,
    abc: &Alphabet,
    format: Option<&str>,
) -> tongs::errors::HmmerResult<sequence::SeqFile<Box<dyn std::io::Read>>> {
    if let Some(format) = format {
        sequence::open_seq_file_with_format(path, abc, SequenceFormat::from_name(format).unwrap())
    } else {
        sequence::open_seq_file(path, abc)
    }
}
