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
