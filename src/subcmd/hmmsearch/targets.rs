use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use tongs::alphabet::Alphabet;
use tongs::sequence::{self, Sequence, SequenceFormat};
use tongs::subcmd::inputs::open_seq_file_as;

use super::Args;

pub(super) struct Reader {
    sqf: sequence::SeqFile<Box<dyn Read>>,
    limit: Option<usize>,
    pub count: u64,
    pub residues: u64,
}

impl Reader {
    pub(super) fn open(args: &Args, abc: &Alphabet) -> Self {
        let sqf = match args.restrictdb_stkey.as_deref() {
            Some(stkey) => open_restricted_target_seq_file(
                &args.seqdb,
                abc,
                args.tformat.as_deref(),
                stkey,
                args.ssifile.as_deref(),
            )
            .unwrap_or_else(|e| {
                eprintln!("Error opening restricted sequence file: {}", e);
                std::process::exit(1);
            }),
            None => {
                open_seq_file_as(&args.seqdb, abc, args.tformat.as_deref()).unwrap_or_else(|e| {
                    eprintln!("Error opening sequence file: {}", e);
                    std::process::exit(1);
                })
            }
        };
        Reader {
            sqf,
            limit: args.restrictdb_n,
            count: 0,
            residues: 0,
        }
    }

    pub(super) fn next_block(
        &mut self,
        budget: usize,
    ) -> Result<Option<(u64, Vec<Sequence>)>, String> {
        let first = self.count;
        let mut block = Vec::new();
        let mut held = 0;
        while held < budget && self.limit.is_none_or(|limit| (self.count as usize) < limit) {
            let mut sq = Sequence::new();
            let more = self.sqf.read(&mut sq).map_err(|e| e.to_string())?;
            if !more {
                break;
            }
            held += sq.n;
            self.count += 1;
            self.residues += sq.n as u64;
            block.push(sq);
        }
        Ok((!block.is_empty()).then_some((first, block)))
    }
}

pub(crate) fn open_restricted_target_seq_file(
    path: &Path,
    abc: &Alphabet,
    tformat: Option<&str>,
    stkey: &str,
    ssifile: Option<&Path>,
) -> Result<sequence::SeqFile<Box<dyn std::io::Read>>, String> {
    if path == Path::new("-") {
        return Err("can't open an SSI index for standard input".to_string());
    }
    if path.extension().is_some_and(|ext| ext == "gz") {
        return Err("can't open an SSI index for a .gz compressed seq file".to_string());
    }
    let ssi_path = ssifile
        .map(Path::to_path_buf)
        .unwrap_or_else(|| tongs::ssi::path_with_appended_suffix(path, ".ssi"));
    let on_disk = tongs::ssi::read_ssi_path(&ssi_path)
        .map_err(|e| format!("failed to read SSI index {}: {e}", ssi_path.display()))?;
    if on_disk.indexed_path.file_name() != path.file_name() {
        return Err(format!(
            "SSI index {} was built for {}, not {}",
            ssi_path.display(),
            on_disk.indexed_path.display(),
            path.display()
        ));
    }
    let offset = on_disk.lookup(stkey).ok_or_else(|| {
        format!(
            "sequence {stkey} not found in SSI index {}",
            ssi_path.display()
        )
    })?;
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("failed to open sequence file {}: {e}", path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("failed to seek sequence file {}: {e}", path.display()))?;
    let reader: Box<dyn Read> = Box::new(file);
    let sqf = sequence::SeqFile::new(reader, abc.clone());
    Ok(if let Some(format) = tformat {
        sqf.with_format(SequenceFormat::from_name(format).unwrap())
    } else {
        sqf
    })
}
