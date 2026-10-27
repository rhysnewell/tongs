//! hmmalign — align sequences to a profile HMM.

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;

use clap::{ArgAction, Parser};

use tongs::alphabet::Alphabet;
use tongs::bg::Bg;
use tongs::dp::gmx::Gmx;
use tongs::hmmfile;
use tongs::logsum;
use tongs::msa;
use tongs::profile::{self, Profile, P7_UNILOCAL};
use tongs::sequence::{self, Sequence, SequenceFormat};
use tongs::trace::State;
use tongs::{
    simd::oprofile::{OProfile, P7O_C, P7O_J, P7O_LOOP, P7O_N},
    simd::probmx::{p_decoding_to_gmx, ProbMx},
};

mod output;
pub(crate) use output::*;
mod align;
use align::*;
mod text_msa;
pub(crate) use text_msa::*;

#[derive(Parser)]
#[command(name = "hmmalign", about = "Align sequences to a profile HMM")]
struct Args {
    /// Output alignment to file instead of stdout
    #[arg(short = 'o')]
    output: Option<PathBuf>,

    /// Trim terminal tails of nonaligned residues from the alignment
    #[arg(long = "trim")]
    trim: bool,

    /// Include alignment in file <f> that the HMM was originally built from
    #[arg(long = "mapali")]
    mapali: Option<PathBuf>,

    /// HMM file
    hmmfile: PathBuf,
    /// Sequence file (FASTA format)
    seqfile: PathBuf,

    /// Output alignment format
    #[arg(long = "outformat", default_value = "Stockholm")]
    outformat: String,

    /// Assert input sequence file format
    #[arg(long = "informat")]
    informat: Option<String>,

    /// Assert protein alphabet
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["dna", "rna"])]
    amino: bool,

    /// Assert DNA alphabet
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["amino", "rna"])]
    dna: bool,

    /// Assert RNA alphabet
    #[arg(long, action = ArgAction::SetTrue, conflicts_with_all = ["amino", "dna"])]
    rna: bool,
}

struct AlignmentRow {
    name: String,
    acc: Option<String>,
    desc: Option<String>,
    aseq: String,
    ppline: Option<String>,
}

pub(crate) struct TextMsa {
    rows: Vec<AlignmentRow>,
    rfline: String,
    pp_cons: String,
}

/// Entry point for `hmmalign`: align sequences to a profile HMM.
///
/// Reads an HMM and a sequence file, runs Forward/Backward + posterior decoding +
/// optimal-accuracy traceback per sequence, then assembles a multiple alignment
/// (Stockholm or A2M). Optionally merges a `--mapali` source alignment via the
/// HMM's map and checksum. Corresponds to `main()` in hmmer/src/hmmalign.c.
pub fn run(args: Vec<String>) -> std::process::ExitCode {
    let args = Args::parse_from(&args);
    if args.hmmfile == std::path::Path::new("-") && args.seqfile == std::path::Path::new("-") {
        eprintln!(
            "ERROR: Either <hmmfile> or <seqfile> may be '-' (to read from stdin), but not both."
        );
        std::process::exit(1);
    }
    // --amino/--dna/--rna mutual exclusion (C: ALPHOPTS toggle group) is
    // enforced at parse time by clap.
    let informat = args.informat.as_ref().map(|informat| {
        SequenceFormat::from_name(informat).unwrap_or_else(|| {
            eprintln!("{informat} is not a recognized input sequence file format");
            std::process::exit(1);
        })
    });

    enum OutFormat {
        Stockholm,
        Pfam,
        A2m,
        Psiblast,
        Afa,
        Clustal,
        ClustalLike,
        Selex,
        Phylip,
        Phylips,
    }
    // Mirrors the format names accepted by Easel's esl_msafile_EncodeFormat()
    // that hmmalign passes through esl_msafile_Write(). "pfam" is the Stockholm
    // single-block variant: Easel writes Stockholm in 200-column blocks but Pfam
    // in a single block spanning the whole alignment.
    let outformat = if args.outformat.eq_ignore_ascii_case("stockholm") {
        OutFormat::Stockholm
    } else if args.outformat.eq_ignore_ascii_case("pfam") {
        OutFormat::Pfam
    } else if args.outformat.eq_ignore_ascii_case("a2m") {
        OutFormat::A2m
    } else if args.outformat.eq_ignore_ascii_case("psiblast") {
        OutFormat::Psiblast
    } else if args.outformat.eq_ignore_ascii_case("afa") {
        OutFormat::Afa
    } else if args.outformat.eq_ignore_ascii_case("clustal") {
        OutFormat::Clustal
    } else if args.outformat.eq_ignore_ascii_case("clustallike") {
        OutFormat::ClustalLike
    } else if args.outformat.eq_ignore_ascii_case("selex") {
        OutFormat::Selex
    } else if args.outformat.eq_ignore_ascii_case("phylip") {
        OutFormat::Phylip
    } else if args.outformat.eq_ignore_ascii_case("phylips") {
        OutFormat::Phylips
    } else {
        eprintln!(
            "{} is not a recognized output MSA file format",
            args.outformat
        );
        return std::process::ExitCode::from(2);
    };

    logsum::p7_flogsuminit();

    let hmms = read_hmms_maybe_stdin(&args.hmmfile).unwrap_or_else(|e| {
        eprintln!("Error reading HMM file: {}", e);
        std::process::exit(1);
    });
    if hmms.is_empty() {
        eprintln!("Error: no HMMs found in {}", args.hmmfile.display());
        std::process::exit(1);
    }
    if hmms.len() != 1 {
        eprintln!(
            "Error: HMM file {} does not contain just one HMM",
            args.hmmfile.display()
        );
        std::process::exit(1);
    }

    let hmm = &hmms[0];
    let abc = if args.amino {
        Alphabet::amino()
    } else if args.dna {
        Alphabet::dna()
    } else if args.rna {
        Alphabet::rna()
    } else {
        Alphabet::new(hmm.abc_type)
    };
    let bg = Bg::new(&abc);

    let mut sequences = Vec::new();
    let mut sqf = open_seq_file_maybe_stdin(&args.seqfile, &abc, informat).unwrap_or_else(|e| {
        eprintln!("Error opening sequence file: {}", e);
        std::process::exit(1);
    });
    let mut sq = Sequence::new();
    while sqf.read(&mut sq).unwrap_or_else(|e| {
        eprintln!("Error reading sequence file: {}", e);
        std::process::exit(1);
    }) {
        sequences.push(sq.clone());
        sq.reuse();
    }

    let msa = if let Some(mapali_path) = &args.mapali {
        let mapped_msas = msa::read_stockholm(mapali_path).unwrap_or_else(|e| {
            eprintln!(
                "Error reading mapped alignment {}: {}",
                mapali_path.display(),
                e
            );
            std::process::exit(1);
        });
        let mapped = mapped_msas.first().unwrap_or_else(|| {
            eprintln!(
                "Mapped alignment {} contained no alignment blocks",
                mapali_path.display()
            );
            std::process::exit(1);
        });
        build_text_msa_with_mapali(hmm, &abc, &bg, &sequences, mapped, args.trim)
    } else {
        build_text_msa(hmm, &abc, &bg, &sequences, args.trim)
    };

    let mut output_file = args.output.as_ref().map(|path| {
        File::create(path).unwrap_or_else(|e| {
            eprintln!("Error opening output file {}: {}", path.display(), e);
            std::process::exit(1);
        })
    });
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();
    let out: &mut dyn Write = match output_file.as_mut() {
        Some(file) => file,
        None => &mut stdout_lock,
    };
    match outformat {
        OutFormat::Stockholm => write_stockholm(out, &msa),
        OutFormat::Pfam => write_stockholm_blocked(out, &msa, msa.rfline.len().max(1)),
        OutFormat::A2m => write_a2m(
            out,
            &msa,
            hmm.abc_type == tongs::alphabet::AlphabetType::Amino,
        ),
        OutFormat::Psiblast => {
            msa::write_psiblast(out, &write_rows(&msa), Some(&msa.rfline)).unwrap();
        }
        OutFormat::Afa => {
            msa::write_afa(out, &write_rows(&msa)).unwrap();
        }
        OutFormat::Clustal => {
            msa::write_clustal(out, &write_rows(&msa), false, EASEL_VERSION).unwrap();
        }
        OutFormat::ClustalLike => {
            msa::write_clustal(out, &write_rows(&msa), true, EASEL_VERSION).unwrap();
        }
        OutFormat::Selex => {
            msa::write_selex(out, &write_rows(&msa), Some(&msa.rfline)).unwrap();
        }
        OutFormat::Phylip => {
            msa::write_phylip(out, &write_rows(&msa), false).unwrap();
        }
        OutFormat::Phylips => {
            msa::write_phylip(out, &write_rows(&msa), true).unwrap();
        }
    }
    std::process::ExitCode::SUCCESS
}

fn read_hmms_maybe_stdin(path: &std::path::Path) -> tongs::errors::HmmerResult<Vec<tongs::Hmm>> {
    if path == std::path::Path::new("-") {
        let stdin = std::io::stdin();
        hmmfile::read_hmms_auto(BufReader::new(stdin.lock()))
    } else {
        hmmfile::read_hmm_file_auto(path)
    }
}

fn open_seq_file_maybe_stdin(
    path: &std::path::Path,
    abc: &Alphabet,
    format: Option<SequenceFormat>,
) -> tongs::errors::HmmerResult<sequence::SeqFile<Box<dyn Read>>> {
    if path == std::path::Path::new("-") {
        let sqf = sequence::SeqFile::new(Box::new(std::io::stdin()) as Box<dyn Read>, abc.clone());
        Ok(if let Some(format) = format {
            sqf.with_format(format)
        } else {
            sqf
        })
    } else if let Some(format) = format {
        sequence::open_seq_file_with_format(path, abc, format)
    } else {
        sequence::open_seq_file(path, abc)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/subcmd/hmmalign_test.rs"]
mod tests;
