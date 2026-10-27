use crate::alphabet::{Alphabet, AlphabetType};
use crate::bg::Bg;
use crate::hmm::Hmm;
use crate::logsum::p7_flogsuminit;
use crate::sequence::Sequence;
use crate::subcmd::hmmalign::{build_text_msa, write_a2m};
use crate::util::simd_env::FlushToZero;

// hmmalign's A2M, every consensus column kept, under the float mode its command runs in. The
// caller's thread gets its own mode back.
pub fn a2m(hmm: &Hmm, sequences: &[Sequence]) -> String {
    let _mode = FlushToZero::enter();
    p7_flogsuminit();
    let abc = Alphabet::new(hmm.abc_type);
    let msa = build_text_msa(hmm, &abc, &Bg::new(&abc), sequences, false);
    let mut out = Vec::new();
    write_a2m(&mut out, &msa, hmm.abc_type == AlphabetType::Amino);
    String::from_utf8(out).expect("A2M rows are ASCII")
}
