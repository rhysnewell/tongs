use super::*;

/// `p7_builder_Create()` default seed (p7_builder.c:77).
pub const DEFAULT_BUILDER_SEED: u32 = 42;

/// Gap-open/gap-extend defaults for single-sequence queries.
///
/// C keeps these in each driver's `ESL_OPTIONS` rather than in `P7_BUILDER`
/// (which initializes `popen`/`pextend` to -1, p7_builder.c:150-151):
/// phmmer.c:71-72 and hmmbuild.c:580-581 use the amino pair; nhmmer.c:96-97
/// and hmmbuild.c:577-578 use the nucleotide pair.
pub const POPEN_AMINO: f64 = 0.02;
pub const PEXTEND_AMINO: f64 = 0.4;
pub const POPEN_NUCLEIC: f64 = 0.03125;
pub const PEXTEND_NUCLEIC: f64 = 0.75;

/// Port of the single-sequence slice of `P7_BUILDER` (hmmer.h:1290-1326).
///
/// Only the fields `p7_SingleBuilder()` reads are carried. The MSA-building
/// fields (`arch_strategy`, `wgt_strategy`, `effn_strategy`, `prior`, ...) stay
/// as the positional parameters of [`build_hmm_from_msa`] and friends; folding
/// those in is a separate change.
///
/// Deviation from C: `P7_BUILDER` holds `abc`, a borrowed `ESL_ALPHABET *`.
/// Carrying that as a Rust lifetime would infect every caller, so the alphabet
/// is passed to the methods that need it and only `abc_type` is stored.
#[derive(Debug, Clone)]
pub struct Builder {
    /// `bld->abc->type`.
    pub abc_type: AlphabetType,
    /// `bld->S`, the score matrix; `None` until a score system is set.
    pub s: Option<ScoreMatrix>,
    /// `bld->Q`, conditional residue probabilities P(b|a), `Kp x Kp`.
    pub q: Option<Vec<Vec<f64>>>,
    /// `bld->popen`, gap-open probability. -1 until set, as in C.
    pub popen: f64,
    /// `bld->pextend`, gap-extend probability. -1 until set, as in C.
    pub pextend: f64,
    /// `bld->w_len`, an explicitly set window length, or -1.
    pub w_len: i32,
    /// `bld->w_beta`, tail mass for deriving the window length.
    pub w_beta: f64,
    /// `bld->r`'s seed. C reseeds to this before calibrating each model when
    /// `do_reseeding` is set (p7_builder.c:127-128).
    pub seed: u32,
    /// `bld->do_reseeding`; FALSE only when `seed == 0`, which C treats as
    /// "choose arbitrarily and allow run-to-run variation".
    pub do_reseeding: bool,
    /// `bld->EmL` .. `bld->Eft`, the calibration sampling config.
    pub calibration: CalibrationConfig,
    /// `bld->errbuf`.
    pub errbuf: String,
}

impl Builder {
    /// Port of `p7_builder_Create()` (p7_builder.c:52) for the single-sequence
    /// fields, with `go == NULL` (i.e. all defaults).
    ///
    /// Note C does **not** initialize `w_len`/`w_beta` here — every caller
    /// assigns them right after (nhmmer.c:882-883, hmmbuild.c:604-605). This
    /// port initializes them to the values those callers use by default rather
    /// than leaving them indeterminate.
    pub fn new(abc_type: AlphabetType) -> Self {
        Builder {
            abc_type,
            s: None,
            q: None,
            popen: -1.0,
            pextend: -1.0,
            w_len: -1,
            w_beta: DEFAULT_WINDOW_BETA,
            seed: DEFAULT_BUILDER_SEED,
            do_reseeding: DEFAULT_BUILDER_SEED != 0,
            calibration: CalibrationConfig::default(),
            errbuf: String::new(),
        }
    }

    /// `p7_builder_Create()` sets the seed from `--seed` (p7_builder.c:92);
    /// `seed == 0` means "arbitrary seed, no reseeding" (p7_builder.c:123-128).
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self.do_reseeding = seed != 0;
        self
    }

    /// `bld->EmL` .. `bld->Eft` (p7_builder.c:113-119).
    pub fn with_calibration(mut self, calibration: CalibrationConfig) -> Self {
        self.calibration = calibration;
        self
    }

    /// `bld->w_len` / `bld->w_beta`, as nhmmer.c:882-883 and hmmbuild.c:604-605
    /// assign them.
    pub fn with_window(mut self, w_len: Option<i32>, w_beta: Option<f64>) -> Self {
        self.w_len = w_len.unwrap_or(-1);
        self.w_beta = w_beta.unwrap_or(DEFAULT_WINDOW_BETA);
        self
    }

    /// Port of `p7_builder_LoadScoreSystem()` (p7_builder.c:196).
    ///
    /// Initializes the builder to parameterize single-sequence queries using
    /// the built-in score matrix named `matrix`, converting its scores to
    /// conditional probabilities against the background frequencies in `bg`.
    pub fn load_score_system(
        &mut self,
        matrix: &str,
        popen: f64,
        pextend: f64,
        bg: &Bg,
        abc: &Alphabet,
    ) -> Result<(), String> {
        self.errbuf.clear();

        let mut s = ScoreMatrix::create(abc);
        s.set(matrix, abc).map_err(|e| match e {
            ScoreMatrixError::NotFound(_) => {
                format!("no matrix named {matrix} is available as a built-in")
            }
            ScoreMatrixError::Invalid(_) => {
                format!("failed to set score matrix {matrix} as a built-in")
            }
        })?;

        self.finish_score_system(s, popen, pextend, bg, abc, matrix)
    }

    /// Port of `p7_builder_SetScoreSystem()` (p7_builder.c:283).
    ///
    /// Like [`Self::load_score_system`], but reads the matrix from `mxfile`. A
    /// `None` `mxfile` means "use the alphabet's default built-in": BLOSUM62
    /// for amino, DNA1 otherwise (p7_builder.c:296-301).
    pub fn set_score_system(
        &mut self,
        mxfile: Option<&std::path::Path>,
        popen: f64,
        pextend: f64,
        bg: &Bg,
        abc: &Alphabet,
    ) -> Result<(), String> {
        self.errbuf.clear();

        let (s, label) = match mxfile {
            None => {
                let name = ScoreMatrix::default_builtin_name(abc.abc_type);
                let mut s = ScoreMatrix::create(abc);
                s.set(name, abc)
                    .map_err(|e| format!("failed to set score matrix {name}: {e}"))?;
                (s, name.to_string())
            }
            Some(path) => {
                let s = ScoreMatrix::from_file_for_alphabet(path, abc).map_err(|e| match e {
                    ScoreMatrixError::Invalid(m) if m.starts_with("Failed to find or open") => m,
                    other => format!("Failed to read matrix from {}:\n{other}", path.display()),
                })?;
                (s, path.display().to_string())
            }
        };

        self.finish_score_system(s, popen, pextend, bg, abc, &label)
    }

    /// The tail shared by `p7_builder_LoadScoreSystem()` (p7_builder.c:219-232)
    /// and `p7_builder_SetScoreSystem()` (p7_builder.c:314-327): probify the
    /// matrix against `bg`, convert the joint probabilities to conditionals,
    /// and record the gap probabilities.
    /// Install an already-constructed score matrix. This is the shared tail of
    /// `p7_builder_LoadScoreSystem()` and `p7_builder_SetScoreSystem()`,
    /// exposed for callers that obtained an `ESL_SCOREMATRIX` some other way.
    pub fn set_score_system_from_matrix(
        &mut self,
        s: ScoreMatrix,
        popen: f64,
        pextend: f64,
        bg: &Bg,
        abc: &Alphabet,
    ) -> Result<(), String> {
        let label = s.name().to_string();
        self.finish_score_system(s, popen, pextend, bg, abc, &label)
    }

    fn finish_score_system(
        &mut self,
        s: ScoreMatrix,
        popen: f64,
        pextend: f64,
        bg: &Bg,
        abc: &Alphabet,
        label: &str,
    ) -> Result<(), String> {
        // "A wasteful conversion of the HMMER single-precision background
        // probs to Easel double-prec" (p7_builder.c:221-222).
        let f: Vec<f64> = bg.f[..abc.k].iter().map(|&x| x as f64).collect();

        let (_lambda, mut q) = s.probify_given_bg(&f, &f, abc).map_err(|e| match e {
            ScoreMatrixError::Invalid(m) if m.contains("bracket") => {
                format!("input score matrix {label} has no valid solution for lambda: {m}")
            }
            ScoreMatrixError::Invalid(m) if m.contains("converge") => {
                format!("failed to solve input score matrix {label} for lambda: are you sure it's valid?")
            }
            other => format!(
                "unexpected error in solving input score matrix {label} for probability parameters: {other}"
            ),
        })?;

        crate::util::scorematrix::joint_to_conditional_on_query(abc, &mut q);

        self.s = Some(s);
        self.q = Some(q);
        self.popen = popen;
        self.pextend = pextend;
        Ok(())
    }

    /// Port of `p7_SingleBuilder()` (p7_builder.c:496).
    ///
    /// Builds a profile HMM from the single sequence `sq`. The score system
    /// must have been set first by [`Self::load_score_system`] or
    /// [`Self::set_score_system`].
    ///
    /// Deviations from C, both deliberate:
    ///  - C's `opt_tr` faux trace (p7_builder.c:520-530) is not produced. No
    ///    upstream caller ever requests it — phmmer.c:557, nhmmer.c:905/:931
    ///    and hmmbuild.c:963/:1040/:1232 all pass `NULL`.
    ///  - C's `opt_gm`/`opt_om` are by-products of its `calibrate()`
    ///    (p7_builder.c:1010). This port's callers build the profile and
    ///    optimized profile themselves from the returned HMM, via
    ///    `profile_config()` / `OProfile::convert()`.
    pub fn single_builder(
        &self,
        name: &str,
        dsq: &[crate::alphabet::Dsq],
        n: usize,
        abc: &Alphabet,
        bg: &Bg,
    ) -> Result<Hmm, String> {
        let q = self
            .q
            .as_ref()
            .ok_or_else(|| "score system not initialized".to_string())?;

        let mut hmm =
            crate::seqmodel::seqmodel(abc, dsq, n, name, q, &bg.f, self.popen, self.pextend);
        crate::hmm::set_composition(&mut hmm);
        crate::hmm::set_consensus(&mut hmm, abc, Some(dsq));

        // calibrate() (p7_builder.c:1010) -> p7_Calibrate(). Its errbuf text
        // is what the drivers print after "build failed: ".
        crate::calibrate::calibrate_with_config(&mut hmm, abc, bg, self.seed, self.calibration)?;

        // p7_builder.c:512-516: nucleotide models carry a window length.
        if matches!(self.abc_type, AlphabetType::Dna | AlphabetType::Rna) {
            if self.w_len > 0 {
                hmm.max_length = self.w_len;
            } else if self.w_beta == 0.0 {
                hmm.max_length = (hmm.m * 4) as i32;
            } else {
                set_max_length_from_beta(&mut hmm, self.w_beta);
            }
        }

        Ok(hmm)
    }
}
