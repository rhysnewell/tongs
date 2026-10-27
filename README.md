# tongs

A fast Rust HMMER 3.4. tongs is a fork of [hmmer-pure-rs](https://github.com/henriksson-lab/hmmer-pure-rs) 0.8.0, a Rust rebuild of [HMMER](http://hmmer.org) translated from its C source. The goal is identical hits to C HMMER at lower wall time, for use as a library inside [rosella](https://github.com/rhysnewell/rosella).

On 183 GTDB marker models against 321,472 metagenome proteins (`-T 10 --domT 10 -Z 1 --domZ 1 --cpu 8`, 4-core Haswell), tongs takes 88 s, C hmmsearch 3.4 141 s, and hmmer-pure-rs 0.8.0 375 s. The per-domain table matches C on every row but one inherited `acc` value.

What changed from the fork point:

* Targets are read once and every model scans whole blocks in parallel, instead of re-reading the file per model.
* The SSV shortcut runs at every model length, on 32-lane AVX2 bands.
* The Viterbi filter runs on 16-lane AVX2.
* Profile tables are shared rather than copied, matrices are reused, and the posterior matrix is never stored: each row is decoded and used straight away.
* Forward and Backward run two targets, and two domain envelopes, through each pass. One lane kernel serves the parser, the full matrix and the pair.
* tongs builds for x86_64 and aarch64 only. The scalar fallback for other targets is gone.


## Build

```bash
cargo build --release
```

`tongs --help` lists the subcommands. Each takes the same flags as its HMMER 3.4 counterpart.

## Tests

```bash
cargo test --release
```

The C parity tests need a built HMMER 3.4 tree under `hmmer/` and the fixtures fetched by `scripts/download_*.sh` and `scripts/setup_real_world_fixture_dirs.sh`.

## Cite

Cite HMMER:

> Eddy SR. *Accelerated profile HMM searches.* PLoS Comput Biol. 2011;7(10):e1002195. doi:[10.1371/journal.pcbi.1002195](https://doi.org/10.1371/journal.pcbi.1002195)

and the hmmer-pure-rs translation:

> Johan Henriksson. Static analysis-guided agentic AI translation enables Rust as a full stack bioinformatics language. arXiv:2608.13029, 2026. https://doi.org/10.48550/arXiv.2608.13029

## License

BSD-3-Clause, the same as HMMER and hmmer-pure-rs. `LICENSE` keeps their copyright lines, and Easel's for the bundled `src/data/esl_scorematrix.c`.
