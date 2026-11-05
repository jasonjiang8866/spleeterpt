# Preregistered mixed-piano evaluation

Reserve seconds 5–17 of two independent performances from Joachim Fritsch's
TRIOS dataset: Brahms Op. 40 (piano, violin, horn) and Schubert D.929 (piano,
violin, cello). These are two pieces from one collection. Keep every excerpt of
each recording in the holdout split. Use the distributed mix, isolated piano,
and the sum of the two remaining tracks as the other reference. Their parts were
recorded separately by musicians. Record input hashes before any inference.
The existing Pistachio recording serves as calibration.

Source: [TRIOS v1](https://doi.org/10.5281/zenodo.6797837). License:
[CC BY-NC-SA 2.0 UK](https://creativecommons.org/licenses/by-nc-sa/2.0/uk/).
Credit Joachim Fritsch, *High Quality Musical Audio Source Separation* (2012),
[Appendix B](https://www.atiam.ircam.fr/Archives/Stages1112/FRITSCH.pdf).
Only code, metadata, hashes, and aggregate metrics belong in git or Actions
artifacts. Fetch audio locally for noncommercial research.

## Before opening holdouts

1. Check package identity against the publisher checksum and freeze per-file
   SHA-256 plus decoded clip hashes. Check common sample geometry, native
   unity gain, and sample-aligned stem sum against the supplied mix. A residual
   above four PCM16 steps (peak) blocks use of that supplied mix as additive
   ground truth. Preserve a failed check; do not silently fit gains or offsets.
2. Freeze candidate revision, checkpoint, reference backend, splits and metrics.
   Run inference only after the historical 25-output strict parity gate passes.
3. For each recording report zero-mean, concatenated-channel SI-SDR for piano
   and other relative to their isolated references, for the official TF model,
   candidate, and unseparated mixture. Report each stem's improvement over the
   mixture and its candidate-minus-TF score. Candidate acceptance requires
   positive improvement over the mixture and no more than 0.1 dB loss against
   TF for each active target on each recording. This is a predeclared engineering
   criterion for these two clips.
4. For leakage, residualize the true piano against true other by least squares.
   Report the squared projection of estimated other onto that piano residual
   relative to true piano power, in dB; lower is better. Report the symmetric
   other-to-piano measure too. Report inactive reference RMS (threshold 1e-5)
   or degenerate projection separately; an undefined required quality score
   blocks a verdict. Do not add epsilons to turn silence into a score.
5. Report all five strict TF/PT waveform verdicts independently. If a frozen
   candidate fails, archive the result and register fresh recordings before
   tuning against these observations. No model execution occurs during registration.

## Preparing and scoring the fixed inputs

Run from the repository root with the locked CPU environment. Preparation checks
the archive and all eight original WAV identities and writes local float WAVs.
For these mono recordings, duplicate each channel to stereo for model input and
for the corresponding targets. Both estimates must use the same channel convention.
Preserve each model's source revision, weights and prediction provenance alongside
its exported WAVs.

```bash
uv run python scripts/prepare_five_stem_corpus.py --work ../holdout \
  --split holdout --require-piano-coverage
# Only after the frozen candidate clears the historical calibration gate.
uv run python scripts/score_piano_holdout.py --inputs ../holdout \
  --tf ../tf_estimates --candidate ../pt_estimates --output ../quality.json
```

Estimate paths are `<root>/trios-brahms/{piano,other}.wav` and
`<root>/trios-schubert/{piano,other}.wav`. Preserve floating-point output without
per-stem normalization. The score command hashes the reference and estimate PCM.
