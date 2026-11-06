# Five-stem CPU audio cross study — 2026-11-06

Local diagnostic extension of Draft PR #49. The inspected remote head remains
`bf419f5c61563ffea908cfc20999845462598f49`; main remains
`b35c45f398de79c8948ec893efdaada1602978c3`. No later PR fix or branch study run
was present when checked. No remote writes, Actions dispatches, paid compute,
weight publication or holdout inference were performed for this preparation.

## Decision

Prioritize the native **magnitude frontend** for the next residual-error isolation.
Keep the Torch encoder and official TensorFlow decoder fixed while splitting
complex absolute value, Hann coefficients, STFT/FFT and window padding. This is
an attribution result on the calibration recording and its repeated version,
not a production fix. The native Torch decoder remains a separate known error
source; do not infer that fixing the frontend will make the native pipeline pass.

## Controlled boundaries

The primary 2x2 crosses TF/Torch magnitude with TF/Torch inverse STFT. The Torch
encoder and official TF decoder always run one window per call. TF softmax,
magnitude multiplication, ratio-mask power/sum/epsilon/extension, complex
multiplication and the **TF complex carrier STFT** remain fixed in this matrix.
The inverse factor includes inverse FFT, synthesis Hann, overlap-add, compensation
and crop. Magnitude also feeds the unchanged post-logit magnitude multiplication.

Twelve additional bridge cells cross post-logit arithmetic and complex carrier
STFT independently (16 cells including the primary four). They connect the
isolated experiment to the complete native-audio/TF-decoder path. They do not
silently fold carrier or softmax changes into the inverse-STFT factor.

Control gates require exact array equality for official masked-spectrum refeed,
every TF post-logit refeed and complete native hybrid replay. Any mismatch makes
attribution invalid. Native 2/4/5-stem audio/model code and all thresholds remain
unchanged. Pointwise waveform acceptance remains
`abs(candidate-reference) <= 1e-5 + 1e-4*abs(reference)`, with exact output length.
No alignment, gain scaling or tolerance adjustment is applied.

## Inputs and environment

Only `audio_example.mp3` from upstream commit
`c8854001ac8acad34a9bc2bd15f28475541828b1` is decoded. The original and repeated
PCM hashes are checked against the existing strict-calibration report before
inference. Both mixed-piano TRIOS holdouts remain sealed. The repeated case is
not an independent recording.

| Input | Stereo samples | STFT frames | Model windows | Last window frames |
|---|---:|---:|---:|---:|
| music_original | 479,832 | 473 | 1 | 473 |
| music_multiwindow | 1,048,713 | 1,029 | 3 | 5 |

Official TF tensor files and `.meta` graph, converted checkpoint, upstream
commit, source MP3, PCM, config, executable source files and `uv.lock` are all
hash-bound in every report. Measurements use the inspected PR head plus the
local diagnostic source files; `working_tree_modified=true` is intentional and
exact source SHA-256 values are included. No claim is made that remote HEAD
already contains this extension.

Observed CPU: **AMD EPYC 9V74 80-Core Processor**. Python **3.12.14**, Torch
**2.8.0+cpu**, TensorFlow **2.18.1**, NumPy **2.0.2**.
TF oneDNN is disabled; Torch MKLDNN remains enabled; Torch intra-op and TF
session intra/inter-op thread counts are 2. OpenBLAS thread count is 1.
CPU flags, build information, libsndfile and all relevant environment variables
are included in the raw reports.

Each configuration is rerun in three fresh Python processes. The secondary
configuration sets only `ATEN_CPU_CAPABILITY=avx2` on the same physical host;
it is not a second machine and does not force every MKL/oneDNN/TF kernel to AVX2.

## Reproduced numerical results

Six final independent Python processes completed: three default **AVX512** and
three explicitly requested **AVX2** on the same host. Every replay gate passed.
Within each configuration all 160 cell/case/stem outputs were bitwise repeatable;
all output hashes also matched across these two configurations. This covers 960
waveform comparisons, with no holdout exposure. It is evidence of repeatability
in this local environment, not proof of universal determinism.

The primary matrix has these results in **every** final process. Counts are
failing channel/sample scalars under the unchanged pointwise gate.

| Magnitude | Inverse STFT | Original failures | Repeated failures | Worst tolerance ratio | Passed stem outputs / 10 |
|---|---|---:|---:|---:|---:|
| TF | TF | 0 | 0 | 0.440354 | 10/10 |
| TF | PT | 0 | 0 | 0.440354 | 10/10 |
| PT | TF | 25 | 43 | 1.160637 | 7/10 |
| PT | PT | 25 | 43 | 1.160949 | 7/10 |

All eight downstream configurations with TF magnitude passed all ten music
outputs. All eight with Torch magnitude retained the same failure pattern:
original bass **25**, repeated drums **17**, repeated bass **26**. Piano and
vocals/other passed in this fixed-TF-decoder experiment. The full native pipeline
still had **2,792 + 1,272 = 4,064** failing scalars on the music pair.

Signed samplewise contrasts separate frontend effects, reconstruction effects
and their interaction; subtracting failure counts would not do this. The
interaction is `(Y11 - Y01) - (Y10 - Y00)`, evaluated in float64 from the
unchanged float32 outputs. These contrast ratios are explanatory scales using
the original reference tolerance, not additional acceptance criteria.

| Primary effect | Largest absolute waveform delta | Largest delta / reference tolerance |
|---|---:|---:|
| frontend_at_tf_inverse | 1.39772892e-05 | 1.174263 |
| frontend_at_pt_inverse | 1.39176846e-05 | 1.174183 |
| inverse_at_tf_frontend | 3.57627869e-07 | 0.012679 |
| inverse_at_pt_frontend | 3.57627869e-07 | 0.011924 |
| interaction | 3.57627869e-07 | 0.009710 |

Across the carrier/post-logit bridge contrasts at fixed Torch inverse STFT, the
largest absolute delta was **3.57627869e-07**.
Reconstruction and downstream arithmetic therefore have nonzero effects, but
neither their swaps nor measured interactions changed the failing-scalar counts
on these inputs. The magnitude intervention is decisive for the residual gate
failures with TF decoder fixed; it does not exonerate the native decoder.

The final historical Actions artifact (42030696242) was independently hash-checked
as `073e51000adedaa49b8f44b24449db76b72526ea79aa0756e3f704acc1ab9d72`.
It recorded 54 hybrid failures, AMD EPYC 9V74, AVX2 and Python 3.12.16.
This local AVX2 setting still produced 68; the local interpreter is 3.12.14.
Changing only the ATen capability label does not reproduce the historical 54,
and no claim is made that Python version, CPU hardware or another particular
runtime difference caused it. Cross-host attribution remains unresolved.

## Per-stem primary waveform metrics

These values are identical in all six final repetitions. Full quantiles, RMSE,
failure fractions, waveform hashes, peak coordinates, boundary errors, bridge
contrasts and source/environment identities are in the raw archive.

| Case | Magnitude / inverse | Stem | Failed scalars | Max absolute error | Max tolerance ratio |
|---|---|---|---:|---:|---:|
| music_original | TF / TF | vocals | 0 | 1.06915832e-06 | 0.094821 |
| music_original | TF / TF | piano | 0 | 2.2649765e-06 | 0.205541 |
| music_original | TF / TF | drums | 0 | 5.80400229e-06 | 0.440354 |
| music_original | TF / TF | bass | 0 | 5.93066216e-06 | 0.375546 |
| music_original | TF / TF | other | 0 | 1.2293458e-06 | 0.082691 |
| music_original | TF / PT | vocals | 0 | 1.05798244e-06 | 0.095219 |
| music_original | TF / PT | piano | 0 | 2.2649765e-06 | 0.206965 |
| music_original | TF / PT | drums | 0 | 5.7965517e-06 | 0.440354 |
| music_original | TF / PT | bass | 0 | 5.9902668e-06 | 0.375099 |
| music_original | TF / PT | other | 0 | 1.22189522e-06 | 0.082691 |
| music_original | PT / TF | vocals | 0 | 3.03983688e-06 | 0.272738 |
| music_original | PT / TF | piano | 0 | 5.7220459e-06 | 0.478734 |
| music_original | PT / TF | drums | 0 | 1.12354755e-05 | 0.918098 |
| music_original | PT / TF | bass | 25 | 1.30459666e-05 | 1.160637 |
| music_original | PT / TF | other | 0 | 1.94460154e-06 | 0.164510 |
| music_original | PT / PT | vocals | 0 | 3.06963921e-06 | 0.270504 |
| music_original | PT / PT | piano | 0 | 5.72949648e-06 | 0.478756 |
| music_original | PT / PT | drums | 0 | 1.12652779e-05 | 0.919992 |
| music_original | PT / PT | bass | 25 | 1.3038516e-05 | 1.160949 |
| music_original | PT / PT | other | 0 | 1.93715096e-06 | 0.164452 |
| music_multiwindow | TF / TF | vocals | 0 | 1.68383121e-06 | 0.144632 |
| music_multiwindow | TF / TF | piano | 0 | 2.2649765e-06 | 0.205541 |
| music_multiwindow | TF / TF | drums | 0 | 5.80400229e-06 | 0.440354 |
| music_multiwindow | TF / TF | bass | 0 | 5.93066216e-06 | 0.375546 |
| music_multiwindow | TF / TF | other | 0 | 1.81794167e-06 | 0.117228 |
| music_multiwindow | TF / PT | vocals | 0 | 1.69873238e-06 | 0.144632 |
| music_multiwindow | TF / PT | piano | 0 | 2.2649765e-06 | 0.206965 |
| music_multiwindow | TF / PT | drums | 0 | 5.7965517e-06 | 0.440354 |
| music_multiwindow | TF / PT | bass | 0 | 5.9902668e-06 | 0.375099 |
| music_multiwindow | TF / PT | other | 0 | 1.83284283e-06 | 0.117444 |
| music_multiwindow | PT / TF | vocals | 0 | 3.65823507e-06 | 0.337311 |
| music_multiwindow | PT / TF | piano | 0 | 5.48362732e-06 | 0.513215 |
| music_multiwindow | PT / TF | drums | 17 | 1.33514404e-05 | 1.136426 |
| music_multiwindow | PT / TF | bass | 26 | 1.44094229e-05 | 1.096141 |
| music_multiwindow | PT / TF | other | 0 | 3.62470746e-06 | 0.268669 |
| music_multiwindow | PT / PT | vocals | 0 | 3.67313623e-06 | 0.339400 |
| music_multiwindow | PT / PT | piano | 0 | 5.48362732e-06 | 0.512859 |
| music_multiwindow | PT / PT | drums | 17 | 1.34706497e-05 | 1.139995 |
| music_multiwindow | PT / PT | bass | 26 | 1.43945217e-05 | 1.096459 |
| music_multiwindow | PT / PT | other | 0 | 3.62470746e-06 | 0.268423 |

[Complete raw evidence](results/audio-cross-20261106/evidence.zip), ZIP SHA-256:
`cabc89b0b66d0c91b53760f08ee5ebf852410f9134be8709e6c22a8116a7cd19`. The archive contains six process reports, two repeat summaries,
validation logs and per-member SHA-256 checksums. It contains no model weights,
audio, predicted stems or holdout material.

## Validation and proposed remote changes

- Final local CPU suite: **320 passed, 3 CUDA tests skipped**.
- **22 new tests** cover signed interaction contrasts, exact native replay for
  2/4/5-stem tiny models across short/exact-window/multiwindow lengths, incomplete
  reports, provenance changes and repeat failures that must not be hidden.
- **82 RunPod offline tests passed**; no RunPod resources were created or queried
  as part of this CPU study. Training/resume/reload/stereo-inference smoke passed.
- Ruff lint/format and Git whitespace checks passed. Manual workflow YAML was
  checked for manual-only triggering, read-only permissions and a 35-minute cap.
- The first full suite had two environment failures because a subprocess found
  host uv 0.13.5 instead of the repository-required 0.13.4. Running the complete
  final suite under pinned uv 0.13.4 resolved both without changing dependencies
  or production code.

Prepared changes add `study_audio_cross.py`, `summarize_audio_cross.py`, their
regressions, a manual-only CPU study workflow, TF CI coverage, this report and
raw numerical evidence. The existing BN/network-halves workflow is untouched.
Publishing this patch to the PR would trigger the existing normal Tests workflow;
the newly prepared numerical study has no push/PR/schedule trigger. Dispatch
would be a separate action requiring approval. PR #49 should remain Draft.

## Reproduction

Use the repository's uv 0.13.4, locked CPU/release groups, verified five-stem
research weights and the pinned upstream checkout. Do not run the holdout corpus
preparer or holdout scorer for this diagnostic.

```bash
uv sync --locked --group dev --group release
export TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2
uv run --no-sync python scripts/prepare_five_stem.py --work /tmp/weights
for repeat in 1 2 3; do
  uv run --no-sync python scripts/study_audio_cross.py \
    --weights /tmp/weights --upstream-root /tmp/upstream \
    --replicate "repeat-${repeat}" --output "/tmp/audio-cross/repeat-${repeat}.json"
done
uv run --no-sync python scripts/summarize_audio_cross.py \
  /tmp/audio-cross/repeat-*.json --output /tmp/audio-cross/summary.json
```

The checked-in manual workflow prepares `/tmp/upstream` at the pinned commit.
For a separate ATen-dispatch experiment, set `ATEN_CPU_CAPABILITY=avx2` before
launching Python and write to a different report directory; do not pool its
records as identical environments. Numerical rejections remain explicit in JSON.
A zero study exit means the diagnostic completed and replay controls held; it
does **not** mean strict waveform acceptance or release eligibility.

## Next experiment

1. Keep the same TF decoder, official downstream arithmetic and calibration PCM.
2. Cross TF/Torch complex STFT with TF/Torch complex absolute value to separate
   STFT drift from magnitude computation. Keep leading padding, bin slicing and
   model windows identical, including the repeated clip's five-frame tail.
3. If the STFT factor dominates, hold Hann coefficients fixed and split window
   generation from FFT implementation/batch shape. Preserve the original FP32
   reference; an FP64 diagnostic is not an automatic replacement baseline.
4. After a reproducible frontend intervention, retest the full native decoder on
   all five existing calibration cases. Only then choose the next decoder-stage
   intervention. Independent host validation remains necessary before claiming a
   portable fix, and the original release/holdout gates remain in force.
