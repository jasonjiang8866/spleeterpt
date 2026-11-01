# Reproducible MUSDB port-quality comparison

`scripts/evaluate_musdb.py` evaluates **all available tracks in the requested
subsets by default**, at full duration, using local assets. It supports MUSDB18
STEMS MP4 and MUSDB18-HQ WAV directories. It does not download datasets, weights,
or upstream repositories, and defaults to CPU. `--device cuda` runs all three PyTorch variants on CUDA
with TF32 disabled; the official TensorFlow reference remains on CPU. This is a
quality comparison, not a cross-framework speed benchmark. There are **no full MUSDB results
yet**. The existing five-preview SI-SDR results remain a separate limited experiment.

## What is compared

Run a separate experiment for each official **2-stem** and **4-stem** checkpoint:

| Variant | Implementation |
| --- | --- |
| `tensorflow_current` | Existing `OfficialGraph`: supplied upstream audio pipeline plus restored official TF network |
| `spleeterpt` | Current unchanged PyTorch runtime |
| `legacy_unmodified` | Supplied fwcd/spleeter-pytorch estimator with original framing/activations |
| `legacy_corrected` | Existing diagnostic padding/cropping correction; also ELU/ELU for four stems |

All variants receive identical decoded float32 stereo mixture samples. The two
legacy instances are loaded separately. The current PyTorch preset must equal the
official preset, and **every converted model-state tensor is checked for exact
equality with the supplied TensorFlow checkpoint** before evaluation. This rejects
fine-tuned/mismatched weights without relaxing any waveform tolerance. The code
never normalizes, shifts, remixes or pads model estimates to improve their scores.

For two stems, references are `vocals` and `drums + bass + other`; the latter is
**not** `mixture - vocals` (AAC mixture and sources need not sum exactly). Four
stems use `vocals`, `drums`, `bass`, `other`. No five-stem scores are produced:
MUSDB has no independent piano target, and its `other` target includes piano.

## Setup and assets

From a checkout of this repository on Linux x86-64, with uv installed:

```bash
uv sync --locked --group dev --extra evaluate
```

The evaluation extra pins `museval==0.4.1` and `tensorflow-cpu==2.18.1`.
Both `ffmpeg` and `ffprobe` must be on PATH, **including for WAV evaluation**:
museval imports musdb/stempeg, which checks these executables at import time.
On Debian/Ubuntu install them with `sudo apt-get update` followed by
`sudo apt-get install -y --no-install-recommends ffmpeg`. The CLI checks this
prerequisite before loading models. Supply:

- The legally obtained complete dataset, or an explicitly labeled subset/preview.
- Existing upstream Deezer Spleeter and legacy fwcd/spleeter-pytorch source checkouts,
  pinned to the commits used in the earlier parity experiments if comparing to them.
- The official TF1 checkpoint prefix with `.index`, `.meta`, and `.data-*` files.
- Its converted spleeterpt checkpoint (see the repository conversion instructions).

Expected layouts:

```text
MUSDB18/test/Artist - Track.stem.mp4
MUSDB18-HQ/test/Artist - Track/{mixture,drums,bass,other,vocals}.wav
```

WAV input must already be stereo, 44.1 kHz, finite, and have equal source lengths.
Mismatches fail rather than silently resampling or cropping reference sources.
The existing ffmpeg preview decoder is reused for MP4 (streams 0 through 4 are
mixture, drums, bass, other, vocals). The manifest records ffmpeg's version.

## Full test-set invocation

Example for a locally supplied MUSDB18-HQ test set and four-stem weights:

```bash
uv run --no-sync python scripts/evaluate_musdb.py \
  --root /data/MUSDB18-HQ --layout wav --subsets test \
  --dataset-label MUSDB18-HQ --expected-tracks 50 \
  --upstream-root /sources/spleeter \
  --legacy-root /sources/spleeter-pytorch \
  --tf-checkpoint /weights/4stems/model \
  --checkpoint /weights/spleeterpt-4stems.pt \
  --output /results/musdb-hq-4stems

uv run --no-sync python scripts/evaluate_musdb.py --validate-output /results/musdb-hq-4stems
```

Repeat with the corresponding two-stem TF/PyTorch checkpoint paths and a **new**
output directory. For STEMS use `--layout stems`. Use `--subsets train test` to
include both directories; train results must not be presented as held-out test
results. Check model training-data overlap separately before interpreting quality
as generalization. Do not compare AAC MUSDB18 and lossless MUSDB18-HQ as if they
were identical inputs.

`--expected-tracks 50` is a count guard, **not proof** that the supplied files are
all original, full-length test songs. The dataset label is a user declaration;
`full_dataset_verified` stays false. Check acquisition provenance and the saved
track IDs/durations before describing an experiment as the complete benchmark.

## Deterministic subsets and smoke runs

- `--track-list selection.txt`: one exact `test/Artist - Track` ID per line, no
  extension, duplicates or unknown IDs. Selected IDs are sorted before inference.
- `--limit 5`: first five sorted IDs. Explicitly a subset, never called a full run.
- `--limit 5 --seed 17`: NumPy `default_rng(17)` sampling without replacement,
  followed by sorting. Exact selected IDs and NumPy version are recorded.
- `--start-seconds 30 --duration-seconds 10`: evaluate that exact interval for all
  models; fail if any song cannot provide it. Defaults use the entire song.
- `--threads 2`: CPU thread limit. This CLI never provisions paid compute.

A preview directory is still a preview even when all its files are selected.
The test/preview corpus is never auto-downloaded by this entry point.

## Two different metrics

**BSSEval-v4 SDR** uses `museval.evaluate`, matching `eval_mus_track` for these
2/4-target scenarios: stereo source-image evaluation, time-invariant 512-sample
filters fitted over the evaluated song, no source permutation, 1-second windows
and hops. Raw SDR, ISR, SIR and SAR arrays are saved per target. The principal SDR
summary is median across non-NaN frames **within each song**, then median across
songs per target. The mean of song medians is also reported; song durations do not
weight the aggregate. museval 0.4.1 omits the final incomplete window on songs
longer than one second; its whole-song filter fit still uses the full evaluated
interval. Songs shorter than a second produce one short window.

**Global stereo SI-SDR** reuses the preview function unchanged: flatten both
channels over the evaluated interval, subtract the global mean, and fit one scale
factor. It has no distortion filter or time alignment, and is **not BSSEval SDR**.
Its per-song scalar values are aggregated separately. Do not compare its numerical
values to published BSSEval SDR tables or treat either metric as a listening test.

Undefined/NaN scores are JSON `null`; infinite scores are explicit `+inf`/`-inf`
strings. Aggregate counts distinguish finite, undefined and infinite observations.
NaNs are omitted from medians, infinities are retained. Paired differences use
only pairs with two finite song scores and report excluded-pair counts as
`undefined`; missing values are never changed to zero or silently counted as wins.
If museval rejects a whole-song silent source/estimate, its BSS scores are kept
as undefined for every target of that variant, with the exact reason in
`bsseval_v4_error`; SI-SDR is still computed independently. Frame coverage counts
are included separately from song statistics. Any other metric/decode/model
exception fails the run and preserves its
completed track records.

### Length fairness

Original legacy inference drops the fractional-hop tail. Default `common-crop`
evaluates **the same prefix for every variant and reference**, allowing only the
original legacy output to be shorter by less than one hop (1024 samples). Current
TF, current PyTorch, and corrected legacy must all retain the complete input
length. Unexpected truncation, added samples, missing stems or NaN audio fails.
Every result records native output lengths, evaluated samples and discarded tail.

This is a controlled port comparison on a common interval. Its SDR numbers can
slightly differ from a full-duration, zero-padded standard submission evaluation;
do not label them as an identical leaderboard protocol. `--length-policy strict`
rejects **any** output-length mismatch instead; it may fail on unmodified legacy.
No estimate is padded or repaired by the scorer. SI-SDR includes the entire common
interval; BSSEval frames follow the museval window convention above.

## Results and integrity

A new output directory is required to prevent stale-result mixing:

- `manifest.json`: frozen IDs and input SHA-256 hashes, requested scope, model
  config, checkpoint hashes, source commits and tracked working-tree hashes,
  package versions, pip freeze, CPU/decoder environment and metric protocol.
- `tracks/0000.json`, etc.: one atomic record for every selected song, all four
  variants and all targets, raw BSS frame arrays, SI-SDR and sample accounting.
- `summary.json`: per-variant/per-target statistics and paired **song-level**
  differences for current minus TF, current minus original legacy, corrected
  minus original legacy, and corrected minus TF. Includes median/mean delta,
  maximum absolute delta, positive/negative/equal counts and named-song deltas.
- `status.json`: running/failed/complete plus manifest, record and summary hashes.

`complete` is set only after the validator checks the expected song × model ×
target coverage, BSS frame counts, lengths, file hashes and a fully recomputed
summary. `--validate-output` repeats that verification without weights or
TensorFlow. It detects missing/extra tracks, changed files and inconsistent
summaries. This is accidental-corruption/incompleteness detection, not an
adversarial signature, a quality threshold or an independent dataset certificate.
An interrupted run retains completed records; restart into a new directory.
There is deliberately no automatic resume that could mix changed assets.

The scorer retains full-song arrays, and upstream/legacy inference is not
streaming. Long songs can require substantial RAM and BSSEval CPU time; use a small
explicit subset first. Failed runs are never included in a successful aggregate.

## Answering whether the port fixed quality

Inspect both metrics per instrument and the paired distributions, separately for
two and four stems. Compare current-vs-TF residuals with current-vs-original gains;
use corrected-legacy-vs-TF to test whether the framing/activation diagnosis explains
the gains. Inspect regressions and excluded pairs, not just one overall median.
Do not assert that every song improves, particularly for two stems. No scalar
pass threshold or automatic “quality fixed” verdict is imposed.

Current status: **entry point implemented; complete-dataset quality conclusion
pending user-supplied data, weights and an actual run**. Synthetic CPU tests verify
selection, target mapping, fair cropping, aggregation, integrity failures, and the
real museval adapter against its official track API; they are not music-quality
measurements. Existing strict waveform-parity thresholds remain unchanged.

References:

- [museval evaluation implementation](https://github.com/sigsep/sigsep-mus-eval/blob/v0.4.1/museval/__init__.py)
- [BSSEval-v4 implementation](https://github.com/sigsep/sigsep-mus-eval/blob/v0.4.1/museval/metrics.py)
- [MUSDB stream/target definitions](https://github.com/sigsep/sigsep-mus-db/blob/master/musdb/configs/mus.yaml)
- [MUSDB dataset and splits](https://sigsep.github.io/datasets/musdb.html)
