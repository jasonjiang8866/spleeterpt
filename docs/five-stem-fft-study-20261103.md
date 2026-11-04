# Five-stem FFT batch candidate: CPU evidence and GPU handoff

## Decision

The candidate is ready for a **GPU experiment**, not a released inference default.
Official 5-stem remains research-only; 2/4-stem `weights-v1.0.0` remain recommended.
No checkpoint tensors, TensorFlow tolerances, ratio-mask formulas, or production
inference APIs changed. No new paid GPU was launched for this report.

The previous [GPU report](five-stem-gpu-20261102.md) and its raw failures remain
unchanged. It localized the 60-second mixture discrepancy to the final 28-frame
FP32 FFT batch. Cached-spectrum and FP64 controls supported that diagnosis but
did not establish acceptance. In particular, the FP64 frontend changed the full
FP32 reference and was not adopted as a fix.

## Candidate and invariants

`scripts/fft_batch_policy.py` implements `pad-long-tail`. For multi-window inputs,
it zero-extends a short final FFT input to 512 frames, computes the same FP32 Hann
STFT, then **crops back to the actual frame count before magnitude/model padding**.
The extra overlapping signal frames therefore never enter the network. Full blocks
and single-window inputs retain their original FFT calls. The full-file reference
is always native. All temporary allocations remain at most one model window.

This is a process-local measurement context, not thread-safe application code.
It temporarily replaces only the streaming frontend and restores it on exceptions.
No default CLI download or inference behavior changes. Equal FFT batch sizes are a
hypothesis about CUDA kernel behavior, not a guarantee of numerical equivalence
for every cuFFT implementation or full-file length.

## Reproducible CPU evidence

Measured source commit: `ed3e4789d940703a057760ed28d89374c99dcfeb`.
Tracked source was unchanged during measurement; `working_tree_modified` in edge
reports includes untracked result JSON being written in the checkout.
Environment: Linux x86_64, Python 3.12.14, Torch 2.8.0+cpu, NumPy 2.0.2,
SciPy 1.15.3, two Torch threads. CUDA was unavailable. Timings are one measured
run after one warmup, include file I/O, and are not speedup estimates; some CPU
checks ran concurrently. No audio or separated output is redistributed.

Official converted 5-stem SHA256:
`0f93e93ad4629baafe6d0a2690fa386907518f8d93667dae0743e886a83f36c7`.
The archive, upstream URL and configuration are pinned in
[`five_stem_source.json`](../scripts/five_stem_source.json). The 2/4-stem files
were downloaded through the released catalog; their exact hashes are in the JSON.

| Check | Result | Worst waveform absolute error |
|---|---|---:|
| Official 2-stem: 15 file edges, silence and mono included | pass | 7.451e-9 |
| Official 4-stem: same 15 cases | pass | 1.118e-8 |
| Official 5-stem: same 15 cases | pass | 1.118e-8 |
| Official 5-stem: 12 s piano, Pistachio at 0 s | pass | 1.490e-8 |
| Official 5-stem: 12 s deterministic synthesis | pass | 2.980e-8 |
| Official 5-stem: 60 s deterministic synthesis | pass | 2.980e-8 |

The frontend-only matrix covers 84 valid lengths: 0/1/5 preceding full windows,
remainders 1/2/7/28/31/32/255/256/511/512, and sample offsets -1/0/+1.
Both native and candidate STFTs were bitwise equal to the CPU full STFT in every
case. This does **not** reproduce the earlier CUDA-specific discrepancy and does
not validate a CUDA fix. Matrix coverage alone is not checkpoint acceptance.

| CPU case | Legacy peak RSS MiB | Candidate peak RSS MiB | Legacy / candidate seconds |
|---|---:|---:|---:|
| 12 s piano | 911.55 | 915.50 | 2.003 / 2.055 |
| 12 s synthetic | 914.58 | 922.51 | 2.107 / 2.148 |
| 60 s synthetic | 2035.18 | 937.98 | 8.181 / 6.511 |

Per-stem waveform error, RMS, reference RMS, **legacy-reference consistency
SI-SDR**, checkpoint/input hashes and memory/time counters are preserved in
[`results/five-stem-fft-20261103/`](results/five-stem-fft-20261103/).
For the piano excerpt, piano RMS was 0.107808 and other RMS was 0.0118291;
consistency SI-SDR was 177.864 and 169.498 dB respectively. These describe agreement
with this project's full-file output. They do not show correct source allocation,
separation quality, or improvement against ground-truth stems.

Local suite: 297 passed, 3 CUDA tests skipped. RunPod offline suite: 82 passed.
The 40 targeted checks include small 2/4/5-stem models, short-input invariance,
network zero-context preservation, exception restoration, corpus leakage checks
and candidate/native report isolation. None replaces the official CUDA experiment.

Reproduce from the measured commit with the locked CPU environment:

```bash
uv sync --locked --group dev --group release
uv run --no-sync python scripts/prepare_five_stem.py --work ../weights
uv run --no-sync python scripts/verify_fft_batches.py --output fft-cpu.json
uv run --no-sync python scripts/verify_bounded_checkpoint.py \
  --checkpoint ../weights/5stems.pt --fft-batch-policy pad-long-tail --output edges.json
uv run --no-sync python scripts/prepare_five_stem_corpus.py \
  --work ../calibration --split calibration
uv run --no-sync python scripts/benchmark_bounded.py \
  --checkpoint ../weights/5stems.pt --fft-batch-policy pad-long-tail \
  --input ../calibration/librosa-pistachio-0.wav --durations 12 --output piano.json
uv run --no-sync python scripts/benchmark_bounded.py \
  --checkpoint ../weights/5stems.pt --fft-batch-policy pad-long-tail \
  --durations 12 60 --output synthetic.json
```

The unchanged gate is max absolute error <=2e-6 and consistency SI-SDR >=100 dB.
For reference RMS <=1e-5, absolute RMS error <=1e-8 and max error <=1e-7 are
required instead of relying on unstable relative metrics. The original stricter
TensorFlow/PyTorch waveform failures remain separate and unresolved.

## Corpus registration and remaining coverage

[`five_stem_corpus.json`](../scripts/five_stem_corpus.json) pins five independent
public recordings, their audio bytes/hashes, upstream metadata hashes and split.
Pistachio and the previously measured Global Volume mixture remain calibration.
Vibe Ace, Choice, Let's Go Fishin', and Hungarian Dance are reserved holdouts;
only source bytes/metadata were inspected, with no model inference on these four.
All clips from a recording stay in one split. Preparation requires an explicit
split; it does not run inference. No new holdout result is claimed here.

**Only one independently recorded piano source is currently confirmed.** Vibe Ace
is described as vibes/bass/kit/synths, Sweet Waltz as synths, and the Brahms excerpt
as string orchestra. They cannot be counted as extra piano recordings. Before
release evaluation, add two independently recorded, licensed piano sources,
including piano in a mixture, and freeze their audio hashes and clip positions.
`--require-piano-coverage` deliberately rejects the current incomplete manifest.
Minimum 3 piano recordings / 2 piano holdouts is a coverage floor, not a claim of
statistical sufficiency. Do not change thresholds after observing the holdout;
register any subsequent calibration and a fresh holdout explicitly.

Upstream text/TOML licenses conflict for several recordings. The manifest records
that ambiguity and exact metadata hashes; research inputs remain local, with no
audio/stem redistribution. Brahms/US Army Strings metadata agree on public-domain
status. Model redistribution licensing remains a separate release review.

## Next GPU experiment, within existing authorization

The existing manual workflow now exposes `fft_batch_study` (default false).
With it true, the worker measures candidate-vs-native-full-file using the same
edge, 1/12/60/600-second synthetic, 1/12/60-second real, and three piano-clip cases.
The 600-second run is needed to measure the **changed candidate's** allocation
bound; the expensive FP64 diagnostic is not repeated. The FFT matrix is recorded
instead. Native historical failures remain in git. Validators bind the policy in
provisioning, worker reports, each isolated measurement, and post-deletion recovery;
a candidate pass cannot be relabeled as a native pass or release approval.

Expected MIG compute for 10–20 minutes: about US$0.12–0.23 at the historical
US$0.69/hour cap; recheck live rate and inventory first. One 60-minute lease caps
this attempt's quoted compute at US$0.69, excluding storage/requests and deletion
latency. The user's **US$2 cumulative project authorization** includes previous
estimated compute US$0.100894; the shared account budget check is not that project
cap. Do not launch concurrent attempts or retry an ambiguous create. Keep fallback
false. No training is involved.

After merging this PR and checking the exact main SHA's CI:

```bash
gh workflow run five-stem-preflight.yml --repo jasonjiang8866/spleeterpt --ref main
gh workflow run five-stem-capacity.yml --repo jasonjiang8866/spleeterpt --ref main
# Inspect fresh inventory, volume, region, image/rate and budget before the paid step.
gh workflow run runpod-five-stem.yml --repo jasonjiang8866/spleeterpt --ref main \
  -f confirm_paid_benchmark=true -f allow_4090_fallback=false -f fft_batch_study=true
```

Equivalent UI: [preflight](https://github.com/jasonjiang8866/spleeterpt/actions/workflows/five-stem-preflight.yml),
[capacity](https://github.com/jasonjiang8866/spleeterpt/actions/workflows/five-stem-capacity.yml),
[paid experiment](https://github.com/jasonjiang8866/spleeterpt/actions/workflows/runpod-five-stem.yml).
Verify the dispatched run's source SHA. Preserve the complete artifact, check
numerical acceptance separately from job status, confirm owned Pod deletion and
run the existing fresh-process recovery verifier. Do not release 5-stem solely
because this candidate or CI passes.

The current editing environment has repository read/write access but no fresh
Actions dispatch tool or authenticated `gh`. Therefore no fresh provider inventory
or paid result is claimed. Repository instructions and the RunPod skill prohibit
changing workflow triggers to bypass this limitation. Manual dispatch is the
remaining operational handoff, not a request for another budget approval.
