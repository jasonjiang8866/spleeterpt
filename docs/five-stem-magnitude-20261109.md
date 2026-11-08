# Five-stem CPU magnitude breakdown — 2026-11-09

This follows the [CPU audio cross](five-stem-audio-cross-20261106.md) from
[PR #49](https://github.com/jasonjiang8866/spleeterpt/pull/49). It crosses
three binary factors in the magnitude frontend: analysis STFT backend,
Hann coefficient source, and complex absolute value backend.

Both inputs use the same hash-pinned historical recording. The repeated
input crosses three model windows and ends with a five-frame tail. The
Torch encoder, official TF decoder, TF carrier STFT, masks and inverse
STFT are fixed. The independently rebuilt TF STFT and the Torch STFT with
native Hann values must each reproduce their own native spectra exactly.
Feeding the TF spectrum through the official magnitude graph must also
reproduce its original feature tensor. The pointwise waveform threshold
remains `1e-5 + 1e-4 * abs(reference)`.

## Observed results

Intel Xeon Platinum 8573C, ATen AVX512, Python 3.12.14, Torch 2.8.0+cpu,
TensorFlow 2.18.1 with oneDNN disabled, NumPy 2.0.2. The table collapses
the absolute-value factor because each TF/PT pair had identical magnitude,
decoder-logit and waveform hashes.

Three fresh processes agreed on all 80 waveform comparisons per process.
[Raw reports](results/magnitude-20261109/evidence.zip) contain all three replicates,
the exact script and model hashes, input PCM hashes, and per-stem metrics.
The ZIP SHA-256 is
`5e394f24bd43b97d15aec18196982063ae20c3e19243a5c96b5d83643c95d560`.
The reports identify base commit `a4c092a01d8308eeecabbc3344bf797573685ad4`
and the local diagnostic source hashes. No audio, weights or holdout material
is included.

| Analysis STFT | Hann | Original worst tolerance ratio | Repeated worst tolerance ratio | Failed scalars, original / repeated |
|---|---|---:|---:|---:|
| TF | TF | 0.308331 | 0.354766 | 0 / 0 |
| TF | Torch | 0.777305 | 0.788843 | 0 / 0 |
| Torch | TF | 0.887487 | 1.023671 | 0 / 1 |
| Torch | Torch | 0.722568 | 0.927264 | 0 / 0 |

The Hann coefficients differed by at most `5.96046448e-8`. Changing FFT
backend at fixed TF Hann shifted the repeated waveform by up to
`1.37537718e-5`. Changing Hann at fixed Torch FFT shifted it by up to
`1.20401382e-5`. The largest signed FFT × Hann interaction was
`1.24387443e-5`. The one failing sample was in repeated-clip drums.
Substituting TF Hann by itself therefore made that Torch-FFT case worse.

The all-Torch frontend passed 10/10 outputs with the fixed TF decoder and
downstream path on this host. The [prior AMD study](five-stem-audio-cross-20261106.md)
found 68 failing scalars under that controlled configuration. The runs
differ in environment and do not isolate CPU architecture as the cause.
The FFT and Hann factors both warrant checking on a host that reproduces
the earlier failures. Native Torch decoder parity and independent mixed-piano
quality validation remain separate release gates. The holdouts were unused.

## Reproduction

Use the pinned CPU/release dependency groups, official weights and upstream
checkout described in the [audio cross setup](five-stem-audio-cross-20261106.md#reproduction).
Run each replicate in a fresh Python process.

```bash
uv run --no-sync python scripts/study_audio_cross.py \
  --magnitude-breakdown --weights /tmp/weights --upstream-root /tmp/upstream \
  --replicate cpu-1 --output /tmp/magnitude-cpu-1.json
```

## Path to five-stem weights publication

The current evidence supports a rounding investigation. It does not establish
a defect in Deezer's trained weights or a separation-quality regression.
The existing [operator and FP64 controls](five-stem-numerics.md) found matching
mathematical networks, with FP32 drift before the five-stem softmax. Two/four
stems use sigmoid masks; five stems uses cross-stem softmax. The measured
response to relative-logit drift helps explain this checkpoint's sensitivity,
but stem count alone does not prove the cause.

| Stage | Work | Completion evidence |
|---|---|---|
| Reproduce across environments | Run this matrix on an AMD environment that reproduces the earlier failure, pinning dependencies and decoded PCM. Record fresh-process repeats and the paired native controls. | A reproducible failing case and comparable Intel/AMD reports with exact source and environment identity. |
| Resolve native inference | Use the failing case to test one frontend intervention at a time, then isolate the remaining decoder convolution/BatchNorm drift. Recheck the full native pipeline after each candidate. | All five calibration inputs pass all five stems under the original pointwise gate, across declared CPU environments. Mixed TF/Torch diagnostic outputs cannot satisfy this native-PyTorch gate. |
| Validate CUDA and bounded inference | Review the current RunPod budget ledger and inventory, then validate the frozen candidate on PRO 6000 MIG 24 GB using the existing controller, Global Volume and cleanup checks. Include long-audio boundaries and the recorded FFT-tail candidate. | Original CUDA waveform gates, edge cases, bounded-memory measurements, artifacts and independently verified temporary-Pod deletion. |
| Open frozen piano holdouts | Freeze candidate source, checkpoint and evaluation protocol, then evaluate the two preregistered TRIOS mixtures against official TF and the mixture baseline. | Positive piano/other improvement over the mixture and at most 0.1 dB loss versus TF for each active target on each recording, plus leakage and waveform reports. Follow the existing holdout protocol on failure. |
| Package and publish | Extend the existing release builder and model catalog after the validation gates pass. Recheck weight redistribution notices, build the model card and verify clean-cache CLI download/inference and all uploaded checksums. | A reviewed draft release with immutable assets and explicit supported environments, followed by release authorization. |

If cross-environment FP32 parity remains unattainable, report that result and
make an explicit release-policy decision before changing acceptance criteria.
Quality-based experimental distribution would require a separately reviewed
policy and supporting measurements. A more permissive gate must not silently
turn previous numerical failures into passes.

Local validation for this change completed with 320 tests passed, three CUDA
tests skipped, and 82 RunPod offline tests passed. Three independent processes
completed the numerical matrix. This PR prepares the first stage of the release
plan; the later stages require their own measured evidence.
