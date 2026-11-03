# Official five-stem GPU validation — 2026-11-02

**Not accepted for release.** This supplements the [CPU report](five-stem-validation-20261102.md),
whose strict TF/PyTorch 6/25 failures remain unchanged. The project-specific budget
is US$2; the initial no-capacity attempt created no compute. PR #42 was merged at
`3ecf45d8d505a314533ad5c0e856f5f5795f4404` before this retry.

## First actual GPU attempt

[Run 41473954513](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41473954513)
used source `3b8dfaf68995588bccb6af414b3f25882a2c17b7`. The catalog now listed
`EUR-IS-2` and `US-PA-1`. One PRO 6000 Blackwell MIG 1g.24gb was created at
$0.69/hour. Runtime confirmed capability 12.0 and 25,367,150,592 visible bytes,
Torch 2.8.0+cu128, CUDA 12.8, Python 3.11.11, NumPy 2.0.2 and SciPy 1.15.3.
Full installed dependency versions are preserved in the raw worker log.
The resolved image digest is
`runpod/pytorch@sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853`.

Fresh conversion matched the pinned official five-stem archive and converted
checkpoint hashes from the CPU report. All 15 file-edge cases passed. Small
CUDA 2/4/5-stem regression tests also passed. These do not substitute for full
official-checkpoint cross-framework acceptance.

Synthetic durations 1/12/60/600 seconds passed the unchanged bounded-vs-legacy
gate. On the 600-second case, CUDA peak allocated memory was 17,197.95 MiB legacy
versus 571.80 MiB bounded; file-to-file times were 11.92 s and 5.49 s respectively.
These are one-run observations on one GPU, not stable performance guarantees.

The 60-second real mixture (`Auctioneer - Our Future Faces`, existing training-pool
validation input) failed the **2e-6 maximum absolute error** gate for two stems:

| Stem | Maximum absolute error | RMS error | Reference RMS | Legacy-reference SI-SDR |
|---|---:|---:|---:|---:|
| drums | 3.23355e-6 | 4.52931e-8 | 0.0270921 | 115.539 dB |
| other | 2.90945e-6 | 4.35878e-8 | 0.0540993 | 121.877 dB |

The SI-SDR values describe consistency with the original PyTorch path, **not
separation quality against true stems**. The piano output was near silent
(reference RMS 5.23424e-6); it passed the predeclared near-silent absolute gate.
The 1/12-second real cases passed. No thresholds were relaxed.

The worker originally stopped at that rejection, so piano excerpts were not run.
The completed raw evidence was downloaded and all archive member hashes verified
after the Pod was deleted. Local fresh-process recovery explicitly records
`acceptance_status=rejected`; archive recoverability is distinct from acceptance.
The controller confirmed deletion; elapsed controller time was 239.48 seconds and
the quoted-rate compute upper estimate was **$0.04590**. This excludes standing
storage/request charges and is not an invoice reconciliation.

Raw [result and worker log](results/five-stem-gpu-20261102/41473954513/result.json),
[provision/cleanup record](results/five-stem-gpu-20261102/41473954513/provision.json),
[recovery](results/five-stem-gpu-20261102/41473954513/recovery.json), and
[hash manifest](results/five-stem-gpu-20261102/41473954513/manifest.json) are retained.

## Follow-up protocol

Before another run, the harness was extended to retain a complete numerical
rejection and continue gathering the remaining cases. Crashes, missing reports
and incomplete duration coverage still abort. The run remains failed if any
comparison rejects. Recovery can verify failed archives without marking numerical
acceptance passed. The production model, streaming algorithm and tolerances are
unchanged.

The diagnostic uses the same failing 60-second input. A cached full FP32 STFT
feeds the bounded path to isolate synthesis grouping from STFT batch effects.
A second control computes FFTs in FP64 with the same FP32 Hann coefficients,
casts spectra to complex64, and keeps the network in FP32. It compares full and
block transforms. This is an explicitly unbounded diagnostic, not a new inference
mode or proposed release threshold. The three pinned piano excerpts will also run.

Local validation before the follow-up: 81 offline lifecycle tests and one new
CPU diagnostic regression passed, including frontend restoration and failed-archive
recovery. The follow-up outcomes are below.


## Completed control experiment

[Run 41477211252](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41477211252)
used clean source `51bcbcca23fe198c0b456e8a94a9bc4f7bb868a2`, after PR #43 merged
at `a7efba41c833875079a29368c47d7796af4ce19c`. The image digest, GPU identity,
CUDA/Python/core dependency versions and weight hashes match the first run.
All expected measurements were collected (`complete=true`), but the worker and
independent acceptance correctly **failed** because `5stems-real` rejected again.
The 60-second error metrics reproduced exactly, including the failed stems.

The normal FP32 STFT blocks matched the full STFT exactly for the first five
512-frame blocks. The final 28-frame block differed by up to **3.14568e-5** in
complex magnitude. Using cached full-STFT blocks reduced the resulting waveform
differences as follows. There are 5,292,000 scalar samples per stem.

| Comparison | Drums max error | Other max error | Max over all stems | Samples >2e-6 (drums / other) |
|---|---:|---:|---:|---:|
| Normal bounded vs original full FP32 | 3.23355e-6 | 2.90945e-6 | 3.23355e-6 | 608 / 167 |
| Cached full FP32 STFT, bounded synthesis vs original full | 7.45058e-9 | 1.49012e-8 | 4.47035e-8 | 0 / 0 |
| FP64 FFT control, bounded vs full | 7.45058e-9 | 1.49012e-8 | 2.98023e-8 | 0 / 0 |

In the FP64 FFT control, all block/full complex64 spectra were exactly equal after
casting. The Hann coefficients, model weights and FP32 network were held fixed.
Together these controls strongly support **FFT batch-shape-dependent FP32
roundoff** as the source of this observed GPU discrepancy, rather than a frame
shift or reconstruction mismatch in this case. This is an inference from the
controls, not a trace of cuFFT's internal kernel selection, and does not prove
correctness for every input or backend.

Crucially, the FP64-FFT full path itself differed from the original FP32 full path
by up to **5.96046e-6**. It is a diagnostic reference, not an automatic fix that
preserves the existing FP32 baseline. Neither production inference nor any
acceptance threshold was changed.

[Full diagnostic](results/five-stem-gpu-20261102/41477211252/diagnostic.json)
contains per-stem maxima, RMS errors, reference RMS and threshold exceedance counts.

## Piano excerpts and memory observations

All three 12-second excerpts from the same pinned solo-piano recording passed
all five stems against the original PyTorch path. They are three excerpts, **not
three independent songs**, and have no isolated reference stems.

| Start | Maximum error over five stems | Minimum consistency SI-SDR | Piano output RMS | Other output RMS |
|---|---:|---:|---:|---:|
| 0 s | 8.94070e-8 | 112.478 dB | 0.107808 | 0.0118291 |
| 15 s | 1.30385e-6 | 102.620 dB | 0.103981 | 0.0162081 |
| 30 s | 5.40167e-8 | 107.997 dB | 0.131641 | 0.00304268 |

The output RMS values describe how this model allocates these recordings; they
are not a purity score or proof of correct piano/other separation. No quality
improvement over four-stem, nor ground-truth separation SI-SDR, is claimed.
The raw result records per-stem metrics, original-source hash and each clip hash.

The repeated 15 edge cases and three small CUDA 2/4/5-stem regressions passed.
Official five-stem synthetic 1/12/60/600 s and real 1/12 s cases passed unchanged.
Previously sufficient full official 2/4-stem GPU benchmarks were not repeated;
this run's 2/4 evidence is limited to the small CUDA regression.

| Synthetic duration | Legacy peak allocated CUDA | Bounded peak allocated CUDA |
|---|---:|---:|
| 1 s | 307.33 MiB | 300.05 MiB |
| 12 s | 575.82 MiB | 571.66 MiB |
| 60 s | 1,930.52 MiB | 571.80 MiB |
| 600 s | 17,197.95 MiB | 571.80 MiB |

The peaks reproduced across both Pods. Host RSS and reserved CUDA memory are also
in the raw records. At 600 s the second run took 6.02 s legacy and 5.21 s bounded,
compared with 11.92 s / 5.49 s in the first run. This variability rules out a
blanket speedup claim; the bounded-memory finding is much more stable. At 60 s
real audio, the second run was 0.65 s / 0.66 s. Timings include file I/O and are
single trials per mode in isolated processes on shared infrastructure.

## Cleanup, cost and durable recovery

The second Pod was deleted and its absence confirmed before a fresh Actions
process revalidated the archive. Recovery passed all member/hash checks and
explicitly recorded `acceptance_status=rejected`, `gpu_acceptance_revalidated=false`.
A separate GET-only status job at 16:09:20 UTC found no project five-stem Pods.
Both independent cleanup snapshots are preserved alongside the raw reports.

| Run | Controller elapsed | Quoted rate | Compute upper estimate |
|---|---:|---:|---:|
| 41473954513 | 239.48 s | $0.69/h | $0.045900 |
| 41477211252 | 286.92 s | $0.69/h | $0.054994 |
| Total | 526.40 s | | **$0.100894** |

These are wall-clock estimates, not invoices; existing persistent storage and
provider request charges are excluded. No new persistent volume was created.
Both paid triggers were restored to manual-only immediately after launch. No
training or further paid experiment was started.

The original small artifact ZIPs are committed so recovery does not depend on
GitHub Actions' 14-day retention. They contain metrics/logs only, not model weights
or source audio:

- [First failed artifact](results/five-stem-gpu-20261102/41473954513/artifact.zip),
  SHA-256 `b57ce2f53640c661340e949076518699afaba1c525ad8741fc9db14f4085f92a`.
- [Complete follow-up artifact](results/five-stem-gpu-20261102/41477211252/artifact.zip),
  SHA-256 `1c935edd3dc1588209a881c00143830b2d038a1a95ddc3a488be51e9673ac2f1`.
- [Follow-up result](results/five-stem-gpu-20261102/41477211252/result.json),
  [cleanup/cost](results/five-stem-gpu-20261102/41477211252/provision.json),
  [recovery](results/five-stem-gpu-20261102/41477211252/recovery.json), and
  [manifest](results/five-stem-gpu-20261102/41477211252/manifest.json).

To verify a downloaded archive without any GPU or RunPod credential:

```bash
unzip docs/results/five-stem-gpu-20261102/41477211252/artifact.zip -d /tmp/spleeterpt-five-stem-evidence
python runpod/five_stem_recovery.py --directory /tmp/spleeterpt-five-stem-evidence --allow-rejected
```

Expected result: archive recovery succeeds, GPU acceptance remains false. To
reproduce the numerical control on an already available CUDA device, prepare the
pinned checkpoint and use the recorded native-rate stereo input:

```bash
uv run --no-sync python scripts/diagnose_bounded.py --checkpoint /path/to/5stems.pt \
  --input /path/to/mixture.wav --device cuda --seconds 60 --fp64-control \
  --output /tmp/five-stem-fft-control.json
```

## Release recommendation

**Continue to withhold ready-to-use five-stem weights and `--model 5stems`.** Two
separate strict contracts remain unsatisfied: the CPU TensorFlow/PyTorch 6/25
waveform comparisons and the GPU bounded/full FP32 comparison documented here.
The controls explain observed errors; they do not turn the original failures into
passes. A future change needs a predeclared numerical contract supported by a
broader mixed-piano corpus and CPU/GPU distributions, or a verified inference
change that preserves the intended reference behavior. Quality evaluation still
needs properly licensed ground-truth stems. Recommended downloads remain the
official converted 2/4-stem baselines.
