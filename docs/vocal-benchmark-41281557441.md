# Spleeter-2 versus Kim Mel-Band RoFormer — measured vocal screen

Completed 2026-11-01 (Asia/Singapore).
Kim Mel-Band RoFormer improves both quality metrics on every song/source in
this screen. Its median paired vocal gain is **+5.030 dB SDR** and **+5.543 dB
SI-SDR**. Native Spleeter-2 has **55.67× faster inference** and lower peak CUDA
allocation with these fixed FP32 recipes. Keep official Spleeter weights as
the compatibility baseline; the vocal quality/resource tradeoff is now measured.

[Successful run](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41281557441) · [Machine-readable measurements](results/vocal-benchmark-41281557441.json) · [Predeclared protocol](roformer-vocal-protocol.md).
Source commit: `086266a8acd75323c344157ee723915e3b39a4e9`.

This is an independent **native two-source** comparison of official Spleeter-2
and the pinned Kim Mel-Band RoFormer checkpoint. It does not reuse scores from
the earlier Spleeter-4/HTDemucs experiment. The same ten complete MUSDB18
**AAC-derived** validation songs total 30.37131 minutes. All 50 test songs
remain unused. The project split is drawn from the training pool; overlap with
upstream pretrained training data remains unresolved. These are measured results
for this screen, not an HQ or unbiased held-out leaderboard.

## Separation quality

BSSEval columns are median one-second frames per song, then median songs.
Paired Δ is the median of within-song candidate-minus-baseline changes;
it is not subtraction of the aggregate medians. SI-SDR is a separate global
stereo metric. Scores use unscaled, full-length audio.

### BSSEval-v4 SDR (dB)

| Source | Native Spleeter-2 | Kim Mel-Band RoFormer | Median paired Δ | Better / worse / equal |
|---|---:|---:|---:|---:|
| vocals | 7.545 | 11.897 | +5.030 | 10 / 0 / 0 |
| accompaniment | 13.549 | 19.209 | +5.218 | 10 / 0 / 0 |

### Global stereo SI-SDR (dB)

| Source | Native Spleeter-2 | Kim Mel-Band RoFormer | Median paired Δ | Better / worse / equal |
|---|---:|---:|---:|---:|
| vocals | 7.463 | 11.862 | +5.543 | 10 / 0 / 0 |
| accompaniment | 13.234 | 18.059 | +5.141 | 10 / 0 / 0 |

Reference accompaniment is the fixed float32 sum **drums + bass + other**.
The AAC mixture need not equal the independently encoded source sum, so the
reference is not mixture minus reference vocals. Spleeter predicts both native
masks; RoFormer predicts vocals and forms accompaniment as mixture minus vocals.

`museval==0.4.1` BSSEval-v4 uses source images, fixed 512-tap filters, no
permutation, one-second windows/hops, no padding and strict alignment.
Undefined/infinite metrics are retained explicitly, never replaced by zeros.

| Source | Spleeter finite / undefined SDR frames | RoFormer finite / undefined SDR frames |
|---|---:|---:|
| vocals | 1381 / 436 (of 1817) | 1381 / 436 (of 1817) |
| accompaniment | 1381 / 436 (of 1817) | 1381 / 436 (of 1817) |

The JSON also includes common-finite-frame SDR, per-song scores, paired
changes and complete finite/undefined/infinite counts. This establishes a new
two-source baseline; the historical four-source 0.001 dB gate does not apply.

## Speed, memory and modeled inference cost

Each model ran in a fresh sequential process on the same PRO 6000 Blackwell
MIG 1g.24gb, with three complete repeats per song. Times sum each song's median
repeat. Inference includes host preprocessing, synchronized CUDA computation
and CPU outputs; local end-to-end adds WAV decoding and FLOAT WAV export.
Both exclude setup, loading, warmup and scoring. Raw repeats and per-song
min/median/max measurements are retained in the JSON.

| Measurement | Native Spleeter-2 | Kim Mel-Band RoFormer |
|---|---:|---:|
| Inference seconds | 5.314 | 295.851 |
| Local WAV end-to-end seconds | 9.252 | 300.236 |
| Inference RTF | 0.00292 | 0.16235 |
| End-to-end RTF | 0.00508 | 0.16476 |
| Inference × realtime | 342.90 | 6.16 |
| End-to-end × realtime | 196.95 | 6.07 |
| Peak allocated CUDA GiB | 3.663 | 4.297 |
| Peak reserved CUDA GiB | 4.662 | 4.477 |
| Process VmHWM GiB | 1.846 | 2.433 |
| Inference USD / audio hour | 0.00201 | 0.11202 |
| End-to-end USD / audio hour | 0.00350 | 0.11368 |

RoFormer takes **55.67×** the inference time and **32.45×** the local
end-to-end time of native Spleeter-2 on this workload. This measures the
declared FP32 eager recipe, not the fastest possible mixed-precision or fused
attention implementation. No AMP, flash attention or silent chunk changes.

CUDA peaks include resident model weights and are reset for each repeat.
Reserved memory is allocator reservation, not additional allocated memory.
CPU memory now uses Linux `/proc/self/status` VmHWM for each fresh process,
avoiding the inherited `ru_maxrss` floor found in the four-source run.
VmHWM includes imports, loading, state hashing and inference; it is not
incremental model-only memory. MIG sibling activity and cache state were
uncontrolled. These are one-host observations, not latency distributions.

Modeled cost multiplies RTF by the actual **$0.69/hour** quote.
It describes steady-state compute occupancy, not a hosted-service invoice.
Setup, idle time, storage, requests and other infrastructure charges are extra.

| Startup measurement (seconds) | Native Spleeter-2 | Kim Mel-Band RoFormer |
|---|---:|---:|
| import_seconds | 1.157 | 0.734 |
| load_seconds | 0.281 | 3.012 |
| warmup_seconds | 0.199 | 1.424 |

RoFormer load timing includes checksum verification and dynamic implementation
imports. These are one observed startup per model, not cold-start distributions.

## Configuration and verification

- Native official Spleeter-2: 512 frames, 1024 bins, zero high-band extension.
- Kim checkpoint: 228,202,852 parameters; pinned MSST source and original-author
  configuration, explicitly `zero_dc=False`; eight-second chunks, overlap two,
  batch one, original reflect/fade stitching and residual accompaniment.
- Both FP32, TF32 disabled, deterministic algorithms, seed 17, two CPU threads,
  fixed eight-second warmup, no extra normalization, gain, clipping or TTA.
- Input, weight, source and protocol checksums; unchanged model state; exact
  finite full-song outputs; independent selection, timing and summary checks.
- Locally revalidated raw metric summaries, export and member hashes, timing
  cost calculations, and all 14 finite stereo FLOAT listening clips. This is
  not a second independent full-audio scoring implementation or subjective test.

| Artifact | SHA-256 |
|---|---|
| Converted official Spleeter-2 | `2032ba2a17765f80e02051ecd0b7a5dd1552a077221275615227c2652f873532` |
| Kim checkpoint | `87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e` |
| Protocol | `540557f24ec32ef7c1ad3e9b05715baa6c531a45ec46e6aadff5f27380eaaf3e` |
| Full export bundle | `45cc5c822ee213a092e6b7701c70e15d8359d2d1e41bcbf0807d800c7020314d` |

## Resources, total cost and outputs

The deleted regional volume was not needed. This run reused the existing
Global Volume and verified 50 validation WAVs plus provenance. It created no
new persistent volume and did not restage the full training pool. Native
Spleeter-2 weights were recreated and matched the CPU preflight fingerprint.

Controller creation-to-cleanup span: **1532.73 seconds**
(25.55 minutes), within the 60-minute lease.
Validation staging took 40.43 seconds; final CPU scoring took
317.01 seconds while the Pod remained allocated. The benchmark worker
took 1403.46 seconds; the remaining controller time includes environment
setup, control/transfer and cleanup rather than model inference.
Estimated compute for this one attempt: **$0.29377**.
This duration-based estimate excludes storage, requests and Actions charges.
The controller confirmed the created Pod had disappeared after deletion;
the workflow also completed its final cleanup. No automatic paid retry.

Full raw metrics and clips are retained at `/global/benchmarks/41281557441-1/`
with checksummed report/export and a completion marker. The Actions artifact
has 14-day retention. Repository JSON contains aggregates, per-song metrics,
all timing repeats, configuration and checksums, without licensed music audio.

The listening ZIP contains 14 stereo 44.1 kHz FLOAT WAVs: centered eight-second
excerpts of the first two sorted songs, mixture/source references and concealed
A/B model labels, with one shared gain per song and a 0.95 peak bound.
`LISTENING.txt` retains dataset/license attribution; `ANSWERS.json` reveals
model labels. No subjective listening verdict is claimed.

Converted official Spleeter weights remain this port's reproducible compatibility
baseline. The frozen-BN pilot does not justify replacing them. Use this separate
vocal screen to assess the quality/resource tradeoff and audition the clips;
do not relabel AAC as HQ or tune on the untouched 50-song test pool.
