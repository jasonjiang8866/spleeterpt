# Spleeter-4 versus HTDemucs — measured validation screen

Completed 2026-11-01 (Asia/Singapore).
HTDemucs improves both quality metrics on every song/stem in this screen and
uses less allocated GPU memory; Spleeter runs 4.53× faster in inference. Keep
official Spleeter weights as the port's compatibility baseline and evaluate the
quality/speed tradeoff for the intended workflow.

[Successful run](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41277234620) ·
[Machine-readable measurements](results/separator-benchmark-41277234620.json) ·
[Predeclared protocol](spleeter4-htdemucs-protocol.md).
Source commit: `04353111e774afc75a43b1ec2d4a550ecdc9fa70`.

This comparison covers the same ten complete MUSDB18 **AAC-derived** validation
songs (30.37131 minutes) on one PRO 6000 Blackwell MIG 1g.24gb. It is a measured
screen for these recordings. The project 90/10 split comes from the training pool;
overlap with upstream pretrained training data is unresolved. It is **not an HQ
or unbiased held-out leaderboard**. All 50 test songs remain unused.

## Separation quality

Model columns are medians across songs, after taking the median of one-second
BSSEval frames within each song. Paired Δ is the median of within-song
HTDemucs-minus-Spleeter changes, not subtraction of the aggregate medians.
Better/worse/equal counts compare HTDemucs to Spleeter over ten songs.

### BSSEval-v4 SDR (dB)

| Stem | Spleeter official | HTDemucs | Median paired Δ | Better / worse / equal |
|---|---:|---:|---:|---:|
| vocals | 7.446 | 9.267 | +2.352 | 10 / 0 / 0 |
| drums | 5.587 | 10.140 | +4.395 | 10 / 0 / 0 |
| bass | 4.645 | 10.594 | +5.152 | 10 / 0 / 0 |
| other | 4.382 | 7.921 | +3.319 | 10 / 0 / 0 |

### Global stereo SI-SDR (dB)

| Stem | Spleeter official | HTDemucs | Median paired Δ | Better / worse / equal |
|---|---:|---:|---:|---:|
| vocals | 7.434 | 9.017 | +2.968 | 10 / 0 / 0 |
| drums | 4.242 | 9.415 | +4.993 | 10 / 0 / 0 |
| bass | 2.908 | 10.348 | +6.569 | 10 / 0 / 0 |
| other | 3.093 | 7.027 | +3.824 | 10 / 0 / 0 |

These are separate metrics. The BSSEval source-image implementation uses
`museval==0.4.1`, fixed 512-tap filters, no permutation, one-second window/hop,
no padding and strict full-song alignment. SI-SDR uses mean removal and one
global scale across stereo. Scores use unscaled full-length exports.

| Stem | Spleeter finite / undefined SDR frames | HTDemucs finite / undefined SDR frames |
|---|---:|---:|
| vocals | 1351 / 466 (of 1817) | 1351 / 466 (of 1817) |
| drums | 1351 / 466 (of 1817) | 1351 / 466 (of 1817) |
| bass | 1351 / 466 (of 1817) | 1351 / 466 (of 1817) |
| other | 1351 / 466 (of 1817) | 1351 / 466 (of 1817) |

The JSON retains finite/undefined/infinite counts, paired common-finite-frame
SDR summaries and per-song metrics; missing frames are not replaced with zeros.
Official baseline aggregate means/medians reproduce the historical evaluation
within **0.000001348 dB**,
passing the predeclared 0.001 dB gate.

## Speed, memory and modeled inference cost

Three complete repeats per song, one fresh sequential process per model. Times
below sum each song's median across those repeats. Raw repeats and per-song
min/median/max values are in the JSON. Inference covers host waveform preprocessing,
CUDA computation and CPU outputs with synchronization. Local end-to-end adds
WAV decoding and FLOAT WAV export. Setup, import, weight loading, warmup and
quality scoring are excluded from both throughput measurements.

| Measurement | Spleeter official | HTDemucs |
|---|---:|---:|
| Inference seconds | 10.532 | 47.701 |
| Local WAV end-to-end seconds | 17.626 | 54.417 |
| Inference RTF | 0.00578 | 0.02618 |
| End-to-end RTF | 0.00967 | 0.02986 |
| Inference × realtime | 173.02 | 38.20 |
| End-to-end × realtime | 103.39 | 33.49 |
| Peak allocated CUDA GiB | 6.926 | 0.560 |
| Peak reserved CUDA GiB | 7.912 | 0.803 |
| Raw `ru_maxrss` GiB (not isolated CPU memory) | 7.360 | 7.360 |
| Inference USD / audio hour | 0.00399 | 0.01806 |
| End-to-end USD / audio hour | 0.00667 | 0.02060 |

HTDemucs takes **4.53×** the inference
time and **3.09×** the local end-to-end
time of this Spleeter port on this workload. Peak CUDA allocation includes the
resident model; it is not an estimate of minimum device capacity. Reserved memory
is allocator reservation, not additional allocated memory to add to the first row.
The physical GPU's sibling tenant activity was not controlled.

**CPU RSS limitation:** both subprocesses report exactly 7,902,781,440 bytes via
`getrusage(RUSAGE_SELF).ru_maxrss`. This counter can retain a launch-time parent
memory floor across exec; a local reproduction returned a 139,520 KiB counter
despite the child's own `/proc/self/status` high-water mark being only 6,500 KiB.
Do not interpret this row as equal or isolated CPU memory consumption. Raw values
are retained for audit. A future run should use process-specific `VmHWM` or an
external sampler. CUDA allocated/reserved counters are independently reset in
each model process; this issue does not affect the GPU memory comparison.

Per-audio-hour prices multiply measured RTF by the actual **$0.69/hour**
Pod quote. They describe steady-state compute occupancy, not an invoice or a
complete hosted-service cost. Short jobs also pay setup/loading/idle time;
storage, requests and other infrastructure charges are additional.

| Measurement (seconds) | Spleeter official | HTDemucs |
|---|---:|---:|
| import_seconds | 1.197 | 1.017 |
| load_seconds | 0.415 | 0.566 |
| warmup_seconds | 0.226 | 1.282 |

Import timing covers dependencies inside the runner, excluding interpreter and
shared protocol imports. Weight loading includes reading/deserialization/device
transfer; filesystem cache state was not controlled. These are one observed
startup per model, not a cold-start distribution.

## Controlled configuration and validation

- Converted official Spleeter-4, 512 frames, 1024 bins, zero high-band extension.
- Single HTDemucs signature `955717e8`, source
  `e976d93ecc3865e5757426930257e200846a520a`, segment 7.8 seconds,
  overlap 0.25, split enabled, **shifts=0**, workers=0, upstream reversible
  mono-reference normalization. This is not the fine-tuned ensemble.
- FP32, TF32 disabled, deterministic algorithms, seed 17, two CPU threads,
  one GPU process at a time, fixed eight-second warmup. Model state remained unchanged.
- PyTorch 2.8.0+cu128 / CUDA 12.8; complete installed versions and actual device
  properties are retained in the JSON. Host CPU: AMD EPYC 9555 64-Core Processor.
- All 51 staged objects, both weight hashes, full-song identity/lengths,
  three repeats, paired summaries and exported archive/member hashes passed
  runner-side checks and independent local artifact validation. Local validation
  recomputed summaries from raw metric arrays; it did not rescore unavailable
  full audio or perform a subjective listening test.

| Artifact | SHA-256 |
|---|---|
| Official converted Spleeter-4 | `b95de4f9460593db5600840d143e0de0fd06685cbe089050ecfc652f73e804d3` |
| HTDemucs checkpoint | `8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4` |
| Protocol | `afe339eddd2aa9be47087c622e5936fdd95935329306d4b98ea1d8e61dd6144c` |
| Full export bundle | `d42b7144569f6af2953b5b7422da8281b0a87ca2bb0415c59ab5b51369927ae9` |

## Resources, total experiment cost and outputs

The old regional volume was deleted by the user. This run reused the existing
Global Volume and verified all 50 validation WAVs plus provenance; it created no
new persistent volume and did not restage the full training pool. Official
Spleeter initialization was recreated and hash checked.

Controller creation-to-cleanup span: **1601.31 seconds**
(26.69 minutes), within the 60-minute lease.
CPU scoring took 1022.99 seconds while the Pod was allocated.
Estimated compute for this successful run: **$0.30692**.
Including the earlier dependency-failure trial gives approximately
**$0.33352**. These duration-based
estimates exclude persistent/container storage, requests and Actions charges.
The controller confirmed deletion of this run's compute Pod; the workflow's
second cleanup also succeeded. No automatic paid retry was used.

Complete raw per-frame metrics and listening excerpts are retained at
`/global/benchmarks/41277234620-1/`, with checksummed export/report and a completion
marker. The Actions artifact has 14-day retention. Repository JSON contains
aggregate and per-song metrics, all timing repeats, runtime/configuration and
checksums, but no licensed music audio; raw metric arrays are in the full bundle.

The listening package contains 26 stereo 44.1 kHz FLOAT WAVs: the same centered
eight seconds of the first two sorted songs, references and concealed A/B model
labels, with one shared gain per song. All clip samples are finite and within
the declared 0.95 peak bound. `LISTENING.txt` retains dataset/license attribution;
`ANSWERS.json` reveals model labels. No subjective listening verdict is claimed.

## Next decision

Keep converted official Spleeter weights as this port's reproducible compatibility
baseline; the frozen-BN pilot does not justify replacing them. Use these paired
quality and resource measurements to choose whether HTDemucs fits a four-stem
workflow, then audition the fixed excerpts. A pinned RoFormer vocal/accompaniment
evaluation remains a separate experiment. Do not tune on the untouched 50-song
test set or relabel the AAC source as HQ.
