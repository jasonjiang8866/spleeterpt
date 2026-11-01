# Spleeter-4 versus HTDemucs: preparation and predeclared screen

Prepared 2026-11-01 (Asia/Singapore), from `fd7d8a844a28cca53f04ec94ee4f9f8123812d29`.
This document defines the next experiment. **No competitor music-quality result
or paid GPU run is claimed.** RoFormer vocal/accompaniment evaluation remains a
separate task. Official converted Spleeter weights remain the baseline.

## Inputs and provenance

Use exactly the ten complete songs in
[`musdb18-training-provenance.json`](results/musdb18-training-provenance.json),
`split_track_ids.validation`: 30.371313681 minutes, 44.1 kHz stereo, 50 WAVs.
These are AAC-derived MUSDB18 files, not MUSDB18-HQ. Do not extract or evaluate
any of the 50 test songs while selecting settings.

The project-specific 90/10 split is **not a common held-out split for pretrained
competitors**. HTDemucs was trained on MUSDB-HQ plus external songs; do not assume
these ten MUSDB training-pool songs were excluded from its training. Spleeter's
private training corpus also prevents a complete overlap audit. This is an
operational/quality screen on these recordings, not an unbiased generalization
leaderboard. A future fixed 50-song evaluation must retain the AAC designation
unless genuine HQ files are independently acquired and verified.

| Input | Existing evidence | Required before inference |
|---|---|---|
| Regional source | `j3aqjpw393`, EU-RO-1, 80 GB STANDARD | Observe identity; no migration or deletion |
| Global Volume | Named `spt-1436489970-step90-inputs-v1` | Resolve unique current ID with GET; reuse it |
| Validation snapshot | `/global/step90-validation-v1`; prior 58-file export includes historical pilot checkpoints | Verify completion marker, pinned provenance and all 50 selected audio hashes; stage only needed inputs to local POSIX disk |
| Full training snapshot | `/global/musdb18-training-v1`, 503 files / 40,342,607,284 bytes in prior report | Do not restage full training pool for this experiment |
| Spleeter baseline | Converted official `4stems.tar.gz`; SHA-256 `b95de4f9460593db5600840d143e0de0fd06685cbe089050ecfc652f73e804d3` | Verify a cached copy or recreate and match hash; retained trained checkpoints are not substitutes |
| HTDemucs | Single signature `955717e8`; official filename `955717e8-8726e21a.th` | Download from official host, verify upstream checksum prefix, lock full SHA-256 and run CPU probe before GPU launch |

The old worker recreated official initialization on local Pod storage; the
repository does **not** establish that the converted official file is cached on
Global Volume. Do not infer cache existence from retained fine-tuned weights.
Repository evidence and a live API volume listing do not prove current object
integrity: mounted read-back checks are a separate launch gate.

## Model and environment recipe

Pin Demucs to
[`e976d93ecc3865e5757426930257e200846a520a`](https://github.com/facebookresearch/demucs/tree/e976d93ecc3865e5757426930257e200846a520a).
Its [`htdemucs.yaml`](https://github.com/facebookresearch/demucs/blob/e976d93ecc3865e5757426930257e200846a520a/demucs/remote/htdemucs.yaml)
contains one model. Do not substitute `htdemucs_ft`, an ensemble, `mdx_extra` or
`mdx_extra_q`. Record installed source commit, full weight hashes, configuration
hashes, all package versions and actual runtime/device details in the result.

| Setting | Spleeter-4 | HTDemucs |
|---|---|---|
| Precision | FP32; no autocast, TF32 disabled | Same |
| Device | One PRO 6000 Blackwell MIG 1g.24gb | Same physical Pod/profile, sequential execution |
| Output | vocals/drums/bass/other, 44.1 kHz stereo FLOAT WAV | Explicitly map drums/bass/other/vocals to the same order |
| Windowing | Official preset: 512 frames, 1024 frequency bins | Split enabled, checkpoint-native segment (record numeric value), overlap 0.25 |
| Shifts | None | `shifts=0` for the first controlled screen; label this non-default recipe |
| Inference preprocessing | Existing port behavior | Preserve upstream reversible mixture normalization and denormalization |
| Export | No clipping or stem-specific gain | Same; disable CLI's default rescaling |
| Model count | One four-stem model | One signature; no fine-tuned bag |
| Parallelism | One model/process, one song at a time; CPU threads 2 | Same; `num_workers=0` |

Do not impose identical window lengths on architectures with different native
constraints. Seed Python/NumPy/Torch to 17; enforce deterministic algorithms and
record exceptions as blockers, not silent fallbacks. Optional recommended settings
(e.g. one seeded shift, AMP) require a separately named experiment; do not mix them
with the controlled FP32 result.

Keep the previous Torch `2.8.0+cu128` runtime. Upstream Demucs metadata constrains
Torchaudio below 2.1, which conflicts with this runtime. The preparation workflow
therefore installs the pinned Demucs commit with `--no-deps` into an isolated
Torch/Torchaudio 2.8 CPU environment and explicitly installs its support libraries.
`scripts/probe_htdemucs.py` checks actual model loading and finite output. A CPU
pass establishes API compatibility only; it does not establish Blackwell support
or performance. The official checkpoint contains a serialized Python class;
explicit `weights_only=False` is confined to that checksum-checked artifact.

## Measurements and acceptance

1. First run a fixed one-song GPU smoke on the first sorted validation ID. Verify
   source mapping, exact length, finite outputs, device identity and memory headroom.
   Abort on a mismatch. This song is included again in the complete ten-song pass.
2. Use a fresh subprocess per model. Record process startup/import separately from
   weight loading (CPU read, deserialize, device transfer). Download and staging are
   setup costs, not model-loading latency. Do not claim filesystem cold-cache timing
   unless cache state is actually controlled.
3. Warm up with a fixed excerpt, synchronize CUDA before/after inference, then
   reset peak allocated/reserved counters. Run three complete inference repetitions
   per song; reseed each repetition, keep first-pass outputs for quality, report
   median and range of timings. Do not retain all repetitions' outputs in GPU memory.
4. Measure inference separately from end-to-end local WAV read + preprocessing +
   inference + transfer + FLOAT WAV export. Both use identical files and export
   format. Score only after timing, in a CPU process with no GPU models loaded.
5. Report seconds, audio duration, RTF (`seconds / audio_seconds`), throughput,
   per-song and corpus peak allocated/reserved VRAM, and per-process peak CPU RSS.
   Include absolute resident-model memory in peaks; do not subtract it away. Record
   MIG identity, visible VRAM, host CPU, driver, image and competing processes.
6. Use the existing `museval==0.4.1` BSSEval-v4 source-image implementation: fixed
   512-tap filters, no permutation, one-second windows/hops, no padding. Separately
   report global stereo SI-SDR (mean removed, one scale). Strict sample alignment;
   no reference-assisted shifting, gain fitting beyond the metric, or truncation.
7. Store per-song/per-stem results and invalid/silent-frame counts. Aggregate median
   frames per song, then median songs per stem; report median **within-song**
   HTDemucs-minus-Spleeter differences and win/loss counts. Also report coverage on
   paired valid frames if coverage differs; do not silently drop difficult songs.
8. Require official baseline mean/median reproduction within the existing 0.001 dB
   MIG gate. Freeze hashes/config before judging candidate results. Export the same
   centered eight-second excerpts from the first two sorted songs, with one shared
   gain per song and concealed model labels; no listening verdict without listening.

## Proposed bounded run and current readiness

Live recheck at **2026-11-01 11:32 SGT**: the existing read-only authentication
job [attempt 2](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41228859225/attempts/2)
authenticated to the Pod API, then failed with `Run persistent-prepare first`:
no regional volume matched the expected name. Its source is
`a9c436e11c59c7e7f92b399e3a5bb0c5bf564b51`; the relevant controller is unchanged
from the current base. This does not establish deletion or absence of the Global
Volume. It also prevented that job from reaching the GPU catalog query. The new
preflight records a missing regional source and still checks Global Volume and
MIG independently. Do not recreate the regional volume based on this old error.

The new [GET-only inventory](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41271997588)
completed at **2026-11-01 12:10 SGT**. It found **zero account Pods** and the existing
Global Volume `cmvwmviz9000007jteci4ajn8` with the expected name. The expected regional
volume was absent. MIG 1g.24gb catalog price was **$0.69/hour**, with `LOW`
availability in `EUR-IS-2`, `US-PA-1` and `US-WA-1`. These are catalog observations,
not a reserved allocation or launch quote. Full
[resource evidence](results/htdemucs-resource-preflight.json) is retained here;
mounted object checks, account credit and current-month spending remain unverified.
No compute or storage was created by this inventory. The resource report lists
generic launch gates; separate CPU probe evidence below resolves its CPU-only gate.

First paid trial proposal: **one MIG Pod, at most $0.69/hour, 60 minutes from
creation**, including staging and setup. Compute ceiling at that rate: **$0.69**;
reserve **$1 total for the trial** to allow container/storage/request overhead,
which must be quoted separately. This is a proposed allocation, not a verified
all-in billing cap or a new recurring budget. The user's $500/month lab budget
does not establish remaining account credit or month-to-date spend.

Reuse existing Global Volume; create no new persistent volume and do not recopy
the full training set. Report existing regional storage (~$5.60/month historically)
separately from incremental trial cost; that regional charge is historical and must
not be treated as current after the missing-volume observation. Global usage/request fees remain separately
billable and must be checked. Do not label historical catalog prices as a live quote.

Before launch, implement and test a creation-based deadline, `finally` cleanup,
independent cleanup/reaper, result validation and immutable result export. Retain
volumes; delete only the newly created compute. No automatic paid retry, GPU
upgrade, second Pod or over-budget continuation. If the one-song timing projects
an overrun, stop with the partial evidence and revise the budget explicitly.

`runpod/benchmark_preflight.py` and its workflow perform GET-only observations;
they cannot create compute or storage. They deliberately report
`ready_for_paid_run=false`: API inventory, mounted-byte verification, credit,
CPU/GPU compatibility and a complete benchmark worker are distinct checks.
This preparation PR does not yet implement the paid benchmark worker or launch it.

## Preparation validation

The local isolated **Python 3.12.14 / Torch 2.8.0+cpu / Torchaudio 2.8.0+cpu**
probe passed with the real official model: 41,984,456 parameters, native 7.8-second
segment, 44.1 kHz stereo, finite output shaped `[1, 4, 2, 44100]` on synthetic input.
The downloaded file is 84,141,911 bytes, SHA-256
`8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4`.
The full hash is now enforced by the probe. See
[local probe evidence](results/htdemucs-cpu-preflight.json).
The [workflow's Python 3.11.17 probe](results/htdemucs-ci-probe.json) also passed
with the same checkpoint SHA-256, parameter count, segment length and finite
four-stem output. The CUDA probe has not run.

All 50 RunPod offline tests pass, including read-only mutation rejection,
credential omission, invalid price rejection and continued Global inventory when
the regional volume is absent. New Python files pass Ruff 0.14.2. That older local
Ruff reports three pre-existing findings in unchanged files; the CI environment
uses the repository's declared dependency range. The resource and real-model
probe jobs are successful in run `41271997588`; standard test results remain
available on the PR checks page.
