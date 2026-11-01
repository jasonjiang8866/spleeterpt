# Bounded Spleeter-4 longer-training screen

Prepared 2026-11-01 (SGT). This extends the matched frozen-BN 90-step pilot;
it is **official-initialized fine-tuning**, not training from scratch or a new
waveform-domain objective. A successful run proves the declared training and
measurement pipeline completed. It does **not** by itself prove convergence or
better separation. Keep converted official weights as the inference baseline.

## Fixed experiment

| Item | Predeclared setting |
|---|---|
| Model / initialization | Native Spleeter-4, converted official checkpoint SHA-256 `b95de4f9460593db5600840d143e0de0fd06685cbe089050ecfc652f73e804d3` |
| Data | Existing MUSDB18 **AAC** training pool; 90 train / 10 validation songs; not HQ |
| Held-out pool | All 50 test songs remain unextracted and unused |
| Arms | Independent LR `1e-5` and `3e-6`, same initialization and seed 17 |
| Updates | 10 epochs × 90 songs × 1 crop, batch 1 = 900 updates per arm |
| Crops | 527,360 samples (11.958 s), deterministic new offset each epoch; record starts and union coverage |
| BN | Official running buffers frozen; affine parameters train; dropout remains active |
| Optimization | Existing summed per-stem magnitude L1, Adam eps=1e-8, clip norm 5, FP32, TF32 off, deterministic |
| Epoch validation | Four fixed crops per validation song; unlike the earlier one-crop pilot, losses are not directly comparable |
| Milestones | 90 / 450 / 900; complete epoch validation before checkpointing |
| Quality curve | Global stereo SI-SDR on all ten complete songs at baseline and every milestone |
| Full scoring | BSSEval-v4 at official baseline and both fixed step-900 endpoints |
| Selection | Fixed endpoints; do not choose or promote a checkpoint using training loss |

All four stems use the same full-length inference and scoring recipe as previous
four-stem evaluations. Reproduce official baseline mean/median to 0.001 dB with
unchanged finite/undefined coverage. Summaries include per-song paired changes,
separate SI-SDR, and common-finite-frame SDR. Upstream pretrained-data overlap
remains unresolved; this is a small, repeatedly used validation screen. One seed
and ten epochs cannot establish robust generalization or convergence.

Training covers changing excerpts, not every sample of each recording. Recorded
crop intervals show the exact loader offsets and union coverage; they are not
an execution trace of minibatch shuffle order. Final RNG states and all data/
recipe signatures except learning rate must match between arms.

## Persistence and recovery

The deleted regional volume is neither required nor recreated. Mount only existing
Global Volume `cmvwmviz9000007jteci4ajn8` (`spt-1436489970-step90-inputs-v1`).
Verify `/global/musdb18-training-v1/snapshot.json` against SHA-256
`172c50e5310b22ceab3d4813a4ba1146c682db682c8aebc4c03ad235de376964`,
then verify all 500 WAVs and both manifests before training. Stage the 37.57 GiB
snapshot onto an 80 GB local Pod disk. Train/checkpoint on the local filesystem.

At each milestone publish `last.pt`, config, arguments, cumulative metrics and
phase timing to `/global/convergence-runs/<run>-<attempt>/<arm>/<step>/`.
Every file is hashed and read back; write `complete.json` last and refuse overwrite.
Six immutable exports are bounded to 512 MiB each (at most 3 GiB total retained
storage; storage/request fees are additional). Incomplete directories have no
completion marker and must not be used as recoverable checkpoints.

At steps 90 and 450, read `last.pt` from Global into a fresh local directory,
compare the complete training state, and use that file with `--resume` in a **new
Python process**. Preserve optimizer, CPU/CUDA RNG, epoch/batch cursors and best
validation scalar. This verifies Global round-trip and process restart, not a
replacement-Pod recovery. Local `best.pt` is a training convenience and is **not**
promised as a durable exported model; fixed milestone `last.pt` files are durable.

## Resource and budget controls

The no-GPU [preflight](../.github/workflows/convergence-preflight.yml) observes
existing Pods/volumes, MIG catalog availability and account budget. Repository
provenance is not a mounted-file check; that check necessarily occurs in the Pod.
The [manual workflow](../.github/workflows/runpod-convergence.yml) runs only on
`main`, under the shared RunPod concurrency group, with explicit paid dispatch.

- One PRO 6000 Blackwell MIG 1g.24gb, quoted rate at most $0.69/hour.
- Existing $500/month account budget, $1 reserve and current balance/headroom check.
- One 60-minute lease including startup/install/staging/scoring, no paid retries,
  no GPU upgrade and no new volume. Actual device/rate checked immediately.
- After the measured baseline scoring and first 90-step phase, project remaining
  training/scoring with 25% margin and 240 seconds for curves/I/O/export. Reject
  if it cannot fit the remaining lease; recheck before final scoring.
- Worker terminates child process groups before expiry. Controller deadline,
  `finally` cleanup and workflow `always()` cleanup delete compute only; verify
  Pod disappearance. Cleanup applies to failed setup or validation as well.
- A rejected projection/failure retains partial report and completed Global
  milestones; it does not trigger a second paid attempt. The next action is to
  diagnose the retained evidence, not label the experiment completed.

Independent runner acceptance checks all 1,800 finite updates, twenty epoch
validation records, six checkpoints/cursors and hashes, frozen BN, changed affine/
other parameters, exact resume links, data/recipe matching, complete quality
curves, baseline reproduction and export integrity. Training timing includes
checkpoint writes and crop validation; full-song quality/scoring time is separate.
The listening export contains fixed centered eight-second excerpts for two songs,
with one shared gain per song, plus full records. It carries no listening verdict.

## Interpreting the outcome

Report training loss, validation loss, per-stem SI-SDR curves, final paired SDR /
SI-SDR changes, duration/peak CUDA allocation, storage and measured compute cost.
A decreasing loss with worsening separation is a negative result. An improvement
on this screen is a candidate for replication, not automatic replacement of the
official baseline. Freeze a validated recipe before any future final test.

CPU tests cover multi-epoch fresh-process resume versus uninterrupted training,
all checkpoint/optimizer/RNG state and metrics, crop offsets/coverage, the actual
spectral loss and gradients, corrupt Global restores, incomplete reports and
single-launch cleanup. [Run 41290811204](convergence-pilot-41290811204.md) completed training and scoring,
but failed post-run acceptance due to an erroneous strict-positive loss/gradient
rule. Retained metrics pass the corrected finite-nonnegative audit; the original
failed status and missing listening export remain explicit. Both endpoints show
negative paired quality medians. No paid retry was launched.
