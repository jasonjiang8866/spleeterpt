# Step-90 full-song validation — 2026-10-31

The step-90 fine-tune regresses substantially: all four stems score lower than official initialization on all ten validation songs, under both BSSEval-v4 SDR and global stereo SI-SDR. Do not promote this checkpoint as a quality improvement.

[Completed GPU run](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41196010747); source commit `7ef7287f66def88edb87f4e10eee60ebd7d214f1`.
The complete 10-song project validation split contains
30.37131 minutes of audio. This compares official converted
4-stem weights against the exact epoch-1 / step-90 checkpoint, using the same
PyTorch inference path. It is not a new TensorFlow oracle comparison or the
official 50-song MUSDB test benchmark. All 50 test songs remain untouched.

## BSSEval-v4 SDR (dB)

| Stem | Official initialization | Step 90 | Median paired Δ | Better / worse |
|---|---:|---:|---:|---:|
| vocals | 7.446 | 1.055 | -6.387 | 0 / 10 |
| drums | 5.587 | 0.606 | -4.695 | 0 / 10 |
| bass | 4.645 | 0.726 | -3.534 | 0 / 10 |
| other | 4.382 | 1.506 | -3.856 | 0 / 10 |

Each model column is the median across songs of each song's median one-second
SDR. Paired Δ is the median of candidate-minus-baseline differences computed
within each song; it is not the difference between the two aggregate columns.
Better/worse counts use finite paired song scores. Aggregate finite/undefined/infinity counts are retained in the published JSON
summary. Per-song and per-window values remain in the Actions/listening export.
Time-invariant 512-tap filters, no source permutation, one-second windows/hops,
no padding; incomplete final windows are omitted by museval 0.4.1.

## Global stereo SI-SDR (dB), separate metric

| Stem | Official initialization | Step 90 | Median paired Δ | Better / worse |
|---|---:|---:|---:|---:|
| vocals | 7.434 | -5.339 | -12.914 | 0 / 10 |
| drums | 4.242 | -6.154 | -10.088 | 0 / 10 |
| bass | 2.908 | -6.042 | -8.410 | 0 / 10 |
| other | 3.093 | -3.909 | -7.979 | 0 / 10 |

These are whole-recording stereo scores with mean removal and one scalar
projection. They are not interchangeable with BSSEval SDR or published results
using a different SI-SDR aggregation convention.

## Integrity and reproducibility

- All 500 prepared WAV files were verified against the exact training-time cloud
  provenance; original source hashes and sample counts match the independently
  pinned official archive inventory.
- The official initialization was reconstructed with the exact original file
  SHA-256 `b95de4f9460593db5600840d143e0de0fd06685cbe089050ecfc652f73e804d3`.
- Candidate: the previously verified, pinned step-90 backup generation.
  The one-epoch pilot used one crop per training song, 90 total updates.
- Both models use FP32, TF32 disabled, deterministic CUDA inference and the same
  full 4-stem preset. The run verified that all model parameters/buffers and the
  persistent training checkpoint generation remained unchanged.
- Every recording was scored from sample zero at its full decoded length, with
  exact output lengths; no selective song/crop removal or output-length repair.
- Actions recomputed the summary and verified all exported member hashes. A
  second local check repeated report/export validation and decoded all 26 WAVs:
  44.1 kHz, stereo, eight seconds, finite samples, peak at most 0.95.

## Listening and retained evidence

The first two sorted validation songs were selected before scoring. Each sample
contains the mixture, four references, four baseline stems and four candidate
stems. Eight-second excerpts come from the center of full-song inference, with
one shared attenuation per song. Metric computation uses unscaled full audio.
No subjective listening verdict is claimed.

The completed Actions run retains the detailed metric report and listening export.
This public summary contains aggregate scores and methodology only; it does not
include per-song records or infrastructure metadata. Audio excerpts are not
bundled in the repository.

Machine-readable aggregate summary: [results/step90-validation-41196010747.json](results/step90-validation-41196010747.json).
Full protocol: [RunPod validation](../runpod/finetune.md).

## Next experiment

Run a controlled BatchNorm-statistics ablation before extending training: compare the current candidate with the same learned parameters but restored official running statistics, and a baseline-parameter model using the candidate statistics. This can separate statistics drift from parameter-update effects; the current result alone does not establish the cause. Keep official converted weights as the inference baseline while investigating.

This small validation set is for model development. Pretrained-data overlap is
unknown; these results do not establish general superiority or convergence.
Keep the test split untouched until model selection and the final protocol are fixed.
