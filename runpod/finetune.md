# Full-song fine-tuning validation

The manually dispatched **RunPod fine-tune validation** workflow compares the
official converted 4-stem initialization against the exact step-90 / epoch-1
checkpoint from experiment `41189465452-1`. It performs inference only and leaves
training checkpoints unchanged. GPU evaluation completed successfully in [run 41196010747](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41196010747).
See [measured results and listening evidence](../docs/step90-validation-41196010747.md).

## Frozen protocol

- All ten songs in the existing seed-17 project validation split, complete audio
  (~30.37 minutes), sorted by ID. This is not the official 50-song test benchmark.
- Both models use the same full 4-stem configuration, FP32 CUDA inference, disabled
  TF32 and deterministic algorithms. The baseline file SHA-256 must equal the
  initialization used for training; candidate generation and every member hash
  must match the previously verified persistent-pilot record.
- Recheck all 500 dataset file hashes, then each selected input immediately before
  use. Decode fingerprints come from the exact cloud `provenance.json` whose SHA
  was recorded during training. Original archive/source hashes, song IDs and sample
  counts must also match the independently recorded release inventory. Local and
  cloud WAV container fingerprints are not assumed interchangeable. No training,
  crop selection, source permutation or output length repair.
- `museval==0.4.1` BSSEval-v4 source-image metrics: time-invariant 512-tap filters,
  one-second windows/hops, no padding. Incomplete final windows are omitted by
  museval. Retain every window, including explicitly undefined values. Report
  median window SDR per song, then median songs per stem, plus the median of
  **paired per-song candidate-minus-baseline differences** and win/loss counts.
  A difference of aggregate medians is not the median paired difference.
- Separately report global stereo SI-SDR, with global mean removal and one scalar
  projection; do not compare it to published BSSEval SDR.
- Listening samples are fixed before scores: first two sorted validation IDs,
  eight seconds at each recording's center, taken from full-song inference.
  Include mixture, four ground-truth stems and both sets of four estimates. One
  shared gain per song prevents clipping without independently normalizing stems.
  Metric computation always uses the original unscaled full audio.

## Execution and outputs

One RTX 4090, existing 80 GB volume `j3aqjpw393`, a **60-minute evaluation lease** and
$1/hour actual compute-rate guard. All GPU workflows share one concurrency group.
The worker ends three minutes ahead of the lease to leave transfer/cleanup time; incomplete
selection fails acceptance rather than quietly reporting a smaller benchmark.
Cleanup deletes compute only. No new storage volume is created.
All other GPU tasks retain their 30-minute leases. At the rate guard, this evaluation
can cost up to approximately $1 compute, excluding delayed cleanup and billing rounding.
Completed per-song JSON and fixed listening excerpts are written under
`/data/evaluations/41189465452-1/<pod-id>` so compute deletion does not discard them.
Partial folders are not accepted as a complete benchmark and are not automatically resumed.

The Actions artifact includes `result.json`, `provision.json`, and
`listening-and-metrics.tar` (under 100 MiB). The runner recomputes summary statistics
and validates bundle/member hashes before deleting the Pod. Results are retained
for 14 days. No training weights or complete songs are exported. Excerpts retain
their original [MUSDB18 source-specific educational-use terms](https://zenodo.org/records/1117372);
they do not inherit this code repository's license. `LISTENING.txt` includes
source attribution and playback guidance.

This ten-song validation set can guide model development, but it is too small to
establish general superiority. Original pretrained data overlap is not verified.
No listening verdict is inferred from numerical scores. Keep the 50-song test
pool untouched until model selection and the final evaluation protocol are fixed.

The [first attempt](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41193783059)
passed verification of all 500 persistent files and reconstructed the exact official
initialization, then failed before scoring because the per-song check referenced
local-preparation WAV hashes. The Pod was deleted. The evaluator now carries and
verifies the original cloud provenance text, using the same prepared files as
training without relaxing source, sample-count or checksum checks.

The [second attempt](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41194464761)
passed the corrected data contract and completed two full-song comparisons. Its
observed CPU scoring speed risked exceeding the 27-minute worker window, so it was
stopped before spending the entire lease on an incomplete run. The 60-minute
evaluation window and persistent per-song outputs address that operational limit;
they do not change the models, selected audio or metric computation.
