# Step-90 BatchNorm intervention

This inference-only diagnostic uses the same ten **complete** validation recordings
and pinned original/step-90 checkpoints as the [previous evaluation](../docs/step90-validation-41196010747.md).
It scores global stereo SI-SDR only, with the exact previous implementation.
BSSEval-v4 is deliberately not recomputed; the two metrics are not interchangeable.
All 50 test songs remain untouched. GPU results are pending.

Initial [launch attempt](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41217176398)
was rejected by the provider with HTTP 400 before a Pod ID was returned. Both cleanup
checks found zero managed Pods. No scoring occurred; the cause is not yet established.
A GET-only catalog check and bounded fixed-label error classification now aid diagnosis;
neither retries provisioning nor prints provider response text.

| Variant | Parameters, including BN gamma/beta | BN running buffers |
|---|---|---|
| official_converted | Official | Official |
| finetuned_step90 | Step 90 | Step 90 |
| learned_official_bn | Step 90 | Official |
| official_learned_bn | Official | Step 90 |

Only `running_mean`, `running_var`, and `num_batches_tracked` are exchanged.
All four models run in eval mode, with dropout disabled, FP32 and TF32 disabled.
The originals are not modified, and no checkpoint is saved or promoted.
Tensor-group SHA-256 fingerprints verify hybrid provenance; state equality checks
ensure no parameter/buffer changes during inference. The persistent checkpoint
is verified before and after the evaluation.

Report per-song scores, aggregate medians/means and finite/undefined counts,
paired contrasts for parameter-only and statistics-only changes, restoring official
statistics, and the parameter/statistics interaction (computed within each song).
Positive SI-SDR changes mean improvement. Effects need not add: the interaction
is `(tuned - official_learned_bn) - (learned_official_bn - official)`.
Do not infer additive attribution from differences of aggregate medians.

The two unmodified controls must reproduce the earlier aggregate SI-SDR means,
medians and coverage within 1e-5 dB, otherwise the run fails acceptance.
Missing songs, wrong lengths, missing variants, changed provenance, altered
summaries and swapped tensor groups are rejected by the Actions-side validator.
Per-song records persist during evaluation; incomplete runs are not accepted.
No new listening audio is exported in this metric-only diagnostic.

Run **RunPod BatchNorm ablation** from main with `confirm_paid_gpu` checked.
One existing-volume RTX 4090 lease, **30 minutes maximum**, **$1/hour rate guard**,
27-minute worker window and automatic compute deletion. Storage is retained at its
existing charge. Maximum intended GPU compute is approximately $0.50, excluding
cleanup delays or billing rounding. No new storage subscription is created.

Interpretation: recovery after restoring official statistics implicates the BN
buffer state in this checkpoint's regression; poor quality with official parameters
and learned statistics tests whether those buffers alone are sufficient to reproduce
it. These are checkpoint-level interventions, not proof of the training-time cause,
convergence, or general superiority. Training with frozen BN statistics is a separate
follow-up experiment, and any proposed fix requires full SDR and listening checks.
