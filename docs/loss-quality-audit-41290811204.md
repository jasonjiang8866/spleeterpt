# Loss falls without consistent reconstructed-audio gains

Offline diagnosis, 2026-11-01. **Do not extend the current training recipe yet.**
Keep the converted official weights. This audit uses the retained
[900-step experiment](convergence-pilot-41290811204.md), checks the production
loss/reconstruction code, and runs a CPU synthetic counterexample. It launches
no GPU, reads no MUSDB audio or checkpoints, and leaves all 50 test songs unused.

The original worker status remains `failed`; its completed measurements were
audited separately. This analysis does not recover the missing listening archive.

## What the recorded measurements establish

Each arm has three checkpoints with both fixed-crop validation loss and full-song
SI-SDR: 90, 450 and 900. The loss averages 40 fixed crops (four per validation
song), while SI-SDR scores the ten complete songs. There is **no recorded official
step-0 crop loss**, no per-song/per-stem crop loss, and no waveform quality on those
exact crops. Thus we cannot calculate a same-example loss/quality relationship or
claim training beat the official baseline's loss. Training epoch averages also
use changing crops and active dropout, unlike evaluation.

| Learning rate | Loss at 90 | Loss at 450 | Loss at 900 | Relative loss change, 90 → 900 |
|---|---:|---:|---:|---:|
| 1e-5 | 1.220972 | 1.217194 | 1.217726 | −0.266% |
| 3e-6 | 1.221168 | 1.218359 | 1.215214 | −0.488% |

For each transition below, subtract **the same song's preceding-checkpoint score**
before taking the ten-song median. These values are not differences between
the earlier report's independently aggregated candidate-minus-official medians.
Those operations can disagree even in sign.

| LR / step transition | Crop loss Δ | Vocals SI-SDR Δ | Drums Δ | Bass Δ | Other Δ |
|---|---:|---:|---:|---:|---:|
| 1e-5, 90 → 450 | −0.003779 | −0.051179 | +0.005168 | −0.062844 | −0.037424 |
| 1e-5, 450 → 900 | +0.000532 | −0.026016 | −0.034070 | −0.018481 | +0.004972 |
| 3e-6, 90 → 450 | −0.002809 | −0.030048 | −0.016263 | −0.002894 | −0.021428 |
| 3e-6, 450 → 900 | −0.003145 | −0.000130 | −0.005682 | −0.040593 | −0.034235 |

Units for quality changes are dB. In the three transitions where loss falls,
**11 of 12 transition/stem paired medians fall**. This is descriptive counting,
not 12 independent trials; effects such as −0.000130 dB are tiny. No significance
test, fitted correlation, early-stopping optimum or causal claim is justified by
three dependent checkpoints, shared songs and two learning rates. Individual-song
improvements still occur. Complete paired values and counts are in the
[machine-readable audit](results/loss-quality-audit-41290811204.json).

Endpoint BSSEval-v4 SDR independently remains below official weights in all four
paired medians for both arms; it was not measured at 90/450 in this experiment.
The current evidence therefore gives no quality-based reason to buy more updates
of this unchanged recipe. It does not prove that longer training can never help.

## Why the objectives can disagree

The implementation at the audited main revision `f08dc2c8` has these paths:

| Stage | Actual operation | Consequence |
|---|---|---|
| [Model](../src/spleeterpt/model.py), four-stem preset | Per-stem sigmoid gain times mixture magnitude | Estimated magnitudes are bounded by mixture magnitude; cancellation can make real source magnitudes larger |
| [Training](../src/spleeterpt/train.py), `spectral_loss` | Sum of per-stem mean absolute errors against target magnitudes, modeled bins only | No ratio-mask normalization, complex reconstruction or waveform metric in the objective |
| [Inference](../src/spleeterpt/audio.py), `ratio_masks` / `separate` | Square estimated magnitudes, normalize jointly across stems, apply to complex mixture, overlap-add | Changing one stem changes other stems' masks; mixture phase is retained |
| Frequency support | Default modeled bins 0…1023; zero extension above them | Loss cannot measure high-band errors; a fixed limitation, not evidence of the regression's cause |
| Reduction | Absolute spectral error vs scale-invariant waveform scoring | Different energy weighting and amplitude sensitivity; neither orders candidates identically in general |

Ignoring epsilon for nonzero bins, with predicted magnitude `P_s` and complex
mixture `X`, training compares `P_s` with `|S_s|`, but inference uses
`Y_s = P_s² / sum_j(P_j²) * X`. Even perfect magnitude estimates need not produce
the reference complex spectra through that reconstruction rule. Mixture-phase
limitations provide another possible gap. These are properties of this objective
and synthesis, **not a newly demonstrated porting error**. Do not change the
inference exponent or baseline weights just to make the loss look aligned.

The current experiment updates all trainable network parameters (with frozen BN
running buffers); its objective is still magnitude-only. It has not optimized a
loss through final waveform reconstruction. A future waveform objective would be
a different experiment requiring its own controlled comparison.

## CPU counterexample through the existing production functions

[Reproduction script](../scripts/diagnose_loss_quality.py) constructs two tones
shared by four synthetic stems, with different source proportions at the two
frequencies. Two hand-constructed, bounded magnitude predictors are substituted
for the U-Nets; the real `spectral_loss` and `separate` functions are unchanged.
This is not a trained network or a realistic music benchmark.

For positive source fractions `q_s` summing to one, predictor A emits
`sqrt(q_s) * |X|`, so the squared ratio recovers `q_s`. Predictor B emits
`q_s * |X|`, nearly matching target magnitudes, but its squared ratio sharpens the
source proportions. Finite-window boundaries prevent exact waveform recovery.

| Synthetic candidate | Summed magnitude L1 | Vocals SI-SDR | Drums | Bass | Other |
|---|---:|---:|---:|---:|---:|
| A: ratio-aligned | 1.796146 | 43.42 | 57.31 | 48.48 | 42.76 |
| B: magnitude-aligned | 0.000000727 | 17.21 | 34.31 | 11.03 | 18.93 |

B has lower loss but worse SI-SDR **and worse waveform MAE on every stem**. This
proves that the implemented objective can rank predictions oppositely to its own
audio output. It does not show that this mechanism caused the MUSDB regression,
or that changing the loss will improve music separation. Regression tests run the
production paths and also guard ID pairing, missing/duplicate songs, lengths and
nonfinite scores.

## Next bounded measurement before any more training

Use existing official and immutable 90/450/900 checkpoints on **the same 40 fixed
validation crops**, in evaluation mode and under a pinned snapshot. No optimizer
steps are needed. Record crop IDs/offsets, per-stem raw magnitude L1, post-ratio
complex error, reconstructed waveform MAE and SI-SDR, plus reference energy and
silent-target handling. Score the identical valid interval for every checkpoint;
explicitly audit training versus inference padding/window boundaries. Include
official step 0. Report per-song/stem pairs and aggregate them without treating
overlapping/repeated crops as independent samples.

Also inspect energy and mixture/source consistency of the identified step-598
crop (`Atlantis Bound - It Was My Fault For Waiting`, start 67181, length 527360).
Its zero-loss cause is still unverified: a synthetic silent-crop test established
that zero is legal, not that this real crop is silent. Check raw estimates and
mask behavior rather than filtering it out based only on zero loss.

If raw loss improves but post-mask/waveform errors worsen on matched material,
test a reconstruction-aware loss in a small controlled pilot. If loss improvement
is confined to crops and disappears on full songs, investigate coverage and
window context first. If both loss and matched quality improve reliably, a modest
extension becomes a hypothesis worth testing. No additional paid run is launched
by this audit, and none of these proposed measurements is claimed complete.

Reproduce offline with PyTorch 2.8.0 CPU (the recorded environment):

```bash
python scripts/diagnose_loss_quality.py \
  --summary docs/results/convergence-pilot-41290811204.json \
  --output docs/results/loss-quality-audit-41290811204.json
pytest -q tests/test_loss_quality.py
```
