# Spleeter-2 versus Kim Mel-Band RoFormer: vocal screen

Prepared 2026-11-01 (Asia/Singapore), following the completed
[Spleeter-4 / HTDemucs comparison](separator-benchmark-41277234620.md).
This is a separate **two-source** task. No RoFormer music-quality result is
available at preparation time. Native official Spleeter-2 is the baseline;
the previous four-stem BSSEval scores cannot be reused as two-source scores.

## Pinned candidate and implementation

Use the checkpoint linked by
[Kimberley Jensen's repository](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/tree/25f44ffb55ee3c301281bba21b2d6d311cb69ae2):

- Hugging Face: `KimberleyJSN/melbandroformer`, revision
  `ac9b0614ab3cd7f77219e18ba494dfd93956c348`.
- `MelBandRoformer.ckpt`: 913,106,900 bytes; SHA-256
  `87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e`.
- The [publisher's model card](https://huggingface.co/KimberleyJSN/melbandroformer/blob/ac9b0614ab3cd7f77219e18ba494dfd93956c348/README.md)
  declares MIT. The full training corpus is not disclosed; the author says more
  data was used than the paper model. This checkpoint is not the paper's exact run.
- MIT [MSST implementation](https://github.com/ZFTurbo/Music-Source-Separation-Training/tree/9cc76c03ccd4da5ddf3477e93d386a1648a77732),
  commit `9cc76c03ccd4da5ddf3477e93d386a1648a77732`. Download only the two model
  source files, original configuration and license, with exact byte/hash checks.
  No unpinned toolkit installation or replacement community weights.
- 384-dimensional, six-layer, 60-band, stereo single-target model; eight attention
  heads of dimension 64; FFT 2048, hop 441, Hann window, 44.1 kHz.

`scripts/roformer_kim.py` records the complete effective configuration and recipe
hash. The state dictionary is loaded strictly with `weights_only=True` after
authenticating its bytes. Source hashes are checked before imports.

**Compatibility detail:** current MSST defaults to zeroing the DC bin; the original
Kim implementation does not. Set `zero_dc=False`, and explicitly pin all added
architecture defaults. Compare real-weight CPU output against the independently
downloaded original author implementation, with both using FP32 eager attention.

## Inputs, targets and inference

Use the existing ten full validation songs in the project 90/10 split: 50
44.1 kHz stereo WAVs, 30.371313681 minutes, decoded from **MUSDB18 AAC**.
Reference accompaniment is `drums + bass + other`, in that fixed summation order.
It is not `mixture - reference vocals`: separately AAC-encoded mixture/stems need
not add exactly. All 50 test songs remain unused. Upstream training overlap is
unresolved, so this is an operational/quality screen, not an unbiased HQ leaderboard.

| Setting | Native Spleeter-2 | Kim Mel-Band RoFormer |
|---|---|---|
| Weights | Converted official `2stems.tar.gz`, archive SHA already pinned in `runpod/quality-assets.json` | Exact publisher checkpoint above |
| Outputs | Predicted vocals and accompaniment from official two-stem masks | Predicted vocals; accompaniment = input mixture − predicted vocals |
| Precision | FP32, no TF32/autocast | Same; explicit eager attention, no SDPA backend selection |
| Chunking | Official preset, 512 frames / 1024 bins / zero high band | Native 352,800 samples (8 seconds), two overlaps, batch 1 |
| Stitching | Existing checked port synthesis | Author's reflect borders, linear 10% fades, final reflect/constant pad |
| Other transforms | No extra normalization or output gain | No normalization, TTA, shifts, ensemble, clipping or gain |
| Output checks | Exact stereo full length, finite FLOAT WAV | Same; nonfinite values are errors, never silently replaced |

The first controlled run deliberately disables the author's AMP/flash settings.
An optimized mixed-precision recipe is a separately named future experiment.
Do not shorten chunks or change overlap silently if memory or time gates fail.

## Measurement and budget gates

1. Run CPU checks first: strict real-weight loading, author-forward parity,
   short/exact/tail chunk boundaries and native official Spleeter-2 conversion.
   These do not establish GPU performance or music quality.
2. Read-only inventory must resolve the existing Global Volume, available PRO 6000
   Blackwell MIG 1g.24gb, no conflicting project Pod, $1 reserve and current-month
   headroom below the existing $500 budget. The deleted regional volume is not needed.
3. Stage and hash-check only the 50 validation WAVs plus provenance. Recreate
   official Spleeter-2 and verify its converted hash against the CPU preflight.
4. One fixed first-song GPU smoke per model plus real two-source CPU scoring.
   Project the full ten-song workload with the existing 50% margin and export
   allowance. Stop if it cannot fit the lease; no automatic paid retry.
5. One independent process per model, seed 17, deterministic algorithms, two CPU
   threads, fixed eight-second warmup, three full-song repeats. Record synchronized
   inference and decode/inference/FLOAT-export timing separately, all repeat values,
   allocated/reserved CUDA peaks and startup timings.
6. CPU memory uses Linux `/proc/self/status` **VmHWM**. Do not reuse `ru_maxrss`,
   whose inherited launch-time floor invalidated the earlier isolated CPU comparison.
7. Score both sources together with the same `museval==0.4.1` BSSEval-v4 source-image
   implementation, fixed 512-tap filters, no permutation, one-second windows/hops,
   no padding and strict full-song alignment. Separately report global stereo
   SI-SDR. Aggregate median frames/song then median songs; paired changes are
   candidate minus baseline within each song. Retain undefined/finite counts and
   common-finite-frame SDR. Establish a new two-source baseline; do not apply the
   historical four-source reproduction gate to these scores.
8. Export fixed centered eight-second excerpts from the first two sorted songs,
   references plus concealed A/B labels, one shared gain per song and a separate
   answer key. Verify package/member hashes before deleting compute.

Bounded paid-run envelope: **one MIG, at most $0.69/hour, 60 minutes from creation**,
$1 reserve. Actual quote and budget are checked again at launch. Existing Global
Volume only; no new persistent storage. Report setup/scoring/idle time separately
from modeled per-audio-hour inference cost. Storage/request charges are additional.
Preserve partial evidence and always delete only the created compute Pod.

## Preparation entry point

The [no-GPU preflight workflow](../.github/workflows/roformer-preflight.yml) checks
resource/budget state and runs the two real-weight CPU probes in an isolated
environment. It cannot create compute or storage. Synthetic probe signals are
not music-quality evidence. [Run 41280553169](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41280553169)
passed both real-weight CPU probes and read-only resource/budget checks. Original
author forward and overlap-add maximum absolute differences were both zero.
Native Spleeter-2 conversion reproduced SHA-256
`2032ba2a17765f80e02051ecd0b7a5dd1552a077221275615227c2652f873532`.
At 06:22 UTC, the account had no Pods, the existing Global Volume was present,
the regional volume was absent, and credit/monthly-headroom checks passed.
The inventory in this first run used the prior four-stem data-plan label; its
resource observations apply, but its baseline-weight field is not vocal evidence.
The updated preflight explicitly selects the vocal profile.

The [paid vocal workflow](../.github/workflows/runpod-vocal-benchmark.yml) is
manual, main-only, defaults to no paid launch, and shares the existing GPU workflow
concurrency lock. Its `vocal` profile selects native two-source inference, scoring,
dependency checks and independent result validation. The original four-stem
profile stays the default for existing entry points. Wrong profile, weight hash,
recipe, dependency versions, output source count, memory counter, song selection
or incomplete results are rejected. Four-stem historical score gates do not apply.

Launch performs fresh resource and budget checks, a single creation attempt,
lease-bounded execution, partial-evidence capture, export verification and compute
cleanup. No automatic paid retry or fallback GPU/precision/chunk configuration.
Successful exports contain 14 listening WAVs (two songs, seven files per song),
full metrics, protocol, provenance and an answer key. A failed first attempt must
be reported and reviewed before another paid attempt.
