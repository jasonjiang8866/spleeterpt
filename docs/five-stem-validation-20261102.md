# Official five-stem validation — 2026-11-02

**Recommendation: do not publish ready-to-use five-stem weights yet.** The official
checkpoint converts reproducibly and the tested bounded path agrees closely with
the legacy path, but the unchanged TF/PyTorch strict waveform gate still fails
6/25 outputs. The project received a $2 budget approval; GPU measurements are blocked by capacity. No runtime model,
audio reconstruction, published catalog, release, or acceptance tolerance changed.

## Repository and provenance

- Starting main: `ac49d6b42e13f3e86120ba7acc726b60736501bb`; PR #40 is merged.
  No other open PR existed at intake. Main [Tests run 41313328569](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41313328569) succeeded.
- `weights-v1.0.0` still contains only the converted official 2/4-stem models.
- Final enriched waveform and piano runs use clean source
  `6126306fa1e11054f23e89b08dd0116d468b461e`. Earlier exploratory reports identify
  their base/dirty state where supported; the evidence manifest records script
  hashes and explains those records. They are not relabeled as later clean runs.
- CPU: AMD EPYC 9V74; Python 3.12.14, Torch 2.8.0+cpu, TensorFlow CPU 2.18.1,
  NumPy 2.0.2, SciPy 1.15.3, two operator threads; TF oneDNN disabled,
  Torch MKLDNN enabled. This differs from historical Torch 2.6.0 evidence.
- Official archive: [Deezer v1.4.0 5stems.tar.gz](https://github.com/deezer/spleeter/releases/download/v1.4.0/5stems.tar.gz),
  182,803,368 bytes, SHA-256
  `25a1e87eb5f75cc72a4d2d5467a0a50ac75f05611f877c278793742513cc7218`.
  Upstream's API exposes no SHA-256 for this old asset; this is a locally measured
  reproducibility pin, not a claimed upstream signature.
- Converted checkpoint SHA-256:
  `0f93e93ad4629baafe6d0a2690fa386907518f8d93667dae0743e886a83f36c7`.
  Fresh conversion and verified-cache reuse both passed. All 350 tensors map
  strictly. Source tensor hashes and configuration are in
  [weight provenance](results/five-stem-20261102/weight-provenance.json).
- Actual stem order is **vocals, piano, drums, bass, other**. The model uses
  ELU/ELU and cross-stem softmax, 44.1 kHz stereo, FFT 4096, hop 1024,
  512-frame windows, 1024 modeled bins, squared ratio masks and zero high-band
  extension. Piano is not appended to the four-stem tensor order.
- Reference audio methods and source configuration are pinned to
  `deezer/spleeter@c8854001ac8acad34a9bc2bd15f28475541828b1`.
  Its current audio frontend is used with the restored official network graph;
  the archived meta graph's older unpadded frontend is not the waveform oracle.

## Reproduced strict failures

The original formula remains `abs(candidate-reference) <= 1e-5 + 1e-4*abs(reference)`
with exact output shape and finite samples. SI-SDR does not replace this gate.
Initial unmodified-harness failure and the enriched clean-source rerun are both
retained. Different runs/backends can shift peak errors slightly; none is hidden.

| Case / stem | Failed scalar samples | Max waveform error | Reference RMS | TF-reference SI-SDR (dB) |
|---|---:|---:|---:|---:|
| Original / piano | 4 | 1.3262e-5 | 0.10727 | 99.891 |
| Original / drums | 1,102 | 2.2922e-5 | 0.14465 | 97.098 |
| Original / bass | 1,686 | 2.5004e-5 | 0.09441 | 93.062 |
| Repeated multiwindow / piano | 20 | 1.3273e-5 | 0.10856 | 99.156 |
| Repeated multiwindow / drums | 772 | 2.4766e-5 | 0.14202 | 97.602 |
| Repeated multiwindow / bass | 480 | 2.0444e-5 | 0.09689 | 94.370 |

Vocals and other pass on both music cases; all five stems pass the short,
silence and impulse cases. All output lengths are correct. Original input is
479,832 samples; the repeated case is 1,048,713 samples, not a second recording.
The failed stems have substantial RMS: this failure is **not explained away by
near-silence SI-SDR instability**. Reports include p50/p95/p99/p99.9 errors,
normalized tolerance ratios, failure fractions, RMS and explicit undefined-score
statuses. Gain fitting is used only for supplementary SI-SDR, never waveform error.

## Numerical diagnosis and controls

The existing independent stage/FP64 diagnostic was rerun with the actual official
checkpoint. These results support FP32 arithmetic sensitivity; they do not define
a universal tolerance over songs, CPU/GPU architectures or dependency versions.

| Control | Maximum absolute difference | Interpretation |
|---|---:|---|
| Independent STFTs, complex bins | 1.2420e-4 | Frontend rounding is measurable |
| Shared magnitude, FP32 logits | 6.7139e-4 | Difference precedes cross-stem softmax |
| Shared magnitude, softmax output | 1.5995e-4 | Logit competition propagates drift |
| Identical TF logits, Torch softmax | 2.3842e-7 | Axis/kernel does not explain main discrepancy |
| Independent TF64 / Torch64 logits | 6.2528e-13 | Mathematical topology converges |
| Independent FP64 softmax | 1.4433e-13 | Cross-stem computation converges |
| Identical estimates, ratio masks | 0 | Ratio-mask formula matches on this control |
| Identical masked STFT, iSTFT waveform | <=3.5763e-7 | Synthesis agrees under fixed inputs |

The stage traces cover encoder/decoder convolutions, ELU, inference BatchNorm and
skip paths for all five U-Nets. Same-logit softmax and fixed-estimate reconstruction
pass; changing CPU backend or using more precise Torch logits does not make the
rounded official TF FP32 oracle exactly reproducible. The magnitude-only
intervention still fails 35 scalar samples on this backend. Padding, frame origin,
frequency extension, periodic Hann synthesis, 2/3 gain and output crop are exercised
by the original upstream frontend, fixed-estimate controls and boundary cases.
No implementation defect requiring a production arithmetic rewrite was found in
these experiments. This is scoped evidence, not proof that no defect exists.

The full-size official network graph comparison also passes its original
`atol=2e-4, rtol=3e-4` criterion. A network-level pass is distinct from the failing
waveform gate.

## Piano clips: agreement, not separation quality

Three preselected 12-second excerpts start at 0, 15 and 30 seconds in
**Pistachio Ice Cream Ragtime**, by The Piano Lady (Lena Orsa). This is **one
independent solo-piano recording**, not three songs. Selection preceded evaluation.
Source: [librosa/data at 38f4b06](https://github.com/librosa/data/tree/38f4b06556fa0ff1acda5e677d8ba05d1bc0fff0),
HQ stereo Ogg; original and decoded clip hashes are recorded.

All 15 TF/PyTorch stem comparisons and all 15 legacy/bounded stem comparisons pass.

| Start | PyTorch piano RMS | PyTorch other RMS | Piano TF/PT max error | Other TF/PT max error |
|---:|---:|---:|---:|---:|
| 0 s | 0.107808 | 0.011829 | 5.2080e-6 | 5.2303e-6 |
| 15 s | 0.103981 | 0.016208 | 4.7367e-6 | 1.6093e-6 |
| 30 s | 0.131641 | 0.003043 | 1.0431e-6 | 9.7603e-7 |

These RMS values describe output allocation. They are not piano recall, leakage
scores, additive energy fractions or evidence of a quality improvement. No
isolated piano/other target was supplied. Even a solo-piano description does not
make the decoded mixture a rigorously defined ground-truth stem. MUSDB's `other`
also cannot be scored as a separate piano target.

The source repository's `.txt` notice says CC-BY-NC-3.0 and its `.toml` says
CC-BY-NC-4.0. Both notices are preserved in the JSON; the disagreement is explicit.
Only numerical research results are committed. No original or separated audio is
redistributed. Mixed-instrument piano coverage is still needed for broader claims.

## Official-checkpoint bounded-memory CPU checks

All **15 file cases / 75 stem outputs** pass using the actual five-stem checkpoint:
one sample; hop/FFT edges ±1 sample; padded model-window transition at
520,192 samples ±1; nominal window at 524,288 ±1; silence and mono WAV. Mono is
actually decoded from a one-channel file. File outputs preserve exact lengths.
The independent bounded criterion remains peak error <=2e-6 and SI-SDR >=100 dB,
with the already documented absolute-error rule for near-silent outputs.

Isolated fresh worker processes compare synthetic 1/12/60/180-second files:

| Duration | Legacy peak RSS (MiB) | Bounded peak RSS (MiB) | Legacy seconds | Bounded seconds |
|---:|---:|---:|---:|---:|
| 1 s | 699.17 | 707.91 | 0.99 | 1.07 |
| 12 s | 899.36 | 906.55 | 2.70 | 2.58 |
| 60 s | 2,049.91 | 941.75 | 7.91 | 7.70 |
| 180 s | 4,928.25 | 995.15 | 25.35 | 22.30 |

Largest waveform error across these cases: **2.9802e-8**. RSS includes interpreter,
checkpoint loading and warmup; timing excludes loading/warmup and includes I/O.
These are one-run measurements on a shared host, not statistically established
speedups. CPU allocator high-water marks vary even with bounded live tensors.
No CUDA memory claim follows from these results.

## Regression and automated checks

- Official converted 2-stem: 10/10 waveform outputs pass; 4-stem: 20/20 pass.
- Local CPU/TensorFlow suite: **259 passed, 3 CUDA tests skipped**.
- RunPod lifecycle, dependency, report rejection and post-deletion recovery mocks:
  **72 tests passed**. These are offline tests, not actual GPU/cleanup evidence.
- Ruff lint/format and whitespace checks pass. No training was started.
- Manual `five-stem-cpu.yml` reproduces checkpoint preparation, network comparison,
  strict waveform failure, stage/precision controls, piano clips and file/memory
  checks. It uploads evidence before its final gate. Known strict failures cause
  a failed workflow; `continue-on-error` only allows evidence collection.
- Existing test CI checks code and small fixtures; it does not certify the official
  checkpoint. The new manual heavyweight workflow has not been dispatched in this
  turn; equivalent local measurements are linked below.

## RunPod plan and approved attempt

[Read-only inventory 41462343827](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41462343827)
at 22:29 Singapore time observed no Pods, no old regional volume, and the existing
Global Volume `spt-1436489970-step90-inputs-v1`. Target MIG catalog price was
$0.69/hour; `EUR-IS-2` had LOW availability. Availability must be rechecked at launch.
The configured base image is the existing Torch 2.8.0/CUDA 12.8.1 image; its digest
is resolved at launch. A fresh pull and actual GPU runtime are not yet verified.

Prepared manual `runpod-five-stem.yml` reuses the main-only explicit paid-dispatch
gate, account budget check, single-POST provisioning, 60-minute lease, worker
watchdog and compute-only cleanup. Proposed scope:

- One RTX PRO 6000 Blackwell MIG 1g.24gb, rate cap $0.69/hour, no training.
- Recreate/hash-check official five-stem weights; 15 CUDA file-edge cases.
- Isolated 1/12/60/600-second synthetic and 1/12/60-second existing validation
  music measurements, plus the same three piano excerpts.
- Small CUDA 2/4/5-stem regressions; no repeat of the already accepted full 2/4
  long-duration GPU benchmark.
- Independently validate actual device, checkpoint, input hashes, all stems,
  numerical thresholds and CUDA memory counters. Retrieve/hash-check the result
  archive, confirm Pod disappearance, then reopen/revalidate results in a fresh
  Actions process and upload the artifact. Recovery requires no extra paid Pod.

Planning estimate: **$0.35–$0.69 compute** for 30–60 minutes; request **$1 total
incremental experiment budget** including reserve for temporary disk/read requests.
The $0.69 rate and 60-minute compute lease are enforced; provider storage/request
billing is not an exact all-in hard cap. No new persistent volume is created;
existing standing storage charges are outside this incremental experiment.
If resource/pricing checks cannot support this envelope, do not launch.
This was the pre-approval estimate. The user subsequently approved **$2 for this
project**, and PR #41 was merged at `c5fc637ca3525976fac296e497c81b340e8714a4`.

### Approved attempt: stopped before provisioning

[Run 41469349316](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41469349316)
used source `dce92135c8aa613e7e0952e2f88b9aea81dfa457`. At 15:13 UTC, fresh
inventory showed zero account Pods, no regional volume, and the existing Global
Volume. The controller stopped at catalog placement before image resolution and
before the only Pod POST. Cleanup found zero managed Pods. No compute was created;
incremental compute charge is **$0**, inferred from this lifecycle, not an invoice
reconciliation. No new storage was created. Existing storage charges continue.
The temporary one-run push trigger was restored to manual-only at
`8e01cad47caf9b2e617eaee014245a3d0337cf30`.

A subsequent [read-only inventory 41471619388](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41471619388)
at 15:26 UTC confirmed `availability=NONE` for MIG ($0.69/h), RTX 4090 ($0.89/h),
RTX 3090 ($0.50/h), RTX 5090 ($1.19/h) and RTX A6000 ($0.59/h), under Secure Cloud,
one GPU, CUDA >=12.8 filters. Account Pod count remained zero. This is a snapshot
under those filters, not a claim about all RunPod inventory. The official
[GPU catalog schema](https://github.com/runpod/docs/blob/main/api-reference-v2/openapi.json)
permits omitted `dataCenters` when unavailable; the explicit `NONE` distinguishes
that case from an incomplete response. Unknown responses must stop, not enable a
fallback.

The follow-up controller permits an **explicitly enabled RTX 4090 24GB fallback**
only for five-stem and only after confirmed MIG capacity absence. Its rate cap is
$1/h, lease 60 minutes: estimated compute $0.45–$0.89 at the observed quote, with
the remainder of the approved $2 reserved for overhead. It rejects a different
actual GPU/rate, validates the actual GPU throughout metrics and recovery, and
keeps MIG mandatory for other profiles. It does not automatically provision any
of the other queried types. Read-only inventory and paid execution are manual.

Validation of this follow-up: **77 offline RunPod tests passed**, plus Ruff.
These include absent/unknown availability, disabled fallback, unrelated profiles,
price/transport failure, and mismatched device reports. They are not GPU evidence.
The raw [launch preflight](results/five-stem-20261102/gpu-launch-preflight.json)
retains its historical static `pending` checklist verbatim; its approval item is
superseded by the user approval above. The [capacity snapshot](results/five-stem-20261102/gpu-capacity.json)
records exact source/filter/time. Their hashes are in
[gpu-attempt-manifest.json](results/five-stem-20261102/gpu-attempt-manifest.json).

No GPU waveform, memory, timing or result-recovery measurements were produced by
this attempt. CPU evidence and all strict failures remain unchanged. A successful
future bounded-memory GPU run alone would still not close the cross-framework
strict-parity or mixed-piano-corpus blockers.

## Release decision and remaining blockers

1. The historical strict contract remains false on two demo music cases. Before a
   different numerical compatibility tier is proposed, use a broader, independently
   selected mixed-piano corpus and CPU/GPU error distributions. Predeclare absolute
   and relative limits, retain all original failures, and label the tier honestly.
   This PR does not introduce a replacement threshold.
2. Real MIG measurement, long CUDA memory behavior and actual post-cleanup recovery
   remain unexecuted. The prepared workflow is not evidence of completion.
3. Piano evidence covers one solo recording and no isolated targets. It cannot
   establish five-stem separation quality or general piano/other discrimination.
4. Weight redistribution basis is unchanged: the authors' [JOSS paper, page 3](https://joss.theoj.org/papers/10.21105/joss.02154.pdf)
   explicitly covers code and pretrained models under MIT. [Upstream issue #959](https://github.com/deezer/spleeter/issues/959)
   was rechecked: open, one non-maintainer follow-up, no maintainer clarification.
   Preserve Deezer's MIT notice in any later converted-weight release; this says
   nothing about input-music rights. No release assets are created here.

Therefore `--model 5stems` remains unavailable. Advanced users may use the pinned
preparation tool and explicit `--checkpoint` at their own chosen numerical
requirements. Recommended ordinary downloads remain 2/4 stems.

## Reproduce

Use Linux/Python 3.12 and the repository lock. Keep work/output outside the checkout
when a clean source-state record is desired. Check out upstream and librosa/data
at the exact revisions above.

```bash
uv sync --locked --group dev --group release
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 TF_ENABLE_ONEDNN_OPTS=0
uv run --no-sync python scripts/prepare_five_stem.py --work ../weights

# Expected exit 1 while the documented strict discrepancy remains.
uv run --no-sync python scripts/verify_audio_parity.py ../upstream-spleeter/audio_example.mp3 \
  --upstream-root ../upstream-spleeter --tf-checkpoint ../weights/5stems/model \
  --checkpoint ../weights/5stems.pt --include-edges --output ../evidence/strict.json

uv run --no-sync python scripts/diagnose_numerics.py ../upstream-spleeter/audio_example.mp3 \
  --upstream-root ../upstream-spleeter --tf-checkpoint ../weights/5stems/model \
  --checkpoint ../weights/5stems.pt --output ../evidence/numerics.json
uv run --no-sync python scripts/verify_piano_clips.py --data-root ../librosa-data \
  --upstream-root ../upstream-spleeter --tf-checkpoint ../weights/5stems/model \
  --checkpoint ../weights/5stems.pt --output ../evidence/piano.json
uv run --no-sync python scripts/verify_bounded_checkpoint.py \
  --checkpoint ../weights/5stems.pt --output ../evidence/edges.json
uv run --no-sync python scripts/benchmark_bounded.py --checkpoint ../weights/5stems.pt \
  --durations 1 12 60 180 --output ../evidence/memory.json
```

Raw evidence: [strict original](results/five-stem-20261102/5stems-original-strict.json),
[strict enriched](results/five-stem-20261102/5stems-strict-final.json),
[stage/precision controls](results/five-stem-20261102/5stems-numerics-current.json),
[piano excerpts](results/five-stem-20261102/5stems-piano-final.json),
[file boundaries](results/five-stem-20261102/5stems-bounded-edges.json),
[memory/time](results/five-stem-20261102/5stems-memory-final.json),
[2-stem regression](results/five-stem-20261102/2stems-regression.json),
[4-stem regression](results/five-stem-20261102/4stems-regression.json),
[network graph](results/five-stem-20261102/5stems-graph.log),
[manifest](results/five-stem-20261102/manifest.json).
