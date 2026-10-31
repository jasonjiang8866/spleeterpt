# Five-stem CPU numerical diagnosis

## Result and scope

The fixed official checkpoint implements the same mathematical U-Nets in the two
frameworks. The remaining tested FP32 disagreement is already present **before
softmax**, arising from inference BatchNorm and convolution numerical paths. The
cross-stem competition converts small logit differences into larger mask
variation. This is supported by shared-input operator replay, stage replacement,
and independent FP64 convergence; it is no longer inferred just from logit range.
No wrong padding, activation, stem axis/order, or checkpoint mapping was found.

**The original strict five-stem waveform gate still fails 6/25 stem outputs.**
This investigation does not relax `atol=1e-5, rtol=1e-4`, mark the Draft ready, or
change runtime model/audio code. Neither switching to FP64 nor changing a CPU
backend is presented as a fix for reproducing the official **FP32** oracle.

All observations below are for the pinned weights and input on this CPU, not a
universal bound across recordings, versions, or architectures. They establish
numerical behavior, not a listening verdict or a music separation benchmark.

## Reproducible controls

- Base revision: `d002eda0425f93a36044f4c22d28c8c8249a712f`.
- Upstream: `deezer/spleeter` at
  `c8854001ac8acad34a9bc2bd15f28475541828b1`.
- Official v1.4.0 five-stem checkpoint; all 350 mapped parameters/statistics are
  unchanged. Checkpoint file and serialized graph hashes are recorded in JSON.
- Upstream `audio_example.mp3`, SHA-256
  `4b431d535d235bd81b62f816f08c3f1afb6679d2706d84b3b75903b7df909507`.
  Both frameworks receive the same decoded stereo float32 PCM, 479,832 samples;
  its byte hash is also recorded. No independent loader/resampler is introduced.
- CPU: AMD EPYC 9V74, PyTorch 2.6.0+cpu, TensorFlow CPU 2.18.1, two operator
  threads. Framework/backend settings are included in each report.
- `OfficialGraph` retains the actual upstream `EstimatorSpecBuilder`, so cuts
  target the original audio pipeline and restored serialized network graph.
- Network traces use the first fixed 512 x 1024 magnitude window, with the exact
  same TF input values. Convolution, activation, and BatchNorm outputs are traced
  for every stem. Each operator is also run independently with its exact TF
  input, distinguishing local arithmetic from accumulated error.
- The trace checks that fetching intermediate tensors does not change its TF
  logits. The reported fetch-invariance checks are all zero.

The strict-failure counts attached to intermediate tensors use the waveform's
formula only as a **drift indicator**, not a specification that each hidden layer
must satisfy that tolerance. Large absolute errors at large internal activation
scales must be read alongside relative L2 errors.

## Earliest divergence and propagation

With default TF oneDNN and Torch MKLDNN enabled, the first convolution is
bit-identical for all five stems on the shared magnitude. The first nonzero
comparison is encoder BatchNorm 1 (zero-based `encoder.0.bn`): maximum absolute
errors range from 7.63e-6 to 6.10e-5. Same-input ELU differences are at most 5.96e-8.
These are small floating-point differences, not evidence of wrong ELU parameters.

The first accumulated tolerance exceedance is encoder convolution 3 for
vocals/piano/other, convolution 2 for drums, and convolution 4 for bass. For local
same-input convolution replay, it first occurs at convolution 3 except bass at
convolution 4. Decoder transposed convolutions introduce further numerical drift;
it is not confined to the final output convolution. See the complete per-stage
`accumulated` and `same_input` entries in the JSON rather than treating the first
nonzero layer as the sole cause.

For the same magnitude input, final logits span -340.78 to 1064.28, with maximum
TF/Torch difference 7.3242e-4 and relative L2 error 3.38e-7. This becomes a maximum
softmax-mask difference of 1.9139e-4. The large logit range alone is not the causal
argument: the controlled replacements below isolate where the difference enters.

## Stage replacements

Each row changes only the named stage; subsequent audio operations use the TF
oracle. These experiments are nonlinear and their errors must not be added.

| Intervention on the original music | Maximum waveform absolute error, all stems | Samples failing strict gate, all stems |
|---|---:|---:|
| Torch magnitude -> TF network/softmax/audio | 9.82e-6 | 0 |
| Torch logits from exact TF magnitude -> TF softmax/audio | 1.93e-5 | 950 |
| Exact TF logits -> runtime-layout Torch softmax -> TF audio | 1.79e-7 | 0 |
| Torch FP64 logits, cast to FP32 -> TF softmax/audio | 1.58e-5 | 176 |

With **identical TF logits**, the runtime-layout Torch softmax differs by at most
2.98e-7 and passes every mask element. Replacing its kernel with TF softmax while
retaining Torch logits leaves essentially the same 1.91e-4 mask discrepancy.
Thus the softmax kernel/axis is not the primary mismatch; its sensitivity to the
incoming stem logit differences matters. The worst actual bin's five logits and
masks are included in `worst_softmax_bin`, without selecting an artificial input. At that bin, the drums/bass logit gap
changes from -0.3616943 to -0.3624878 (difference 7.93e-4), moving the bass
probability from 0.5764427 to 0.5766341. This directly exhibits sensitivity to
relative stem logits despite their common level near 869.

With **identical TF network estimates**, the five ratio masks are bit-identical.
Their multiplication by the exact TF mixture STFT and Torch synthesis differs by
at most 3.58e-7 in waveform. Independently feeding identical masked complex STFTs
into both inverse transforms gives the same bound. Keeping TF estimates but
using the Torch mixture STFT gives at most 4.18e-7. All these waveforms pass.
This rules out ratio masks, frequency extension, and inverse STFT as the main
source of the measured failure.

The independent STFTs differ by up to 1.24e-4 in complex bins (relative L2
1.85e-7); input magnitudes differ by up to 1.22e-4 (relative L2 1.21e-7). Their
effect through the network is measurable, even though the default-backend
magnitude-only intervention passes. Frontend and network rounding both contribute;
fixing only one does not establish complete pipeline parity.

## Precision and backend evidence

The FP64 reference independently reconstructs TF convolution/transposed
convolution, ELU, concatenation, and inference BatchNorm using the exact FP32
checkpoint values promoted to float64. TF fused BatchNorm does not support
float64, so this reference uses its mathematical inference formula, not the
archived FP32 kernel. It is a convergence control, not a replacement official
oracle. A regression test checks its full five-U-Net topology with nontrivial
running statistics, scales and biases.

| Comparison on identical TF magnitude | Maximum logit error | Relative L2 error |
|---|---:|---:|
| Official TF FP32 vs Torch FP32 | 7.32e-4 | 3.38e-7 |
| Reconstructed TF FP64 vs Torch FP64 | 5.68e-13 | 4.05e-16 |
| Reconstructed TF FP64 vs official TF FP32 | 4.39e-4 | 2.69e-7 |
| Reconstructed TF FP64 vs Torch FP32 | 6.86e-4 | 3.47e-7 |

The FP64 softmax outputs agree to 1.44e-13. Both frameworks' FP32 logits differ
from the converged result; merely making Torch more precise cannot guarantee
agreement with a rounded TF FP32 computation. The FP64-to-FP32 waveform
intervention still fails bass, demonstrating this directly.

Disabling Torch MKLDNN or changing to channels-last changes the FP32 errors but
does not eliminate logit drift. Their logit-only waveform replacements still
fail 439 and 775 samples respectively. A separate process with
`TF_ENABLE_ONEDNN_OPTS=0` also retains network/magnitude sensitivity, while
same-logit softmax and fixed-estimate synthesis remain close. On that backend,
even the magnitude-only intervention fails 35 samples (drums/bass). Therefore an
argument that the frontend is always harmless, or that one backend toggle solves
the problem, would exceed this evidence.

Several explicit first-BatchNorm arithmetic rearrangements were also tried as an
exploratory check; none made all five first-BN outputs exact. No production
BatchNorm rewrite is justified by these experiments. The committed harness
reproduces the operator/precision/backend evidence; it does not depend on that
exploratory check.

## Validation and interpretation

- Full waveform reruns retain the original outcomes: all 10 two-stem and 20
  four-stem outputs pass; five stems fails the same 6/25 outputs with maximum
  waveform error 2.3976e-5. Exact output lengths are retained.
- CPU suite: 38 tests pass, plus Ruff lint/format and the existing CPU/fp32
  synthetic training/resume/inference smoke test. The new three tests check the
  diagnostic's strict failure counting, actual five-stem softmax tensor layout,
  and independent FP64 topology. No training on real data or paid GPU was used.
- The existing waveform gate remains an independent check. A green unit-test CI
  does **not** mean the five-stem official-checkpoint waveform gate passed.

The original tested music agreement SNR is at least 89.85 dB and its largest
waveform difference is 2.40e-5. These are strong numerical closeness measurements
against TF, while strict samplewise equivalence remains false. They do not measure
separation accuracy against isolated ground truth and cannot establish audibility.
There was no listening test or new full-dataset evaluation. The prior five MUSDB
preview clips cover 2/4 stems only and remain a limited sanity check, not a
five-stem quality benchmark. Whether this FP32 discrepancy is acceptable is a
separate application requirement; this PR continues to expose the strict failure.

## Run the experiments

From the repository root, install CPU Torch 2.6.0 and `.[dev,verify]`; separately
check out the pinned upstream repository. Download official `5stems.tar.gz` from
Deezer's v1.4.0 release and extract into `weights/5stems` with
`tar --no-same-owner -xzf ...`. Convert it with the existing strict converter.
No weights or copyrighted audio are committed.

```bash
spleeterpt convert --checkpoint weights/5stems/model --stems 5 --output weights/5stems.pt

TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=2 \
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 \
python scripts/diagnose_numerics.py ../upstream-spleeter/audio_example.mp3 \
  --upstream-root ../upstream-spleeter --tf-checkpoint weights/5stems/model \
  --checkpoint weights/5stems.pt --output runs/numerics5.json

TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=2 \
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 \
python scripts/diagnose_numerics.py ../upstream-spleeter/audio_example.mp3 \
  --upstream-root ../upstream-spleeter --tf-checkpoint weights/5stems/model \
  --checkpoint weights/5stems.pt --skip-float64 --output runs/numerics5-no-onednn.json

# Expected nonzero exit while the documented strict failure persists.
python scripts/verify_audio_parity.py ../upstream-spleeter/audio_example.mp3 \
  --upstream-root ../upstream-spleeter --tf-checkpoint weights/5stems/model \
  --checkpoint weights/5stems.pt --include-edges --output runs/parity5.json

pytest -q
ruff check .
ruff format --check .
python scripts/smoke_test.py
```

Repeat the waveform command with 2/4-stem checkpoints to check those guarantees.
The diagnostic supports multi-window inputs for its full frontend/backend checks;
its layer/precision trace deliberately uses only the first network window, and
waveform logit replacements are emitted only for a clip fitting one window.

Raw diagnostics: [default backend](results/numerics-5stems.json),
[TF oneDNN disabled](results/numerics-5stems-no-onednn.json),
[five-stem waveform rerun](results/audio-parity-5stems-rechecked.json),
[two-stem rerun](results/audio-parity-2stems-rechecked.json),
[four-stem rerun](results/audio-parity-4stems-rechecked.json).
