# Audio parity and the older PyTorch port

## Findings

The current `spleeterpt` implementation matches the current official TensorFlow
audio pipeline for the tested 2/4-stem cases, including real music and boundaries.
An ablation of `fwcd/spleeter-pytorch` identifies two concrete incompatibilities:

1. **Two stems: audio frame alignment, not the learned U-Net.** Given the same
   magnitude tensor, its converted U-Net output is bit-identical to `spleeterpt`
   in this experiment. Its centered STFT presents different time frames/windows
   to the network. Padding/cropping around the otherwise unchanged legacy
   estimator removes nearly all of its difference from current upstream.
2. **Four stems: wrong activations as well as audio alignment.** The legacy code
   hardcodes LeakyReLU/ReLU, while official 4-stem weights require ELU/ELU. Fixing
   activations alone substantially improves agreement; fixing both restores close
   agreement. Five stems additionally requires cross-stem softmax, absent in that
   legacy implementation; the ablation here covers 2/4 stems only.

This establishes causes of **output mismatch**, not a proof that every legacy
2-stem result sounds worse. On five ground-truth preview clips, the legacy 2-stem
scores sometimes improve and sometimes worsen. The severe 4-stem degradation is
reproduced quantitatively. No subjective listening verdict is claimed.

Local regression suite: 35 tests pass, including three analytic checks for the
evaluation metrics. This suite result is separate from the five-stem waveform
gate failure documented below; the latter is not hidden or counted as passing.

## Fixed references and fair comparison

- Current official source: `deezer/spleeter` at
  [`c8854001ac8acad34a9bc2bd15f28475541828b1`](https://github.com/deezer/spleeter/tree/c8854001ac8acad34a9bc2bd15f28475541828b1).
- Legacy source: `fwcd/spleeter-pytorch` at
  [`8015f4309f51e84d8a9b270711f4c1cf2ae2389e`](https://github.com/fwcd/spleeter-pytorch/tree/8015f4309f51e84d8a9b270711f4c1cf2ae2389e).
- All models use the same official v1.4.0 pretrained weights. No training, weight
  tuning, source permutation, time alignment, or gain matching is used for parity.
- CPU, PyTorch 2.6.0+cpu, TensorFlow 2.18.1; identical decoded 44.1 kHz float32 PCM
  feeds both frameworks. Different MP3 loaders are therefore not a confounder.
- Legacy package CLI initialization is bypassed to avoid its optional audio/Core ML
  dependencies; its actual estimator, separator, network, and weight converter run.

**Reference version matters.** The `.meta` graph shipped inside the v1.4.0 weight
archive has an older audio frontend with no leading FFT-frame padding. Current
official `EstimatorSpecBuilder` adds 4096 leading zeros and crops those samples
after synthesis. Raw archived waveform outputs are therefore not a suitable
oracle for a port of current upstream. The earlier network-only checkpoint
comparison remains valid; the network graph and weights are reused here.

`verify_audio_parity.py` executes the actual current upstream
`EstimatorSpecBuilder` STFT, partitioning, ratio-mask, extension, and inverse-STFT
methods. Its model outputs come from the restored official serialized U-Net graph
fed with the builder's spectrogram. The reference audio implementation is not a
copy of our Torch implementation.

## Complete waveform parity

Input: upstream `audio_example.mp3`, 479,832 samples (10.881 seconds), SHA-256
`4b431d535d235bd81b62f816f08c3f1afb6679d2706d84b3b75903b7df909507`.
It is an excerpt of “Slow Motion Dream” by Steven M Bryant, supplied by upstream
under CC Attribution 3.0 (see upstream README for credits). No audio is committed.

Cases per preset: original music; music repeated to 1,048,713 samples to cross
multiple network windows and end off the hop grid; a 2,000-sample music fragment;
4,097 samples of silence; and impulses at the start, middle, and end of 8,201 samples.
The repeated track is a boundary case, not an independent recording.

| Preset | Largest music waveform absolute error | Lowest music agreement SNR | Strict samplewise check |
|---|---:|---:|---|
| 2 stems | 4.77e-7 | 128.69 dB | Pass all 5 cases / 10 stem outputs |
| 4 stems | 1.37e-6 | 123.24 dB | Pass all 5 cases / 20 stem outputs |
| 5 stems | 2.40e-5 | 89.85 dB | Fail 6 of 25 stem outputs |

The check is `allclose(atol=1e-5, rtol=1e-4)` plus exact output length. For five
stems, piano/drums/bass fail on each music case; all short/silent/impulse cases
pass. **The threshold was not loosened to turn these failures into passes.**

Five-stem pre-softmax logits on this music span approximately -341 to 1064, and
TF/Torch logit differences reach about 6.7e-4. The follow-up [stage/precision diagnosis](five-stem-numerics.md) now isolates
pre-softmax FP32 BatchNorm/convolution drift, a measurable frontend contribution,
and cross-stem competition sensitivity. Identical-logit softmax and fixed-estimate
audio reconstruction pass; independent FP64 logits converge to 5.68e-13. Neither
backend changes nor higher Torch precision reproduces all TF FP32 samples. Do not
claim bitwise or strict samplewise five-stem equivalence from these results.

“Agreement SNR” treats the **TensorFlow output** as reference. It is not music
separation SDR against isolated instruments and is not a perceptual quality score.

Raw results: [2 stems](results/audio-parity-2stems.json),
[4 stems](results/audio-parity-4stems.json), [5 stems](results/audio-parity-5stems.json).

## Causal ablations

The legacy analysis uses `center=True`, which supplies half an FFT of leading
padding. Current upstream supplies a full FFT. With FFT=4096 and hop=1024, this
changes the network's frame origin by two hops. Strided U-Nets and fixed windows
are not invariant to that shift. Consistent inverse-STFT alignment is needed too;
simply shifting final audio does not recover the original masks.

The experiment adds external padding so the unchanged legacy STFT receives the
official frame layout, then crops its output back to the input length. It does
not replace the legacy network or weights. Legacy `istft` also omits `length`,
dropping 600 samples on the demo and 137 on the repeated case. Fixing only that
length leaves the main mismatch intact.

**Original demo, vocals, reference = current TensorFlow:**

| Variant | 2-stem agreement SNR | 4-stem agreement SNR |
|---|---:|---:|
| Legacy unchanged | 23.85 dB | -0.42 dB |
| Only supply correct output length | 23.85 dB | -0.42 dB |
| Only correct frame padding/cropping | 128.53 dB | -0.44 dB |
| Only change to ELU/ELU | Not applicable | 20.40 dB |
| Correct frame padding + ELU/ELU | Not applicable | 129.08 dB |
| Current spleeterpt | 128.69 dB | 129.27 dB |

The same-input raw network relative L2 error is **0** for 2 stems and **0.966**
for unmodified 4 stems on this demo. This rules out convolution padding, skip
selection, or weight conversion as the cause of the measured 2-stem mismatch.

Additional code observations, not the main demo causes:

- Legacy CLI does not resample to the model's expected 44.1 kHz. This can matter
  for other sample rates; the tested inputs were all 44.1 kHz.
- Legacy ratio-mask epsilon is divided by 2 even when there are 4 instruments.
  Residual differences near silence can remain after the two major fixes. Our
  implementation divides by the actual instrument count.
- The README warning does not identify a specific failed recording or a confirmed
  root cause; these experiments explain reproducible differences at the pinned
  commit, not every historical subjective report.

Raw ablations: [2 stems](results/legacy-ablation-2stems.json),
[4 stems](results/legacy-ablation-4stems.json).

## Ground-truth preview evaluation

Dataset: SigSep's publicly provided [MUSDB18 7-second preview](https://github.com/sigsep/sigsep-mus-db/releases/tag/v0.4.0).
Before inference, selected the first five lexicographically sorted filenames in
the **test** subset: AM Contra, Al James, Angels In Amplifiers, Arise, and BKS.
Full filenames and source hashes are in the result files. No cherry-picking after
seeing scores, no fine-tuning, and no use of the training subset.

Metric: whole-clip, global-stereo **SI-SDR**, with channels flattened, mean removed,
and a single scale fit. This is **not museval/BSSEval-v4 SDR** and these numbers
must not be compared directly with its published benchmark tables. All methods
are scored on the same prefix, truncated to the shortest output (legacy omits the
fractional-hop tail). Accompaniment target = drums + bass + other.

Median SI-SDR in dB across the five tracks:

| Preset / instrument | Official TF | spleeterpt | Legacy unchanged | Legacy corrected |
|---|---:|---:|---:|---:|
| 2 / vocals | 5.363 | 5.363 | 5.272 | 5.363 |
| 2 / accompaniment | 11.187 | 11.187 | 11.359 | 11.187 |
| 4 / vocals | 6.313 | 6.313 | -35.267 | 6.313 |
| 4 / drums | 1.926 | 1.926 | -6.143 | 1.926 |
| 4 / bass | 1.793 | 1.793 | -27.582 | 1.793 |
| 4 / other | 0.474 | 0.474 | -4.928 | 0.474 |

Maximum individual TF-vs-spleeterpt score difference: **1.41e-6 dB** (2 stems),
**3.89e-6 dB** (4 stems). Legacy 2-stem vocals score slightly higher on four of
these five clips; thus its framing mismatch is not uniformly a quality loss.
Legacy 4-stem degradation, however, is substantial and is removed by the fixes.

This is a five-clip sanity check, not a complete MUSDB evaluation or a statistical
quality claim. Five stems is not scored: MUSDB supplies no separate piano target.

Raw scores: [2 stems](results/musdb-preview-2stems.json),
[4 stems](results/musdb-preview-4stems.json).

## Reproduce

Install `pip install -e '.[dev,verify]'` and ffmpeg. Separately clone/check out the
two pinned source revisions above, download/convert the official weights as in
the root README, and download/unzip SigSep's `MUSDB18-7-STEMS.zip` preview into a
directory containing `test/*.stem.mp4`. Scripts do not download datasets, modify
the external repositories, or start training.

```bash
python scripts/verify_audio_parity.py ../upstream-spleeter/audio_example.mp3 \
  --upstream-root ../upstream-spleeter \
  --tf-checkpoint weights/2stems/model --checkpoint weights/2stems.pt \
  --include-edges --output runs/audio-parity-2stems.json

python scripts/diagnose_legacy_port.py ../upstream-spleeter/audio_example.mp3 \
  --upstream-root ../upstream-spleeter --legacy-root ../comparison-fwcd \
  --tf-checkpoint weights/2stems/model --checkpoint weights/2stems.pt \
  --include-multiwindow --output runs/legacy-ablation-2stems.json

python scripts/evaluate_preview.py --root ../musdb-preview \
  --upstream-root ../upstream-spleeter --legacy-root ../comparison-fwcd \
  --tf-checkpoint weights/2stems/model --checkpoint weights/2stems.pt \
  --tracks 5 --output runs/musdb-preview-2stems.json
```

Repeat with 4 stems. Waveform parity also supports 5 stems and currently returns
a failing exit status on the strict full-music checks noted above. No runtime model
changes were needed for the confirmed 2/4-stem matches; this change adds evaluation
tools and records both successes and the remaining five-stem discrepancy.
