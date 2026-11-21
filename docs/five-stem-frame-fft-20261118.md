# Five-stem RFFT replay on shared windowed frames

**Decision.** Keep the production analysis transform unchanged. Widening the
Torch RFFT improved spectral agreement on AMD, but the full five-stem pipeline
still passed 19 of 25 waveform comparisons on each tested CPU. Substituting
TensorFlow's RFFT also left six full-pipeline failures on both machines.

## Inputs and controls

Both hosts used commit `2590f6703d77988093befd51ead20bbb29662a7d`, the pinned
converted weights, the restored official TF graph, Torch 2.8.0+cpu and TF 2.18.1
with oneDNN disabled. Each ran three fresh processes over the original music,
multiwindow repeat, short excerpt, silence and boundary impulses. The pointwise
waveform limit was `1e-5 + 1e-4 * abs(reference)` against the official TF
waveform. Each process performed 150 comparisons (six paths × five cases ×
five stems). All 150 output hashes repeated within each host.

Torch generated the FP32 Hann coefficients once per case. TF and Torch
independently framed and windowed the decoded PCM with those coefficients.
The frame hashes agreed in all five cases, both within and across the hosts.
Both RFFTs received the same contiguous FP32 buffer in channel/frame/sample
order. Their results reproduced their respective enclosing STFTs with the
shared window. Refeeding the Torch spectrum through the real inference entry
point exactly reproduced the ordinary native waveform for every stem and case.

The three transform arms were Torch FP32, TF FP32, and a Torch FP64 RFFT
cast back to complex64 immediately after the transform. In the fixed path,
only the magnitude changed; the Torch encoder, TF decoder, official TF carrier
and TF reconstruction were held fixed. The full path used the supplied spectrum
for both magnitude and carrier and retained the production Torch model, masks
and inverse STFT. The TF-transform arm in this path is a diagnostic control.

## Results

Entries show **passing case/stem outputs out of 25** and the total number of
**failing scalar samples per process**. Counts are not multiplied by repeats.

| Transform | Intel fixed | Intel full | AMD fixed | AMD full |
|---|---:|---:|---:|---:|
| Torch FP32 | 25 / 0 | 19 / 3,737 | 22 / 68 | 19 / 4,413 |
| TF FP32 | 25 / 0 | 19 / 2,913 | 25 / 0 | 19 / 3,142 |
| Torch FP64 → complex64 | 25 / 0 | 19 / 3,655 | 23 / 16 | 19 / 3,325 |

On AMD, the ten fixed-path Torch FP32 music waveform hashes exactly reproduced
the [earlier 9V74 study](five-stem-audio-cross-20261106.md). The 68 failing
scalars split 25 on the original and 43 on the repeat. The widened transform
reduced this to 8 and 8. On the original music, the maximum complex spectral
difference from TF fell from `1.24198e-4` to `6.11542e-5`. FFT execution thus
contributes measurable drift even after aligning framing, window values and
window multiplication.

On the complete AMD path, widening reduced failures on the original music
from 3,002 to 1,562 but increased failures on the repeat from 1,411 to 1,763.
All three transform arms failed piano, drums and bass on both music inputs on
both hosts. The short, silence and impulse outputs passed throughout.
These counts measure implementation agreement on calibration audio.

## Evidence and next experiment

| Host | Environment | Raw reports and repeat summary | ZIP SHA-256 |
|---|---|---|---|
| Intel Xeon Platinum 8370C | Python 3.12.14, AVX512 | [Local Intel archive](results/frame-fft-20261118/intel.zip) | `adba3b407b0afce3a4824292f4cf9dd6a53c4ca7ba1663a7f1fc23f6da3b9844` |
| AMD EPYC 9V74 | Ubuntu 22.04 hosted runner, Python 3.12.16, AVX512 | [Actions run](https://github.com/jasonjiang8866/spleeterpt/actions/runs/44305071329), [AMD archive](results/frame-fft-20261118/amd.zip) | `dbfe95e8f8c756526d1e19d3542ffbe1e28c271f7ab7bb42a02ddf241f3f9539` |

The AMD archive matches the Actions digest. The Ubuntu 24.04 allocation
skipped the experiment at CPU selection. To reproduce, select `frame-fft`
in the manual CPU audio-cross workflow, or supply `--frame-fft` to the
[existing CLI setup](five-stem-magnitude-20261109.md#reproduction).

The next comparison should freeze one magnitude, the encoder skip tensors
and bottleneck, and the carrier and reconstruction. Move the TF/Torch decoder
cut across successive blocks on the same two failing music inputs. This
measures cumulative decoder drift while avoiding the already-tested isolated
second transposed convolution. Test the smallest effective pure Torch
candidate through all 25 complete waveform comparisons. CUDA, piano holdouts
and weight distribution remain later release gates.

The [decoder block follow-up](five-stem-decoder-cuts-20261118.md) traces both
directions of this cut and tests the sixth transposed convolution in the full
native path on Intel and AMD.
