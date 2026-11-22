# Five-stem analysis FFT and output-head cross

Widening the analysis RFFT and output head together, with the sixth decoder
convolution already widened, cleared all 44 historical EPYC 7763 failures
and introduced zero. This pure PyTorch combination passed 25/25 on each of
the three AMD environments. Its worst observed error divided by the allowed
error was 0.924634 on the 7763, 0.932652 on the hosted 9V74 and 0.952342 on
the local 9V74. Advance it to independent validation as an opt-in CPU
candidate. The earlier Intel result covered only the widened sixth convolution.

## Calibration results

Each cell gives passing case/stem pairs out of 25 and failing stereo channel/sample
values. The fourth column tracks retained historical failures and newly failing
coordinates on the 7763. Any unretained historical coordinate passed that arm.
All arms widen decoder convolution six. A widened RFFT returns to complex64
before the magnitude and carrier are used. A widened head returns to FP32
before the five-stem softmax.

| Analysis RFFT | Output head | EPYC 7763 | 7763 retained / new | EPYC 9V74 hosted | EPYC 9V74 local |
|---|---|---:|---:|---:|---:|
| Torch FP32 | Torch FP32 | 22/25 · 44 | 44 / 0 | 24/25 · 2 | 24/25 · 2 |
| Torch FP32 | TF FP32 | 24/25 · 18 | 0 / 18 | 24/25 · 2 | 24/25 · 2 |
| Torch FP32 | Torch FP64 | 24/25 · 5 | 5 / 0 | 24/25 · 4 | 24/25 · 2 |
| TF FP32 | Torch FP32 | 22/25 · 90 | 39 / 51 | 25/25 · 0 | 25/25 · 0 |
| TF FP32 | TF FP32 | 24/25 · 4 | 4 / 0 | 25/25 · 0 | 25/25 · 0 |
| TF FP32 | Torch FP64 | 24/25 · 5 | 5 / 0 | 23/25 · 15 | 23/25 · 9 |
| Torch FP64 | Torch FP32 | 24/25 · 2 | 2 / 0 | 24/25 · 2 | 24/25 · 2 |
| Torch FP64 | TF FP32 | 24/25 · 1 | 0 / 1 | 24/25 · 2 | 24/25 · 2 |
| Torch FP64 | Torch FP64 | 25/25 · 0 | 0 / 0 | 25/25 · 0 | 25/25 · 0 |

All 25 widened-D6 baseline waveform hashes and all 25 official reference
hashes on the 7763 matched the [previous experiment](five-stem-decoder-cuts-20261118.md).
The 44 historical failures consisted of nine original drums, five multiwindow
drums and thirty multiwindow bass scalars. Widening only the head fixed 39
and retained five original drums failures. Widening only the FFT fixed 42
and retained two multiwindow drums failures. Neither introduced a new failure.
Replacing only the native head with the TF FP32 head cleared all 44 but
introduced 18 multiwindow piano failures. Accounting for the coordinates
distinguishes that change from the all-Torch arm which passed every output.

On both 9V74 hosts the two baseline failures were original drums, channel 1,
samples 323606 and 323607 (zero-based). With the native head, widening the
RFFT cleared both but introduced multiwindow drums failures at channel 0,
samples 528256 and 528258. Widening the head as well cleared those. The two
9V74 environments share a CPU model but differ in OS and Python patch version.
The raw archives retain their individual waveform hashes.

## Method and reproduction

Select `fft-head` in the [manual CPU workflow](../.github/workflows/five-stem-audio-cross.yml)
or use `--fft-head` in the [study CLI](../scripts/study_audio_cross.py).
Each of three fresh processes per host ran five previously used calibration
inputs, all five stems and nine configurations. Every process used the same
pinned converted checkpoint and official TF graph. The acceptance tolerance
was `1e-5 + 1e-4 * abs(reference)` per unaligned scalar. The archive records
the exact failing channel/sample coordinates in each arm.

The separator consumed the selected analysis spectrum for both network magnitude
and carrier. All RFFTs received the same FP32 Hann-windowed frames. Torch and TF
each had to reproduce their own enclosing STFT. Each TF output head had to
reproduce its official logits on a refed official state before consuming a Torch
decoder state. The injected Torch FP32 spectrum had to reproduce the unpatched
widened-D6 waveform exactly. The rest of the separator, including masking and
inverse STFT, stayed native. The FP64 hooks preserve the original output layout
and also execute the displaced operation, so this diagnostic does not measure
deployment latency.

Each archive contains three raw reports and a summary. Recomputing the
summaries reproduced all 225 rows per host. All within-host waveform hashes
and failure coordinates repeated across the three processes. This study uses
CPU calibration audio only. GPU, bounded inference, and independent piano
quality have separate acceptance criteria.

| Raw reports | Python | Torch CPU capability | SHA-256 of ZIP |
|---|---|---|---|
| [7763](results/fft-head-20261119/7763.zip) | 3.12.16 | AVX2 | `1fcc7fc9d032b57da03b7266f4e36c4fd5d543475c85af5acfeaa7edc3c55adb` |
| [9v74-hosted](results/fft-head-20261119/9v74-hosted.zip) | 3.12.16 | AVX512 | `e2f2d6b9fb6344c25a3aa8fd505bbd375fb19f7a920f1618d91568dbe5de7ee2` |
| [9v74](results/fft-head-20261119/9v74.zip) | 3.12.14 | AVX512 | `f75d294dee6752a99753837c7553ad5b486ebe59b9a2efb6b731ee4070903010` |

Sources were pinned to [220ae14](https://github.com/jasonjiang8866/spleeterpt/commit/220ae14d8a5f89f718c46f319c16e849c20b8e8c).
The [7763 run](https://github.com/jasonjiang8866/spleeterpt/actions/runs/44412570597)
and [hosted 9V74 run](https://github.com/jasonjiang8866/spleeterpt/actions/runs/44411491059)
contain the original Actions artifacts. The local 9V74 run used the same source
and repository lockfile. The ZIP digests of both hosted archives match Actions.
