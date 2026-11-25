# Five-stem CUDA precision on Blackwell MIG

Only the FP64 analysis FFT passed the two-input precision screen when paired with disabled TF32 and deterministic cuDNN. On the original music, it reduced the control's 184 failures (38 drums, 146 bass) to zero across 4,798,320 stereo channel samples summed over five stems. Every output met the official TF samplewise tolerance in both full and bounded mode, in both repeats. The largest tolerance ratio was 0.939 on drums (limit 1.0). On the 60 s Auctioneer mix, all five bounded outputs passed against full-file output with a worst absolute difference of 2.98e-8 (limit 2e-6).

[Run 45293725620](https://github.com/jasonjiang8866/spleeterpt/actions/runs/45293725620) measured main commit `1c8f972c905b5a1ba9f3a6c8286d19925d920299`, on one RTX PRO 6000 Blackwell MIG 1g.24gb with PyTorch 2.8.0+cu128 and cuDNN 91002. In each of 64 CLI invocations, the experiment used fresh processes, the pinned checkpoint, and deterministic cuDNN with TF32 off and no warmup. Each input saw all eight profiles twice in opposite order. The original music used the official TensorFlow CPU graph for comparison. The 60 s real mix measured streaming consistency. The control PCM for both inputs and the TF reference hashes matched the [previous study](five-stem-cuda-settings-20261123.md).

| Operations using FP64 | Original vs TF, failing samples (full, R1 / R2) | Original vs TF, failing samples (bounded, R1 / R2) | Bounded vs full, passed stems (original, R1 / R2) | Bounded vs full, passed stems (60 s, R1 / R2) |
| --- | ---: | ---: | ---: | ---: |
| FP32 control | 184 / 184 | 184 / 184 | 5 / 5 | 5 / 5 |
| Analysis FFT | 0 / 0 | 0 / 0 | 5 / 5 | 5 / 5 |
| Last decoder | 11 / 11 | 11 / 11 | 5 / 5 | 5 / 5 |
| Output | 178 / 178 | 178 / 178 | 5 / 5 | 5 / 5 |
| Analysis FFT + Last decoder | 161 / 161 | 161 / 161 | 5 / 5 | 5 / 5 |
| Analysis FFT + Output | 17 / 17 | 17 / 17 | 5 / 5 | 5 / 5 |
| Last decoder + Output | 105 / 105 | 105 / 105 | 5 / 5 | 5 / 5 |
| Analysis FFT + Last decoder + Output | 3 / 3 | 3 / 3 | 5 / 5 | 5 / 5 |

Both repeats produced identical output PCM per mode for all profiles. Using FP64 in both the analysis FFT and final decoder yielded 161 failures, and enabling all three wide operations yielded three. The release-validation candidate therefore keeps the decoder and output convolutions at FP32. The diagnostic computes its forward RFFT on FP32-windowed frames in FP64, then casts the spectrum back to complex64. Inverse FFT and waveform reconstruction retain the existing implementation.

## Default CLI and release status

The unmodified default CLI passed 10/25 frozen TF comparisons in each mode. Numerical acceptance failed and the workflow returned a failure status. The eight-profile diagnostic completed and its results were recovered. The default bounded path used 571.80 MiB on the 600 s synthetic file. The FFT-only candidate still needs the full calibration, 60 s and 600 s waveform and memory checks, and the frozen TRIOS holdout through a supported CLI entry point. The CUDA holdout remained unopened.

## Recovery and cost

The controller confirmed Pod deletion. A [separate inventory](https://github.com/jasonjiang8866/spleeterpt/actions/runs/45295451613) found zero project Pods. The controller measured 683.49 s at US$0.69/hour, an estimated US$0.1310 of compute excluding storage and requests. After verifying the artifact ZIP and every TAR member, the [independent verifier](https://github.com/jasonjiang8866/spleeterpt/pull/58) recomputed the sole passing candidate from per-stem metrics and PCM hashes.

[Raw two-repeat precision study](results/cuda-precision-20261124/cuda-precision.json.gz) · [Default CLI calibration and memory](results/cuda-precision-20261124/cuda-cli.json.gz) · [Source, archive checksums, cleanup, cost and recovery](results/cuda-precision-20261124/provenance.json)


## Comparing the public CLI to this screen

For a public `fft64` run, the post-deletion verifier writes `cuda_cli_screen_replay` in
`recovery.json`. It checks the checkpoint and original input PCM, then lists any
TensorFlow reference stems and public CLI output stems whose SHA-256 differs from
this screen. Full and bounded outputs are compared separately. Empty difference
lists with status `compared` mean all five reference hashes and ten output hashes
matched. Interrupted runs can report a missing CLI report or pending original music.
The full calibration, long-file and TRIOS gates determine the formal acceptance result.
