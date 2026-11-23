# Five-stem CUDA CLI acceptance on Blackwell MIG

[Run 45022340621](https://github.com/jasonjiang8866/spleeterpt/actions/runs/45022340621) measured commit `d7f0e58a2edc85fbd733d8540e0cc52fb2a412bf` on one RTX PRO 6000 Blackwell MIG 1g.24gb. **Numerical acceptance failed.** The memory bound passed. Each measurement used a fresh process running the installed CLI entry point with its default CUDA settings, the same converted checkpoint and float WAV input for each pair. The test exercised both full-file inference and the automatic five-stem FFT tail path in bounded inference.

| Input | Bounded vs full-file waveform checks | CLI max absolute error | Controlled benchmark max error | CLI peak allocated MiB, full / bounded |
| --- | ---: | ---: | ---: | ---: |
| Auctioneer real mix, 60 s | 2/5 | 2.921e-6 | 4.470e-8 | 1,970.71 / 571.80 |
| Synthetic, 60 s | 2/5 | 1.567e-5 | 2.980e-8 | 1,970.71 / 571.80 |
| Synthetic, 600 s | 5/5 | 1.207e-6 | 4.470e-8 | 17,419.95 / 571.80 |

The absolute-error ceiling was 2e-6, with the existing 100 dB consistency or near-silence criteria. Drums, bass and other exceeded that ceiling on the real mix. Vocals, drums and other exceeded it on the 60 s synthetic input. The controlled benchmark ran on the same Pod and passed all three cases. All 84 padded FFT matrix cases matched the full-file spectrogram exactly. These controls verify the tail FFT in the measured environment. The CLI and controlled benchmark differ in execution settings, including TF32, cuDNN determinism and warmup. This run did not isolate the effect of each setting.

## Official TensorFlow reference

The official TF CPU graph used the pinned upstream revision with oneDNN disabled. Both CUDA modes used the same five frozen input PCM hashes. Passing required every channel/sample of a stem to meet the original samplewise limits (atol 1e-5 and rtol 1e-4).

| Calibration input | CUDA full vs TF | CUDA bounded vs TF |
| --- | ---: | ---: |
| Original music | 0/5 | 0/5 |
| Multiwindow music | 0/5 | 0/5 |
| Short music | 0/5 | 0/5 |
| Silence | 5/5 | 5/5 |
| Boundary impulses | 5/5 | 5/5 |
| **Total** | **10/25** | **10/25** |

The frozen protocol opens the two TRIOS excerpts only after the calibration, long-file waveform and memory gates pass. They remained unopened for this CUDA run. The earlier [AMD and Intel CPU results](five-stem-native-cpu-20261119.md) still apply to the opt-in CPU path. Before testing more music, isolate CUDA precision and determinism on the original music case and the 60 s real mix, checking both streaming consistency and TF agreement.

## Evidence and cost

The worker completed the full benchmark matrix and reported a numerical rejection. The Actions archive SHA-256 was `22c8f1d2c643099e1c64c36c67a8247245a5cf7cc767d33ab68ecd54a38aa6f4`. Independent recovery verified all member hashes after deletion. The controller measured 332.36 s at US$0.69/h, an estimated US$0.0637 of compute. The [post-run inventory](https://github.com/jasonjiang8866/spleeterpt/actions/runs/45022914478) found zero Pods.

[All per-stem measurements](results/cuda-cli-20261122/measurements.json.gz) · [Provisioning and cleanup](results/cuda-cli-20261122/provision.json.gz) · [Recovery verdict](results/cuda-cli-20261122/recovery.json.gz) · [Post-run Pod count](results/cuda-cli-20261122/post-run-inventory.json.gz)

## CUDA settings experiment

On the original music excerpt, 1,239,048 of 4,798,320 channel samples across all five stems exceeded the official TF tolerance in full-file mode and 1,238,160 exceeded it in bounded mode. To isolate the effect of the CLI settings, the manual five-stem workflow accepts `cuda_settings_study` alongside `cli_acceptance` and `fft_batch_study`. It uses the existing one-Pod lease.

The study crosses three switches: disable TF32 for both matmul and cuDNN, enable cuDNN determinism, and warm the loaded model with one silent window. Unset switches inherit the installed CLI defaults. A ninth profile adds the controlled benchmark's thread count, seed, cuDNN benchmark setting and cuBLAS workspace setting. Each profile runs twice in opposite order on the original music and the same 60 s Auctioneer mix. Full-file and bounded inference use separate fresh processes. Both modes are compared with the same official TF output used for the original calibration. The 60 s mix checks bounded against full-file output.

The Actions artifact contains `cuda-settings.json` inside `listening-and-metrics.tar`, with effective backend settings, per-stem errors, failed sample counts and PCM hashes for both repeats. A candidate for full validation must pass all five stems for bounded vs full on both inputs and for both modes vs TF on the original music, twice. Repeat hashes help distinguish numerical drift from exact reproducibility. Any candidate still needs the complete CLI calibration, 60 s and 600 s waveform and memory checks, and the frozen TRIOS holdout. If a setting restores streaming agreement alone, use the remaining TF errors to guide the next decoder precision experiment.

## Isolating the remaining CUDA precision errors

The [backend experiment](https://github.com/jasonjiang8866/spleeterpt/actions/runs/45058973840) identified a repeatable control. With TF32 disabled and cuDNN determinism enabled, the full and bounded modes agreed on both music inputs. Against official TF, the original music still had 38 failing drum samples and 146 failing bass samples in each mode and repeat.

The manual five-stem workflow's `cuda_precision_study` option runs a separate eight-arm screen. Select CLI acceptance and the FFT batch study, and leave the earlier CUDA settings study off. The control uses FP32 throughout, disables TF32 and enables deterministic cuDNN with no warmup. The other seven arms cross three independent changes on that control:

| Operation | FP64 intervention |
| --- | --- |
| Analysis FFT | Use the existing FP64 RFFT on FP32-windowed frames and round the spectrum to complex64. Apply it in full-file and bounded mode, including the bounded tail. |
| Last decoder transposed convolution | Evaluate only the final transposed convolution in each of the five U-Nets in FP64 and round its output to FP32. |
| Output convolution | Evaluate the output convolution in each U-Net in FP64 and round its logits to FP32. |

Run all eight arms twice, reversing the order on the second pass. Each of the two modes and two frozen inputs uses a fresh CLI process, for 64 invocations. First check whether the control reproduces the historical counts and repeat PCM. If the control changes, report that drift before attributing an improvement to precision. Use the same-run control for each contrast and inspect all five stems.

The archive member `cuda-precision.json` includes per-stem TF failure counts, peak errors, PCM hashes and effective settings. A profile enters `screen_passed_profiles` only if both repeats pass all five original-music TF comparisons in both modes, both inputs pass bounded vs full, and repeat PCM hashes match. A passing profile still needs validation through a user-facing CLI on the full calibration, long-file memory gates and frozen TRIOS protocol before weight publication.
