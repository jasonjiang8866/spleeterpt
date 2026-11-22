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
