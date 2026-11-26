# Five-stem CUDA acceptance with the FP64 analysis FFT

The opt-in CUDA CLI passed 23 of 25 calibration waveforms in each mode on one RTX PRO 6000 Blackwell MIG 24 GB. The [default CLI](five-stem-cuda-cli-20261122.md) passed 10 of 25. Drums and bass on the multiwindow music fixture still exceed the frozen tolerance. The CUDA TRIOS holdout remains unopened and five-stem weights remain research-only. [GPU run, attempt 2](https://github.com/jasonjiang8866/spleeterpt/actions/runs/45855512496/attempts/2).

## Official TensorFlow waveform comparison

The worker ran the installed CLI at commit `50e3de8092eaf3505c0fb6b553ff8ab67252552a` with `--five-stem-cuda-precision`, with and without `--bounded-memory`. Each cell counts stems whose entire stereo waveform agrees with the frozen TensorFlow CPU reference at `atol=1e-5, rtol=1e-4`.

| Calibration input | Full-file | Bounded |
| --- | ---: | ---: |
| Original music | 5/5 | 5/5 |
| Repeated music across windows | 3/5 | 3/5 |
| First 2,000 frames of music | 5/5 | 5/5 |
| Silence | 5/5 | 5/5 |
| Boundary impulses | 5/5 | 5/5 |

The repeated fixture ends at 1,048,713 stereo frames, off the STFT hop grid. Each mode exceeded tolerance on 56 drum samples and 214 bass samples out of 2,097,426 channel samples per stem. The worst error relative to the allowed error was 1.187× for drums and 1.504× for bass. The saved metrics give counts and extrema. Locating the failing samples by window and channel would distinguish errors around a join from errors elsewhere in the repeated recording.

On the original music, the input and checkpoint hashes matched the [precision screen](five-stem-cuda-precision-20261124.md). An independent replay compared all five reference PCM hashes and all ten CLI output hashes with the screen. All matched. [Verifier revision](https://github.com/jasonjiang8866/spleeterpt/commit/ac0066f3f9e259722babeda3bd55cbb748aa7eff).

## Long-file waveform agreement and GPU memory

These measurements compare bounded output with full-file output from the same CUDA CLI profile. Peak allocated memory includes model loading in a fresh process for each mode.

| Audio | Stems passed | Max absolute difference | Full-file peak (MiB) | Bounded peak (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 60 s real mix | 5/5 | 2.98e-8 | 1,970.7 | 571.8 |
| 60 s synthetic | 5/5 | 2.98e-8 | 1,970.7 | 571.8 |
| 600 s synthetic | 5/5 | 2.98e-8 | 17,419.9 | 571.8 |

## Worker completion and evidence

After the CLI wrote its report, the separate FFT shape diagnostic stopped at a call to `_block_stft` that omitted the analysis transform. The worker marked the suite incomplete. Its three CUDA tests and 15 bounded edge cases had passed, and the later standalone piano benchmarks did not run. Passing the standard analysis STFT explicitly restores this diagnostic's historical FP32 comparison. The paid measurement above predates that repair.

The first allocation attempt found no capacity. The second obtained one GPU at US$0.69/hour. Controller elapsed time was 384.8 s, an estimated US$0.074 for compute, excluding storage and request charges. The workflow deleted the Pod, verified the recovered archive members and recorded a rejected acceptance verdict. A subsequent read-only inventory found zero project Pods. [Post-run inventory](https://github.com/jasonjiang8866/spleeterpt/actions/runs/45857429700).

The repository preserves the gzip-compressed original CLI report and the source, checkpoint, artifact and recovery digests. The uncompressed CLI report has SHA-256 `8cdfd6d06bd9309fbb58be88bff363b23c0b6acee447d9351c300d1984b5afdc`. [Per-stem metrics and PCM hashes](results/cuda-cli-fft64-20261127/cuda-cli.json.gz) · [Provenance and both recovery results](results/cuda-cli-fft64-20261127/provenance.json).
