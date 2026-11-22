# Which CUDA settings change five-stem agreement?

Disabling cuDNN TF32 and enabling cuDNN determinism made full-file and bounded inference agree on both music inputs in both repeats. The two modes produced identical PCM on the original music. The largest difference on the 60 s Auctioneer mix was 4.47e-8. Against the official TensorFlow CPU graph, drums still exceeded tolerance at 38 channel samples and bass at 146 of the 4,798,320 tested across five stems. Both modes reproduced the same counts and PCM hashes in both repeats. **None of the nine configurations met all screening criteria.**

[Run 45058973840](https://github.com/jasonjiang8866/spleeterpt/actions/runs/45058973840) used one RTX PRO 6000 Blackwell MIG 1g.24gb, PyTorch 2.8.0+cu128, CUDA 12.8 and cuDNN 91002. The CLI used the same pinned five-stem checkpoint and original-music TF reference throughout. The real 60 s input came from the hashed Auctioneer mix. Each of nine configurations ran twice in reverse profile order, in separate processes for each input and mode (72 CLI invocations). The table shows five-stem pass counts in repeat 1 / repeat 2.

| CUDA profile | Original vs TF, full | Original vs TF, bounded | Bounded vs full, original | Bounded vs full, 60 s |
| --- | ---: | ---: | ---: | ---: |
| `default` | 0 / 0 | 0 / 0 | 2 / 2 | 3 / 3 |
| `no_tf32` | 3 / 3 | 3 / 3 | 2 / 2 | 5 / 3 |
| `deterministic` | 0 / 0 | 0 / 0 | 5 / 5 | 5 / 5 |
| `warmup` | 0 / 0 | 0 / 0 | 2 / 2 | 3 / 3 |
| `no_tf32+deterministic` | 3 / 3 | 3 / 3 | 5 / 5 | 5 / 5 |
| `no_tf32+warmup` | 4 / 5 | 5 / 4 | 2 / 2 | 3 / 5 |
| `deterministic+warmup` | 0 / 0 | 0 / 0 | 5 / 5 | 5 / 5 |
| `no_tf32+deterministic+warmup` | 3 / 3 | 3 / 3 | 5 / 5 | 5 / 5 |
| `benchmark` | 3 / 3 | 3 / 3 | 5 / 5 | 5 / 5 |

Matmul TF32 was already off by default on this image. The no-TF32 intervention changed the observed cuDNN flag. On the original music, disabling TF32 alone reduced full-file TF failures from 1,239,195 and 1,240,198 channel samples to 26 and 59. With warmup, one repeat passed all five stems in each mode, but the other repeat failed and the streaming comparison failed on the original clip both times. Determinism alone made repeat PCM reproducible and increased the peak error against TF to 0.01086. Repeatability and reference accuracy therefore need separate checks.

With TF32 disabled and deterministic cuDNN, adding warmup preserved the observed pass counts and peak errors. The controlled benchmark profile produced the same metrics. The remaining full-file and bounded errors against TF peaked at 1.41e-5 for drums and 1.58e-5 for bass. Vocals, piano and other passed the original samplewise criterion. The next precision experiment should use this repeatable configuration as the control and test the analysis FFT and final decoder convolutions independently, following the [validated CPU path](five-stem-native-cpu-20261119.md).

## Release gates and recovery

In the separate, unmodified CLI acceptance run on the same Pod, each mode passed 10/25 frozen TF checks. The 600 s synthetic file used 571.80 MiB peak allocated memory in bounded mode and 17,419.95 MiB in full-file mode. The default CLI exceeded the streaming threshold on the real 60 s case. The study profiles have only the two-input screening measurements. The CUDA TRIOS holdout stayed unopened.

The controller measured 604.03 seconds at US$0.69/hour, an estimated US$0.1158 of compute excluding storage and requests. Recovery verified the exported files after compute deletion. A [separate inventory](https://github.com/jasonjiang8866/spleeterpt/actions/runs/45060051104) recorded zero project Pods. The workflow ended with a numerical rejection and retained the complete diagnostic.

[Per-profile raw measurements](results/cuda-settings-20261123/cuda-settings.json.gz) · [Default CLI calibration and memory](results/cuda-settings-20261123/cuda-cli.json.gz) · [GPU, input identity, archive hash, cost and cleanup](results/cuda-settings-20261123/provenance.json)
