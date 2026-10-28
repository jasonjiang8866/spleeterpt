# Initial validation

Environment: Linux CPU, Python 3.12, PyTorch 2.6.0+cpu, TensorFlow CPU 2.18.1
(optional oracle/conversion dependency). No RunPod or other paid GPU was used.

## Automated checks

- 32 tests with TensorFlow installed; 28 core tests plus 4 optional TF tests.
- Gradient propagation through every trainable U-Net parameter for 2/4/5-stem presets.
- TF SAME-convolution/transposed-convolution adjoint identity (catches spatial shifts).
- STFT reconstruction including 1-sample, short, and non-hop-aligned waveforms.
- Finite silent masks, high-frequency extension, and mixture consistency with
  `mask_extension="average"`.
- Aligned source loading, deterministic crops, split overlap/misalignment rejection.
- A four-batch CPU run interrupted after batch two and resumed produces exactly
  the same final state tensors and validation loss as the uninterrupted run.
- Resume rejects changed settings; existing run/output files are protected.
- Training checkpoint reload, stereo WAV inference, and exact output length/rate.
- Strict conversion rejects missing, incompatible, nonfinite, and extra-stem tensors.
- TF operator/topology comparisons with shared synthetic weights and nontrivial BN stats.

`python scripts/smoke_test.py` additionally passes a two-epoch synthetic training /
resume / checkpoint reload / inference sequence. This is a plumbing test, not a
convergence or perceptual-quality benchmark.

## Official pretrained checkpoint comparisons

Downloaded the official 2/4/5-stem archives from
`https://github.com/deezer/spleeter/releases/download/v1.4.0/{N}stems.tar.gz`.
Converted 140 / 280 / 350 active model tensors respectively. Loaded each archive's
original serialized `.meta` graph and restored its own TensorFlow checkpoint.

Input: one full `[1, 512, 1024, 2]` float32 magnitude spectrogram, uniform `[0,20)`
with NumPy seed 901. Compared per-stem magnitude estimates before ratio masking.
All tensors pass `numpy.testing.assert_allclose(atol=2e-4, rtol=3e-4)`.

| Preset | Largest absolute error across stems | Largest per-stem RMSE |
|---|---:|---:|
| 2 stems | 6.87e-5 | 8.73e-7 |
| 4 stems | 1.79e-4 | 4.39e-6 |
| 5 stems | 4.22e-4 | 3.11e-5 |

The tolerance combines absolute and relative terms; the maximum absolute error
alone is not the acceptance threshold. Floating-point differences depend on the
CPU kernels and environment. Reproduce each comparison with:

```bash
python scripts/verify_official_checkpoint.py \
  --tf-checkpoint weights/2stems/model --checkpoint weights/2stems.pt
```

This directly validates checkpoint tensor mapping, graph topology, activations,
spatial padding, and inference BN state against the original graph. Separate tests
compare STFT/iSTFT to TensorFlow. It does not establish bitwise equality, identical
cross-framework training trajectories, or music-separation quality on a dataset.
The downloaded weights are not committed; the converter stores source file hashes
inside each resulting `.pt` checkpoint.

## Remaining validation on RunPod

1. CUDA fp32 smoke test, then fp16/bf16 where supported.
2. Full-size model forward/backward, actual VRAM and steps/sec measurement.
3. Real dataset validation and held-out SDR/listening evaluation.
4. Persistent-volume checkpoint survival and restart/resume on the actual Pod.

Training defaults are a starting baseline, not tuned hyperparameters or quality claims.
