# Official Spleeter conversions — weights-v1.0.0

These are **Deezer's pretrained 2-stem and 4-stem weights converted by spleeterpt**.
This is an independent PyTorch conversion, not an official Deezer release, a newly
trained model, or a quality improvement over the original weights. No fine-tuned
weights, optimizer state, music or dataset are distributed. Five-stem weights are
excluded because strict FP32 waveform parity still has unresolved discrepancies.

## Origin, license and scope

Source: `deezer/spleeter` release `v1.4.0`, `2stems.tar.gz` and `4stems.tar.gz`.
The packaged `pretrained.json` pins the official URLs, archive lengths/SHA-256 and
converted checkpoint lengths/SHA-256. `provenance.json` records conversion code,
source tensor-file hashes, environment and validation report hashes. Rebuild with
Python 3.12, torch 2.8.0+cpu, TensorFlow CPU 2.18.1 and NumPy 2.0.2; ordinary inference
needs PyTorch and this package, not TensorFlow.

Copyright (c) 2019-present, Deezer SA. The authors' JOSS paper explicitly states
that source code **and pretrained models** are distributed under MIT:
https://doi.org/10.21105/joss.02154 (page 3). The full upstream notice accompanies
the assets as `Deezer-Spleeter-MIT.txt`. Preserve it when redistributing. The source
README mentions code specifically; upstream clarification issue
https://github.com/deezer/spleeter/issues/959 remains unanswered as checked on
2026-11-01. The paper is the recorded basis for the weight license, not a claim of
a newer maintainer confirmation. Model licensing does not grant rights to input
music or redistribute the original training dataset.

## Validation and limitations

Every published weight must pass:

1. Exact source archive size and SHA-256 checks, strict finite tensor conversion.
2. Full-size magnitude output comparison against the restored official TF1 graph
   (`atol=2e-4`, `rtol=3e-4`).
3. Waveform comparison against pinned upstream audio methods plus the original
   network: original demo music, repeated multi-window music, short music, silence
   and boundary impulses. Exact output lengths and `atol=1e-5`, `rtol=1e-4`.
4. Equality with the checked-in release catalog; post-upload asset checksums.

The upstream audio implementation is pinned to
`c8854001ac8acad34a9bc2bd15f28475541828b1`. The demo music is used only during
acceptance and is not included. These tests measure numerical compatibility, not
new separation quality scores or production-service load acceptance. Historical
full-song quality experiments are documented in the repository README.

Both models use 44.1 kHz stereo internally, mixture-phase reconstruction, squared
ratio masks and zero extension above the default modeled band near 11 kHz. The
entire STFT remains resident in memory: constrain audio lengths/concurrency when
deploying. Output stems are float WAV. The checkpoint format is spleeterpt's
version 1 dictionary, loaded via `torch.load(..., weights_only=True)`.

## Download and use

From a checkout of this repository, with [uv installed](../README.md#install):

```bash
uv sync --locked
uv run --no-sync spleeterpt download --stems 4
uv run --no-sync spleeterpt separate song.wav --model 4stems --output outputs/song --device cpu
# For CUDA, first sync the cu128 group as described in the README, then use --device cuda.
```

`--model 2stems` selects vocals/accompaniment; `4stems` produces vocals/drums/bass/
other. The first call downloads from the fixed `weights-v1.0.0` Release. Subsequent
calls verify the cached file before reuse. The default cache is
`$XDG_CACHE_HOME/spleeterpt` or `~/.cache/spleeterpt`; override with `--cache-dir`.
The full MIT notice is copied alongside downloaded weights. Failed/incomplete
downloads never replace an existing cache file. For offline use, prefetch weights
or supply a local verified file with `--checkpoint /path/to/4stems.pt`.

The ordinary Git tree contains metadata, not weight blobs. Each release includes
two `.pt` files, configs, conversion/validation logs, waveform reports,
`pretrained.json`, `provenance.json`, this model card, the license and `SHA256SUMS`.
The manual **Publish pretrained weights** workflow rebuilds, validates and stages
a draft Release, verifies every uploaded asset, then publishes. It refuses to
overwrite an existing release. A failure before publication leaves a draft for
inspection rather than silently replacing prior assets. Creating or dispatching
this workflow does not rent any RunPod compute.
