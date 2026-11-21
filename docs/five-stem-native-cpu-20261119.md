# Native five-stem CPU inference and mixed-piano holdouts

The opt-in native path at [e63c898](https://github.com/jasonjiang8866/spleeterpt/commit/e63c8987fdf18a30c8d85785f1d680e08bc9430d)
passed all 25 historical waveforms on an AMD EPYC 9V74. Its output hashes
matched all 25 archived outputs from the [same-host diagnostic](five-stem-fft-head-20261119.md).
The native implementation runs the analysis RFFT, final decoder transposed
convolution and output convolution in FP64, then casts each result back.
It uses the existing checkpoint and performs each convolution once.

## Two preregistered mixed-piano recordings

The candidate and official TensorFlow graph processed seconds 5–17 of each
TRIOS recording. All ten candidate stem waveforms matched TF within the
original samplewise tolerance. All four active piano/other targets improved
over the mix and stayed within the preregistered 0.1 dB allowance versus TF.

| Recording | Stem | SI-SDR gain over mixture (dB) | Candidate minus TF (dB) |
|---|---|---:|---:|
| Brahms Op. 40 | Piano | +6.9013 | −0.000047 |
| Brahms Op. 40 | Other | +2.4417 | −0.000028 |
| Schubert D.929 | Piano | +7.3376 | +0.000009 |
| Schubert D.929 | Other | +6.3296 | +0.000007 |

Input identity, geometry and mixture additivity matched the frozen registration
before inference. These are two chamber recordings from the same collection.
These recordings are now observed for this candidate. Future tuning needs fresh
holdouts for independent confirmation. See the [protocol and dataset credits](five-stem-piano-holdout.md)
and [raw metrics, including leakage, PCM hashes and every timing](results/native-cpu-20261119/epyc-9v74.json.gz).

## Runtime and usage

On 10.88 s of stereo music with two CPU threads, median full-track inference
time was 1.381 s for default FP32 and 1.533 s for the candidate (+11.0%).
Each mode had one warmup and six timed runs in alternating order. Timing
excludes model loading, decoding and WAV export. Intel validation of this
exact native implementation and bounded-memory CUDA acceptance remain open.

From a checkout containing the candidate and an already converted checkpoint:

```bash
uv run --no-sync spleeterpt separate song.wav --checkpoint weights/5stems.pt \
  --device cpu --five-stem-cpu-precision --output outputs/song-5stems
```

The CLI flag selects full-track CPU inference. Released download presets are
2stems and 4stems. To reproduce the calibration, timings and registered holdouts,
use the pinned upstream checkout and weights from the [CPU study](five-stem-magnitude-20261109.md)
and the original TRIOS archive:

```bash
TF_ENABLE_ONEDNN_OPTS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  uv run --no-sync python scripts/verify_native_five_stem.py \
  --weights weights --upstream-root ../spleeter --trios-archive ../trios.zip \
  --output native-five-stem.json
```

## Intel replay

The manual `native-intel` study checks the CPU vendor before downloading weights
and checks out the same native implementation as the 9V74 run. The
[Ubuntu 22.04 attempt](https://github.com/jasonjiang8866/spleeterpt/actions/runs/44526567093)
received an EPYC 9V74. The [Ubuntu 24.04 attempt](https://github.com/jasonjiang8866/spleeterpt/actions/runs/44840963148)
received an EPYC 7763. Both stopped before inference.

The [Xeon 8370C archive](results/decoder-cuts-20261118/intel.zip) and the
[9V74 native report](results/native-cpu-20261119/epyc-9v74.json.gz) have identical
decoded PCM hashes for all five inputs and identical checkpoint files. In all
three Xeon repeats, only the five silence outputs had the same TensorFlow
reference hashes as the 9V74. All 20 references for music and boundary impulses
differed at the byte level. The hashes identify exact replays. Measuring numerical
agreement requires comparing candidate and reference on the same Intel host
with the existing samplewise tolerance (`atol=1e-5`, `rtol=1e-4`).

On a known Intel x86-64 Linux machine, the same calibration can run outside
Actions. From a spleeterpt clone with uv 0.13.4 available, run in Bash:

```bash
(
  set -euo pipefail
  grep -qm1 '^vendor_id[[:space:]]*:[[:space:]]*GenuineIntel$' /proc/cpuinfo
  replay_dir=$(mktemp -d)
  echo "Replay directory: $replay_dir"
  git worktree add --detach "$replay_dir/spleeterpt" e63c8987fdf18a30c8d85785f1d680e08bc9430d
  cd "$replay_dir/spleeterpt"
  git clone -q https://github.com/deezer/spleeter.git "$replay_dir/upstream"
  git -C "$replay_dir/upstream" checkout -q c8854001ac8acad34a9bc2bd15f28475541828b1
  export TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=2 OPENBLAS_NUM_THREADS=1
  export OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2
  uv sync --python 3.12.16 --locked --group dev --group release
  uv run --no-sync python scripts/prepare_five_stem.py --work "$replay_dir/weights"
  uv run --no-sync python scripts/verify_native_five_stem.py \
    --weights "$replay_dir/weights" --upstream-root "$replay_dir/upstream" \
    --output "$replay_dir/native-intel.json"
)
```
