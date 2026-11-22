# Native five-stem CPU inference and mixed-piano holdouts

The opt-in native path at [e63c898](https://github.com/jasonjiang8866/spleeterpt/commit/e63c8987fdf18a30c8d85785f1d680e08bc9430d)
passed all 25 historical waveforms on an AMD EPYC 9V74. Its output hashes
matched all 25 archived outputs from the [same-host diagnostic](five-stem-fft-head-20261119.md).
The native implementation runs the analysis RFFT, final decoder transposed
convolution and output convolution in FP64, then casts each result back.
It uses the existing checkpoint and performs each convolution once.

## AMD results on two preregistered mixed-piano recordings

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
excludes model loading, decoding and WAV export. Bounded-memory CUDA acceptance remains open.

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

[DigitalOcean run #14](https://github.com/jasonjiang8866/spleeterpt/actions/runs/45017698867) completed the same five-input calibration on a four-vCPU, 8 GiB Premium Intel Droplet in Amsterdam. The guest reported vendor `GenuineIntel` and model `DO-Premium-Intel`. The candidate matched the official same-host TensorFlow outputs on **25/25** waveforms at `atol=1e-5, rtol=1e-4`. Default FP32 passed **20/25**. Both used two PyTorch threads, Python 3.12.14, PyTorch 2.8.0+cpu, TensorFlow 2.18.1 and NumPy 2.0.2. All input PCM and checkpoint hashes matched the frozen calibration.

| Input | Default FP32 | Opt-in candidate |
| --- | ---: | ---: |
| Original music | 3/5 | 5/5 |
| Multiwindow music | 2/5 | 5/5 |
| Short music | 5/5 | 5/5 |
| Silence | 5/5 | 5/5 |
| Boundary impulses | 5/5 | 5/5 |

Median inference on the original clip took 1.562 s by default and 1.835 s with the candidate on this shared VM. The [archived run artifact](results/native-cpu-20261119/do-intel-45017698867.zip) contains the raw waveform metrics, CPU identity, quote and deletion confirmation. Its SHA-256 is `9525f17c5657b4103f2f3e1711ebe398eab1c738aa3b0d395a3c12b3fc17af9e`.

[Run #13](https://github.com/jasonjiang8866/spleeterpt/actions/runs/45017291194) stopped at Python setup. Its Droplet and the successful run's Droplet were each verified absent after deletion. At US$0.08333/hour their combined 346.7 seconds from request to confirmed deletion cost an estimated US$0.0080.

### Intel replication of the TRIOS recordings

[Run #15](https://github.com/jasonjiang8866/spleeterpt/actions/runs/45018853332) repeated the 25/25 calibration on a fresh Premium Intel Droplet and compared each stem with the official TensorFlow output on that host. All ten waveforms passed the original samplewise gate across the two 12-second excerpts. Each of the four active targets improved over the mixture and stayed within 0.1 dB of TensorFlow. Leakage measures the contribution of the other active source in the indicated output. Lower values indicate less leakage.

| Recording | Output | SI-SDR gain over mixture (dB) | Candidate minus TF (dB) | Candidate leakage (dB) | TF leakage (dB) |
| --- | --- | ---: | ---: | ---: | ---: |
| Brahms Op. 40 | Piano | +6.9013 | -0.000041 | -11.8153 | -11.8154 |
| Brahms Op. 40 | Other | +2.4417 | -0.000025 | -19.4675 | -19.4674 |
| Schubert D.929 | Piano | +7.3376 | +0.000007 | -27.8402 | -27.8402 |
| Schubert D.929 | Other | +6.3296 | +0.000005 | -13.2528 | -13.2528 |

Both CPUs used the same candidate, checkpoint, registered audio and scoring protocol. The Intel run replicates the AMD result on the same two recordings. Generalization to new music requires new recordings.

The [raw metrics and cleanup record](results/native-cpu-20261119/do-intel-45018853332.zip) have ZIP SHA-256 `1bb2065645402ab6754cc102e9439f892f97610820003dfdc640c9ef0927fa2c`. The run verified the Droplet was absent by ID after deletion. At US$0.08333/h, the 285.4 seconds from create request to confirmed deletion cost an estimated US$0.0066.

### Earlier placement attempts

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

On 22 November 2026, [RunPod placement run #9](https://github.com/jasonjiang8866/spleeterpt/actions/runs/45011267899) requested one Secure Cloud CPU3g Pod at a time (4 vCPUs, 16 GB). Each worker read `/proc/cpuinfo` before fetching the checkpoint. All three reported `AuthenticAMD`.

| Data center | Reported processor | Request to confirmed deletion |
| --- | --- | ---: |
| EU-CZ-1 | AMD EPYC 9B45 | 39 s |
| EU-NL-1 | AMD EPYC 9654 | 31 s |
| EU-RO-1 | AMD EPYC 9655P | 104 s |

The three placements exhausted the agreed limit. The Intel worker executed **0 of 25** waveform checks. Each Pod was confirmed absent before the next request or exit. The three intervals totaled 174 s, an estimated US$0.0077 at US$0.16/h. The run artifact `native-intel-45011267899` contains the three placement records (ZIP SHA-256 `14c6e8e009428ddf14a102434515c37f9799f71cbb4bad57212a6d07eee4a1b9`).

A [second placement run (#10)](https://github.com/jasonjiang8866/spleeterpt/actions/runs/45014722775) found 13 available CPU3g data centers and tried five new locations. Each worker again reported `AuthenticAMD` before downloading weights.

| Data center | Reported processor | Request to confirmed deletion |
| --- | --- | ---: |
| EUR-IS-1 | AMD EPYC 9754 | 29 s |
| US-CA-2 | AMD EPYC 7713 | 28 s |
| US-GA-2 | AMD EPYC 9354 | 28 s |
| US-IL-1 | AMD EPYC 9355 | 27 s |
| US-MD-1 | AMD EPYC 7702P | 20 s |

All five Pods were confirmed absent after cleanup. The intervals totaled 132 s, an estimated US$0.0059 at US$0.16/h. The run artifact `native-intel-45014722775` contains the catalog snapshot and all five placement records (ZIP SHA-256 `3bc3e7106f84a65686e08c6be6b53d8c5883e50c4d0ef4b00d7e25331e45ec45`). Across the two runs, eight distinct locations supplied AMD CPUs and zero Intel waveform checks ran.

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
  uv sync --python 3.12.14 --locked --group dev --group release
  uv run --no-sync python scripts/prepare_five_stem.py --work "$replay_dir/weights"
  uv run --no-sync python scripts/verify_native_five_stem.py \
    --weights "$replay_dir/weights" --upstream-root "$replay_dir/upstream" \
    --output "$replay_dir/native-intel.json"
)
```
