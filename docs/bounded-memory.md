# Bounded-memory inference (opt-in)

The existing `audio.separate()` remains the default and numerical reference.
`streaming.separate_file()` and CLI `--bounded-memory` bound both host and device
working memory with respect to file duration. Model size, stem count and fixed
segment geometry still determine the required memory. No weights, FFT parameters,
model window origins, mask exponent, high-frequency policy or synthesis gain change.
Official converted 2/4-stem weights remain the recommended baseline; this feature
does not establish that fine-tuning improves quality or promote the 5-stem release.

## Where the old path grows

Let `L` be original samples, `C=2`, `N=n_fft`, `H=N/4`, `B=N/2+1`,
`T=ceil((L+N)/H)`, `W=segment_frames`, and `K` be stem count. Sizes below are
FP32/complex64 payloads, excluding allocators, model weights and convolution workspaces.

| Allocation | Default path | Bounded file path |
|---|---|---|
| Decoded input on CPU | `4*C*L`, plus possible full-file resampling copies | At most one `N+(W-1)*H` sample block |
| Input/padding on model device | Grows with `L` | One block |
| Mixture complex STFT | `8*C*T*B` | At most `8*C*W*B` |
| Stem complex spectra | `8*K*C*T*B` | At most `8*K*C*W*B` |
| iRFFT frames and windowed copy | Each `4*K*C*T*N` | Each at most `4*K*C*W*N` |
| Folded audio and returned CPU stems | Grows with `K*C*L` | One block plus `K*C*(N-H)` carry |
| Files on disk | Grows with duration | Still grows with duration |

The tensor iterator excludes caller-owned input storage. Keeping its yielded
blocks in a list, concatenating them, or putting them in an unbounded queue
defeats its output-memory bound. A service must cap concurrent jobs and apply
backpressure; sequential jobs can reuse one loaded model.

## Exact frame grid and reconstruction

1. The reference prepends **N zeros**, uses `center=False`, and right-pads until
   there are `T=ceil((L+N)/H)` frames. There is no reflection padding.
2. Window `j` starts at global frame `j*W`, exactly as in `separate()`.
   For `A` actual frames starting at frame `s`, read original samples from
   `s*H-N` through `(s+A-1)*H` (exclusive), supplying zeros outside `[0,L)`.
   Neighboring reads overlap by `N-H`. They reproduce the reference STFT frames;
   they are not independent calls to whole-song `separate()`.
3. Use the same periodic Hann window and `center=False` STFT. Compute magnitudes
   at the same frequency bins; zero-pad only the final **magnitude** window to W.
   Run the unmodified model and `ratio_masks`, then discard padded output frames.
   High-frequency `zeros`/`average` extension and complex mixture phase are unchanged.
4. For this block only, iRFFT, multiply by the same Hann window and overlap-add
   with `F.fold`. Add the previous block's unfinished `N-H` samples to its beginning.
   Emit the first `A*H` samples, retaining a cloned tail of `N-H` samples.
5. Intersect emitted positions with padded coordinates `[N,N+L)`, multiply by
   the existing **2/3 gain**, and send float32 stereo chunks to CPU/the file writer.
   Do **not** replace this with `torch.istft` or per-block window normalization.
   The final carried tail lies outside the original crop and is discarded.

The accumulation groups at block boundaries differ from one global `fold`;
float32 roundoff is expected, not bit-identical output. The model context does
not change. For official geometry, reads are at most 527,360 samples/channel,
emitted blocks at most 524,288, and carry is 3,072 samples/channel/stem. This is
roughly a 12-second network window, not a realtime low-latency design.

## Usage

```bash
uv sync --locked
uv run --no-sync spleeterpt separate input-44100.wav --model 4stems \
  --bounded-memory --output outputs/job-001 --device cpu
```

CUDA: sync with `--no-default-groups --group cu128` and select `--device cuda`.
Local checkpoints, including advanced 5-stem experiments, work via `--checkpoint`.
Never change a checkpoint's segment geometry to save memory: that changes model context.

```python
from spleeterpt.checkpoint import load_model
from spleeterpt.streaming import separate_file, separate_iter

model = load_model("weights/2stems.pt", "cpu")  # or "cuda"
for job_id, input_path in jobs:
    paths = separate_file(model, input_path, f"outputs/{job_id}")
    # Use paths only after success; do not run unbounded concurrent jobs.

# For an existing CPU tensor at the checkpoint sample rate:
for stems in separate_iter(model, waveform):  # [samples], [1,L], or [2,L]
    consume(stems)  # each value is CPU float32 [2, block_samples]
```

Mono is duplicated to stereo, as in `read_audio`. Empty, non-finite and unsupported
channel inputs are rejected. Tensor input must be floating point on CPU. Iterator
validation is incremental: a late NaN can be detected after earlier blocks were
consumed; a custom sink owns its own rollback policy.

The file path requires seekable audio at the checkpoint rate. WAV and FLAC are
regression-tested. Other libsndfile formats depend on decoder/seek support.
Automatic whole-file `resample_poly` would break the CPU bound; independently
resampling each block would alter edge samples. Neither happens implicitly. For
other sample rates, resample once to a disk file, for example:

```bash
ffmpeg -i input.wav -ar 44100 -ac 2 -c:a pcm_f32le -rf64 auto input-44100.wav
```

FFmpeg resampling is not a claim of parity with the default SciPy resampler.
Compare both inference modes using the **same resulting file**. Live pipes and
streaming resampling are outside the current API.

Output directories must not exist. Stems are written in a sibling temporary
directory, closed, then the completed directory is renamed into place. Handled
errors remove temporary outputs; process kill/power loss can leave hidden staging
directories and no crash-durability guarantee is made. Disk space must accommodate
all float32 stereo stems (`8*K*L` bytes plus headers); RF64 is selected for stems
exceeding the ordinary WAV size limit. Multi-hour RF64 interoperability needs
validation with your downstream tools.

## Validation and measurement protocol

Regression tests cover 2/4/5-stem U-Nets with small test channel widths, both mask
extensions, one-sample/short inputs, silence, mono, stereo, FFT/hop/model-window
boundaries and multiple windows. They also check exact length, mixture consistency
for `average`, bounded source reads/STFT sizes, file publication/error cleanup,
CLI output and absence of leaked inference mode. The CUDA regression is explicitly
skipped without hardware; small-model 5-stem tests do not establish official TF parity.

The benchmark launches each mode/duration in a **fresh process**, loads identical
checkpoint weights, performs the same one-window warmup, and measures complete
file-to-file execution including decoding/writing. Modes run sequentially. It
records Linux/macOS process peak RSS (including loading/warmup), time, and on CUDA
devices synchronized peak allocated/reserved memory after resetting peak counters.
Host RSS is not device memory, and reserved memory is not allocated memory.
The default is a single trial, so times are descriptive, not a speedup guarantee.

```bash
uv run --no-sync spleeterpt download --stems 2 --cache-dir weights
uv run --no-sync python scripts/benchmark_bounded.py \
  --checkpoint weights/weights-v1.0.0/2stems.pt --device cpu \
  --durations 1 12 60 180 --output outputs/bounded-2-cpu.json
# Repeat with 4stems.pt; use --device cuda for REAL GPU measurements.
# Optional --input real-music-44100.wav compares prefixes (no artificial looping).
```

Without `--input`, audio is deterministic stereo tones, a chirp and amplitude
modulation. Benchmark SI-SDR is zero-mean global-stereo, using **legacy output as
reference**. It quantifies implementation agreement, not separation quality
against isolated sources or MUSDB SDR. Silence/constant-reference cases are
marked undefined; exact matches are accepted without fabricating an infinite score.
Defaults require maximum absolute waveform error <= 2e-6 and SI-SDR >= 100 dB
(or an exact match), suitable for these normalized FP32 fixtures. JSON includes
checkpoint/input hashes, configuration, versions and per-stem metrics.

### Measured CPU results (2026-11-01)

Official released weights, PyTorch 2.8.0+cpu, two Torch threads, normalized
synthetic stereo input, one isolated trial per cell. These are local working-tree
measurements of this implementation, not GPU results. RSS includes imports,
checkpoint loading, warmup and output I/O; the checkpoint/configuration hashes
are in the raw [2-stem](results/bounded-2stems-cpu.json) and
[4-stem](results/bounded-4stems-cpu.json) reports.

| Stems | Duration | Legacy peak RSS (MiB) | Bounded peak RSS (MiB) | Legacy / bounded time (s) |
|---|---:|---:|---:|---:|
| 2 | 1 s | 506 | 501 | 0.35 / 0.29 |
| 2 | 12 s | 642 | 620 | 0.75 / 0.78 |
| 2 | 60 s | 1,136 | 633 | 4.89 / 2.35 |
| 2 | 180 s | 2,401 | 637 | 14.71 / 7.98 |
| 4 | 1 s | 617 | 613 | 0.79 / 0.79 |
| 4 | 12 s | 794 | 786 | 1.93 / 1.83 |
| 4 | 60 s | 1,714 | 870 | 6.01 / 5.51 |
| 4 | 180 s | 4,002 | 872 | 25.44 / 18.60 |

At 180 seconds, peak process RSS falls about 73.5% (2-stem) and 78.2% (4-stem).
The longest inputs show almost constant bounded-path RSS from 60 to 180 seconds.
All compared stems restore exact input length. One-second outputs are identical;
other cases have maximum absolute error <= **2.98023224e-8** and legacy-reference
SI-SDR >= **166.77 dB**. These are agreement measurements, not quality improvements.

### RunPod GPU protocol

The paid workflow `runpod-bounded.yml` reuses the existing budget check, a single
RTX PRO 6000 Blackwell **MIG 1g.24gb**, <= $0.69/hour quote, 60-minute lease,
compute-only cleanup and expired-pod reaper. It uses the existing Global Volume,
never creates/replaces a volume, and copies only one previously evaluated
validation mixture. API credentials remain on the Actions runner.

The worker measures 1/12/60/600-second synthetic inputs plus 1/12/60-second real
music prefixes for both official checkpoints, and runs CUDA 2/4/5-stem regression
tests. Each comparison runs in fresh processes. The Actions-side validator checks
identity, checkpoint hashes, CUDA device/runtime, every stem/duration, numerical
thresholds, and actual allocated-memory improvement. For 60 to 600 seconds it
requires bounded allocated memory <= 1.10x + 32 MiB and at 600 seconds less than
half the legacy allocation. Partial evidence is retained if acceptance fails.
Shared-MIG timings do not control for other tenants. A single experiment is not
a throughput/concurrency or reliability qualification.

Production acceptance remains separate: test official weights and representative recordings
on target GPUs, memory across longer durations and repeated jobs, numerical
agreement, concurrent-job limits, failure recovery and downstream output consumers.
