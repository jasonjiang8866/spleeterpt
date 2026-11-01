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
(or an exact match), suitable for these normalized FP32 fixtures. For near-silent
reference stems (RMS <= 1e-5, -100 dBFS), the alternative is **both** RMS error
<= 1e-8 (-160 dBFS) **and** peak error <= 1e-7 (-140 dBFS). Every stem records
its reference RMS and the acceptance rule used; non-silent stems retain the
100 dB requirement. JSON includes
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
The final workflow is **manual dispatch only on main**, with an explicit paid-run
input. The temporary feature-branch push trigger used for this experiment has
been removed; routine pushes and pull requests do not rent GPUs.

The worker measures 1/12/60/600-second synthetic inputs plus 1/12/60-second real
music prefixes for both official checkpoints, and runs CUDA 2/4/5-stem regression
tests. Each comparison runs in fresh processes. The Actions-side validator checks
identity, checkpoint hashes, CUDA device/runtime, every stem/duration, numerical
thresholds, and actual allocated-memory improvement. For 60 to 600 seconds it
requires bounded allocated memory <= 1.10x + 32 MiB and at 600 seconds less than
half the legacy allocation. Partial evidence is retained if acceptance fails.
Shared-MIG timings do not control for other tenants. A single experiment is not
a throughput/concurrency or reliability qualification.

The [first GPU experiment](results/bounded-mig-initial.json) was retained as a
**failed strict-gate run**, not relabeled as passing. All durations completed,
but the real 12-second bass stem scored 94.23 dB against a universal 100 dB gate,
despite peak error 2.41e-9 and RMS error 6.45e-11. This prompted the explicit
near-silence rule above and a second run with a 12-second FFT diagnostic.
`scripts/diagnose_bounded.py` compares normal block FFTs with cached full-song
STFT slices to isolate FFT-batch differences from synthesis grouping. This short
diagnostic intentionally holds a full spectrum and is excluded from timing and
memory measurement processes. Its cache is never used by production inference.

### Measured RTX PRO 6000 24GB MIG results (2026-11-01)

[Accepted Actions run 41308828220](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41308828220)
and [raw report, diagnostic and cleanup evidence](results/bounded-mig-41308828220.json).
Measured source: `6482108518162423d940a810c2db8cf07edf940d`; subsequent changes
document results, improve failed-worker error reporting and remove temporary launch
authorization. The inference implementation is unchanged. Runtime: Torch 2.8.0+cu128,
CUDA 12.8, capability 12.0, one MIG device with 25,367,150,592 visible bytes.

| Stems | Synthetic duration | Legacy allocated (MiB) | Bounded allocated (MiB) | Legacy / bounded file-to-file seconds |
|---|---:|---:|---:|---:|
| 2 | 60 s | 823.93 | 243.50 | 0.321 / 0.296 |
| 2 | 600 s | 7,367.14 | 243.50 | 2.482 / 2.196 |
| 4 | 60 s | 1,562.94 | 462.25 | 0.653 / 0.611 |
| 4 | 600 s | 13,920.76 | 462.25 | 4.897 / 4.154 |

At ten minutes, allocated CUDA memory decreases **96.7%** for both presets.
Reserved peaks at ten minutes are **9,526 -> 324 MiB** (2-stem) and
**16,070 -> 572 MiB** (4-stem). Bounded allocated peaks stay constant between
60 and 600 seconds. Small clips can have similar or slightly higher overhead;
this change is principally a duration-scaling improvement.

All 14 checkpoint/input/duration comparisons completed with exact output lengths.
Synthetic lengths were 1, 12, 60 and 600 seconds; real lengths were 1, 12 and 60
seconds from the previously evaluated validation mixture **Auctioneer - Our Future
Faces**. No final test-pool audio was used. Maximum waveform error over all stems
and cases was **5.216e-7**. Non-exact, non-near-silent cases had legacy-reference
SI-SDR >= **100.668 dB** (real audio >= **136.77 dB**).

The sole near-silence exception was the real 12-second bass stem: reference RMS
**3.224e-6** (~-109.8 dBFS), RMS error **6.448e-11**, peak error **2.408e-9**,
SI-SDR **94.228 dB**. It passed the documented absolute-error branch; its SI-SDR
was not hidden or relabeled. The diagnostic found identical first-window spectra
and a maximum **7.762e-6 complex STFT difference** in the short final FFT batch.
Reusing full-STFT slices reduced bass peak error to **2.274e-13** and RMS error
to **2.256e-15**, supporting FFT batch roundoff as the dominant source of that
difference; residual synthesis differences remain ordinary float32 accumulation.

Three actual CUDA 2/4/5-stem small-U-Net regressions passed. Official 5-stem
checkpoint/TF parity was not measured here. Local full-suite verification passed
**245 tests**, with 6 explicit environment skips (CUDA/TensorFlow), plus **67**
offline RunPod lifecycle/report checks; lint, format and training/resume smoke
checks passed. No subjective listening
verdict, full MUSDB quality rescore, multi-hour RF64 consumer test or sustained
concurrent-service acceptance is implied.

The accepted run's controller elapsed time was 141.87 seconds at $0.69/hour,
an upper compute-time estimate of **$0.0272**, excluding persistent storage/request
charges. The initial strict-gate run cost estimate was $0.0266. A duplicate trigger
was [canceled](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41308281564)
and its short-lived Pod deleted. The accepted and initial run reports
both confirm compute deletion; the Global Volume was retained.

Production acceptance remains separate: test official weights and representative recordings
on target GPUs, memory across longer durations and repeated jobs, numerical
agreement, concurrent-job limits, failure recovery and downstream output consumers.
