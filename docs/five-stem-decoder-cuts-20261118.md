# Five-stem decoder block study

**The sixth decoder block produced the largest change in waveform failures in
both switch directions on both CPUs.** On the complete native pipeline,
widening just its transposed convolution passed 25/25 on the Intel Xeon 8370C
and 22/25 on the AMD EPYC 7763. Keep it as a CPU research candidate until
the residual cases pass the original strict gate.

## Full native waveform results

Each configuration used the same five calibration inputs and official
checkpoint. Each entry gives passing case/stem outputs out of 25, followed by
the number of failing channel/sample scalars in one process.

| Sixth transposed convolution | Intel 8370C | AMD 7763 | AMD worst error / allowed error |
|---|---:|---:|---:|
| Ordinary Torch FP32 | 19/25 · 3,737 | 20/25 · 1,918 | 1.791487 |
| TF FP32 on the incoming Torch tensor | 24/25 · 2 | 22/25 · 246 | 1.485768 |
| Torch FP64, cast back to FP32 before activation | 25/25 · 0 | 22/25 · 44 | 1.160756 |

The widened candidate left nine failing scalars in original drums, five in
multiwindow drums, and thirty in multiwindow bass on AMD. On Intel its worst
tolerance ratio was 0.906127. The short excerpt, silence and boundary impulses
passed on both machines. The tolerance throughout was
`abs(candidate-reference) <= 1e-5 + 1e-4 * abs(reference)`, without alignment
or gain changes. The 9V74 baseline in the previous FFT experiment came from
a different AMD host.

## Moving the decoder boundary

Failing waveform scalars are summed over the five cases once per host.
Each column starts with the other backend for the complete decoder and
replaces one more stage per row. The last row replaces the output head.

| Stages in prefix | Intel TF prefix | Intel Torch prefix | AMD TF prefix | AMD Torch prefix |
|---|---:|---:|---:|---:|
| 0 | 1,850 | 0 | 2,304 | 0 |
| 1 | 1,880 | 0 | 2,297 | 0 |
| 2 | 1,860 | 0 | 2,222 | 0 |
| 3 | 1,821 | 0 | 2,270 | 0 |
| 4 | 1,856 | 0 | 2,436 | 0 |
| 5 | 1,853 | 0 | 2,244 | 0 |
| **6** | **0** | **1,850** | **22** | **1,405** |
| 6 plus head | 0 | 1,850 | 0 | 2,304 |

On AMD the sixth switch in the TF-prefix direction removed all 2,244
previously failing scalars and introduced 22 others. Changing the output
head to TF then cleared those 22. This makes the head a specific follow-up
target. These 22 came from a controlled path with TF magnitude and synthesis;
their relationship to the 44 remaining full-pipeline failures is untested.

## Controlled comparison

Each U-Net received the official TF magnitude. Its Torch encoder ran once per
model window, producing the same five raw skips and bottleneck for every cut.
One sweep used the first *k* TF decoder blocks and finished in Torch. The
reverse sweep used a Torch prefix and finished in TF. Each of the six blocks
contains a transposed convolution, activation and inference BatchNorm. The
cut is immediately after normalization and before the next skip concatenation.
The final output convolution is a separate, seventh switch. Logits in all
14 controlled paths went through the same official TF masks, carrier and
waveform reconstruction. At every boundary each backend had to reproduce its
own endpoint logits exactly when refed its own state.

For the complete native comparison, the ordinary PyTorch separator supplied
the analysis spectrum, encoder, skips, masking and inverse transform. Across
the five stem networks we replaced only the sixth transposed convolution,
before its activation and BatchNorm. The TF control supplied that one
operator's output from the official graph on its true Torch input. The
pure-Torch candidate cast that layer's input, weights and bias to FP64,
executed the usual cropped transposed convolution, and copied its output back
to the native FP32 layout. All other operations retained their native precision.
The diagnostic hooks also execute the original convolution, so elapsed time
from this experiment cannot establish the overhead of an optimized implementation.

## Reproduction and evidence

Select `decoder-cuts` in the manual five-stem CPU audio-cross workflow or
add `--decoder-cuts` to the [CPU study CLI](five-stem-magnitude-20261109.md#reproduction).
Both hosts used commit `67dbecd87ca86f4b4054ca2190ea1947dda9804b`, pinned source
weights and PCM, Torch 2.8.0+cpu and TensorFlow 2.18.1 with oneDNN disabled.
The Intel interpreter was Python 3.12.14 (AVX512), the AMD interpreter Python
3.12.16 (AVX2). Each host ran three independent processes. All 425 waveform
hashes per process repeated within each host, as did decoder logits and the
samplewise counts of fixes and regressions at each adjacent switch. One
additional Intel process was killed before completing the multiwindow case;
its partial report was excluded and a fresh process reran all five cases.
The diagnostic spools logits to temporary files. On memory-constrained hosts
with a RAM-backed temporary directory, set `TMPDIR` to a disk-backed directory.

- [AMD raw reports and summary](results/decoder-cuts-20261118/amd.zip)
  match the [Actions artifact](https://github.com/jasonjiang8866/spleeterpt/actions/runs/44309357625)
  digest `51ac82e741430a60a0e437c776d3ba31f47e4246134c6378bddd3fb25bab065a`.
- [Intel raw reports and summary](results/decoder-cuts-20261118/intel.zip)
  have ZIP SHA-256 `8d373e491bdc0d87b2ed7838c1195b923e1205349c2fcf1280aa54236c17ad82`
  and include a supplemental layout check. An identity FP32 copy at the same
  hook reproduced all 25 baseline waveform hashes. Direct FP64 convolution
  with a contiguous FP32 output reproduced all 25 widened-candidate hashes
  without computing the displaced FP32 convolution. Run the included
  `layout.py` from the repository root with the same weights, upstream
  checkout and a matching Intel report.

The next controlled experiment should freeze the widened sixth convolution on
the failing AMD configuration, then cross analysis FFT and output-head
implementation independently while retaining the native mask and inverse
transform. Compare the actual failing samples on drums and bass, then
recheck any pure Torch candidate on the complete five-input matrix. CUDA,
bounded inference and independent piano quality remain separate release gates.
