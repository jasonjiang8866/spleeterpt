# Official five-stem GPU validation — 2026-11-02

**Not accepted for release.** This supplements the [CPU report](five-stem-validation-20261102.md),
whose strict TF/PyTorch 6/25 failures remain unchanged. The project-specific budget
is US$2; the initial no-capacity attempt created no compute. PR #42 was merged at
`3ecf45d8d505a314533ad5c0e856f5f5795f4404` before this retry.

## First actual GPU attempt

[Run 41473954513](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41473954513)
used source `3b8dfaf68995588bccb6af414b3f25882a2c17b7`. The catalog now listed
`EUR-IS-2` and `US-PA-1`. One PRO 6000 Blackwell MIG 1g.24gb was created at
$0.69/hour. Runtime confirmed capability 12.0 and 25,367,150,592 visible bytes,
Torch 2.8.0+cu128, CUDA 12.8, Python 3.11.11, NumPy 2.0.2 and SciPy 1.15.3.
Full installed dependency versions are preserved in the raw worker log.
The resolved image digest is
`runpod/pytorch@sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853`.

Fresh conversion matched the pinned official five-stem archive and converted
checkpoint hashes from the CPU report. All 15 file-edge cases passed. Small
CUDA 2/4/5-stem regression tests also passed. These do not substitute for full
official-checkpoint cross-framework acceptance.

Synthetic durations 1/12/60/600 seconds passed the unchanged bounded-vs-legacy
gate. On the 600-second case, CUDA peak allocated memory was 17,197.95 MiB legacy
versus 571.80 MiB bounded; file-to-file times were 11.92 s and 5.49 s respectively.
These are one-run observations on one GPU, not stable performance guarantees.

The 60-second real mixture (`Auctioneer - Our Future Faces`, existing training-pool
validation input) failed the **2e-6 maximum absolute error** gate for two stems:

| Stem | Maximum absolute error | RMS error | Reference RMS | Legacy-reference SI-SDR |
|---|---:|---:|---:|---:|
| drums | 3.23355e-6 | 4.52931e-8 | 0.0270921 | 115.539 dB |
| other | 2.90945e-6 | 4.35878e-8 | 0.0540993 | 121.877 dB |

The SI-SDR values describe consistency with the original PyTorch path, **not
separation quality against true stems**. The piano output was near silent
(reference RMS 5.23424e-6); it passed the predeclared near-silent absolute gate.
The 1/12-second real cases passed. No thresholds were relaxed.

The worker originally stopped at that rejection, so piano excerpts were not run.
The completed raw evidence was downloaded and all archive member hashes verified
after the Pod was deleted. Local fresh-process recovery explicitly records
`acceptance_status=rejected`; archive recoverability is distinct from acceptance.
The controller confirmed deletion; elapsed controller time was 239.48 seconds and
the quoted-rate compute upper estimate was **$0.04590**. This excludes standing
storage/request charges and is not an invoice reconciliation.

Raw [result and worker log](results/five-stem-gpu-20261102/41473954513/result.json),
[provision/cleanup record](results/five-stem-gpu-20261102/41473954513/provision.json),
[recovery](results/five-stem-gpu-20261102/41473954513/recovery.json), and
[hash manifest](results/five-stem-gpu-20261102/41473954513/manifest.json) are retained.

## Follow-up protocol

Before another run, the harness was extended to retain a complete numerical
rejection and continue gathering the remaining cases. Crashes, missing reports
and incomplete duration coverage still abort. The run remains failed if any
comparison rejects. Recovery can verify failed archives without marking numerical
acceptance passed. The production model, streaming algorithm and tolerances are
unchanged.

The diagnostic uses the same failing 60-second input. A cached full FP32 STFT
feeds the bounded path to isolate synthesis grouping from STFT batch effects.
A second control computes FFTs in FP64 with the same FP32 Hann coefficients,
casts spectra to complex64, and keeps the network in FP32. It compares full and
block transforms. This is an explicitly unbounded diagnostic, not a new inference
mode or proposed release threshold. The three pinned piano excerpts will also run.

Local validation before the follow-up: 81 offline lifecycle tests and one new
CPU diagnostic regression passed, including frontend restoration and failed-archive
recovery. Final GPU outcomes will be recorded after the follow-up completes.
