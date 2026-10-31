# CUDA music-quality preview pilot

Run **RunPod quality preview** from Actions on `main`, with the paid-GPU checkbox.
It shares the smoke workflow's concurrency group, single RTX 4090, 30-minute lease,
post-allocation $1/hour compute-rate guard, and cleanup. See [limitations](actions.md).
Pushes and PRs cannot rent GPUs. This quality-preview workflow does not start full-data training.

The worker creates an isolated venv with stable `torch==2.8.0+cu128`,
`tensorflow-cpu==2.18.1`, `museval==0.4.1`, NumPy 2.0.2 and SciPy 1.15.3.
It asserts the actual torch version and CUDA availability, and runs `pip check`.
The container tag alone is not accepted as evidence of package versions.

`quality-assets.json` pins public SigSep MUSDB18-7-STEMS and official Deezer v1.4.0
2/4-stem archive SHA-256 values, source commits, and five preview file hashes.
These are the same five test clips selected in the earlier CPU preview experiment.
The hashes were verified from official release downloads before launching compute.
Only those five clips are extracted. Raw audio, weights and archives are not uploaded.
Use these preview assets for this research comparison; this is not MUSDB18-HQ or
the complete MUSDB test set, and is not evidence of training-data independence.

Both 2-stem and 4-stem runs compare official TensorFlow on CPU with current
spleeterpt, original legacy PyTorch, and corrected legacy PyTorch on CUDA.
All variants use the same decoded inputs and matching official weights. TF32 is
disabled. Existing common-prefix length policy and BSSEval-v4 / global stereo
SI-SDR protocols are unchanged. No waveform pass threshold is relaxed; this run
measures quality scores and does not impose an automatic “quality fixed” verdict.
Five-stem evaluation is excluded because MUSDB has no isolated piano reference.

The artifact retains exact manifest/status/summary/per-track JSON, runtime peak
allocated CUDA memory, package versions, source and input hashes, and bounded logs.
The Actions runner verifies the fixed scope, code commits, stable CUDA runtime,
and all five input hashes for both presets. It then repeats the existing result
validator, including file hashes, coverage, lengths and recomputed aggregates.
`passed` means complete validated measurements, not acceptable perceptual quality.

See [metric interpretation](../docs/musdb-evaluation.md). The principal comparison
is paired song-level SDR/SI-SDR difference against TensorFlow and original legacy,
separately for each instrument. Never interpret the mixed CPU/GPU timings as a
framework performance benchmark.
