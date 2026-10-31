# Real-music training acceptance

Manually dispatch **RunPod real music training** on `main` with the paid GPU
checkbox. It shares the existing single-4090 concurrency group, 30-minute lease,
post-creation $1/hour rate guard and cleanup/reaper. No push or PR provisions GPUs.
The stable environment is torch 2.8.0+cu128 with TF32 off; deterministic kernels
and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are enabled.

This is an engineering acceptance test, not a trained model release or a quality
improvement claim. Three hash-pinned seven-second clips come **only** from the
public MUSDB preview archive's `train/` pool: two training tracks and one disjoint
validation track. They are not claimed to be unseen by the official pretrained
model. `training-assets.json` freezes the split and source file hashes;
`quality-assets.json` freezes the archive and official checkpoint downloads.
The test pool is not used. AAC streams are decoded individually to float WAV;
2-stem accompaniment uses drums + bass + other, not mixture minus vocals.

Both full 2/4-stem presets initialize from converted official weights. FP32,
batch size 1, learning rate 1e-5, seed 17, two crops per training track, one
validation crop and two epochs give eight optimizer steps per run. Short clips
are zero-padded by the existing dataset to the standard 527360-sample (~11.96 s)
training segment. Repeated crops of these short clips are identical audio; this
does not test diverse long-song crops or convergence.

For each preset, an uninterrupted eight-step run is compared against another
run stopped after step three and resumed in a **fresh Python process** on the
same Pod. Validation must execute at steps four and eight. Acceptance requires:

- Every loss/gradient finite, no skipped updates, all optimizer step counters eight.
- Changed, finite model weights and successful best-checkpoint CUDA inference.
- Exact continuous/resumed final model, optimizer, scaler, RNG, cursor and best-loss
  states, exact best-checkpoint states, and identical complete metric histories.
- Full preset configuration, pinned runtime, all track hashes and both presets.

The stdlib Actions validator checks the returned structured evidence independently
of the worker's pass flag. Raw metric histories, per-process wall time (including
initialization/checkpoint I/O), peak allocated CUDA bytes, decoded file hashes,
checkpoint SHA-256s and package versions are kept in the Actions artifact. No audio
or model weights are uploaded: acceptance checkpoints are disposable and deleted
with the Pod. This is process-restart validation, not a Pod replacement/storage
recovery test; persistent storage recovery must be exercised before long training.

A short-run validation loss is diagnostic, not a generalization or SDR result.
Formal training still requires a complete authorized training corpus, a defined
validation split, durable checkpoint export, and held-out quality evaluation.
