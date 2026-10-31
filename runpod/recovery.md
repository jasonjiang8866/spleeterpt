# Checkpoint recovery across two separate Pods

Manually dispatch **RunPod checkpoint recovery** on `main`. Two RTX 4090 Pods
run sequentially, each with the existing 30-minute lease and $1/hour
post-allocation rate guard. Pod A must finish and be deleted before Pod B is
created. Both use the same workflow commit, stable torch 2.8.0+cu128, full 4-stem
preset, FP32 and deterministic training. No network volume or extra secret is
required. The RunPod account key remains on the Actions runner.

Pod A trains a continuous eight-step reference, then independently trains three
steps from the same official initialization. It exports the interrupted
checkpoint with optimizer/RNG/cursor state, metric history and audio provenance.
The runner downloads the bundle over the token-protected HTTPS proxy, validates
its SHA-256, exact file inventory and every member hash, and deletes the Pod.
The first job uploads that bundle to an immutable Actions artifact. A separate
job downloads the artifact, validates it before provisioning, creates Pod B,
and transfers it over authenticated HTTPS. Pod B redownloads the pinned preview
clips and verifies every decoded WAV hash before restoring the original mtimes
needed by the trainer's data signature. Absolute data paths remain fixed.

Success requires a different provider Pod ID, exact complete metric history,
and identical type-aware hashes of every final/best model, optimizer, scaler,
RNG, signature, cursor and config state against Pod A's continuous reference.
Best-checkpoint inference must be finite. The final artifact also retains the
actual `last.pt` and `best.pt` weights in `checkpoint.tar`, not only their hashes.
Music and credentials are never included. Archive members are strictly allowlisted;
links, duplicate paths, oversize content and mismatched hashes fail closed.
Downloads use authenticated JSON with 4 MiB base64 chunks and per-chunk offset/size/hash
checks, followed by whole-archive validation. This avoids HTTP Range handling in
the proxy. A small transport probe runs before waiting for the training result.

Artifacts:

- `recovery-export-<run>-<attempt>`: interrupted checkpoint and reference evidence.
- `recovery-final-<run>-<attempt>`: resumed final/best checkpoints and validation.

Both expire after **14 days**. Download them before expiry if needed. Artifact
storage can be billed by GitHub. This is bounded engineering acceptance on three
short training-pool clips, not a release of an improved model, a full training
experiment, or automatic recovery from arbitrary host failure. Tensor hashes
establish equality in the tested environment, not cross-GPU reproducibility.

The runner receives the checkpoint before Pod deletion; artifact upload happens
after deletion. If that runner fails before uploading, that copy may be lost.
Long training needs periodic durable checkpoint exports before relying on this
flow for fault tolerance. Never confuse an Actions timeout with a provider stop.
The existing cleanup/reaper still refuses persistent volumes and only deletes
managed temporary Pods.

## Complete training data preparation

The official [MUSDB18 release](https://zenodo.org/records/1117372) currently
publishes a 4,684,228,845-byte `musdb18.zip` with MD5
`af06762477334799bfc5abf237648207`. It is provided for educational use; the
individual source recordings retain their stated licenses. This project uses it
for personal research; do not redistribute the recordings or assume commercial
rights. This is AAC MUSDB18, not MUSDB18-HQ.

Download the official archive on CPU/storage first (no paid GPU is needed):

```bash
curl --fail --location --continue-at - \
  'https://zenodo.org/api/records/1117372/files/musdb18.zip/content' \
  --output musdb18.zip
python scripts/prepare_musdb_training.py \
  --archive musdb18.zip --output /data/musdb18-prepared
```

The preparation command verifies the complete official archive size/checksum,
requires at least 45 GiB free space, checks exactly 100 train and 50 test entries,
and decodes only the 100-song training pool to float WAV. It verifies stereo
44.1 kHz and equal source lengths and records source/decoded hashes, full lengths,
archive SHA-256 and decoder version. A deterministic **project-specific** 90/10
track split (seed 17) is created; it is not the conventional 86/14 split. Test
recordings remain unextracted. Existing output directories are rejected; a failed
preparation is not complete unless `provenance.json` is present.

Full-data training is not launched by this recovery workflow. Before a convergence
run: finish data preparation, provide a durable dataset location, implement
periodic durable checkpoint exports, and select a bounded training schedule and
validation/SDR protocol. The preview recovery result does not fill those gaps.
