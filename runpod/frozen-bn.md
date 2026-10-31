# Matched frozen-BN pilot

Manual workflow `runpod-frozen-bn.yml` starts two independent 90-step runs on the
same PRO 6000 MIG 24GB GPU, from the exact official four-stem initialization used
by the earlier pilot. The sole training intervention is `--freeze-bn-stats`:
BN modules use their stored statistics while their affine parameters and normal
training dropout remain active. The flag is recorded in the resume signature;
changing it during resume fails. Older checkpoints default to ordinary BN.

Both arms use the pinned 90/10 song split, seed 17, batch size 1, one crop per
song, one epoch, Adam at 1e-5, fp32, deterministic operations and no TF32. Checks
require 90 optimizer updates, finite losses, changed affine parameters, identical
input signatures and final RNG states. Frozen BN buffers must match initialization
bit for bit; ordinary BN buffers must change. This is one seed and one short
pilot, not a convergence or generalization claim. The 50 test songs remain unused.

A CPU Pod copies the existing decoded training pool to a separate immutable
`musdb18-training-v1` prefix on the already managed Global Volume. The 503-file
snapshot is bounded to 40 GiB; all original source/WAV and manifest hashes are
verified. The earlier validation snapshot is retained unchanged. The GPU mounts
only Global Volume, stages checked files on its 80 GB local disk and trains with
local POSIX checkpoints. Each completed arm publishes one immutable last.pt plus
metadata (at most 512 MiB) with read-back checks and a completion marker written
last. No locks or atomic rename are used on Global Volume. Partial runs retain
any completed arm, but do not report a successful trial.

Evaluate official, ordinary-BN and frozen-BN models on all 10 complete validation
songs using the existing BSSEval v4 SDR and global stereo SI-SDR protocol. Report
three paired comparisons. Official aggregate mean/median and coverage must
reproduce the prior run within the predeclared 0.001 dB cross-architecture bound.
The new ordinary-BN arm is the primary training control; its correspondence to
the older 4090 checkpoint is descriptive, not a forced acceptance threshold.
Export centered eight-second excerpts of the first two sorted validation songs,
with shared gain across all variants. Automated metrics do not imply a subjective
listening verdict. Raw per-song reports and audio stay in private result artifacts;
only aggregate findings should be added to the repository.

CPU: at most $0.20/hour for 30 minutes. GPU: exact MIG identity, at most $0.69/hour
for 60 minutes including setup, training and evaluation. Both Pods are deleted in
finally and workflow cleanup steps; the expiry reaper is a fallback. No automatic
provisioning retry. Volumes and completed checkpoints are retained and incur
separate storage/request charges. No paid job runs on push.
