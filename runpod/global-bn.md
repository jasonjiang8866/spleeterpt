# Global Volume BN trial

Run `RunPod Global Volume BN trial` on `main`. This explicitly creates or reuses
one named Global Volume, `spt-1436489970-step90-inputs-v1`, billed for occupied
storage and underlying requests. It never converts, overwrites or deletes the
original 80 GB STANDARD network volume. Neither volume is deleted by cleanup.

1. Select available 2-vCPU capacity in the existing source volume's region, with
   catalog and actual rates at most $0.20/hour and a 30-minute lease. Attach the
   original network volume at `/data` and the global volume at `/global`.
2. Copy only the ten complete validation songs (50 WAV files), their pinned
   provenance, and the exact final step-90 backup. The input snapshot is capped at
   8 GiB. Every byte count and SHA-256 is derived from the already-verified
   provenance and backup evidence. Read back each destination before writing the
   completion marker last. Incomplete copies are resumable; a different or corrupt
   completed snapshot is rejected. A source lock and before/after generation
   checks prevent copying across an active checkpoint update.
3. Delete the copy Pod, then start one PRO 6000 MIG 24GB in an available region,
   with a 30-minute lease and actual rate at most $0.69/hour. A fresh mount must
   match the copy job's snapshot digest. Stage and verify inputs onto the Pod's
   local `/data`, then run the existing four-cell BN intervention. Reports identify
   both the original and global volume and explicitly count 50 verified audio
   files, rather than claiming the full 500-file training dataset was rechecked.
4. Download and independently validate the metric report, then delete compute.
   All metric, checkpoint, runtime and cross-architecture baseline gates remain.

Compute lease ceilings total $0.445, excluding storage/request charges. The local
GPU staging disk is 40 GB; the copy Pod has a 10 GB container disk. Timeouts,
always-run cleanup and the existing expired-Pod reaper apply to both mount forms.
Provisioning POSTs are never automatically retried. No credentials enter worker
subprocesses or artifacts; report access uses the existing temporary result token.

Global storage is object-backed: no file locks, atomic rename or permission bits.
Only the copy stage writes global objects; evaluation uses a fresh mount and local
files. Training and periodic checkpoint publication remain on regional storage.
This is a validation-input snapshot, not a full offsite training backup.

References (checked 2026-10-31):
- https://docs.runpod.io/storage/globalvolume/overview
- https://docs.runpod.io/storage/globalvolume/globalvolume-pods
- REST v2 `POST /global-volumes`, `mounts.global`, and CPU catalog availability.
