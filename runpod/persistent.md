# Persistent data and periodic checkpoint pilot

This manually dispatched workflow is a bounded storage/training acceptance test,
not convergence training or a separation-quality benchmark. It uses the full
100-song MUSDB18 **training pool**, the recorded project-specific 90/10 split,
one deterministic crop per training song and one centered crop per validation
song. It does not process the held-out 50-song test pool or every audio sample.

## Workflow and cost

Run **RunPod persistent training pilot** on `main`, with its explicit cost checkbox.
It creates or reuses exactly one STANDARD 80 GB network volume named
`spt-1436489970-musdb18-v1`. New placement chooses a compatible datacenter with
available Secure Cloud RTX 4090s from the live REST v2 catalog. Datacenter pinning
can restrict subsequent GPU availability. Duplicate names, changed size/tier or
missing mounts fail closed. Creation is never blindly retried after a lost response.

Three Pods run sequentially, with existing 30-minute leases and $1/hour actual-rate
guards (at most $1.50 compute for a normal three-Pod run, excluding billing rounding,
boot overhead and delayed cleanup). All use the pinned-at-launch official image
and isolated stable torch 2.8.0+cu128 environment. The preparation job temporarily
uses the same GPU environment to reuse the tested deployment/cleanup path; it does
not train. A future CPU-only preparer can reduce that one-time cost.

1. Download and verify the official 4.68 GB archive, decode only the 100 train songs
   on `/data`, and verify all 500 WAVs and the exact manifest split. Subsequent runs
   reuse prepared audio but recheck all file hashes.
2. Fresh full 4-stem FP32 initialization from the official converted weights;
   train ten batches with Adam, lr 1e-5, batch 1, seed 17; save checkpoint generations
   on the network volume. Delete this Pod.
3. Mount the same volume on a new Pod. Verify data, checkpoint hashes and source
   Pod identity; restore step 10 and its metrics, finish the 90-step epoch, evaluate
   all ten validation songs, and persist `last.pt`/`best.pt`. Delete this Pod.

The standard volume continues to cost approximately **$5.60/month** while retained
([official pricing](https://www.runpod.io/pricing), checked 2026-10-31), including
when no compute is running. The workflow never deletes the volume. The account
key stays in GitHub Actions; it is never put on a Pod. There is no new secret.

## Checkpoint contract

`--save-every 10 --backup-dir /data/experiments/<run>-<attempt>/backups` publishes a
new generation after each `last.pt` save. Files and the containing directory are
fsynced before `latest.json` is atomically replaced. A generation includes the
checkpoint, the matching metric-log prefix, configuration/arguments and any best
checkpoint. Every file has a size and SHA-256 in `manifest.json`. Only after the new
pointer is committed are generations older than the latest three removed.

An interrupted copy leaves the previous pointer usable. Restoration validates
all files before copying into a new output directory; extra metrics from a failed
attempt are not replayed. Checkpoint cursors and the trainer's data/config signature
remain mandatory. All writers hold a volume-level advisory lock. This workflow
is serialized with the other GPU workflows. External/manual writers must honor
that lock too.

The trainer snapshots every ten completed batches and at epoch end, not on a wall
clock schedule. Losing a Pod can lose work since the last committed generation.
The network volume survives compute deletion; it is **not an independent offsite
backup** and does not protect against volume loss or account/storage deletion.
Metadata reports go to 14-day GitHub artifacts; actual periodic weights stay on
the volume. The earlier recovery workflow remains available for exporting actual
weights to GitHub artifacts. A production training campaign still needs an
independent offsite checkpoint policy and a multi-epoch/SDR evaluation protocol.

## Failure and recovery boundaries

- The cleanup/reaper accepts only the exact declared network mount at `/data` and
  deletes the managed **Pod**, never the volume. Unknown mounts remain protected.
- Partial data preparation has no completed `provenance.json`; it is never treated
  as a valid dataset. Inspect/recover the incomplete directory before re-running
  preparation. Verified archive parts can be reused.
- Each workflow attempt has its own experiment directory. Re-running the whole
  workflow starts a new bounded experiment; it does not overwrite older results.
- The scripted resume acceptance expects step 10 from the same commit on a distinct
  Pod. General recovery at arbitrary steps uses `spleeterpt.backup.restore()` and
  the normal training CLI with identical paths, data and settings.
- Old experiments and the retained volume require deliberate lifecycle management;
  this pilot does not silently remove them or automatically expand storage.

Cloud execution status and run links are recorded in the main README once observed.
Future executions expose allowlisted download-chunk, decoded-track, verified-track
and checkpoint-step progress in Actions without forwarding arbitrary live logs.

## Resume the stored step-10 pilot after an orchestration fix

The separate **RunPod resume persistent pilot** workflow takes the existing
experiment directory (`<original-run>-<attempt>`) and its exact source commit.
It allocates only the replacement Pod, validates data and checkpoint hashes,
checks the source metadata, and resumes step 10. A different current commit is
permitted only with that explicit source commit; review that the model, data,
optimizer and training behavior remain compatible. The trainer still enforces its
full signature. This is not arbitrary checkpoint migration or arbitrary-step retry.
The original run saved a valid step-10 checkpoint before an acceptance bug compared
Python tuples to JSON lists; the normalized comparison and a regression fixture
now cover both in-memory and serialized reports.
