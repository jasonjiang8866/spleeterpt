# GitHub Actions → RunPod

## One-time setup

1. Add the RunPod key as repository Actions Secret `RUNPOD_API_KEY`. The key must
   allow Pod management (GraphQL Read/Write in the current key UI); AI endpoint
   permissions are not needed. `preflight` verifies the REST v2 management API with
   an authenticated **read-only** list request. It does not prove write access.
2. Merge the RunPod Actions PR into `main`. The authentication workflow also runs
   on pushes to its setup branch, `codex/runpod-actions`, so this initial check can
   run before merging. No key or account inventory is printed.
3. Under Actions, verify **RunPod authentication** succeeds. HTTP 401/403 means
   the request was rejected; check key status, permissions, and transient network
   filtering before changing credentials. Do not widen permissions blindly.
4. Open **RunPod GPU smoke → Run workflow**, select `main`, and tick the paid-GPU
   checkbox. GPU creation is never triggered by a push or pull request.

No RunPod CLI, permanent SSH key, manual Pod, network volume, model weights, or
music dataset is required. This initial workflow supports only this repository,
one Secure Cloud RTX 4090, full-size 2-stem, FP32, batch size 1, synthetic audio.
The initial real RTX 4090 smoke passed on 2026-10-31:
[run 41175739814](https://github.com/jasonjiang8866/spleeterpt/actions/runs/41175739814).
That image contained a torch development build despite its tag. The separate
[quality pilot](quality.md) installs and checks stable torch in an isolated venv.

## What runs

The controller resolves the public RunPod PyTorch 2.8.0/Python 3.11/CUDA 12.8.1
image tag to a registry digest **before** creating the Pod, and records it with the
exact workflow commit. The Pod clones that public commit and installs the project.
Its entrypoint overrides the image's usual interactive startup; only an HTTP
result endpoint on port 8888 is exposed, accessed through RunPod's HTTPS proxy.
A fresh bearer token protects that endpoint. There is no remote shell interface.
The account-level RunPod key stays on the Actions runner; it is not sent to the Pod.

The full-size smoke test runs two training steps, saves/reloads optimizer state,
resumes to step three, completes validation, loads `best.pt`, and checks both
output WAVs. Success requires finite losses/gradients, no skipped updates, changed
weights, correct step cursors, finite stereo audio of the expected length, and
checkpoint SHA-256 values. This tests execution, **not separation quality**.

Download `runpod-smoke-<run-id>-<attempt>` from the Actions run. It contains the
container digest, source commit, Pod ID/rate/expiry, validated report, metrics,
package versions, peak allocated CUDA memory, elapsed time, and bounded log tail.
Synthetic audio and large checkpoints are validated on the Pod then discarded;
this smoke workflow is not suitable for retaining a real training run.

## Cost and cleanup

- Maximum one managed smoke Pod at a time. Existing managed Pods block creation.
- Lease: 30 minutes from provisioning, including image download and setup.
- Actual `cost` (hourly compute rate) is checked immediately **after** allocation. A missing rate
  or a rate above **US$1/hour** triggers cleanup. A rejected Pod can still incur a
  brief charge; this is not a pre-allocation price guarantee or total spend cap.
- Controller `finally` cleanup and an `always()` Actions cleanup step delete only
  this run/attempt's temporary Pod. Provisioning POST is never retried, avoiding
  duplicate rentals after ambiguous timeouts.
- A separate workflow checks expired managed Pods every five minutes after merge.
  It recognizes exact repository/run/attempt/expiry names plus an ownership env
  marker and refuses Pods with persistent volumes. It never deletes network volumes.
- GitHub schedules can be delayed, disabled, or unavailable; API outages and
  cancellations can prevent cleanup. **The 30-minute lease is not a provider-side
  shutdown guarantee.** Check the RunPod console if cleanup fails. Do not treat
  GitHub's job timeout, closing a browser, or killing the worker as stopping billing.
- The scheduled cleanup uses GitHub runner minutes. Disable its schedule when no
  managed Pods remain and you stop using this workflow; re-enable before new runs.

The independent cleanup can be run manually from Actions. It deletes only expired
smoke Pods. To release a nonexpired orphan immediately, identify it from the run's
artifact / `spt-<repository-id>-<run-id>-<attempt>-<expiry>` name in the RunPod
console and terminate that temporary Pod.

## Failure diagnosis

- Image resolution error: no GPU was allocated. Check Docker Hub connectivity/tag.
- No GPU stock: choose another time; this version does not silently switch GPU type.
- Proxy unavailable/startup failure: runner times out and attempts cleanup. Inspect
  Pod logs while it exists; failures before the worker starts cannot return logs.
- OOM/nonfinite loss/resume failure: the worker returns failed with a bounded log
  tail, and Actions fails after collecting it. Do not mark this as quality success.
- Credential errors: replace the GitHub Secret, rerun authentication. Never put the
  key in workflow YAML, command-line arguments, artifact files, or a PR comment.

Offline verification: `python -m unittest discover -s runpod -p 'test_*.py' -v`.
Tests simulate creation responses/timeouts, price rejection, ownership boundaries,
persistent-volume refusal, incomplete reports, and actual local HTTP auth handling.

References: [RunPod credentials](https://docs.runpod.io/get-started/credentials),
[Pod API](https://docs.runpod.io/api-reference-v2/pods/create-a-pod),
[GitHub schedules](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).
