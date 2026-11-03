# RunPod handoff for a new conversation

Reviewed against main `bf6398f0cfa5d4850615b596d71657c827a0f771` on
**2026-11-03 (Asia/Singapore)**. Re-read current files before executing commands.
This guide preserves the operating method, not a live resource inventory or a
paid-run authorization.

## How the connection works

The reusable path is **GitHub Actions → repository controller → RunPod API →
temporary compute → verified artifacts / Global Volume → compute deletion**.

| Layer | Existing entry point | What the next conversation needs |
|---|---|---|
| GitHub execution | Manual `workflow_dispatch` on `main` | A supported dispatcher, or an already authenticated GitHub CLI with workflow access |
| RunPod credentials | Actions repository Secret `RUNPOD_API_KEY` | The Secret name only; its value remains on the runner |
| Provider operations | [control.py](control.py), [benchmark_control.py](benchmark_control.py) | Reuse the existing REST v2 controller; do not recreate raw provisioning calls |
| Account budget | [benchmark_budget.py](benchmark_budget.py) | GraphQL read of balance/monthly charges; inspect the current checks and user-approved experiment budget |
| Persistent inputs | [global_storage.py](global_storage.py), [global_inputs.py](global_inputs.py) | Discover the configured Global Volume; verify the mounted completion marker and hashes |
| Worker and evidence | [worker.py](worker.py), task-specific result validators | Exact source commit, image digest, configuration, artifact hashes and acceptance outcome |
| Cleanup | Controller `finally`, workflow `always()`, [expired-Pod workflow](../.github/workflows/runpod-cleanup.yml) | Verify deletion of this project's owned compute; preserve persistent data |

The controller uses `https://api.runpod.io/v2` for resource management and
`https://api.runpod.io/graphql` for the budget query. These endpoints are used
inside Actions. The worker receives a temporary result-endpoint token, not the
account API key. Never serialize raw Pod environments: they contain that token.

The repository Secret does **not** automatically authenticate a new conversation's
shell, and GitHub repository read/write access does not necessarily include
Actions dispatch. Do not ask the user to paste the RunPod key into chat or export
it from Actions. [Actions setup](actions.md) explains credential setup if it
actually needs repair.

## Start with fresh observations

Read current main, open PRs, the relevant workflow YAML, recent run results and
the current experiment report. Check the exact source commit's CI before paid
execution. Historical run, Pod and volume IDs are evidence, not reusable targets.

| Purpose | Workflow file | Effect |
|---|---|---|
| Global Volume / project Pod inventory | [five-stem-preflight.yml](../.github/workflows/five-stem-preflight.yml) | Read-only resource observation; does not verify mounted data or authorize a launch |
| Preferred MIG and fallback catalog | [five-stem-capacity.yml](../.github/workflows/five-stem-capacity.yml) | Read-only availability; stock is advisory |
| Existing five-stem worker progress | [five-stem-status.yml](../.github/workflows/five-stem-status.yml) | Read-only, allowlisted status for matching owned workers |
| Legacy authentication diagnostic | [runpod-auth.yml](../.github/workflows/runpod-auth.yml) | Also checks the old regional-volume capacity path; a missing volume is not proof of a bad key |

Do not recreate a deleted regional volume just because an old instruction names
it. Current benchmark launches require an existing Global Volume
(`ensure_global(create=False)`). An inventory's generic `pending` fields are not
a substitute for current code, experiment reports or approval scope.

## Select the workflow, not a remembered command

| Requested experiment | Workflow | Paid input / additional choice |
|---|---|---|
| Official five-stem inference | [runpod-five-stem.yml](../.github/workflows/runpod-five-stem.yml) | `confirm_paid_benchmark`; optional `allow_4090_fallback` defaults false |
| Bounded-memory inference | [runpod-bounded.yml](../.github/workflows/runpod-bounded.yml) | `confirm_paid_benchmark` |
| Spleeter-4 vs HTDemucs | [runpod-benchmark.yml](../.github/workflows/runpod-benchmark.yml) | `confirm_paid_benchmark` |
| Spleeter-2 vs Mel-Band RoFormer | [runpod-vocal-benchmark.yml](../.github/workflows/runpod-vocal-benchmark.yml) | `confirm_paid_vocal` |

Other training/recovery workflows are indexed by [README.md](README.md).
The initial `runpod-smoke.yml` is a legacy synthetic RTX 4090 workflow; it is not
the preferred MIG benchmark entry point.

At the reviewed commit, benchmark defaults prefer one RTX PRO 6000 MIG 24 GB,
cap its hourly compute rate at US$0.69 and use a 60-minute lease. The five-stem
workflow can explicitly allow one RTX 4090 at up to US$1/hour and describes a
US$2 project budget for that fallback. Recheck these values before dispatch.
The shared budget helper's US$1 reserve and US$500 monthly check are **not**
a total experiment cost cap or authorization for the fallback. Honor the user's
actual project budget, including already consumed attempts and storage/request
costs. Existing valid approval need not be requested again within its scope.

## Dispatch from the next environment

1. Discover current GitHub tools. Prefer a documented workflow-dispatch operation
   if exposed. A GET-only fetch tool or a tool for rerunning an old workflow is
   not a fresh dispatch API.
2. If no dispatcher exists, use an **already authenticated** `gh` environment
   with access to this repository. Do not print tokens or assume the connector
   login is also available to the CLI.
3. If neither exists, name the precise blocker: **workflow dispatch unavailable
   in this environment**, not “RunPod is unavailable.” Provide the existing
   Actions workflow and concrete inputs so the user can start it. Resume
   monitoring from the resulting run ID. Browser interaction is optional and
   should follow the user's environment preference.
4. Do not silently alter workflow triggers, bypass paid-input checks, or retry a
   potentially successful provisioning request to overcome a tool limitation.

For an authenticated CLI, these commands run **read-only provider checks**:

```bash
gh workflow run five-stem-preflight.yml --repo jasonjiang8866/spleeterpt --ref main
gh workflow run five-stem-capacity.yml --repo jasonjiang8866/spleeterpt --ref main
```

After the exact five-stem experiment and budget are authorized, the MIG-only
launch is:

```bash
gh workflow run runpod-five-stem.yml \
  --repo jasonjiang8866/spleeterpt --ref main \
  -f confirm_paid_benchmark=true -f allow_4090_fallback=false
```

Set `allow_4090_fallback=true` only when that fallback and its cost are within the
user's authorization. Confirm the run's recorded `head_sha` matches the reviewed
main commit; a moving branch must not be treated as an immutable revision.
Record the returned run URL/ID. If the CLI does not return one, list recent runs
and correlate workflow, branch, event, creation time and head SHA instead of
assuming the latest run belongs to this request.

Replace `RUN_ID` with that verified numeric run ID:

```bash
gh run view RUN_ID --repo jasonjiang8866/spleeterpt
gh run download RUN_ID --repo jasonjiang8866/spleeterpt --dir runpod-evidence
```

Read results and logs through supported GitHub tools when available. Download
artifacts into a new directory per run to avoid mixing evidence. A failed run can
still have useful artifacts. Official CLI references:
[workflow run](https://cli.github.com/manual/gh_workflow_run),
[run view](https://cli.github.com/manual/gh_run_view),
[run download](https://cli.github.com/manual/gh_run_download).

## Finish the lifecycle

- Check both job outcome and task-specific result acceptance. Downloaded audio or
  a completed worker alone does not mean a validation gate passed.
- Verify `cleanup_confirmed` where provided and refresh owned-Pod inventory.
  A job timeout, cancellation or closed conversation does not stop billing.
- The scheduled `runpod-cleanup.yml` reaps **expired, ownership-verified**
  temporary Pods; it is not an immediate “delete any Pod” operation or a
  provider-side shutdown guarantee. Reconcile an ambiguous create response by
  exact run/attempt identity before considering another rental.
- Retain Global Volume data and verify result/input hashes. Do not delete or
  recreate storage as part of compute cleanup. Actions artifact retention is
  finite; archive durable evidence before it expires.
- Report source commit, workflow run link, acceptance result, estimated compute
  charges and cleanup status. Do not include credentials, raw environments or
  private account balances.

## Paste this into a new conversation

> 请继续仓库 jasonjiang8866/spleeterpt。先读取 main 的 AGENTS.md、
> runpod/agent-handoff.md 和本次实验对应的 workflow YAML，再核实当前 PR、
> Actions、资源和已有授权。RunPod 通过 GitHub Actions 内的 RUNPOD_API_KEY
> Secret 调用，复用现有控制器、Global Volume 校验、结果验证和清理流程。
> 不需要把 API key 给新对话。先做只读预检；付费运行沿用明确有效的项目预算，
> 不把历史额度当作新授权。如果缺少 Actions dispatch 工具，准确说明触发接口
> 的限制，并使用已有认证的 gh 或提供现有工作流的手动启动入口。
