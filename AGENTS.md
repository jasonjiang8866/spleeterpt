# Repository handoff for coding agents

For RunPod requests, first read [runpod/agent-handoff.md](runpod/agent-handoff.md)
and the current workflow YAML for the requested experiment. The handoff documents
how another conversation can reuse GitHub Actions and the existing
`RUNPOD_API_KEY` repository Secret without receiving the key.

- Refresh main, open PRs and relevant Actions runs before continuing; old chat
  summaries and scratch checkouts are not the current resource inventory.
- Distinguish GitHub workflow-dispatch access from RunPod authentication.
  Inspect available tools before declaring a cloud task unsupported.
- Preserve valid user authorization within its stated project, experiment and
  budget. A monthly account budget or a historical run is not blanket spending
  authorization. Read-only preparation can continue independently.
- Use existing controllers, result validators, Global Volume verification and
  ownership-scoped cleanup. Do not dump Pod environment variables or credentials.
- Keep paid execution manual by default; do not add automatic paid push/PR
  triggers as an implicit workaround for missing workflow-dispatch capability.
- For documentation-only work, do not rent compute. Check links and relevant CI
  before merging; no new unit tests are needed for prose changes.
