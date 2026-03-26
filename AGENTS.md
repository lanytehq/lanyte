# AI Agent Guide — lanyte

Start every session with:

1. `/Users/davethompson/dev/lanytehq/AGENTS.md`
2. `/Users/davethompson/dev/lanytehq/lanyte-crucible/docs/guides/dev-warmup.md`
3. This repo's `REPOSITORY_SAFETY_PROTOCOLS.md`

## Working rules

- This repo is the Rust core workspace (TCB). Keep peer logic out of gateway routing code.
- Follow schemas-before-code: IPC contract changes land in crucible first.
- Use feature branches and PRs; no direct pushes to `main`.
- Before marking any PR ready for review, read and follow `docs/PR_CHECKLIST.md`.
  Run `make pr-final` as the local merge gate — it mirrors CI exactly.
- Keep Rust MSRV at `1.85.0` and avoid nightly features.

## Local machine overrides

Create `AGENTS.local.md` for machine-specific notes. This file is gitignored.
