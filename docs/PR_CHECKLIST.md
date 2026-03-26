# PR Validation Checklist

A PR is not ready for review or merge until the local validation surface matches
the repository's CI gate. Package-scoped checks are useful during implementation,
but final validation must be workspace-wide.

Any code change made in response to review reopens the validation gate. Rerun the
full PR-final checklist after every review-driven commit.

## PR-Final Gate

Run `make pr-final` before marking a PR ready for review or pushing review fixes.

This target mirrors CI exactly. If it passes locally, CI should pass remotely.

```bash
make pr-final
```

The target runs, in order:

1. `cargo fmt --check` — formatting must already be applied
2. `cargo clippy --workspace --all-targets -- -D warnings` — workspace-wide lint
3. `cargo test --workspace --all-targets` — workspace-wide tests
4. `cargo deny check` — license and advisory audit
5. `cargo +$(MSRV) check --workspace --locked` — MSRV compatibility

## Step-by-step procedure

### 1. Confirm repo and branch context

```bash
pwd
git status --short --branch
```

Verify you are in the `lanyte` repo on the correct branch. Wrong CWD is a common
source of `gh` CLI misdirection in a multi-repo workspace.

### 2. Sync with remote

```bash
git fetch --all --prune
```

Confirm the branch is tracking the intended remote and is not behind `main`.

### 3. Run the PR-final gate

```bash
make pr-final
```

If any step fails, fix and restart from step 3. Do not skip ahead.

### 4. If formatting was applied, recheck tree state

If you ran `cargo fmt` (or `make fmt`) to fix formatting issues:

```bash
cargo fmt --check          # must pass cleanly now
git status --short         # must show no unexpected changes
git diff --stat            # review what the formatter changed
```

Stage and commit the formatting fix, then rerun `make pr-final` from step 3.

### 5. Post-review changes require full revalidation

After any code change in response to review feedback:

```bash
# stage and commit the fix, then:
make pr-final
```

Do not rely on package-scoped checks (`cargo test -p <crate>`) for final validation.
Package-scoped runs miss workspace-level interactions (shared feature flags,
cross-crate test dependencies, workspace-level clippy lints).

### 6. Push only from a clean working tree

```bash
git status --short         # must show nothing
git push
```

### 7. If CI still fails

Compare the exact failing CI command to the local command. Do not assume equivalence
from similar command names. The CI workflow is at `.github/workflows/check.yml`.

## CI/Makefile alignment note

The `make pr-final` target matches CI exactly (no feature flags). The `make check`
target includes `--features $(WORKSPACE_TEST_FEATURES)` for broader local coverage
during development. These are intentionally different:

| Target                  | Scope          | Features              | Matches CI?  |
| ----------------------- | -------------- | --------------------- | ------------ |
| `make pr-final`         | workspace      | none (matches CI)     | yes          |
| `make check`            | workspace      | test-support features | no (broader) |
| `cargo test -p <crate>` | single package | varies                | no           |

Use `make pr-final` as the merge gate. Use `make check` for local development.

## Optional Live Provider Checks

Live provider checks are useful during adapter development, but they are not part of
CI and must not be treated as a substitute for `make pr-final`.

Maintainers can keep provider credentials in a local-only file such as:

```bash
~/.config/lanytehq/llm.env
```

Example contents:

```bash
export LANYTE_LLM_GROK_API_KEY='...'
export LANYTE_LLM_GROK_MODEL='grok-4.20-beta-latest-reasoning'
export LANYTE_LLM_CLAUDE_API_KEY='...'
export LANYTE_LLM_CLAUDE_MODEL='claude-sonnet-4-6'
export LANYTE_LLM_OPENAI_API_KEY='...'
export LANYTE_LLM_OPENAI_MODEL='gpt-5.4'
```

For Grok/xAI and OpenAI, maintainers can run:

```bash
source ~/.config/lanytehq/llm.env
make live-llm
```

This runs the opt-in `lanyte-llm` live integration test files:

```bash
cargo test -p lanyte-llm --test claude_live -- --nocapture
cargo test -p lanyte-llm --test grok_live -- --nocapture
cargo test -p lanyte-llm --test openai_live -- --nocapture
```

The Claude live coverage currently includes:

1. sync text completion
2. streaming text completion
3. function-tool round-trip using Anthropic's native message + tool-result flow

The OpenAI live coverage currently includes:

1. sync text completion
2. streaming text completion
3. function-tool round-trip using streamed tool-call capture followed by a tool-result follow-up

For a local OpenAI-compatible endpoint such as Ollama, set a local base URL and
optionally a local model name:

```bash
export LANYTE_LLM_OPENAI_BASE_URL='http://127.0.0.1:11434/v1'
export LANYTE_LLM_OPENAI_MODEL='gpt-oss:20b'
```

If the local endpoint does not require auth, the live OpenAI tests will use a
placeholder key automatically when the base URL is `127.0.0.1` or `localhost`.

Rules:

- Secrets must stay in local-only files or injected env vars. Never commit them.
- Live checks are for adapter reality-checks, not merge gating.
- If a live check reveals a provider-shape mismatch, add or update a deterministic unit test
  capturing that shape before merging.
