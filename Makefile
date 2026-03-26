SHELL := /bin/bash
MSRV := $(shell awk -F\" '/^rust-version =/ {print $$2; exit}' Cargo.toml)
WORKSPACE_TEST_FEATURES := lanyte-gateway/test-support,lanyte-orchestrator/test-support

.PHONY: all clean help check fmt quality test build deny msrv check-all precommit prepush pr-final live-llm install-hooks

all: check

clean:
	cargo clean

help:
	@echo "Targets: all clean help check fmt quality test build deny msrv check-all precommit prepush pr-final install-hooks"
	@echo ""
	@echo "  pr-final    CI-exact merge gate (see docs/PR_CHECKLIST.md)"
	@echo "  live-llm    optional provider-backed LLM smoke tests (local secrets only)"

check:
	cargo fmt --check
	cargo clippy --workspace --all-targets --features $(WORKSPACE_TEST_FEATURES) -- -D warnings
	cargo test --workspace --all-targets --features $(WORKSPACE_TEST_FEATURES)

fmt:
	cargo fmt
	@if command -v goneat >/dev/null 2>&1; then \
		goneat format --types yaml,json,markdown --folders . --finalize-eof --quiet; \
	else \
		echo "goneat not found; skipping non-Rust formatting"; \
	fi

quality:
	cargo clippy --workspace --all-targets --features $(WORKSPACE_TEST_FEATURES) -- -D warnings
	@if command -v goneat >/dev/null 2>&1; then \
		goneat assess . --categories lint --check; \
	else \
		echo "goneat not found; skipping goneat assess"; \
	fi

test:
	cargo test --workspace --all-targets --features $(WORKSPACE_TEST_FEATURES)

build:
	cargo build --workspace --all-targets --features $(WORKSPACE_TEST_FEATURES)

deny:
	cargo deny check

msrv:
	@echo "Checking MSRV $(MSRV)..."
	@if ! rustup toolchain list | grep -q "$(MSRV)"; then \
		echo "Installing toolchain $(MSRV)..."; \
		rustup toolchain install $(MSRV) --profile minimal; \
	fi
	cargo +$(MSRV) check --workspace --locked
	@echo "[ok] MSRV $(MSRV) verified"

# PR-final gate: mirrors CI exactly (.github/workflows/check.yml).
# Use this before marking a PR ready for review. See docs/PR_CHECKLIST.md.
pr-final:
	cargo fmt --check
	cargo clippy --workspace --all-targets -- -D warnings
	cargo test --workspace --all-targets
	cargo deny check
	@echo "Checking MSRV $(MSRV)..."
	@if ! rustup toolchain list | grep -q "$(MSRV)"; then \
		echo "Installing toolchain $(MSRV)..."; \
		rustup toolchain install $(MSRV) --profile minimal; \
	fi
	cargo +$(MSRV) check --workspace --locked
	@echo "[ok] pr-final gate passed"

live-llm:
	cargo test -p lanyte-llm --test claude_live -- --nocapture
	cargo test -p lanyte-llm --test grok_live -- --nocapture
	cargo test -p lanyte-llm --test openai_live -- --nocapture

check-all: check deny

precommit: check fmt quality

prepush: precommit test build deny

install-hooks:
	@git config core.hooksPath .githooks
	@echo "Configured git hooks path to .githooks"
