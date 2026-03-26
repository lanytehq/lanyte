# REPOSITORY SAFETY PROTOCOLS

This repository contains the Lanyte Rust core workspace (TCB). Treat it as high-sensitivity code.

## Never Commit

- secrets (API keys, tokens, credentials)
- private keys or signing materials
- customer data / PII

## Core Constraints

- Contracts first: schema and spec changes land in crucible before implementation.
- Gateway remains a validated router, not a business logic host.
- Prefer safe Rust. Any `unsafe` requires explicit security review.
