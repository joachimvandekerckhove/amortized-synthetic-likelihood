# Agent instructions

This directory contains prompts for AI coding agents working with the
companion repository. Each file is self-contained: give the full contents to
your agent, substitute placeholders where noted, and let the agent run commands
on the user's machine.

For human-readable setup and reproduction, see the repository root
[`README.md`](../README.md).

## Files

| File | Use when |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | Re-running an **existing** model (`ddm3`, `ddm4`, `ddmcollapsesig`, `dw`) on a fresh machine and confirming automated gates pass |
| [`EXTEND.md`](EXTEND.md) | **Adding a new model** — install the repo, implement the simulator and pipeline, run the full train/wire/recovery workflow |

`EXTEND.md` is written as an LLM skill: it instructs the agent to install the
repository, gather the generative process from a naive external user, follow
repository conventions, and treat pipeline gates as hard success criteria.

## Placeholders

| Placeholder | File | Meaning |
|---|---|---|
| `MODEL` | `REPRODUCE.md` | One of `ddm3`, `ddm4`, `ddmcollapsesig`, `dw` |
| `SLUG` | `EXTEND.md` | Short identifier for a new model (lowercase, no spaces) |

## Expected agent behavior

- Read repository documentation and match existing conventions
- Run commands on the user's machine; do not only describe them
- Treat training R² and recovery coverage gates as pass/fail criteria
- Avoid unrelated refactors, preset edits, or commits unless explicitly asked

`EXTEND.md` additionally requires parameter-sensitivity tests for every free
parameter before emulator training.
