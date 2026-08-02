# Agent prompts

Copy-paste instructions for AI coding agents working with this repository.
Each file is a self-contained prompt: open it, copy the full contents into
your agent, substitute any placeholders, and send.

For human-readable setup and pipeline documentation, see the repository root
[`README.md`](../README.md).

For most use cases, relatively cheap "frontier-adjacent" coding models suffice.

## Files

| File | Use when |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | Running the existing pipeline for `ddm3`, `ddm4`, `ddmcollapsesig`, or `dw` on a fresh machine and confirming results pass the automated gates |
| [`EXTEND.md`](EXTEND.md) | Implementing a **new** generative model and wiring it into the multivariate synthetic-likelihood framework |

## Placeholders

Replace these before sending the prompt:

| Placeholder | Where | Meaning |
|---|---|---|
| `MODEL` | `REPRODUCE.md` | One of `ddm3`, `ddm4`, `ddmcollapsesig`, `dw` |
| `SLUG` | `EXTEND.md` | Short identifier for your new model (lowercase, no spaces), used in paths and JAGS node names |
| `## The process` | `EXTEND.md` | Fill in this section with a detailed description of the generative process before sending |

## Expected agent behavior

Both prompts assume the agent can:

- read repository documentation and follow existing conventions
- create and build plans as needed
- run commands on the user's machine (not just describe them)
- treat pipeline gates as hard success criteria (training R², recovery coverage)
- avoid unrelated refactors, preset edits, or commits unless explicitly asked

`EXTEND.md` additionally requires parameter-sensitivity tests for every
free parameter before any emulator training.
