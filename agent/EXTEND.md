---
name: extend-asl-companion
description: >-
  Install the amortized synthetic likelihood companion repository and implement
  a new generative model end-to-end (simulator, Model spec, pipeline, tests,
  full train/wire/recovery). Use when an external user asks to add a model to
  this repo or extend the ASL framework with a new example.
---

# Extend the ASL companion repository with a new model

You are an AI coding agent invoked on behalf of an **external user** who does
not know this codebase. They will describe a generative cognitive or statistical
model in plain language. Your job is to:

1. Install and verify this repository on their machine.
2. Elicit any missing specification details in simple terms.
3. Implement the model following existing conventions exactly.
4. Run the full four-stage pipeline and report pass/fail gate results.

**Run commands yourself.** Do not only list instructions for the user to follow
unless a step genuinely requires their credentials or physical action (e.g.
`sudo apt install`).

Do not refactor unrelated code. Do not edit `src/asl/presets/full.toml`.
Do not commit or push unless the user explicitly asks.

## When to use this skill

Use this skill when:

- The user wants a **new model slug** added to this repository.
- They have (or will provide) parameters, summary statistics, and a simulator.
- They expect the same workflow as the shipped examples (`ddm3`, `ddm4`,
  `ddmcollapsesig`, `dw`).

**Stop and use `agent/REPRODUCE.md` instead** if the user only wants to re-run
an existing model.

## How to work with a naive external user

The user may not know repository jargon. Translate between their language and
implementation choices:

| They might say | You need to determine |
|---|---|
| "fit my model" | Parameter names, bounds, summaries, trial structure |
| "recovery study" | `confirm-recovery` stage (500 synthetic subjects by default) |
| "train the network" | `train-emulator` stage |
| "hook it into JAGS/Bayesian inference" | `wire-to-jags` stage |

**Before editing any file:**

1. Restate the model in your own words (parameters, summaries, simulator).
2. Propose `SLUG`, architecture (`DeepWide_24x4` vs `DeepWide_32x6`), and
   which existing example to copy (`ddm3`, `ddmcollapsesig`, or `dw`).
3. Ask clarifying questions if bounds, summary definitions, or transforms are
   ambiguous.
4. Wait for user approval of the plan.

Use short, non-technical status updates while long pipeline stages run.

## Phase 0 — Lock the specification

Obtain and record:

| Item | What you need |
|---|---|
| **SLUG** | Short lowercase identifier (e.g. `mytask`). Used in paths and JAGS. |
| **Parameters** | Names, meanings, training bounds, inference/prior bounds. |
| **Summaries** | Names, definitions, count, which need `log1p` vs `identity`. |
| **Simulator** | How `(params, n_trials, seed)` produces one summary vector. |
| **Architecture** | `DeepWide_24x4` if ≤3 params and ≤3 summaries; else `DeepWide_32x6`. |

Read these templates in the repository before writing code (do not guess APIs):

1. `models/ddm/ddm3.py` — minimal three-parameter example
2. `models/ddm/ddmcollapsesig.py` — four parameters, non-trivial simulator
3. `models/social/dw.py` + `models/social/dw_bounds.py` — custom training
   draws, separate training vs prior bounds
4. `src/asl/spec.py` — `Model` dataclass contract
5. `scripts/ddm3/run.py` and `scripts/ddm3/Makefile` — pipeline wiring
6. `tests/test_ddmcollapsesig.py` — parameter-sensitivity tests

## Phase 1 — Install and verify the environment

Run on the user's machine from a clean shell. Use their fork URL if they
provide one.

```bash
git clone https://github.com/joachimvandekerckhove/amortized-synthetic-likelihood.git
cd amortized-synthetic-likelihood

# System packages (Debian/Ubuntu example)
sudo apt install jags pkg-config g++ make

python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[jags,dev]"
```

Optional GPU training (Pascal / sm_61, e.g. GTX 1080 Ti):

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

Verify before implementing anything new:

```bash
python -c "import asl; import jnnx; print('OK')"
pytest
```

Both commands must succeed. Python **3.11+** is required (`tomllib` in stdlib).

ONNX Runtime for JAGS module compilation downloads automatically on the first
`wire-to-jags` step into `vendor/`. Optional prefetch:

```bash
make bootstrap-ort
```

Optional user overrides: `asl.toml` at the repo root, or `configs/<slug>.toml`
merged via `ASL_CONFIG` (see `configs/dw.toml`).

## Phase 2 — Implement the simulator

Create `models/<family>/<slug>.py`.

Minimum exports:

```python
PARAM_NAMES = ("...", ...)
PARAM_BOUNDS = ((lo, hi), ...)          # training / emulator support
PRIOR_PARAM_BOUNDS = ((lo, hi), ...)    # recovery + JAGS priors (may equal PARAM_BOUNDS)
SUMMARY_NAMES = ("...", ...)
N_SUMMARIES = len(SUMMARY_NAMES)
SUMMARY_TRANSFORMS = ("log1p", ..., "identity")  # one per summary

def simulate_summaries(params: np.ndarray, n_trials: int, seed: int) -> np.ndarray:
    ...
```

Contract for `simulate_summaries`:

- **Input:** `params` shape `(n_params,)`, `n_trials >= 1`, integer `seed`
- **Output:** 1-D `float64` array of length `N_SUMMARIES`
- **Deterministic:** same `(params, n_trials, seed)` → same output
- **Failure:** return `np.full(N_SUMMARIES, np.nan)` when undefined; training
  data generation drops these rows
- **No side effects:** no file I/O or global mutable state

Reuse existing simulators (`models/ddm/simulator.py`, etc.) when applicable.

If training draws need a distribution other than uniform on `param_bounds`, set
`draw_cov_parameters` on the `Model` (see `models/social/dw.py`).

## Phase 3 — Prove every parameter moves at least one summary

**Mandatory before training.** For **each** parameter `p_j`, add a test that:

1. fixes other parameters at interior values
2. uses two substantially different values of `p_j` within bounds
3. uses the **same** `seed` and large `n_trials` (typically ~2000)
4. asserts summaries are finite and not all equal

```python
def test_k_affects_summaries():
    low = simulate_summaries(params_low, n_trials=2000, seed=23)
    high = simulate_summaries(params_high, n_trials=2000, seed=23)
    assert np.all(np.isfinite(low))
    assert np.all(np.isfinite(high))
    assert not np.allclose(low, high)
```

If a parameter does not affect any summary, revise the model design. Do not
proceed to emulator training until every parameter passes.

## Phase 4 — Define the Model spec and register it

In the same module, add JAGS hooks and the `Model` instance:

```python
from asl.cholesky import build_sl_likelihood_line, emulator_output_names_for
from asl.spec import Model

RECOVERY_PRIORS = {"param": "param ~ dunif(lo, hi)", ...}

def build_jags_likelihood(obs: dict) -> list[str]:
    del obs
    return build_sl_likelihood_line("SLUG", PARAM_NAMES, N_SUMMARIES)

SLUG_MODEL = Model(
    slug="SLUG",
    param_names=PARAM_NAMES,
    param_bounds=PARAM_BOUNDS,
    prior_bounds=PRIOR_PARAM_BOUNDS,
    summary_names=SUMMARY_NAMES,
    summary_transforms=SUMMARY_TRANSFORMS,
    emulator_output_names=emulator_output_names_for(N_SUMMARIES, SUMMARY_NAMES),
    simulate_summaries=simulate_summaries,
    recovery_priors=RECOVERY_PRIORS,
    build_jags_likelihood=build_jags_likelihood,
    default_architecture="DeepWide_32x6",
)
```

Register in `models/catalog.py` inside `get_model()`.

Checklist:

- `slug` matches `data/SLUG/`, `results/SLUG/`, `scripts/SLUG/`
- `build_jags_likelihood` returns `obs[1:p] ~ SLUG_sl(...)` via
  `build_sl_likelihood_line`
- `prior_bounds` is set on every model
- `n_outputs == n_summaries + n_summaries * (n_summaries + 1) // 2`

## Phase 5 — Add the pipeline

Copy and adapt:

- `scripts/ddm3/run.py` → `scripts/SLUG/run.py` (change imported `Model`)
- `scripts/ddm3/Makefile` → `scripts/SLUG/Makefile` (set `MODEL := SLUG`)

The four Makefile targets are fixed:

1. `generate-data`
2. `train-emulator`
3. `wire-to-jags`
4. `confirm-recovery`

Plus `clean` and `clean-generated` only. Do not add other targets.

Add `make SLUG` to the root `Makefile` if this is a paper example (see `dw`).

If the model needs non-default hyperparameters, add `configs/SLUG.toml` and set
`ASL_CONFIG` in the model Makefile (see `configs/dw.toml`).

## Phase 6 — Add unit tests

Create `tests/test_<slug>.py` with:

1. sensitivity tests (Phase 3) — one per parameter
2. model spec tests — names, `slug`, `n_outputs`, `supports_recovery()`
3. catalog test — `get_model("SLUG")` returns your model
4. simulator sanity — shape, finiteness for typical params

```bash
pytest tests/test_<slug>.py -v
pytest
```

All tests must pass before running the pipeline.

## Phase 7 — Run the full pipeline

From the repository root:

```bash
make -C scripts/SLUG all
```

This runs generate-data (if needed) → train → wire → recovery. To regenerate
training data from scratch:

```bash
make -C scripts/SLUG clean
make -C scripts/SLUG all
```

On failure, clean generated artifacts and retry:

```bash
make -C scripts/SLUG clean-generated
```

Do not skip `wire-to-jags`. Recovery requires the compiled JAGS module.

Tell the user when each stage starts and when it finishes. Pipeline stages can
take hours on CPU.

## Phase 8 — Verify automated gates

Success requires **all** of the following:

| Stage | Gate |
|---|---|
| Training | `[train] PASS` — `overall_r2` in `results/SLUG/final_summary.json` meets threshold (default **≥ 0.999**; override in `configs/SLUG.toml` if needed, as `dw` uses **≥ 0.995**) |
| Recovery | `[recovery] PASS` printed |
| Recovery | every `coverages_95ci` in **(0.90, 0.99)** |
| Recovery | `n_converged` close to `n_attempted` (default 500) |
| Recovery | all `mean_rhat` < 1.01 |

Report `final_summary.json` and `recovery_summary.json` to the user in plain
language (what passed, what failed, and what it means).

Paper-scale defaults (`src/asl/presets/full.toml`): 20,000 parameter draws,
25,000 training epochs (20,000 for `dw`), 500 recovery subjects.

## Phase 9 — Commit training data (only if asked)

If the user asks to commit artifacts:

- commit `data/SLUG/cov_train.csv` and `data/SLUG/cov_settings.json`
- do **not** commit `results/`, `figures/`, or `models/*.jnnx/` (gitignored)

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Many invalid rows in `generate-data` | simulator returns NaN too often | tighten bounds; increase `trials_per_replicate` in config |
| Training R² below threshold | insensitive summaries; wrong transforms | redo Phase 3; check `summary_transforms`; try larger architecture |
| `wire-to-jags` fails | missing `pkg-config`, g++, or ONNX Runtime | install build tools; run `make bootstrap-ort` |
| Recovery coverage out of range | emulator bias; wiring error | confirm `wire-to-jags` succeeded; inspect `figures/SLUG/recovery.pdf` |
| Parameter not recovered | summary insensitive to that param | redo Phase 3; revise summaries |

## Forbidden actions

- Do not add smoke-test configs, abbreviated pipelines, diagnostic-only scripts,
  or Makefile targets beyond the four pipeline steps plus `clean` /
  `clean-generated`.
- Do not add a second synthetic-likelihood node for the same parameter vector.
- Do not add top-level CLI entrypoints beyond `scripts/SLUG/run.py`.
- Do not modify core `asl` library code unless a hook is genuinely missing; if
  you must, preserve backward compatibility for existing models.
- Do not skip parameter-sensitivity tests.
- Do not commit or push unless the user explicitly asks.

## Deliverables

When finished, report to the user:

1. Files created or modified
2. Parameter → summary sensitivity map (from tests)
3. `final_summary.json` and `recovery_summary.json` gate results
4. Approximate wall time per pipeline stage
5. Any deviation from this checklist and why
