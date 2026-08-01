## The process

**Here the user describes the generative process in as much detail as possible.**

## Goal

Implement a new model **SLUG** end to end:

1. using the process information, define parameters, bounds, and summary statistics
2. implement a stochastic simulator: `(params, n_trials, seed) -> summaries`
3. register a multivariate `Model` spec with JAGS synthetic-likelihood hooks
4. add a `scripts/SLUG/` pipeline (Makefile + run.py)
5. add unit tests, including **parameter-sensitivity checks for every summary**
6. run smoke mode, then full mode, and confirm all gates pass

Success means:

- `pytest` passes (including your new tests)
- smoke pipeline completes (`asl.toml`: `[run] smoke = true`)
- full pipeline completes with `[train_mv] PASS` (R² >= threshold) and
  `[recovery_mv] PASS` (all coverages in **(0.90, 0.99)**)

## Required reading (in order)

Study these existing models as templates (simplest to more complex):

1. `models/ddm/ddm3.py` + `models/ddm/ddm3mv.py` — three summaries, three params
2. `models/ddm/ddmcollapsesig.py` + `models/ddm/ddmcollapsesigmv.py` — ten
   summaries, four params, parameter-sensitivity tests
3. `src/asl/spec.py` — `Model` dataclass contract
4. `src/asl/data.py` — how summary names map to log1p transforms
5. `src/asl/mv.py` — `build_sl_likelihood_line`, `emulator_output_names_for`
6. `scripts/ddm3mv/run.py` and `scripts/ddm3mv/Makefile` — pipeline wiring
7. `tests/test_ddmcollapsesig.py` — sensitivity and model-spec tests
8. `agent/REPRODUCE.md` — environment setup and gate definitions

Do **not** edit `src/asl/presets/full.toml` or `smoke.toml` unless I explicitly
ask. Override hyperparameters in `asl.toml` instead.

## Repository layout for a new model

Add files (choose informative <family> paths if your model is not a DDM variant):

```
models/<family>/<slug_base>.py      # simulator + constants
models/<family>/<slug>mv.py         # Model spec (or single file if small)
scripts/SLUG/run.py                 # four-step dispatcher
scripts/SLUG/Makefile               # make targets
tests/test_<slug>.py                # simulator + spec tests
```

Generated at runtime (gitignored):

```
data/SLUG/cov_train.csv
results/SLUG/model.onnx
results/SLUG/final_summary.json
results/SLUG/recovery_summary.json
models/SLUG.jnnx/
  obs_transform.json                  raw-summary transforms (JNNX v2; written by wire-to-jags)
figures/SLUG/recovery.pdf
```

## Step 1: Specify the generative model

Before writing code, write down (in comments or a brief design note):

| Item | Your choice |
|---|---|
| Parameters | names, meanings, bounds |
| Summary statistics | names, definitions, which are proportions vs RT-based |
| Fixed constants | anything not inferred (e.g. fixed bias, known stimulus set) |
| Architecture | `DeepWide_24x4` (<= 3 params, <= 3 summaries) or `DeepWide_32x6` (larger) |

Rules from the existing codebase:

- **Parameter bounds** must keep the simulator numerically stable (avoid
  degenerate trials where almost no paths absorb, or summaries are always NaN).
- **Summary names** drive automatic transforms: columns whose names contain
  `acc`, `rate`, or `prob` (and not `rt`) are treated as proportions; all
  others receive `log1p` before standardization (`asl.data.summary_column_masks`).
  Name RT summaries accordingly (e.g. `rt_mean`, `rt_q50`, not `mean_rt`).
- **Order matters**: `param_names`, `param_bounds`, and the `params` array passed
  to `simulate_summaries` must use the same order everywhere.

Create a plan from these instructions, then obtain user approval before proceeding.

## Step 2: Implement the simulator

Create `models/<family>/<slug_base>.py` with at minimum:

```python
PARAM_NAMES = ("...", ...)
PARAM_BOUNDS = ((lo, hi), ...)
SUMMARY_NAMES = ("...", ...)
N_SUMMARIES = len(SUMMARY_NAMES)
RECOVERY_PRIORS = {"param": "param ~ dunif(lo, hi)", ...}

def simulate_summaries(params: np.ndarray, n_trials: int, seed: int) -> np.ndarray:
    ...
```

Contract for `simulate_summaries`:

- **Input**: `params` shape `(n_params,)`, `n_trials >= 1`, integer `seed`
- **Output**: 1-D float array of length `N_SUMMARIES`
- **Deterministic given seed**: same `(params, n_trials, seed)` -> same output
- **NaN on failure**: return `np.full(N_SUMMARIES, np.nan)` when summaries are
  undefined (too few valid trials, division by zero, etc.). The training-data
  generator drops these rows silently.
- **No side effects**: do not write files or depend on global mutable state

Reuse existing simulators where possible (`models/ddm/simulator.py`, etc.)
rather than reimplementing from scratch.

## Step 3: Confirm summary statistics are sensitive to parameters

**This step is mandatory before training.** An emulator cannot support inference
if summaries do not change when a parameter changes.

For **each parameter** `p_j`, write at least one test that:

1. fixes all other parameters at interior values (not on bounds)
2. draws two values of `p_j` that differ substantially (low vs high within bounds)
3. uses the **same seed** and a large enough `n_trials` (typically ~2000)
4. asserts at least one summary differs meaningfully between the two runs

Example pattern (from `tests/test_ddmcollapsesig.py`):

```python
def test_k_affects_rt_median():
    params_low_k = np.array([1.2, 0.2, 0.1, 0.2])
    params_high_k = np.array([1.2, 0.2, 8.0, 0.2])
    low = simulate_summaries(params_low_k, n_trials=2000, seed=23)
    high = simulate_summaries(params_high_k, n_trials=2000, seed=23)
    assert np.all(np.isfinite(low))
    assert np.all(np.isfinite(high))
    assert not np.allclose(low, high)          # at least one summary differs
    assert high[3] < low[3]                    # optional: directional check
```

Also test:

- output length matches `SUMMARY_NAMES`
- finite summaries for a typical interior parameter vector
- edge cases you care about (e.g. a parameter at its lower bound if valid)

If a parameter does not move any summary, either **remove it from the model** or
**choose different summaries**. Do not proceed to training until every
parameter has demonstrated sensitivity.

Document which summary indices respond to which parameters in test docstrings.

## Step 4: Define the multivariate Model spec

Create `models/<family>/<slug>mv.py` (or combine with Step 2 if small):

```python
from asl.mv import build_sl_likelihood_line, emulator_output_names_for
from asl.spec import Model
from models.<family>.<slug_base> import (
    N_SUMMARIES, PARAM_BOUNDS, PARAM_NAMES, RECOVERY_PRIORS,
    SUMMARY_NAMES, simulate_summaries,
)

def build_<slug>_jags_likelihood(obs: dict) -> list[str]:
    return build_sl_likelihood_line(
        "SLUG", PARAM_NAMES, N_SUMMARIES,
    )

<SLUG_UPPER>MV = Model(
    slug="SLUG",
    param_names=PARAM_NAMES,
    param_bounds=PARAM_BOUNDS,
    summary_names=SUMMARY_NAMES,
    emulator_output_names=emulator_output_names_for(N_SUMMARIES, SUMMARY_NAMES),
    simulate_summaries=simulate_summaries,
    recovery_priors=RECOVERY_PRIORS,
    build_jags_likelihood=build_<slug>_jags_likelihood,
    default_architecture="DeepWide_32x6",   # or DeepWide_24x4
    default_n_epochs=10000,
)
```

Checklist:

- `slug` matches directory names in `data/`, `results/`, `scripts/`
- `emulator_output_names` is set (required for multivariate / SL path)
- `build_jags_likelihood` returns one line: `obs[1:p] ~ SLUG_sl(...)` (raw
  summaries; JNNX applies transforms from `obs_transform.json` at compile time)
- `wire-to-jags` must succeed and produce `models/SLUG.jnnx/obs_transform.json`
- `recovery_priors` use JAGS syntax consistent with `param_bounds`
- `supports_mv_recovery()` is True (automatic when hooks are set)
- `n_outputs == n_summaries + n_summaries * (n_summaries + 1) // 2` (mu + chol)

Use `source_slug` only if training data should live under a different directory
name than `slug` (unusual; see `ddm3mv`).

## Step 5: Add the pipeline script and Makefile

Copy `scripts/ddm3mv/run.py` to `scripts/SLUG/run.py`. Change:

- `SLUG` constant
- import and `register(...)` call for your `Model` instance

Copy `scripts/ddm3mv/Makefile` to `scripts/SLUG/Makefile`. Change:

- `MODEL := SLUG`

The four steps are fixed: `generate-data`, `train-emulator`, `wire-to-jags`,
`confirm-recovery`. Do not add extra CLI modes.

## Step 6: Add unit tests

Create `tests/test_<slug>.py` with at least:

1. **Sensitivity tests** (Step 3) — one per parameter
2. **Model spec tests** — `param_names`, `summary_names`, `slug`, `n_outputs`,
   `supports_mv_recovery()`
3. **Simulator sanity** — finite output for typical params, correct length

Run:

```bash
pytest tests/test_<slug>.py -v
pytest
```

All tests must pass before running the pipeline.

## Step 7: Environment and configuration

Follow `agent/REPRODUCE.md` for system packages, Python venv, and ONNX Runtime.

Edit `asl.toml`:

```toml
[run]
smoke = true    # start with smoke; switch to false for full run

[wire]
onnxruntime_dir = "/absolute/path/to/onnxruntime-linux-x64-1.18.0"
```

Optional overrides for development (in `asl.toml`, not presets):

```toml
[training]
architecture = "DeepWide_32x6"

[cov_data]
parallel_workers = 8

[recovery]
parallel_workers = 4
```

## Step 8: Smoke test (required before full run)

From the repo root:

```bash
make -C scripts/SLUG all
```

Smoke mode uses 300 epochs, 800 parameter draws, 50 recovery subjects. Expect
roughly 5–15 minutes depending on simulator cost and hardware.

Verify:

- `data/SLUG/cov_train.csv` created
- `results/SLUG/final_summary.json` — R² >= 0.995 (smoke threshold)
- `results/SLUG/recovery_summary.json` — pipeline prints `[recovery_mv] PASS`

If smoke fails, fix the simulator or summaries before enabling full mode. Common
causes: too many NaN rows in training data, summaries not sensitive to params,
bounds too wide, or architecture too small for the parameter count.

## Step 9: Full pipeline run

Set `smoke = false` in `asl.toml`, then:

```bash
make -C scripts/SLUG clean
make -C scripts/SLUG all
```

Full mode: 20,000 parameter draws, 10,000 training epochs, 500 recovery
subjects x 500 trials. This can take hours on CPU.

Gates (same as `agent/REPRODUCE.md`):

| Stage | Gate |
|---|---|
| Training | `overall_r2 >= 0.999` in `final_summary.json` |
| Recovery | every `coverages_95ci` value in **(0.90, 0.99)** |
| Recovery | `n_converged` close to `n_attempted` (500) |
| Recovery | all `mean_rhat` < 1.01 |

Report the contents of `final_summary.json` and `recovery_summary.json`.

## Step 10: Optional — commit training data

If I ask you to commit artifacts, follow the pattern of existing models:

- commit `data/SLUG/cov_train.csv` and `data/SLUG/cov_settings.json`
- do **not** commit `results/`, `figures/`, or `models/*.jnnx/` (gitignored) unless
  instructed otherwise

## What not to do

- Do not add a second emulator for the same parameter vector (one SL node per model)
- Do not add top-level CLI entrypoints beyond `scripts/SLUG/run.py`
- Do not modify core `asl` library code unless a hook is genuinely missing, and in
  that case backwards compatibility is inviolable
- Do not skip parameter-sensitivity tests
- Do not treat smoke-mode recovery as paper-scale validation
- Do not commit or push unless I explicitly ask

## If something fails

| Symptom | Likely cause | Action |
|---|---|---|
| Many invalid rows in `generate-data` | simulator returns NaN too often | tighten bounds; increase `n_trials` in cov_data preset via smoke first |
| Training R² below threshold | insensitive summaries; wrong transforms; too few rows | rerun Step 3; check summary names; try larger architecture |
| Recovery coverage too low | emulator bias; poor MLE init | check `wire-to-jags` succeeded; inspect `figures/SLUG/recovery.pdf` |
| Recovery coverage too high | overdispersed emulator; SL miswired | confirm `build_sl_likelihood_line` slug matches JNNX package name |
| `wire-to-jags` fails | missing ONNX Runtime or g++ | set `onnxruntime_dir`; install build tools |
| Parameter not recovered | summary not sensitive to that param | return to Step 3; add or replace summaries |

## Deliverables

When finished, report:

1. file list created or modified
2. parameter -> summary sensitivity map (from tests)
3. smoke and full gate results (JSON summaries)
4. total wall time per stage
5. anything that required deviation from this checklist
