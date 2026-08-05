## The process

**Here the user describes the generative process in as much detail as possible.**

## Goal

Implement a new model **SLUG** end to end:

1. using the process information, define parameters, bounds, and summary statistics
2. implement a stochastic simulator: `(params, n_trials, seed) -> summaries`
3. define a `Model` spec with JAGS synthetic-likelihood hooks and register it in `models/catalog.py`
4. add a `scripts/SLUG/` pipeline (Makefile + run.py)
5. add unit tests, including **parameter-sensitivity checks for every summary**
6. run the full pipeline and confirm all gates pass

Success means:

- `pytest` passes (including your new tests)
- full pipeline completes with `[train] PASS` (R² >= threshold) and
  `[recovery] PASS` (all coverages in **(0.90, 0.99)**)

## Required reading (in order)

Study these existing models as templates (simplest to more complex):

1. `models/ddm/ddm3.py` — three summaries, three params
2. `models/ddm/ddmcollapsesig.py` — four summaries, four params
3. `models/social/dw.py` — non-RT summaries, custom training draws, prior bounds
4. `src/asl/spec.py` — `Model` dataclass contract
5. `src/asl/data.py` — per-summary transforms via `Model.summary_transforms`
6. `src/asl/cholesky.py` — `build_sl_likelihood_line`, `emulator_output_names_for`
7. `scripts/ddm3/run.py` and `scripts/ddm3/Makefile` — pipeline wiring
8. `tests/test_ddmcollapsesig.py` — sensitivity and model-spec tests
9. `agent/REPRODUCE.md` — environment setup and gate definitions

Do **not** edit `src/asl/presets/full.toml` unless I explicitly ask. Override
hyperparameters in `asl.toml` or via `ASL_CONFIG` instead.

## Repository layout for a new model

Add files (choose informative <family> paths if your model is not a DDM variant):

```
models/<family>/<slug>.py           # simulator + Model spec
scripts/SLUG/run.py                 # four-step dispatcher
scripts/SLUG/Makefile               # make targets
tests/test_<slug>.py                # simulator + spec tests
data/SLUG/cov_train.csv             # committed training data (after full run)
```

Generated at runtime (gitignored):

```
data/SLUG/cov_train.csv
results/SLUG/model.onnx
results/SLUG/final_summary.json
results/SLUG/recovery_summary.json
results/SLUG/recovery_subjects.json
models/SLUG.jnnx/
  obs_transform.json                  raw-summary transforms (JNNX v2; written by wire-to-jags)
figures/SLUG/recovery.pdf             2-column paper-style recovery multipanel
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
- **Training vs prior bounds**: set `param_bounds` for emulator training / cov_data
  draws and `prior_bounds` for recovery subject draws (required on every model).
- **Summary transforms**: set `summary_transforms` explicitly (`"identity"` or
  `"log1p"`) — one entry per `summary_names` column.
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

## Step 4: Define the Model spec

In `models/<family>/<slug>.py` (or extend the simulator file from Step 2), add the
`Model` instance. Follow `models/ddm/ddm3.py` or `models/social/dw.py`:

```python
from asl.cholesky import build_sl_likelihood_line, emulator_output_names_for
from asl.spec import Model
# import simulator constants and simulate_summaries from the same module

def build_jags_likelihood(obs: dict) -> list[str]:
    del obs
    return build_sl_likelihood_line("SLUG", PARAM_NAMES, N_SUMMARIES)

SLUG_MODEL = Model(
    slug="SLUG",
    param_names=PARAM_NAMES,
    param_bounds=PARAM_BOUNDS,
    prior_bounds=PRIOR_BOUNDS,
    summary_names=SUMMARY_NAMES,
    summary_transforms=SUMMARY_TRANSFORMS,
    emulator_output_names=emulator_output_names_for(N_SUMMARIES, SUMMARY_NAMES),
    simulate_summaries=simulate_summaries,
    recovery_priors=RECOVERY_PRIORS,
    build_jags_likelihood=build_jags_likelihood,
    default_architecture="DeepWide_32x6",   # or DeepWide_24x4
)
```

If training draws need a different distribution than uniform `param_bounds`, set
`draw_cov_parameters` (see `models/social/dw.py`). `prior_bounds` is required
on every model.

Register the model in `models/catalog.py` (`get_model` lookup).

Checklist:

- `slug` matches directory names in `data/`, `results/`, `scripts/`
- `emulator_output_names` is set (required for multivariate / SL path)
- `build_jags_likelihood` returns one line: `obs[1:p] ~ SLUG_sl(...)` (raw
  summaries; JNNX applies transforms from `obs_transform.json` at compile time)
- `wire-to-jags` must succeed and produce `models/SLUG.jnnx/obs_transform.json`
- `recovery_priors` use JAGS syntax consistent with inference bounds
- `supports_recovery()` is True (automatic when hooks are set)
- `n_outputs == n_summaries + n_summaries * (n_summaries + 1) // 2` (mu + chol)

## Step 5: Add the pipeline script and Makefile

Copy `scripts/ddm3/run.py` to `scripts/SLUG/run.py`. Change the imported
`Model` constant (keep lazy step imports).

Copy `scripts/ddm3/Makefile` to `scripts/SLUG/Makefile`. Change:

- `MODEL := SLUG`

For a paper example, also add `make SLUG` to the root `Makefile` (see `dw`).

The four steps are fixed: `generate-data`, `train-emulator`, `wire-to-jags`,
`confirm-recovery`. Do not add extra CLI modes.

## Step 6: Add unit tests

Create `tests/test_<slug>.py` with at least:

1. **Sensitivity tests** (Step 3) — one per parameter
2. **Model spec tests** — `param_names`, `summary_names`, `slug`, `n_outputs`,
   `supports_recovery()`
3. **Catalog** — `get_model("SLUG")` returns your `Model` (see `tests/test_catalog.py`)
4. **Simulator sanity** — finite output for typical params, correct length

Run:

```bash
pytest tests/test_<slug>.py -v
pytest
```

All tests must pass before running the pipeline.

## Step 7: Environment and configuration

Follow `agent/REPRODUCE.md` for system packages, Python venv, and ONNX Runtime.

`asl.toml` is optional for most runs. Override hyperparameters there or via
`ASL_CONFIG` (see README).

```toml
[wire]
onnxruntime_dir = "/absolute/path/to/onnxruntime"  # optional
```

Optional overrides for development:

```toml
[training]
architecture = "DeepWide_32x6"

[cov_data]
parallel_workers = 8

[recovery]
parallel_workers = 4
```

## Step 8: Run the pipeline

From the repo root:

```bash
make -C scripts/SLUG all
```

For a quicker recovery check during development, layer a smaller override:

```bash
ASL_CONFIG=configs/recovery_highn.toml make -C scripts/SLUG confirm-recovery
```

(Customize a config file with fewer subjects or trials as needed.)

Verify:

- `data/SLUG/cov_train.csv` created (or committed data present)
- `results/SLUG/final_summary.json` — R² >= 0.999 in `final_summary.json`
- `results/SLUG/recovery_summary.json` — pipeline prints `[recovery] PASS`
- `results/SLUG/recovery_subjects.json` — per-subject arrays for paper figures

If training fails, fix the simulator or summaries before a full rerun. Common
causes: too many NaN rows in training data, summaries not sensitive to params,
bounds too wide, or architecture too small for the parameter count.

## Step 9: Gates

Paper-scale defaults: 20,000 parameter draws, 10,000 training epochs, 500
recovery subjects x 500 trials. Wall time depends on hardware (see README
reference machine).

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
- Do not commit or push unless I explicitly ask

## If something fails

| Symptom | Likely cause | Action |
|---|---|---|
| Many invalid rows in `generate-data` | simulator returns NaN too often | tighten bounds; increase `trials_per_replicate` via `asl.toml` |
| Training R² below threshold | insensitive summaries; wrong transforms; too few rows | rerun Step 3; check summary names; try larger architecture |
| Recovery coverage too low | emulator bias; poor chain mixing | check `wire-to-jags` succeeded; inspect `figures/SLUG/recovery.pdf` |
| Recovery coverage too high | overdispersed emulator; SL miswired | confirm `build_sl_likelihood_line` slug matches JNNX package name |
| `wire-to-jags` fails | missing ONNX Runtime or g++ | set `onnxruntime_dir`; install build tools |
| Parameter not recovered | summary not sensitive to that param | return to Step 3; add or replace summaries |

## Deliverables

When finished, report:

1. file list created or modified
2. parameter -> summary sensitivity map (from tests)
3. gate results (JSON summaries from training and recovery)
4. total wall time per stage
5. anything that required deviation from this checklist
