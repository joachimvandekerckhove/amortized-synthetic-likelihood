## Goal

Reproduce the complete four-stage pipeline for **MODEL**:

1. generate training data (optional if committed data exists)
2. train the dual-head emulator
3. wire the emulator into JAGS via JNNX
4. run the simulate-and-recover study

The run is successful only if every stage completes with exit code 0 and the
recovery step reports `PASS` with all parameter coverages in **(0.90, 0.99)**.

## Required reading (in order)

1. `README.md` in the repository root — full reproduction guide
2. `asl.toml` — user configuration file you must edit
3. `scripts/MODEL/Makefile` — exact make targets for this model
4. Do **not** edit `src/asl/presets/full.toml`

## Repository

Clone (or use an existing checkout of):

```
https://github.com/joachimvandekerckhove/amortized-synthetic-likelihood.git
```

All commands below are run from the **repository root**.

## System prerequisites

Install before starting:

| Requirement | Purpose |
|---|---|
| Python 3.10+ | pipeline runtime |
| `make` | orchestrates the four stages |
| JAGS 4.x | Bayesian recovery via `py2jags` |
| `g++` | compiles the JNNX JAGS module |
| ONNX Runtime (pre-built) | links the JAGS module to the ONNX emulator |

On Debian/Ubuntu:

```bash
sudo apt install jags g++ make
```

Download ONNX Runtime (example for Linux x86-64):

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar xf onnxruntime-linux-x64-1.18.0.tgz
```

Note the extracted directory path; you will need it below.

## Python environment

```bash
python -m venv .venv
source .venv/bin/activate

# If a GPU is available, obtain a suitable version of torch
# For older GPUs (Pascal / sm_61, e.g. GTX 1080 Ti), use this exact wheel:
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# CPU-only alternative:
# pip install torch

pip install -e ".[jags,dev]"
```

Verify:

```bash
python -c "import asl; import jnnx; print('OK')"
pytest
```

All tests must pass before running the pipeline.

## Configuration

Edit `asl.toml` at the repo root only if you need overrides (see README).
Most users need no configuration: ONNX Runtime SDK auto-downloads on first
`wire-to-jags`, and paper-scale defaults live in `src/asl/presets/full.toml`.

```toml
[wire]
onnxruntime_dir = "/absolute/path/to/onnxruntime"  # optional override
```

Rules:

- Set `onnxruntime_dir` only to override the auto-downloaded SDK in `vendor/`.
- Do not change architecture, epoch count, or recovery subject count unless I
  explicitly ask. Defaults come from `src/asl/presets/full.toml`.

## Model-specific notes

| Model | Architecture | Committed training data? |
|---|---|---|
| `ddm3` | DeepWide_24x4 | yes (`data/ddm3/`) |
| `ddm4` | DeepWide_32x6 | yes (`data/ddm4/`) |
| `ddmcollapsesig` | DeepWide_32x6 | yes (`data/ddmcollapsesig/`) |
| `dw` | DeepWide_32x6 | yes (`data/dw/`) |

Training data generation (`make -C scripts/MODEL generate-data`) is optional
when `data/MODEL/cov_train.csv` already exists. Skip it unless I ask to
regenerate from scratch.

On CPU, training and recovery take much longer. Use available CPU cores: the
pipeline defaults to ~90% of cores for data generation and recovery parallelism.

## Run the pipeline

Execute from the repo root. Make handles dependency ordering.

**Full run (recommended):**

```bash
make -C scripts/MODEL all
```

**Step by step (use if debugging a failure):**

```bash
make -C scripts/MODEL train-emulator
make -C scripts/MODEL wire-to-jags
make -C scripts/MODEL confirm-recovery
```

If a previous partial run left corrupt artifacts, clean first:

```bash
make -C scripts/MODEL clean
make -C scripts/MODEL all
```

Do not skip `wire-to-jags`. Recovery depends on the compiled JAGS module.

After upgrading to JNNX v2, re-run `wire-to-jags` for every model even if an
older `.jnnx` package exists locally (packages are gitignored and must include
`obs_transform.json`).

## Success criteria

After the run, verify these artifacts exist and inspect their contents.

### JNNX package (`models/MODEL.jnnx/`)

Created by `wire-to-jags` (gitignored; not in the repository). Required files:

- `metadata.json` with `"version": "2.0.0"`
- `obs_transform.json` (column transforms for raw JAGS observations)
- `model.onnx`, `likelihood.json`, `scalers.json`

Optional: if `fixtures/MODEL_sl_regression.json` is present, `wire-to-jags`
also runs numerical SL regression checks. Without that file, wiring still
succeeds and only skips the regression step.

### Training (`results/MODEL/final_summary.json`)

- `overall_r2` must be **>= 0.999** (the pipeline exits non-zero otherwise)
- Architecture must match the table above

Reference values (approximate; small differences on CPU are fine):

| Model | Expected overall R² |
|---|---|
| `ddm3` | 0.99992 |
| `ddm4` | similar |
| `ddmcollapsesig` | 0.99969 |

### Recovery (`results/MODEL/recovery_summary.json`)

The pipeline prints `[recovery] PASS` on success. Check:

1. **`n_converged` / `n_attempted`** — should be close to 500/500
2. **`coverages_95ci`** — every parameter must fall strictly in **(0.90, 0.99)**
3. **`mean_rhat`** — all values should be below 1.01
4. **`correlations`** — should be high (typically > 0.94) but need not match
   reference values exactly because MCMC is stochastic

Reference recovery results (from the reference machine; yours should be
comparable, not identical):

**ddm3**

| Parameter | Correlation | Coverage |
|---|---|---|
| v | 0.997 | 0.950 |
| a | 0.995 | 0.944 |
| t0 | 0.977 | 0.938 |

**ddm4**

| Parameter | Correlation | Coverage |
|---|---|---|
| v | 0.991 | 0.948 |
| a | 0.993 | 0.954 |
| t0 | 0.971 | 0.950 |
| w | 0.990 | 0.964 |

**ddmcollapsesig**

| Parameter | Correlation | Coverage |
|---|---|---|
| a0 | 0.942 | 0.952 |
| v | 0.990 | 0.940 |
| k | 0.958 | 0.940 |
| t0 | 0.994 | 0.962 |

### Diagnostic plot

`figures/MODEL/recovery.pdf` should be created after recovery.

## What not to do

- Do not edit `src/asl/presets/full.toml`
- Do not enable `ASL_LEGACY_LIKELIHOOD` or other removed code paths (they no
  longer exist)
- Do not refactor the repository or add CLI wrappers
- Do not commit or push unless I explicitly ask
- Do not skip the coverage gate: a finished run that prints `FAIL` is not a
  successful reproduction

## If something fails

Diagnose in stage order:

1. **`pytest` fails** — fix the Python environment before running the pipeline
2. **`train-emulator` fails on R²** — check that committed training data is
   intact; try `make -C scripts/MODEL clean` and rerun
3. **`wire-to-jags` fails** — confirm `onnxruntime_dir` if overridden, or run
   `make bootstrap-ort`; ensure `g++` is installed
4. **`confirm-recovery` fails coverage** — confirm the JAGS module compiled
   successfully and `wire-to-jags` was not skipped. Check
   `results/MODEL/recovery_summary.json` for which parameter failed
5. **JAGS module not found** — rerun `wire-to-jags`; if `sudo make install`
   failed, the pipeline falls back to `LTDL_LIBRARY_PATH` — check wire output

Report back with:

- which model was run
- the exact commands executed
- contents of `final_summary.json` and `recovery_summary.json`
- whether each gate passed or failed
- any error messages from the failing stage

## Optional: high-N supplemental recovery

Only if I ask for it:

```bash
ASL_CONFIG=configs/recovery_highn.toml make -C scripts/MODEL confirm-recovery
```

This runs 50 subjects x 10,000 trials (supported for `ddm3` and `ddm4`).
