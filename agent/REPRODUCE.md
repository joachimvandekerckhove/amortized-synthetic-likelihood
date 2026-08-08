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
2. `asl.toml` — optional user overrides (see Configuration)
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
| Python 3.11+ | pipeline runtime (`tomllib` in stdlib; 3.10 fails at import) |
| `make` | orchestrates the four stages |
| JAGS 4.x | Bayesian recovery via `py2jags` |
| `g++` | compiles the JNNX JAGS module |
| `pkg-config` | JNNX-generated Makefile locates JAGS headers |
| ONNX Runtime C/C++ SDK | links the JAGS module to the ONNX emulator (auto-downloaded; see Configuration) |

On Debian/Ubuntu:

```bash
sudo apt install jags pkg-config g++ make
```

The ONNX Runtime SDK is downloaded automatically into `vendor/` on the first
`wire-to-jags` step. To prefetch:

```bash
make bootstrap-ort
```

Manual download (only if auto-download is unavailable on your platform):

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
tar xf onnxruntime-linux-x64-1.23.2.tgz
```

Set `wire.onnxruntime_dir` in `asl.toml` to the extracted directory if needed.

## Python environment

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ".[jags,dev]"
```

For GPU training on Pascal / sm_61 GPUs (e.g. GTX 1080 Ti), reinstall PyTorch
with CUDA after the editable install:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

Verify:

```bash
python -c "import asl; import jnnx; print('OK')"
pytest
```

All tests must pass before running the pipeline.

## Tested platform

The paper-scale pipeline was validated on the following stack. Other machines
may differ in wall time and small numerical details; gates and pass/fail
criteria are the same.

### Hardware

| Component | Specification |
|---|---|
| CPU | 32 logical cores |
| GPU | 2× NVIDIA GeForce GTX 1080 Ti (Pascal, sm_61) |

### Software

| Component | Version / notes |
|---|---|
| OS | Linux (Oracle Cloud VM, Ubuntu-based) |
| Python | 3.11 |
| PyTorch | 2.6.0+cu126 (CUDA 12.6 wheel) |
| JAGS | 4.x (`jags` system package) |
| g++ | system package |
| ONNX Runtime SDK | 1.23.2 (auto-downloaded to `vendor/`) |
| Package install | `pip install -e ".[jags,dev]"` |

### Verification commands

| Command | Result on tested platform |
|---|---|
| `pytest` | 105 tests passed |
| `make all` | all four models: train + wire + recovery **PASS** |

### Observed wall time (`make all`)

Recovery dominated total runtime. On the hardware above:

| Model | Recovery (approx.) |
|---|---|
| `ddm3` | 30 minutes |
| `ddm4` | 1 hour |
| `ddmcollapsesig` | 2 hours |
| `dw` | about 5.5 hours |
| **Total** | about 9 hours |

Report your own OS, Python, and GPU when filing issues or reproduction reports.

### Clean-environment check (Docker)

Independent validation in a fresh `docker run --rm` container after `git clone`
from `origin/main` (commit `16109b6`). Host: Ubuntu 22.04 with Docker 29.x.
Container image: `python:3.11-slim-bookworm` (Debian 12).

| Stage | Result | Notes |
|---|---|---|
| `pip install -e ".[jags,dev]"` | pass | no extra steps |
| `pytest` | pass | 105 tests |
| `wire-to-jags` | pass with `pkg-config` | without it: clear error from wire preflight |

**Ubuntu 22.04 default Python (3.10):** `pytest` fails immediately with
`ModuleNotFoundError: No module named 'tomllib'`. Use Python 3.11+ (e.g.
`python:3.11-slim-bookworm` or Ubuntu 24.04).

**`wire-to-jags` without `pkg-config`:** wire fails early with an install hint.
JNNX's generated Makefile needs `pkg-config` to locate JAGS module headers.

**Recovery / JAGS module load:** `recovery.py` prepends the ONNX Runtime SDK
`lib/` directory to `LD_LIBRARY_PATH` before spawning JAGS. No manual export
needed after `wire-to-jags`. If chains still fail, confirm with
`ldd /usr/lib/x86_64-linux-gnu/JAGS/modules-4/ddm3_emulator.so` that
`libonnxruntime.so.1` resolves.

**Training epochs via `ASL_CONFIG`:** `[training] training_epochs` in
`asl.toml` or `ASL_CONFIG` overrides the bundled default from `full.toml`.

**Minimum subjects for recovery summary:** `confirm-recovery` needs at least three
converged subjects to produce `recovery_summary.json` and run the coverage gate.

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

**`dw` parameterization (canonical uniform).** Training draws use
`epsilon ~ Unif(0.125, 0.375)` and `mu ~ Unif(0.075, 0.425)`; JAGS priors
and recovery true values use `epsilon ~ Unif(0.15, 0.35)` and
`mu ~ Unif(0.1, 0.4)`. See `models/social/dw_bounds.py` and `configs/dw.toml`
(R=1000 replicates per parameter draw). To regenerate training data from
scratch, run `make -C scripts/dw clean` then `make -C scripts/dw all`.

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

### Training (`results/MODEL/final_summary.json`)

- `overall_r2` must meet the model's training gate (default **>= 0.999** in
  `src/asl/presets/full.toml`; `dw` overrides to **>= 0.995** in
  `configs/dw.toml`)
- Architecture must match the table above

Reference values (approximate; small differences on CPU are fine):

| Model | Expected overall R² |
|---|---|
| `ddm3` | 0.99992 |
| `ddm4` | similar |
| `ddmcollapsesig` | 0.99969 |
| `dw` | 0.997 |

### Recovery (`results/MODEL/recovery_summary.json`, `results/MODEL/recovery_subjects.json`)

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

**dw** (canonical uniform parameters; `DeepWide_32x6`, R=1000)

| Parameter | Correlation | Coverage |
|---|---|---|
| epsilon | 0.985 | 0.96 |
| mu | 0.949 | 0.972 |

### Diagnostic plot

`figures/MODEL/recovery.pdf` should be created after recovery (2-column paper-style
multipanel plot). `results/MODEL/recovery_subjects.json` stores per-subject true,
estimated, CI, and R-hat arrays for paper figure scripts.

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
3. **`wire-to-jags` fails** — install `pkg-config` and `g++`; confirm
   `onnxruntime_dir` if overridden, or run `make bootstrap-ort`
4. **`confirm-recovery` fails with zero converged / chain exit codes** — rerun
   `wire-to-jags`; check `ldd` on the installed `ddm3_emulator.so` for missing
   `libonnxruntime.so.1` (recovery sets `LD_LIBRARY_PATH` automatically)
5. **`confirm-recovery` fails coverage** — confirm the JAGS module compiled
   successfully and `wire-to-jags` was not skipped. Check
   `results/MODEL/recovery_summary.json` for which parameter failed
6. **JAGS module not found** — rerun `wire-to-jags`; if `sudo make install`
   failed, the pipeline falls back to `LTDL_LIBRARY_PATH` — check wire output

Report back with:

- which model was run
- the exact commands executed
- contents of `final_summary.json` and `recovery_summary.json`
- whether each gate passed or failed
- any error messages from the failing stage
