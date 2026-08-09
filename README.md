# Amortized synthetic likelihood — Reproduction Guide

Reproducibility package for the paper *Amortized synthetic likelihoods in
Bayesian graphical models*. Four models are covered:

| Model | Parameters | Summaries | Architecture | Epochs |
|---|---|---|---|---|
| `ddm3` | v, a, t0 | acc, rt\_mean, rt\_var | DeepWide\_24x4 | 25,000 |
| `ddm4` | v, a, t0, w | rt\_mean/var (corr + err), err\_rate | DeepWide\_32x6 | 25,000 |
| `ddmcollapsesig` | a0, v, k, t0 | acc, rt\_q10, var\_t1, var\_t3\_minus\_t1 | DeepWide\_32x6 | 25,000 |
| `dw` | epsilon, mu | 6 opinion-dynamics summaries | DeepWide\_32x6 | 20,000 |

The `ddmcollapsesig` model has a sigmoid collapsing boundary
`a(t) = a0 / (1 + exp(k t))`. The collapse rate k has no known forward
equations relating it to observable statistics, which makes this model a
natural showcase for the ASL approach.

The `dw` model implements bounded-confidence opinion dynamics (Deffuant--Weisbuch):
agents with opinions in `[0, 1]` interact pairwise and move toward each other
when their opinions differ by less than epsilon. Parameters are inferred on the
logit scale and mapped to canonical epsilon and mu in `(0, 1)` via the logistic
sigmoid. Training draws are uniform on the logit image of epsilon in
`[0.125, 0.375]` and mu in `[0.075, 0.425]`; JAGS priors and recovery true
values use the logit image of epsilon in `[0.15, 0.35]` and mu in `[0.1, 0.4]`.
Recovery metrics are reported on the canonical scale.
Pipeline overrides live in `configs/dw.toml` (R=1000 replicates per draw).

## How it works

Each model follows the same four-stage pipeline:

1. **Generate training data** — draw parameters uniformly over bounded training
   supports and simulate summary statistics (with replicate-based covariances).
2. **Train emulator** — fit a dual-head neural network that predicts summary
   means and covariances from parameters.
3. **Wire to JAGS** — compile the trained network into a JAGS module that
   provides a synthetic likelihood for Bayesian inference.
4. **Recover** — simulate-and-recover study (500 synthetic subjects) to
   confirm the emulator supports valid inference.

## 1. Prerequisites

- **Python 3.11+** (3.10 lacks stdlib `tomllib`)
- **JAGS 4.x** — used by `py2jags` for Bayesian recovery
- **g++** — needed to compile the JAGS module
- **pkg-config** — JNNX module build locates JAGS headers
- **make**

Install on Debian/Ubuntu:

```bash
sudo apt install jags pkg-config g++ make
```

## 2. Python environment

```bash
git clone https://github.com/joachimvandekerckhove/amortized-synthetic-likelihood.git
cd amortized-synthetic-likelihood

python -m venv .venv
source .venv/bin/activate

pip install -e ".[jags,dev]"
```

`pip install` pulls PyTorch (CPU wheel). For GPU training on Pascal / sm\_61
GPUs (e.g. GTX 1080 Ti), reinstall with the CUDA build:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

Verify:

```bash
python -c "import asl; import jnnx; print('OK')"
pytest
```

## 3. Configuration

Most users need no configuration. The first `wire-to-jags` step downloads the
ONNX Runtime C/C++ SDK into `vendor/` automatically (headers and `lib/` for
JAGS module compilation). The pip `onnxruntime` package is used separately for
recovery inference. Recovery prepends the SDK `lib/` directory to
`LD_LIBRARY_PATH` automatically; no manual export is needed.

Optional overrides in `asl.toml`:

```toml
[wire]
onnxruntime_dir = "/path/to/onnxruntime"  # skip auto-download
```

Prefetch the SDK without running a full pipeline:

```bash
make bootstrap-ort
```

Default pipeline parameters live in `src/asl/presets/full.toml`. Override
individual keys in `asl.toml` or via `ASL_CONFIG=configs/<model>.toml` when
running a model-specific pipeline.

| Parameter | Default |
|---|---|
| `training_epochs` | 25,000 |
| `batch_size` | 4,096 |
| `parameter_draws` (training data) | 20,000 |
| `trials_per_replicate` | 600 (`dw`: 150 via `configs/dw.toml`) |
| `replicates_per_parameter` | 120 (`dw`: 1000) |
| `synthetic_subjects` (recovery) | 500 |
| `trials_per_subject` (recovery) | 500 (`dw`: 150 via `configs/dw.toml`) |

`make dw` layers `configs/dw.toml` (R=1000 replicates, DeepWide\_32x6,
lr=0.0003, batch 512).
Regenerate `data/dw/` after changing cov-data settings.

## 4. Reproduction steps

All commands run from the **repo root**.

### Quick start

```bash
make preflight         # run pytest (recommended before any pipeline)
make ddm3              # full pipeline for 3-parameter DDM
make ddm4              # full pipeline for 4-parameter DDM
make ddmcollapsesig    # full pipeline for collapsing-bounds DDM
make dw                # full pipeline for Deffuant-Weisbuch (DW) opinion dynamics
make all               # all four models
```

Training data is committed for all four models (`data/<model>/cov_train.csv`).
To force regeneration: `make -C scripts/<model> regenerate-data` (or
`make -C scripts/<model> reproduce` for the full pipeline from scratch).

If you already have `results/<model>/model.onnx` from a prior run, skip
training and run only `make -C scripts/<model> wire-to-jags`.

### Reference machine

On a 32-core CPU with two NVIDIA GTX 1080 Ti GPUs, a full `make all` run
(train + wire + recovery for all four models) took about nine hours. Per-model
recovery dominated wall time (ddm3 about 30 minutes, ddm4 about one hour,
`ddmcollapsesig` about two hours, `dw` about 5.5 hours). Your machine will
differ.

### 4.1 Walkthrough: `ddm3` (step by step)

```bash
make -C scripts/ddm3 train-emulator
make -C scripts/ddm3 wire-to-jags
make -C scripts/ddm3 confirm-recovery
```

Inference in JAGS uses one stochastic node:

```
obs[1:3] ~ ddm3_sl(v, a, t0, n_trials)
```

Expected emulator accuracy (reference machine):

| Summary | R² |
|---|---|
| Accuracy | 0.99990 |
| log(1 + mean RT) | 0.99991 |
| log(1 + Var[RT]) | 0.99979 |
| Overall | 0.99986 |

Expected recovery (MCMC is stochastic; expect small differences across machines):

| Parameter | Correlation | 95% CI coverage |
|---|---|---|
| v | 0.998 | 0.962 |
| a | 0.989 | 0.964 |
| t0 | 0.941 | 0.966 |

### 4.2 `ddm4`

```bash
make ddm4
```

Expected recovery:

| Parameter | Correlation | 95% CI coverage |
|---|---|---|
| v | 0.995 | 0.960 |
| a | 0.986 | 0.958 |
| t0 | 0.912 | 0.962 |
| w | 0.974 | 0.976 |

### 4.3 `ddmcollapsesig`

```bash
make ddmcollapsesig
```

Inference:

```
obs[1:4] ~ ddmcollapsesig_sl(a0, v, k, t0, n_trials)
```

Expected recovery:

| Parameter | Correlation | 95% CI coverage |
|---|---|---|
| a0 | 0.951 | 0.964 |
| v | 0.996 | 0.960 |
| k | 0.947 | 0.966 |
| t0 | 0.982 | 0.962 |

### 4.4 `dw`

```bash
make dw
```

Inference:

```
obs[1:6] ~ dw_sl(logit_epsilon, logit_mu, n_agents)
```

Expected emulator accuracy (reference machine):

| Summary | R² |
|---|---|
| Effective clusters (final) | 0.998 |
| Opinion entropy (final) | 0.998 |
| Mean opinion shift | 0.999 |
| Late opinion variance | 0.998 |
| Abs. variance change | 0.999 |
| Large-move rate | 0.997 |
| Overall | 0.998 |

Expected recovery (MCMC is stochastic; expect small differences across machines):

| Parameter | Correlation | 95% CI coverage |
|---|---|---|
| epsilon | 0.987 | 0.956 |
| mu | 0.968 | 0.950 |

Recovery infers logit-scale parameters and reports coverage on canonical
epsilon and mu after the logistic sigmoid map.

## 5. Coverage gates

Training-data generation applies a **parameter MI gate** (unless disabled via
`cov_data.parameter_mi_gate = false`): each parameter must carry detectable
mutual information with the **full summary vector jointly**. Summaries with
no associated parameter produce a warning only.

Emulator training gates on **held-out validation R²** (`val_r2` in
`final_summary.json`), default **>= 0.999** (`dw`: **>= 0.995**).

Each recovery study applies automated gates: at least **98%** of subjects must
converge, and every parameter's empirical 95% CI coverage must fall in
**(0.90, 0.99)**. The pipeline exits non-zero if any gate fails.

## 6. Repository layout

```
asl.toml                            user overrides (edit this)
src/asl/presets/full.toml           default parameters (do not edit)
configs/dw.toml                     DW pipeline overrides (R, architecture)
Makefile                            top-level entrypoint (make ddm3, etc.)

data/<model>/cov_train.csv          training data (committed)
data/<model>/cov_settings.json      metadata: n_rep, R, seed
data/<model>/generation_summary.json generation statistics (after regenerate)

scripts/<model>/run.py              pipeline entry point
scripts/<model>/Makefile            step targets

models/ddm/                         DDM simulators and specs
models/social/                      opinion-dynamics models (dw)
src/asl/                            pipeline library
tests/                              unit tests (pytest)
vendor/                             ONNX Runtime SDK (auto-downloaded, gitignored)
agent/                              AI agent prompts (optional)
```

Running the pipeline also produces `results/`, `models/*.jnnx/`, and
`figures/` (all gitignored).

## 7. Cleaning artifacts

```bash
make -C scripts/ddm3 clean-generated   # remove gitignored outputs only
make -C scripts/ddm3 clean               # also removes committed training data (prompts for confirmation)
```

## 8. Tests

```bash
make preflight
pytest
```

Unit tests cover configuration, data loading, Cholesky math, emulator
training helpers, and model simulators.
