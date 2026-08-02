# Amortized synthetic likelihood — Reproduction Guide

Reproducibility package for the paper *Amortized synthetic likelihoods for
cognitive models with intractable likelihoods*. Three models are covered:

| Model | Parameters | Summaries | Architecture | Epochs |
|---|---|---|---|---|
| `ddm3` | v, a, t0 | acc, rt\_mean, rt\_var | DeepWide\_24x4 | 10,000 |
| `ddm4` | v, a, t0, w | rt\_mean/var (corr + err), err\_rate | DeepWide\_32x6 | 10,000 |
| `ddmcollapsesig` | a0, v, k, t0 | acc, rt quantiles (5th–95th, corr + err) | DeepWide\_32x6 | 10,000 |

The `ddmcollapsesig` model has a sigmoid collapsing boundary
`a(t) = a0 / (1 + exp(k t))`. The collapse rate k has no known forward
equations relating it to observable statistics, which makes this model a
natural showcase for the ASL approach.

## How it works

Each model follows the same four-stage pipeline:

1. **Generate training data** — draw parameters uniformly and simulate summary
   statistics (with replicate-based covariances) across the parameter space.
2. **Train emulator** — fit a dual-head neural network that predicts summary
   means and covariances from parameters.
3. **Wire to JAGS** — compile the trained network into a JAGS module that
   provides a synthetic likelihood for Bayesian inference.
4. **Recover** — simulate-and-recover study (500 synthetic subjects) to
   confirm the emulator supports valid inference.

## 1. Prerequisites

- **Python 3.10+**
- **JAGS 4.x** — used by `py2jags` for Bayesian recovery
- **g++** — needed to compile the JAGS module
- **make**

Install JAGS on Debian/Ubuntu:

```bash
sudo apt install jags
```

## 2. Python environment

```bash
git clone https://github.com/joachimvandekerckhove/amortized-synthetic-likelihood.git
cd amortized-synthetic-likelihood

python -m venv .venv
source .venv/bin/activate

# GPU training (Pascal / sm_61 GPUs such as GTX 1080 Ti)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
# CPU-only alternative: pip install torch

pip install -e ".[jags,dev]"
```

Verify:

```bash
python -c "import asl; import jnnx; print('OK')"
pytest
```

## 3. Configuration

Edit `asl.toml` at the repo root:

```toml
[wire]
onnxruntime_dir = "/path/to/onnxruntime"  # required for wire-to-jags
```

Download the ONNX Runtime C/C++ SDK from
https://github.com/microsoft/onnxruntime/releases and set `onnxruntime_dir`
to the extracted directory (headers and `lib/` are required for JAGS module
compilation; the pip `onnxruntime` package is used separately for training).

Default pipeline parameters live in `src/asl/presets/full.toml`. Override
individual keys in `asl.toml` if needed. Layer scenario overrides via:

```bash
ASL_CONFIG=configs/recovery_highn.toml make -C scripts/ddm3 confirm-recovery
```

| Parameter | Default |
|---|---|
| `training_epochs` | 10,000 |
| `batch_size` | 4,096 |
| `parameter_draws` (training data) | 20,000 |
| `trials_per_replicate` | 600 |
| `replicates_per_parameter` | 120 |
| `synthetic_subjects` (recovery) | 500 |
| `trials_per_subject` (recovery) | 500 |

## 4. Reproduction steps

All commands run from the **repo root**.

### Quick start

```bash
make ddm3              # full pipeline for 3-parameter DDM
make ddm4              # full pipeline for 4-parameter DDM
make ddmcollapsesig    # full pipeline for collapsing-bounds DDM
make all               # all three models
```

Training data is committed for all three models (`data/<model>/cov_train.csv`).
To regenerate: `make -C scripts/<model> generate-data`.

If you already have `results/<model>/model.onnx` from a prior run, skip
training and run only `make -C scripts/<model> wire-to-jags`.

### 4.1 Walkthrough: `ddm3` (step by step)

```bash
make -C scripts/ddm3 train-emulator    # ~20 min GPU
make -C scripts/ddm3 wire-to-jags
make -C scripts/ddm3 confirm-recovery    # ~2–4 h
```

Inference in JAGS uses one stochastic node:

```
obs[1:3] ~ ddm3_sl(v, a, t0, n_trials)
```

Expected emulator accuracy (reference machine):

| Summary | R² |
|---|---|
| Accuracy | 0.99994 |
| log(1 + mean RT) | 0.99994 |
| log(1 + Var[RT]) | 0.99978 |
| Overall | 0.99992 |

Expected recovery (MCMC is stochastic; expect small differences across machines):

| Parameter | Correlation | 95% CI coverage |
|---|---|---|
| v | 0.997 | 0.950 |
| a | 0.995 | 0.944 |
| t0 | 0.977 | 0.938 |

High-N supplemental recovery (50 subjects × 10,000 trials):

```bash
ASL_CONFIG=configs/recovery_highn.toml make -C scripts/ddm3 confirm-recovery
```

### 4.2 `ddm4`

```bash
make ddm4
```

Expected recovery:

| Parameter | Correlation | 95% CI coverage |
|---|---|---|
| v | 0.991 | 0.948 |
| a | 0.993 | 0.954 |
| t0 | 0.971 | 0.950 |
| w | 0.990 | 0.964 |

### 4.3 `ddmcollapsesig`

```bash
make ddmcollapsesig
```

Inference:

```
obs[1:10] ~ ddmcollapsesig_sl(a0, v, k, t0, n_trials)
```

Expected recovery:

| Parameter | Correlation | 95% CI coverage |
|---|---|---|
| a0 | 0.942 | 0.952 |
| v | 0.990 | 0.940 |
| k | 0.958 | 0.940 |
| t0 | 0.994 | 0.962 |

## 5. Coverage gates

Each recovery study applies an automated gate: every parameter's empirical
95% CI coverage must fall in **(0.90, 0.99)**. The pipeline exits non-zero
if any parameter fails. All three examples pass with the default settings.

## 6. Repository layout

```
asl.toml                            user overrides (edit this)
src/asl/presets/full.toml           default parameters (do not edit)
configs/recovery_highn.toml         high-N recovery override
Makefile                            top-level entrypoint (make ddm3, etc.)

data/<model>/cov_train.csv          training data (committed)
data/<model>/cov_settings.json      metadata: n_rep, R, seed

scripts/<model>/run.py              pipeline entry point
scripts/<model>/Makefile            step targets

models/ddm/                         model simulators and specs
src/asl/                            pipeline library
tests/                              unit tests (pytest)
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
pytest
```

Unit tests cover configuration, data loading, Cholesky math, emulator
training helpers, and model simulators.
