# Amortized synthetic likelihood — Reproduction Guide

Reproducibility package for the paper *Amortized synthetic likelihoods for
cognitive models with intractable likelihoods*. Three models are covered:

| Model | Parameters | Summaries | Architecture | Epochs |
|---|---|---|---|---|
| `ddm3mv` | v, a, t0 | acc, rt\_mean, rt\_var | DeepWide\_24x4 | 10,000 |
| `ddm4mv` | v, a, t0, w | rt\_mean/var (corr + err), err\_rate | DeepWide\_32x6 | 10,000 |
| `ddmcollapsesig` | a0, v, k, t0 | acc, rt quantiles (5th–95th, corr + err) | DeepWide\_32x6 | 10,000 |

Each pipeline has four stages: generate training data → train dual-head
emulator → compile into JAGS via JNNX → run simulate-and-recover.


## 1. Prerequisites

### System packages

- **Python 3.10+**
- **JAGS 4.x** — used by `py2jags` for Bayesian recovery
- **g++** — needed by JNNX to compile the JAGS module
- **make**

Install JAGS on Debian/Ubuntu:

```bash
sudo apt install jags
```

### ONNX Runtime (for JNNX module compilation)

The wiring step compiles a C++ JAGS module that links against ONNX Runtime.
Download a pre-built release:

```bash
# Example: ONNX Runtime 1.18.0 for Linux x86-64
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar xf onnxruntime-linux-x64-1.18.0.tgz
# Note the path, e.g. /opt/onnxruntime-linux-x64-1.18.0
```


## 2. Python environment

```bash
git clone https://github.com/joachimvandekerckhove/amortized-synthetic-likelihood.git
cd amortized-synthetic-likelihood

python -m venv .venv
source .venv/bin/activate

# GPU training (Pascal / sm_61 GPUs such as GTX 1080 Ti — use this exact wheel)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
# CPU-only alternative: pip install torch

pip install -e ".[jags,dev]"
```

The `jags` extra installs `py2jags` and `jnnx` (JNNX v1.1.2) from GitHub.
The `dev` extra adds `pytest`.

Verify the installation:

```bash
python -c "import asl; import jnnx; print('OK')"
pytest          # 70 tests, ~3 s
```


## 3. Configuration

Edit `asl.toml` at the repo root before running anything:

```toml
[run]
smoke = false       # set true for a fast smoke test (~5 min total)

[wire]
onnxruntime_dir = "/opt/onnxruntime-linux-x64-1.18.0"  # <-- set this
```

All other pipeline parameters (epoch counts, batch size, subject counts, etc.)
are set in `src/asl/presets/full.toml`. Do not edit that file; use `asl.toml`
to override individual keys.

Full-scale defaults (active when `smoke = false`):

| Parameter | Value |
|---|---|
| `training_epochs` | 10,000 |
| `batch_size` | 4,096 |
| `learning_rate` | 0.001 |
| `parameter_draws` (training data) | 20,000 |
| `trials_per_replicate` | 600 |
| `replicates_per_parameter` | 120 |
| `synthetic_subjects` (recovery) | 500 |
| `trials_per_subject` (recovery) | 500 |

Smoke-test defaults (active when `smoke = true`): 300 epochs, 800 parameter
draws, 50 recovery subjects. Use smoke mode for a quick sanity check before
committing to the full run.


## 4. Reproduction steps

All commands are run from the **repo root**.  Make targets handle dependency
ordering: a target only runs if its output file is missing.

### 4.1 Example 1: three-parameter DDM (`ddm3mv`)

Training data is already committed to `data/ddm3mv/cov_train.csv`
(20,000 parameter draws). To regenerate from scratch (slow, ~30 min):

```bash
make -C scripts/ddm3mv generate-data
```

Train the emulator (~20 min on GPU, longer on CPU):

```bash
make -C scripts/ddm3mv train-emulator
```

Output: `results/ddm3mv/model.onnx`, `results/ddm3mv/final_summary.json`

Expected emulator accuracy:

| Summary | R² |
|---|---|
| Accuracy | 0.99994 |
| log(1 + mean RT) | 0.99994 |
| log(1 + Var[RT]) | 0.99978 |
| Overall | 0.99992 |

Compile the ONNX emulator into a JAGS synthetic-likelihood module:

```bash
make -C scripts/ddm3mv wire-to-jags
```

Output: `models/ddm3mv.jnnx/` package, JAGS module installed system-wide.
Inference in JAGS uses one stochastic node:

```
obs_std[1:3] ~ ddm3mv_sl(v, a, t0, n_trials)
```

Run the simulate-and-recover study (500 subjects × 500 trials, ~2–4 h):

```bash
make -C scripts/ddm3mv confirm-recovery
```

Output: `results/ddm3mv/recovery_summary.json`

Expected result (500/500 converged, all Rhat < 1.01):

| Parameter | Correlation | 95% CI coverage |
|---|---|---|
| v | 0.997 | 0.950 |
| a | 0.995 | 0.944 |
| t0 | 0.977 | 0.938 |

Run all four steps in sequence with one command:

```bash
make -C scripts/ddm3mv all
```

**High-N supplemental recovery** (50 subjects × 10,000 trials):

```bash
ASL_CONFIG=configs/recovery_highn.toml make -C scripts/ddm3mv confirm-recovery
```

### 4.2 Example 2: four-parameter DDM (`ddm4mv`)

Identical pipeline. Training data committed to `data/ddm4mv/cov_train.csv`.

```bash
make -C scripts/ddm4mv all
# or step by step:
make -C scripts/ddm4mv train-emulator
make -C scripts/ddm4mv wire-to-jags
make -C scripts/ddm4mv confirm-recovery
```

Output: `results/ddm4mv/recovery_summary.json`

Expected result (498/500 converged):

| Parameter | Correlation | 95% CI coverage |
|---|---|---|
| v | 0.991 | 0.948 |
| a | 0.993 | 0.954 |
| t0 | 0.971 | 0.950 |
| w | 0.990 | 0.964 |

High-N:

```bash
ASL_CONFIG=configs/recovery_highn.toml make -C scripts/ddm4mv confirm-recovery
```

### 4.3 Example 3: collapsing-bounds DDM, joint recovery (`ddmcollapsesig`)

Two condition emulators (`ddmcollapsesig_fixed` and `ddmcollapsesig_collapse`)
share one parameter vector (a0, v, k, t0) and are inferred jointly. Training
data is **not** committed and must be generated first (~45 min per condition):

```bash
make -C scripts/ddmcollapsesig generate-data   # both conditions
```

Train both emulators (~40 min each on GPU):

```bash
make -C scripts/ddmcollapsesig train           # train-fixed + train-collapse
```

Compile both into JAGS modules:

```bash
make -C scripts/ddmcollapsesig wire            # wire-fixed + wire-collapse
```

The joint JAGS model uses one SL node per condition:

```
obs_fixed_std[1:10]    ~ ddmcollapsesig_fixed_sl(a0, v, k, t0, n_trials_fixed)
obs_collapse_std[1:10] ~ ddmcollapsesig_collapse_sl(a0, v, k, t0, n_trials_collapse)
```

Run joint recovery (500 subjects × 500 trials, ~6–8 h):

```bash
make -C scripts/ddmcollapsesig joint-recovery
```

Output: `results/ddmcollapsesig_joint/recovery_summary.json`

Expected result (500/500 converged):

| Parameter | Correlation | 95% CI coverage |
|---|---|---|
| a0 | 0.997 | 0.932 |
| v | 0.996 | 0.940 |
| k | 0.988 | 0.966 |
| t0 | 0.999 | 0.960 |

Run the full pipeline in one command:

```bash
make -C scripts/ddmcollapsesig all
```


## 5. Coverage gates

Each recovery study applies an automated gate: every parameter's empirical
95% CI coverage must fall in **(0.90, 0.99)**. The pipeline exits non-zero
and prints a diagnostic if any parameter fails. All three examples pass with
the committed emulators and the default full-scale settings.


## 6. Repository layout

```
asl.toml                            user overrides (edit this)
src/asl/presets/full.toml           full-scale preset (do not edit)
src/asl/presets/smoke.toml          smoke preset (do not edit)
configs/recovery_highn.toml         high-N scenario override

data/<model>/cov_train.csv          training data (committed for ddm3mv, ddm4mv)
data/<model>/cov_settings.json      metadata: n_rep, R, seed

models/<model>.jnnx/                JAGS wiring package
  metadata.json                       JNNX package manifest
  likelihood.json                     sigma_emu (emulator residual covariance)
  model.onnx                          trained ONNX emulator
  scalers.{json,pkl}                  I/O scaling metadata

results/<model>/model.onnx          trained emulator (generated)
results/<model>/final_summary.json  training metrics (generated)
results/<model>/recovery_summary.json recovery metrics (generated)

scripts/<model>/Makefile            pipeline targets
scripts/<model>/run.py              step dispatcher

src/asl/                            pipeline library
  config.py                         TOML configuration loader
  cov_data.py                       training-data generator
  data.py                           dataset loading and target transforms
  export_mv.py                      ONNX export
  figures.py                        recovery diagnostic plots
  mlp.py                            network architectures (DeepWide MLP)
  mv.py                             Cholesky math and SL likelihood helpers
  recovery.py                       recovery study utilities
  recovery_mv.py                    multivariate simulate-and-recover
  registry.py                       model registry
  spec.py                           Model dataclass
  train_mv.py                       dual-head emulator training
  wire.py                           JNNX package assembly and JAGS wiring

models/ddm/                         model definitions
  simulator.py                      Euler DDM simulator
  ddm3.py  ddm4.py                  scalar model specs
  ddm3mv.py  ddm4mv.py              multivariate emulator model specs
  ddmcollapsesig*.py                collapsing-bounds model specs

tests/                              unit tests (pytest)
```


## 7. Tests

```bash
pytest
```

70 tests covering the configuration system, data loading, MLP architectures,
Cholesky math, emulator training helpers, and the model registry.
