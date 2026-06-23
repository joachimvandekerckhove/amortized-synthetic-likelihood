# Amortized synthetic likelihood

Minimal reproducibility package for multivariate DDM emulators (`ddm3mv`, `ddm4mv`).

## Models and defaults

| Model | Default architecture | Epochs |
|-------|---------------------|--------|
| `ddm3mv` | `DeepWide_24x4` | 10,000 |
| `ddm4mv` | `DeepWide_32x6` | 10,000 |

Training data in `data/` was generated with `n_theta=20000`, `n_rep=600`, `R=120`, `seed=42`. Committed CSVs are canonical; `generate-data` is optional and slow.

## Dependencies

**Python** (3.10+):

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[jags]"
```

Use the `cu126` wheel above for Pascal GPUs (e.g. GTX 1080 Ti). A default `pip install torch` may install a build that drops sm_61 and silently falls back to CPU.

**External:**

- [JNNX](https://github.com/joachimvandekerckhove/jnnx) -- `pip install -e /path/to/jnnx`
- **JAGS** and **g++** for recovery
- **ONNXRUNTIME_DIR** -- path to ONNX Runtime install (required for `wire-to-jags`)

```bash
export ONNXRUNTIME_DIR=/path/to/onnxruntime
```

## Pipeline

From the repo root:

```bash
make -C scripts/ddm3mv all    # 500 subjects x 500 trials recovery
make -C scripts/ddm4mv all
```

Individual steps:

```bash
make -C scripts/ddm3mv train-emulator
make -C scripts/ddm3mv wire-to-jags
make -C scripts/ddm3mv confirm-recovery
```

High-N recovery (50 subjects x 10,000 trials):

```bash
make -C scripts/ddm3mv confirm-recovery-highn
make -C scripts/ddm4mv confirm-recovery-highn
```

Override architecture or epochs:

```bash
ESL_ARCHITECTURE=DeepWide_32x6 ESL_N_EPOCHS=10000 make -C scripts/ddm3mv train-emulator
```

## Recovery gates

Coverage gate: each parameter's 95% CI coverage must fall in **(0.90, 0.99)**.

- `ddm3mv` @ `DeepWide_24x4`: passes train + 500x500 + 50x10k in eval
- `ddm4mv` @ `DeepWide_32x6`: passes all gates; leaner arches fail 50x10k on over-coverage

## Layout

```
data/<model>/cov_train.csv     # training data (committed)
results/<model>/model.onnx     # trained emulator
models/<model>.jnnx/           # JAGS wiring package
results/<model>/recovery_summary.json
```
