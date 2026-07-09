# Amortized synthetic likelihood

Minimal reproducibility package for multivariate DDM emulators (`ddm3mv`, `ddm4mv`, `ddmcollapsesig`).

## Models and defaults

| Model | Default architecture | Epochs |
|-------|---------------------|--------|
| `ddm3mv` | `DeepWide_24x4` | 10,000 |
| `ddm4mv` | `DeepWide_32x6` | 10,000 |
| `ddmcollapsesig_fixed` | `DeepWide_32x6` | 10,000 |
| `ddmcollapsesig_collapse` | `DeepWide_32x6` | 10,000 |

Training data in `data/` was generated with `parameter_draws=20000`, `trials_per_replicate=600`, `replicates_per_parameter=120`, `random_seed=42`. Committed CSVs are canonical; `generate-data` is optional and slow.

Inference uses JNNX v1.1.2 `{slug}_sl` stochastic nodes (one line per
emulator).

## Dependencies

**Python** (3.10+):

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[jags,dev]"
```

Use the `cu126` wheel above for Pascal GPUs (e.g. GTX 1080 Ti). A default `pip install torch` may install a build that drops sm_61 and silently falls back to CPU.

**External:**

- [JNNX](https://github.com/joachimvandekerckhove/jnnx) v1.1.2+ -- `pip install "jnnx @ git+https://github.com/joachimvandekerckhove/jnnx.git@v1.1.2"`
- **JAGS** and **g++** for recovery

## Configuration

Pipeline presets ship in `src/asl/presets/{full,smoke}.toml` and are not meant to be edited. User overrides go in `asl.toml` at the repo root. Set `[run].smoke = true` to select the smoke preset.

Example overrides in `asl.toml`:

```toml
[run]
smoke = false

[training]
architecture = "DeepWide_32x6"
training_epochs = 10000

[wire]
onnxruntime_dir = "/path/to/onnxruntime"
```

Layer scenario-specific overrides via `ASL_CONFIG` (merged over `asl.toml`):

```bash
ASL_CONFIG=configs/recovery_highn.toml make -C scripts/ddm3mv confirm-recovery
```

## Pipeline

From the repo root:

```bash
make -C scripts/ddm3mv all    # 500 subjects x 500 trials recovery
make -C scripts/ddm4mv all
make -C scripts/ddmcollapsesig all    # joint fixed + collapse recovery
```

Individual steps:

```bash
make -C scripts/ddm3mv train-emulator
make -C scripts/ddm3mv wire-to-jags
make -C scripts/ddm3mv confirm-recovery
make -C scripts/ddmcollapsesig generate-data
make -C scripts/ddmcollapsesig joint-recovery
```

The `ddmcollapsesig` pipeline trains two condition emulators (`fixed`, `collapse`) that share parameters `(a0, v, k, t0)`, then runs joint recovery via `results/ddmcollapsesig_joint/recovery_summary.json`.

High-N recovery (50 subjects x 10,000 trials):

```bash
make -C scripts/ddm3mv confirm-recovery-highn
make -C scripts/ddm4mv confirm-recovery-highn
```

## Recovery gates

Coverage gate: each parameter's 95% CI coverage must fall in **(0.90, 0.99)**.

- `ddm3mv` @ `DeepWide_24x4`: passes train + 500x500 + 50x10k in eval
- `ddm4mv` @ `DeepWide_32x6`: passes all gates; leaner arches fail 50x10k on over-coverage

## Layout

```
asl.toml                       # user overrides (edit this)
src/asl/presets/full.toml      # full-scale preset (do not edit)
src/asl/presets/smoke.toml     # smoke preset (do not edit)
configs/recovery_highn.toml    # example ASL_CONFIG scenario override
data/<model>/cov_train.csv     # training data (committed)
results/<model>/model.onnx     # trained emulator
models/<model>.jnnx/           # JAGS wiring package (v1.1 SL: likelihood.json + metadata)
results/<model>/recovery_summary.json
```

## Tests

```bash
pytest
```
