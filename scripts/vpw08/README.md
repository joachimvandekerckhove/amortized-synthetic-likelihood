# VPW08 shape-perception illustrations

Reproduces the three hierarchical MCMC analyses and figures from the VPW08
paper example using the `ddm3`, `ddm4`, and `ddmcollapsesig` emulators.

## Prerequisites

- Python environment: `pip install -e ".[jags,dev]"` from repo root
- JAGS 4.x and GNU `parallel` (each fit uses four parallel chains via py2jags)
- Emulators wired: `make ddm3 ddm4 ddmcollapsesig` (installs JAGS `.so` modules)
- **EZ reference fit:** R + R2jags (`fit_ez.R`). If R is unavailable, place
  `data/vpw08/ez_fit_reference.json` (normalized EZ posterior summaries).

## Reproduce

From the repository root:

```bash
make vpw08
```

Or from this directory:

```bash
make -C scripts/vpw08 all
```

## Outputs

| File | Description |
|------|-------------|
| `results/vpw08/ez_fit.json` | EZ hierarchical reference |
| `results/vpw08/ddm3_ezmatched_fit.json` | DDM3 + EZ-matched priors |
| `results/vpw08/ddm4_fit.json` | DDM4 hierarchical fit |
| `results/vpw08/collapse_delta_kappa_fit.json` | Collapsing-bound delta-kappa |
| `figures/vpw08/*.pdf` | Star plots and delta-kappa figure |

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ASL_VPW08_FORCE` | `0` | Set to `1` to re-run despite converged JSON |
| `ASL_VPW08_N_ITER` | `2500` | MCMC iterations per chain |
| `ASL_VPW08_N_BURNIN` | `250` | Burn-in per chain |
| `ASL_VPW08_RHAT_GATE` | `1.05` | Convergence threshold |

## Tests

```bash
make -C scripts/vpw08 test
```
