# MCMC_pump_test_analysis (updated 3/21/26)

**Updates**
- 3/21/2026 — Refactored scripts into shared core module; added batch runner; no more copy/paste into well folders.
- 1/1/2026 — Modified FFT convolution routine for ~4× speed improvement.

Python-based MCMC analysis of step-drawdown pumping test parameters and operational costs.

---

## Overview

This repository contains Python codes to perform Markov-chain Monte Carlo (MCMC) analysis of step-drawdown pumping tests. The codes are easily modified for variable-rate or constant-rate pumping tests. The MCMC sampler used is [emcee](https://emcee.readthedocs.io).

---

## Repository layout

```
MCMC_pump_test_analysis/
├── env.yml                          # Conda environment (all required packages)
├── wells_to_run.txt                 # List of well directories to process
├── README.md
├── main python files/
│   ├── run_wells.py                 # Batch runner — entry point for all analyses
│   ├── mcmc_core.py                         # Shared model/math/utility functions
│   ├── step_test_mcmc.py                    # Single-chain MCMC workflow
│   ├── step_test_mcmc_GR.py                 # Gelman-Rubin multi-chain MCMC workflow
│   └── archive/
│       ├── step_test_mcmc.py        # Original single-chain script (reference)
│       ├── step_test_mcmc_GR.py     # Original GR script (reference)
│       └── README_original.md       # Original README (reference)
├── well CR-15/
│   ├── well_data.txt                # Workflow flags + well parameters
│   ├── bounds.py                    # Parameter initial guess and bounds
│   ├── CR15_pump_dd.txt             # Drawdown time series
│   └── CR15_pump_Q.txt              # Pumping rate time series
├── well CR-16/  ...
└── well CR-NNN/  ...                # Additional wells follow the same structure
```

Each well folder contains only its own data and configuration — **no Python scripts need to be copied into well folders**.

---

## Setup

### 1. Create the conda environment

```bash
conda env create -f env.yml
conda activate mcmc-pump-test-analysis
```

The environment installs: `python=3.12`, `numpy`, `scipy`, `matplotlib`, `emcee`, `corner`, `h5py`.

### 2. Select wells to run

Edit `wells_to_run.txt` in the repo root. Each line is a path (absolute or relative to the repo root) to a well directory. Lines starting with `#` are comments.

```
# wells_to_run.txt
../well CR-15
../well CR-16
../well CR-226
```

### 3. Run the analysis

From inside the `main python files` directory:

```bash
cd "main python files"

# Single-chain MCMC (default — best for final production runs):
python run_wells.py

# Gelman-Rubin multi-chain MCMC (better for convergence checking):
python run_wells.py --gr

# Custom wells list:
python run_wells.py --wells /path/to/my_wells.txt
```

The runner processes each well in sequence, printing progress and elapsed time. If a well fails it is reported at the end and the rest continue.

---

## Per-well configuration

### `well_data.txt`

Each well directory must contain a `well_data.txt`. Only the first token on each line is read; inline comments are for readability.

```
True        # optimize flag    — run classical Levenberg-Marquardt optimization
True        # minimize flag    — run log-likelihood minimization (estimates error magnitude)
True        # MCMC flag        — run MCMC
True        # prediction flag  — run Monte Carlo energy/cost predictions
False       # IES flag         — not yet implemented
226_dd.txt  # drawdown file    (columns: time, drawdown)
226_Q.txt   # discharge file   (columns: step start time, Q)
0.5         # well radius (ft)
241         # screened interval length (ft)
1086        # depth to static water (ft)
1720        # total well / discharge pipe depth (ft)
0.417       # piping diameter (ft)
0.7         # pump efficiency (not well efficiency)
120         # Hazen-Williams friction coefficient for piping
0.3         # porosity of gravel pack
3.0         # duration of a single pumping episode for prediction (days)
4000        # total pump hours per year for prediction
0.1         # electricity cost ($/kWh)
```

**Workflow flags** — set to `False` to skip steps that have already completed and saved results to disk. For example, once MCMC has finished and `mcmc_save.h5` is saved, set `mcmc=False` to skip straight to predictions on future runs.

### `bounds.py`

Each well directory must also contain a `bounds.py` specifying the parameter initial guess and search bounds for `[S, T, ln(C), p, ln(k)]`:

```python
import numpy as np
STCPK_guess = np.array([ 0.1,  200, -14,  3.0, -2.35])
lower_bound = np.array([1e-3,    1., -60,  1.0, -10. ])
upper_bound = np.array([ 0.5, 2000.,   0,  5.0,   5. ])
```

---

## Outputs (written to each well directory)

| File | Stage | Description |
|---|---|---|
| `final_params.txt` | optimize | Best-fit `[S, T, ln(C), p]` from Levenberg-Marquardt |
| `COV.txt`, `SD.txt` | optimize | Approximate covariance matrix and standard deviations |
| `final_dd.txt`, `final_residual.txt` | optimize | Modeled drawdown and weighted residuals |
| `final_params_min.txt` | minimize | Best-fit `[S, T, ln(C), p, ln(k)]` from log-likelihood minimization |
| `mcmc_save.h5` | MCMC | Full MCMC chain (can be large; reusable — set `mcmc=False` once done) |
| `corner.pdf` | MCMC | Corner plot of posterior marginals |
| `model_spread.pdf` | MCMC | Posterior predictive spread vs. observed drawdown |
| `efficiency.svg`, `blah.svg` | predict | Well efficiency and pumping cost distributions |

---

## User-adjustable settings

Two settings inside the scripts are the most likely candidates for adjustment.

**Normalized pump rate for predictions** (used in the `predict_E` block):
```python
norm_pump_rate = 1.0 * 2 * 3.14 * r * screen_intvl * por * 1440 / 7.48
# 1 gpm per ft² of screen, converted to ft³/day
```

**Observation error model** — the magnitude `k` is estimated by the minimization; the user specifies the functional form:
```python
obs_err = np.where(obs_dd < 1, 1, obs_dd)   # error proportional to drawdown (default)
# obs_err = np.ones_like(obs_dd)             # uniform error
```

---

## Script descriptions

| Script | Description |
|---|---|
| `run_wells.py` | Batch runner. Reads `wells_to_run.txt` and calls the appropriate `main(well_dir)` for each well. |
| `mcmc_core.py` | Shared library: `ProbabilityFunction` class, all physics/math functions (`fun_pump`, `make_tvec`, `run_model`, `log_likelihood`, etc.), and helpers. |
| `step_test_mcmc.py` | Single multi-walker chain using `emcee.EnsembleSampler`. Best for final production runs. |
| `step_test_mcmc_GR.py` | Runs multiple shorter independent chains and applies the Gelman-Rubin statistic per parameter to assess convergence. |
| `archive/` | Original pre-refactor scripts kept for reference. |
