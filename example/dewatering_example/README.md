# Dewatering Synthetic Example

This directory isolates dewatering-specific synthetic analysis from the main pump-test workflow. It demonstrates the full probabilistic workflow and includes a **deterministic vs probabilistic comparison** and **QAQC diagnostics** to show the advantage of the MCMC approach over a single best-fit estimate.

## Contents

- `dewatering_analysis.py` — Pump-rate sweep, volume/price risk analysis, deterministic comparison, and QAQC utilities.
- `mine_dewatering/` — Synthetic mine dewatering test folder and generated outputs.

## Run

From the repository root:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcmc-pump-test-analysis
python "example/dewatering_example/dewatering_analysis.py"
```

This script writes sweep summaries and figures inside `example/dewatering_example/mine_dewatering/`.

## Run via `run_wells.py`

If you prefer the batch runner, use `wells_to_run_ex.txt` (already pointing to this synthetic example):

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcmc-pump-test-analysis
cd "main python files"
python run_wells.py --wells ../wells_to_run_ex.txt
```

For Gelman-Rubin multi-chain mode:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mcmc-pump-test-analysis
cd "main python files"
python run_wells.py --gr --wells ../wells_to_run_ex.txt
```

## Outputs

All files are written to `mine_dewatering/`.

### Probabilistic sweep

| File | Description |
|---|---|
| `annual_total_cost_vs_rate.svg` | Annual cost vs production rate — linear and log scale |
| `annual_energy_vs_volume.svg` | Annual energy vs production rate — linear and log scale |
| `annual_total_cost_vs_rate_summary.txt` | Cost and energy deciles at each pump rate |
| `volume_price_risk_analysis.svg` | Probability of meeting a cost threshold vs production rate |
| `volume_price_risk_summary.txt` | Tabulated success probability and failure risk |

### Deterministic vs probabilistic comparison

Requires `mcmc_save.h5` (MCMC chain) and `final_params_min.txt` (MAP parameters) to already exist in `mine_dewatering/`. Both are produced by the main `run_wells.py` workflow before `dewatering_analysis.py` is called.

| File | Description |
|---|---|
| `deterministic_vs_probabilistic.svg` | MAP (deterministic) cost and energy lines overlaid on MCMC 10–90% posterior bands across the full pump-rate sweep |
| `cost_uncertainty_at_targets.svg` | Histograms of MCMC cost samples at low, medium, and high production rates with the MAP point estimate marked — shows the uncertainty missed by a single-point estimate |

### QAQC diagnostics

| File | Description |
|---|---|
| `qaqc_map_fit.svg` | Three-panel figure: observed vs MAP-fit drawdown; residual time series; normal Q-Q plot of residuals |
| `qaqc_posterior_predictive.svg` | MCMC 90% predictive interval vs observed data with MAP fit overlaid; % coverage of observations reported in title (ideal ≈ 90%) |
| `qaqc_summary.txt` | MAP fit statistics (RMSE, bias, relative RMSE, max residual, std) and posterior predictive coverage |

## Key functions in `dewatering_analysis.py`

| Function | Description |
|---|---|
| `run_pump_rate_sweep_predictions` | Draws posterior samples and evaluates cost/energy at each pump rate |
| `run_volume_price_risk_analysis` | Computes probability of meeting a cost threshold across production rates |
| `compute_deterministic_sweep` | Loads MAP parameters from `final_params_min.txt` and runs the forward model at each pump rate — produces a single (no-uncertainty) cost and energy estimate for comparison |
| `plot_deterministic_vs_probabilistic` | Overlays the MAP line on the MCMC posterior bands; saves `deterministic_vs_probabilistic.svg` and `cost_uncertainty_at_targets.svg` |
| `run_qaqc` | Computes residual diagnostics for the MAP fit and checks posterior predictive coverage of the observed data |

