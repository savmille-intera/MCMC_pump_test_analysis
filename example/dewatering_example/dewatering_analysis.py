#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np

# Allow this example script to import the shared core model utilities.
REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY_DIR = REPO_ROOT / "main python files"
if str(MAIN_PY_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_PY_DIR))

from mcmc_core import load_bounds, load_well_config, sample_annual_power_energy_costs


def run_pump_rate_sweep_predictions(cfg, n_samp=400):
    print("--------------------------------------------------------------------------")
    print("Setting up posterior pump-rate sweep (dewatering synthetic example) ...")

    reader = emcee.backends.HDFBackend("mcmc_save.h5")
    tau = reader.get_autocorr_time(tol=0)
    burnin = int(1 * np.max(tau))
    thin = int(1 * np.min(tau))
    flat_samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

    norm_pump_rate = 1.0 * 2 * 3.14 * cfg.r * cfg.screen_intvl * cfg.por * 1440 / 7.48
    rate_scale = np.linspace(0.5, 3, 50)
    pump_rates = norm_pump_rate * rate_scale
    annual_volume_m3 = pump_rates * 0.028316846592 * 365.0
    annual_volume_million_m3 = annual_volume_m3 / 1.0e6

    q10 = np.zeros_like(pump_rates)
    q50 = np.zeros_like(pump_rates)
    q90 = np.zeros_like(pump_rates)
    e10 = np.zeros_like(pump_rates)
    e50 = np.zeros_like(pump_rates)
    e90 = np.zeros_like(pump_rates)
    annual_cost_samples_usd = np.zeros((len(pump_rates), n_samp))
    annual_energy_samples_kwh = np.zeros((len(pump_rates), n_samp))
    rng = np.random.default_rng(42)

    for i, pump_rate in enumerate(pump_rates):
        _, annual_energy_kwh, annual_costs = sample_annual_power_energy_costs(flat_samples, cfg.r, pump_rate, cfg, n_samp=n_samp, rng=rng)
        annual_cost_samples_usd[i, :] = annual_costs
        annual_energy_samples_kwh[i, :] = annual_energy_kwh
        q10[i], q50[i], q90[i] = np.quantile(annual_costs, [0.1, 0.5, 0.90])
        e10[i], e50[i], e90[i] = np.quantile(annual_energy_kwh, [0.1, 0.5, 0.90])

    q10_kusd = q10 / 1.0e3
    q50_kusd = q50 / 1.0e3
    q90_kusd = q90 / 1.0e3

    np.savetxt(
        "annual_total_cost_vs_rate_summary.txt",
        np.column_stack(
            [
                pump_rates,
                annual_volume_m3,
                annual_volume_million_m3,
                q10,
                q50,
                q90,
                q10_kusd,
                q50_kusd,
                q90_kusd,
                e10,
                e50,
                e90,
            ]
        ),
        header=(
            "pump_rate_ft3_per_day annual_volume_m3_per_year annual_volume_million_m3_per_year "
            "cost_q10_usd_per_year cost_q50_usd_per_year cost_q90_usd_per_year "
            "cost_q10_kusd_per_year cost_q50_kusd_per_year cost_q90_kusd_per_year "
            "energy_q10_kwh_per_year energy_q50_kwh_per_year energy_q90_kwh_per_year"
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    axes[0].plot(annual_volume_million_m3, q50_kusd, color="C0", lw=2, label="Median annual total cost")
    axes[0].fill_between(annual_volume_million_m3, q10_kusd, q90_kusd, color="C0", alpha=0.25, label="10-90% posterior range")
    axes[0].set_xlabel("Production Rate (million m^3/yr)")
    axes[0].set_ylabel("Estimated Costs ($k/yr)")
    axes[0].set_title("Linear scale")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    q10_log = np.maximum(q10_kusd, 1e-9)
    q50_log = np.maximum(q50_kusd, 1e-9)
    q90_log = np.maximum(q90_kusd, 1e-9)
    axes[1].plot(annual_volume_million_m3, q50_log, color="C0", lw=2, label="Median annual total cost")
    axes[1].fill_between(annual_volume_million_m3, q10_log, q90_log, color="C0", alpha=0.25, label="10-90% posterior range")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Production Rate (million m^3/yr)")
    axes[1].set_ylabel("Estimated Costs ($k/yr)")
    axes[1].set_title("Log scale")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend()

    fig.tight_layout()
    plt.savefig("annual_total_cost_vs_rate.svg")
    plt.show()

    e10_mwh = e10 / 1.0e3
    e50_mwh = e50 / 1.0e3
    e90_mwh = e90 / 1.0e3

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    axes[0].plot(annual_volume_million_m3, e50_mwh, color="C1", lw=2, label="Median annual energy")
    axes[0].fill_between(annual_volume_million_m3, e10_mwh, e90_mwh, color="C1", alpha=0.25, label="10-90% posterior range")
    axes[0].set_xlabel("Production Rate (million m^3/yr)")
    axes[0].set_ylabel("Energy Consumption (MWh/yr)")
    axes[0].set_title("Linear scale")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    e10_log = np.maximum(e10_mwh, 1e-9)
    e50_log = np.maximum(e50_mwh, 1e-9)
    e90_log = np.maximum(e90_mwh, 1e-9)
    axes[1].plot(annual_volume_million_m3, e50_log, color="C1", lw=2, label="Median annual energy")
    axes[1].fill_between(annual_volume_million_m3, e10_log, e90_log, color="C1", alpha=0.25, label="10-90% posterior range")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Production Rate (million m^3/yr)")
    axes[1].set_ylabel("Energy Consumption (MWh/yr)")
    axes[1].set_title("Log scale")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend()

    fig.tight_layout()
    plt.savefig("annual_energy_vs_volume.svg")
    plt.show()

    return {
        "pump_rates_ft3_day": pump_rates,
        "annual_volume_m3": annual_volume_m3,
        "annual_volume_million_m3": annual_volume_million_m3,
        "annual_cost_samples_usd": annual_cost_samples_usd,
        "annual_energy_samples_kwh": annual_energy_samples_kwh,
        "cost_q10_kusd": q10_kusd,
        "cost_q50_kusd": q50_kusd,
        "cost_q90_kusd": q90_kusd,
        "energy_q10_kwh": e10,
        "energy_q50_kwh": e50,
        "energy_q90_kwh": e90,
    }


def run_volume_price_risk_analysis(
    sweep_results,
    price_threshold_kusd,
    target_annual_volume_million_m3,
    n_price_levels=60,
):
    annual_volume_million_m3 = sweep_results["annual_volume_million_m3"]
    annual_cost_samples_kusd = sweep_results["annual_cost_samples_usd"] / 1.0e3

    success_prob = np.mean(annual_cost_samples_kusd <= price_threshold_kusd, axis=1)
    failure_risk = 1.0 - success_prob

    target_idx = int(np.argmin(np.abs(annual_volume_million_m3 - target_annual_volume_million_m3)))
    target_volume = annual_volume_million_m3[target_idx]
    target_success = success_prob[target_idx]
    target_risk = failure_risk[target_idx]

    print("Volume-price risk summary:")
    print(f"  Price threshold: {price_threshold_kusd:.1f} $k/yr")
    print(f"  Target volume:   {target_volume:.3f} million m^3/yr")
    print(f"  P(cost <= threshold): {target_success:.3f}")
    print(f"  Risk of exceeding threshold: {target_risk:.3f}")

    np.savetxt(
        "volume_price_risk_summary.txt",
        np.column_stack([annual_volume_million_m3, success_prob, failure_risk]),
        header=(
            "annual_volume_million_m3_per_year "
            "prob_cost_below_threshold "
            "risk_cost_above_threshold"
        ),
    )

    price_min = np.min(annual_cost_samples_kusd)
    price_max = np.max(annual_cost_samples_kusd)
    price_levels_kusd = np.linspace(price_min, price_max, n_price_levels)
    prob_grid = np.zeros((len(price_levels_kusd), len(annual_volume_million_m3)))
    for i, level in enumerate(price_levels_kusd):
        prob_grid[i, :] = np.mean(annual_cost_samples_kusd <= level, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    contour = axes[0].contourf(
        annual_volume_million_m3,
        price_levels_kusd,
        prob_grid,
        levels=np.linspace(0, 1, 11),
        cmap="viridis",
    )
    axes[0].axhline(price_threshold_kusd, color="w", ls="--", lw=1.5)
    axes[0].axvline(target_volume, color="w", ls=":", lw=1.5)
    axes[0].set_xlabel("Production Rate (million m^3/yr)")
    axes[0].set_ylabel("Price Threshold ($k/yr)")
    axes[0].set_title("P(cost <= threshold)")
    fig.colorbar(contour, ax=axes[0], label="Probability")

    axes[1].plot(annual_volume_million_m3, success_prob, color="C2", lw=2, label="Success probability")
    axes[1].plot(annual_volume_million_m3, failure_risk, color="C3", lw=2, label="Failure risk")
    axes[1].axvline(target_volume, color="k", ls=":", lw=1.5)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("Production Rate (million m^3/yr)")
    axes[1].set_ylabel("Probability")
    axes[1].set_title(f"At price threshold = {price_threshold_kusd:.1f} $k/yr")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    plt.savefig("volume_price_risk_analysis.svg")
    plt.show()


def run_dewatering_example(mine_dewatering_dir, n_samp=400):
    mine_dewatering_dir = Path(mine_dewatering_dir).resolve()
    original_dir = Path.cwd()
    try:
        os.chdir(mine_dewatering_dir)
        bounds = load_bounds(str(mine_dewatering_dir))
        cfg = load_well_config("well_data.txt")

        sweep_results = run_pump_rate_sweep_predictions(cfg, n_samp=n_samp)
        default_price_threshold_kusd = float(np.median(sweep_results["cost_q50_kusd"]))
        default_target_volume_million_m3 = float(np.median(sweep_results["annual_volume_million_m3"]))
        run_volume_price_risk_analysis(
            sweep_results,
            price_threshold_kusd=default_price_threshold_kusd,
            target_annual_volume_million_m3=default_target_volume_million_m3,
        )
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":

    example_dir = Path(__file__).resolve().parent / "mine_dewatering"
    run_dewatering_example(example_dir)

