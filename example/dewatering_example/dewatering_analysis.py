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

from mcmc_core import (
    load_bounds,
    load_well_config,
    sample_annual_power_energy_costs,
    fun_pump,
    make_tvec,
    makeQ_of_t,
    make_Q_at_data,
    make_ds_dt,
    energy_calc,
)


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

    decile_levels = np.arange(0.1, 1.0, 0.1)
    decile_labels = [f"p{int(level * 100)}" for level in decile_levels]
    cost_deciles_usd = np.zeros((len(pump_rates), len(decile_levels)))
    energy_deciles_kwh = np.zeros((len(pump_rates), len(decile_levels)))
    annual_cost_samples_usd = np.zeros((len(pump_rates), n_samp))
    annual_energy_samples_kwh = np.zeros((len(pump_rates), n_samp))
    rng = np.random.default_rng(42)

    for i, pump_rate in enumerate(pump_rates):
        _, annual_energy_kwh, annual_costs = sample_annual_power_energy_costs(flat_samples, cfg.r, pump_rate, cfg, n_samp=n_samp, rng=rng)
        annual_cost_samples_usd[i, :] = annual_costs
        annual_energy_samples_kwh[i, :] = annual_energy_kwh
        cost_deciles_usd[i, :] = np.quantile(annual_costs, decile_levels)
        energy_deciles_kwh[i, :] = np.quantile(annual_energy_kwh, decile_levels)

    cost_deciles_kusd = cost_deciles_usd / 1.0e3
    q10_kusd = cost_deciles_kusd[:, 0]
    q50_kusd = cost_deciles_kusd[:, 4]
    q70_kusd = cost_deciles_kusd[:, 6]
    q90_kusd = cost_deciles_kusd[:, 8]

    summary_cols = [pump_rates, annual_volume_m3, annual_volume_million_m3]
    summary_headers = [
        "pump_rate_ft3_per_day",
        "annual_volume_m3_per_year",
        "annual_volume_million_m3_per_year",
    ]
    for i, label in enumerate(decile_labels):
        summary_cols.append(cost_deciles_usd[:, i])
        summary_headers.append(f"cost_{label}_usd_per_year")
    for i, label in enumerate(decile_labels):
        summary_cols.append(cost_deciles_kusd[:, i])
        summary_headers.append(f"cost_{label}_kusd_per_year")
    for i, label in enumerate(decile_labels):
        summary_cols.append(energy_deciles_kwh[:, i])
        summary_headers.append(f"energy_{label}_kwh_per_year")

    np.savetxt(
        "annual_total_cost_vs_rate_summary.txt",
        np.column_stack(summary_cols),
        header=" ".join(summary_headers),
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

    e10_mwh = energy_deciles_kwh[:, 0] / 1.0e3
    e50_mwh = energy_deciles_kwh[:, 4] / 1.0e3
    e90_mwh = energy_deciles_kwh[:, 8] / 1.0e3

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
        "decile_levels": decile_levels,
        "cost_deciles_usd": cost_deciles_usd,
        "cost_deciles_kusd": cost_deciles_kusd,
        "energy_deciles_kwh": energy_deciles_kwh,
        "cost_q10_kusd": q10_kusd,
        "cost_q50_kusd": q50_kusd,
        "cost_q70_kusd": q70_kusd,
        "cost_q90_kusd": q90_kusd,
        "energy_q10_kwh": energy_deciles_kwh[:, 0],
        "energy_q50_kwh": energy_deciles_kwh[:, 4],
        "energy_q90_kwh": energy_deciles_kwh[:, 8],
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

    if np.isscalar(target_annual_volume_million_m3):
        requested_target_volumes = np.array([float(target_annual_volume_million_m3)])
    else:
        requested_target_volumes = np.asarray(target_annual_volume_million_m3, dtype=float).ravel()

    target_idx = np.array(
        [int(np.argmin(np.abs(annual_volume_million_m3 - target_v))) for target_v in requested_target_volumes],
        dtype=int,
    )
    target_volume = annual_volume_million_m3[target_idx]
    target_success = success_prob[target_idx]
    target_risk = failure_risk[target_idx]

    print("Volume-price risk summary:")
    print(f"  Price threshold: {price_threshold_kusd:.1f} $k/yr")
    print("  Target volume checks (requested -> modeled):")
    for req_v, model_v, p_succ, p_fail in zip(requested_target_volumes, target_volume, target_success, target_risk):
        print(
            "    "
            f"{req_v:.3f} -> {model_v:.3f} million m^3/yr | "
            f"P(cost <= threshold)={p_succ:.3f}, risk={p_fail:.3f}"
        )

    np.savetxt(
        "volume_price_risk_summary.txt",
        np.column_stack([annual_volume_million_m3, success_prob, failure_risk]),
        header=(
            "annual_volume_million_m3_per_year "
            "prob_cost_below_threshold "
            "risk_cost_above_threshold"
        ),
    )

    np.savetxt(
        "volume_price_target_risk_summary.txt",
        np.column_stack([requested_target_volumes, target_volume, target_success, target_risk]),
        header=(
            "requested_target_volume_million_m3_per_year "
            "modeled_target_volume_million_m3_per_year "
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
        cmap="coolwarm_r",
    )
    axes[0].axhline(price_threshold_kusd, color="w", ls="--", lw=1.5)
    axes[0].scatter(target_volume, np.full_like(target_volume, price_threshold_kusd), color="w", edgecolors="k", s=45, zorder=4)
    axes[0].set_xlabel("Production Rate (million m^3/yr)")
    axes[0].set_ylabel("Price Threshold ($k/yr)")
    axes[0].set_title("P(cost <= threshold)")
    fig.colorbar(contour, ax=axes[0], label="Success probability (blue high, red low)")

    axes[1].plot(annual_volume_million_m3, success_prob, color="blue", lw=2, label="Success probability")
    axes[1].plot(annual_volume_million_m3, failure_risk, color="C3", lw=2, label="Failure risk")
    axes[1].scatter(target_volume, target_success, color="k", s=40, zorder=4, label="Selected target rates")
    for model_v, p_succ in zip(target_volume, target_success):
        axes[1].axvline(model_v, color="k", ls=":", lw=1.0, alpha=0.5)
        text_offset_y = -10 if p_succ > 0.9 else 6
        vertical_align = "top" if p_succ > 0.9 else "bottom"
        axes[1].annotate(
            f"{model_v:.2f}",
            xy=(model_v, p_succ),
            xytext=(4, text_offset_y),
            textcoords="offset points",
            fontsize=8,
            color="k",
            va=vertical_align,
        )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("Production Rate (million m^3/yr)")
    axes[1].set_ylabel("Probability")
    axes[1].set_title(f"At price threshold = {price_threshold_kusd:.1f} $k/yr", pad=12)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    plt.savefig("volume_price_risk_analysis.svg")
    plt.show()

    return {
        "annual_volume_million_m3": annual_volume_million_m3,
        "success_prob": success_prob,
        "failure_risk": failure_risk,
        "requested_target_volumes_million_m3": requested_target_volumes,
        "modeled_target_volumes_million_m3": target_volume,
        "target_success_prob": target_success,
        "target_failure_risk": target_risk,
        "price_threshold_kusd": price_threshold_kusd,
    }


def compute_deterministic_sweep(cfg, pump_rates):
    """Run the forward model at MAP parameters for each pump rate.

    Returns cost ($k/yr) and energy (MWh/yr) arrays – one value per rate.
    """
    map_params = np.loadtxt("final_params_min.txt")  # [S, T, ln(C), p, ln(k_mag)]

    obs_time_pred = np.logspace(np.log10(0.001), np.log10(cfg.pump_duration), 40)
    ds_dt_pred = np.zeros_like(obs_time_pred)

    det_cost_kusd = np.zeros(len(pump_rates))
    det_energy_mwh = np.zeros(len(pump_rates))

    S, T = map_params[0], map_params[1]
    for i, pump_rate in enumerate(pump_rates):
        q_data_pred = np.array([[0.0, pump_rate]])
        q_at_data_pred = make_Q_at_data(obs_time_pred, q_data_pred)
        tvec = make_tvec(S, T, max(obs_time_pred), cfg.r)
        Q = makeQ_of_t(tvec, q_data_pred)
        model_dd, efficiency = fun_pump(
            map_params[:4], Q, q_at_data_pred, tvec, ds_dt_pred, obs_time_pred, cfg.r
        )
        energy_kW, _ = energy_calc(
            cfg.d_total_well,
            cfg.d_to_water,
            model_dd[-1],
            cfg.pipe_D,
            cfg.pump_E,
            efficiency[-1],
            cfg.hazen,
            q_at_data_pred[-1],
        )
        annual_energy_kwh = energy_kW * cfg.pump_hr_year
        det_cost_kusd[i] = annual_energy_kwh * cfg.elec_cost / 1.0e3
        det_energy_mwh[i] = annual_energy_kwh / 1.0e3

    return det_cost_kusd, det_energy_mwh, map_params


def plot_deterministic_vs_probabilistic(sweep_results, det_cost_kusd, det_energy_mwh):
    """Overlay the MAP (deterministic) line on the MCMC probabilistic bands."""
    vol = sweep_results["annual_volume_million_m3"]
    q10 = sweep_results["cost_q10_kusd"]
    q50 = sweep_results["cost_q50_kusd"]
    q90 = sweep_results["cost_q90_kusd"]
    e10 = sweep_results["energy_q10_kwh"] / 1.0e3
    e50 = sweep_results["energy_q50_kwh"] / 1.0e3
    e90 = sweep_results["energy_q90_kwh"] / 1.0e3

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    axes[0].fill_between(vol, q10, q90, color="C0", alpha=0.25, label="MCMC 10–90% range")
    axes[0].plot(vol, q50, color="C0", lw=2, label="MCMC median")
    axes[0].plot(vol, det_cost_kusd, color="C3", lw=2, ls="--", label="Deterministic (MAP)")
    axes[0].set_xlabel("Production Rate (million m³/yr)")
    axes[0].set_ylabel("Annual Cost ($k/yr)")
    axes[0].set_title("Annual Cost: Deterministic vs Probabilistic")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].fill_between(vol, e10, e90, color="C1", alpha=0.25, label="MCMC 10–90% range")
    axes[1].plot(vol, e50, color="C1", lw=2, label="MCMC median")
    axes[1].plot(vol, det_energy_mwh, color="C3", lw=2, ls="--", label="Deterministic (MAP)")
    axes[1].set_xlabel("Production Rate (million m³/yr)")
    axes[1].set_ylabel("Annual Energy (MWh/yr)")
    axes[1].set_title("Annual Energy: Deterministic vs Probabilistic")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    plt.savefig("deterministic_vs_probabilistic.svg")
    plt.show()

    # Histogram at three representative pump rates
    annual_cost_samples_kusd = sweep_results["annual_cost_samples_usd"] / 1.0e3
    n_vol = len(vol)
    target_idxs = [n_vol // 4, n_vol // 2, 3 * n_vol // 4]

    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
    for ax, idx in zip(axes2, target_idxs):
        samples = annual_cost_samples_kusd[idx, :]
        p10, p50, p90 = np.percentile(samples, [10, 50, 90])
        det_val = det_cost_kusd[idx]
        ax.hist(samples, bins=30, color="C0", alpha=0.6, density=True, label="MCMC samples")
        ax.axvline(det_val, color="C3", lw=2, ls="--", label=f"MAP: {det_val:.1f} $k/yr")
        ax.axvline(p50, color="C0", lw=1.5, label=f"MCMC P50: {p50:.1f} $k/yr")
        ax.axvspan(p10, p90, color="C0", alpha=0.15, label="MCMC 10–90%")
        ax.set_xlabel("Annual Cost ($k/yr)")
        ax.set_ylabel("Probability density")
        ax.set_title(f"Rate ≈ {vol[idx]:.2f} Mm³/yr")
        ax.legend(fontsize=7)

    fig2.suptitle("Cost Uncertainty at Selected Production Rates", fontsize=11)
    fig2.tight_layout()
    plt.savefig("cost_uncertainty_at_targets.svg")
    plt.show()


def run_qaqc(cfg, map_params):
    """QAQC figures: observed vs MAP fit, residuals, and posterior predictive coverage."""
    obs_data = np.loadtxt(cfg.obs_filename)
    obs_time = obs_data[:, 0]
    obs_dd = obs_data[:, 1]
    obs_err = np.ones_like(obs_dd)

    q_data = np.loadtxt(cfg.q_filename)
    q_at_data = make_Q_at_data(obs_time, q_data)
    ds_dt_at_data = make_ds_dt(obs_time, obs_dd)

    S, T = map_params[0], map_params[1]
    tvec = make_tvec(S, T, max(obs_time), cfg.r)
    Q = makeQ_of_t(tvec, q_data)
    map_dd, _ = fun_pump(map_params[:4], Q, q_at_data, tvec, ds_dt_at_data, obs_time, cfg.r)

    residuals = obs_dd - map_dd
    rmse = np.sqrt(np.mean(residuals**2))
    bias = np.mean(residuals)
    rel_rmse_pct = 100.0 * rmse / np.mean(obs_dd)

    print(f"  QAQC — MAP fit: RMSE = {rmse:.3f} ft, bias = {bias:.3f} ft, relative RMSE = {rel_rmse_pct:.1f}%")

    # Figure: observed vs modeled drawdown + residual time series
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(obs_time, obs_dd, "ok", ms=4, label="Observed")
    axes[0].plot(obs_time, map_dd, "-r", lw=2, label=f"MAP fit")
    axes[0].set_xlabel("Time (d)")
    axes[0].set_ylabel("Drawdown (ft)")
    axes[0].set_title(f"Observed vs MAP Fit\nRMSE = {rmse:.3f} ft, bias = {bias:.3f} ft")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].axhline(0, color="k", ls="--", lw=1)
    axes[1].plot(obs_time, residuals, "ok", ms=4)
    axes[1].set_xlabel("Time (d)")
    axes[1].set_ylabel("Residual (obs – MAP, ft)")
    axes[1].set_title("Residual Time Series")
    axes[1].grid(alpha=0.25)

    # Q-Q plot of normalised residuals
    n = len(residuals)
    sorted_res = np.sort(residuals / np.std(residuals))
    theoretical_q = np.array([np.percentile(np.random.randn(10000), 100 * (i + 0.5) / n) for i in range(n)])
    axes[2].plot(theoretical_q, sorted_res, "ok", ms=4)
    lim = max(np.abs(theoretical_q).max(), np.abs(sorted_res).max()) * 1.1
    axes[2].plot([-lim, lim], [-lim, lim], "r--", lw=1)
    axes[2].set_xlabel("Theoretical quantile")
    axes[2].set_ylabel("Sample quantile (normalised residual)")
    axes[2].set_title("Normal Q-Q Plot of Residuals")
    axes[2].set_xlim(-lim, lim)
    axes[2].set_ylim(-lim, lim)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    plt.savefig("qaqc_map_fit.svg")
    plt.show()

    # Save QAQC summary
    with open("qaqc_summary.txt", "w") as fh:
        fh.write("MAP fit QAQC summary\n")
        fh.write(f"  RMSE (ft):           {rmse:.4f}\n")
        fh.write(f"  Bias (ft):           {bias:.4f}\n")
        fh.write(f"  Relative RMSE (%):   {rel_rmse_pct:.2f}\n")
        fh.write(f"  Max abs residual:    {np.max(np.abs(residuals)):.4f} ft\n")
        fh.write(f"  Std of residuals:    {np.std(residuals):.4f} ft\n")

    # Posterior predictive coverage check using MCMC chain
    try:
        reader = emcee.backends.HDFBackend("mcmc_save.h5")
        tau = reader.get_autocorr_time(tol=0)
        burnin = int(1 * np.max(tau))
        thin = int(1 * np.min(tau))
        flat_samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

        n_check = min(300, len(flat_samples))
        rng = np.random.default_rng(0)
        inds = rng.integers(0, len(flat_samples), size=n_check)
        pp_models = np.zeros((n_check, len(obs_time)))
        for j, ind in enumerate(inds):
            samp = flat_samples[ind]
            S_s, T_s = samp[0], samp[1]
            tvec_s = make_tvec(S_s, T_s, max(obs_time), cfg.r)
            Q_s = makeQ_of_t(tvec_s, q_data)
            dd_s, _ = fun_pump(samp[:4], Q_s, q_at_data, tvec_s, ds_dt_at_data, obs_time, cfg.r)
            noise_s = np.exp(samp[-1]) * dd_s * rng.standard_normal(len(dd_s))
            pp_models[j, :] = dd_s + noise_s

        pp_lo = np.percentile(pp_models, 5, axis=0)
        pp_hi = np.percentile(pp_models, 95, axis=0)
        pp_med = np.median(pp_models, axis=0)
        coverage = np.mean((obs_dd >= pp_lo) & (obs_dd <= pp_hi))

        print(f"  Posterior predictive 90% coverage: {100*coverage:.1f}% of observations")

        fig2, ax2 = plt.subplots(figsize=(9, 5))
        ax2.fill_between(obs_time, pp_lo, pp_hi, color="grey", alpha=0.35, label="MCMC 90% predictive interval")
        ax2.plot(obs_time, pp_med, color="grey", lw=1.5, label="MCMC median prediction")
        ax2.plot(obs_time, map_dd, "-r", lw=2, label="MAP deterministic fit")
        ax2.plot(obs_time, obs_dd, "ok", ms=4, label="Observed")
        ax2.set_xlabel("Time (d)")
        ax2.set_ylabel("Drawdown (ft)")
        ax2.set_title(
            f"Posterior Predictive Check\n90% interval coverage = {100*coverage:.1f}% of observations"
        )
        ax2.legend()
        ax2.grid(alpha=0.25)
        fig2.tight_layout()
        plt.savefig("qaqc_posterior_predictive.svg")
        plt.show()

        with open("qaqc_summary.txt", "a") as fh:
            fh.write(f"\nPosterior predictive coverage (90% interval): {100*coverage:.2f}%\n")
            fh.write(f"  (ideal ≈ 90%; <90% → underestimated uncertainty; >90% → overestimated)\n")

    except Exception as exc:
        print(f"  Posterior predictive check skipped: {exc}")


def run_dewatering_example(mine_dewatering_dir, n_samp=400):
    mine_dewatering_dir = Path(mine_dewatering_dir).resolve()
    original_dir = Path.cwd()
    try:
        os.chdir(mine_dewatering_dir)
        bounds = load_bounds(str(mine_dewatering_dir))
        cfg = load_well_config("well_data.txt")

        sweep_results = run_pump_rate_sweep_predictions(cfg, n_samp=n_samp)
        # Use the 70th percentile annual cost curve as the default threshold for risk.
        default_price_threshold_kusd = float(np.quantile(sweep_results["cost_q70_kusd"], 0.8))
        default_target_volume_million_m3 = np.quantile(
            sweep_results["annual_volume_million_m3"],
            [0.6, 0.8, 0.9],
        )
        run_volume_price_risk_analysis(
            sweep_results,
            price_threshold_kusd=default_price_threshold_kusd,
            target_annual_volume_million_m3=default_target_volume_million_m3,
        )

        # Deterministic vs probabilistic comparison (Reviewer 1)
        print("--------------------------------------------------------------------------")
        print("Running deterministic vs probabilistic comparison ...")
        det_cost_kusd, det_energy_mwh, map_params = compute_deterministic_sweep(
            cfg, sweep_results["pump_rates_ft3_day"]
        )
        plot_deterministic_vs_probabilistic(sweep_results, det_cost_kusd, det_energy_mwh)

        print("Running QAQC diagnostics ...")
        run_qaqc(cfg, map_params)
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":

    example_dir = Path(__file__).resolve().parent / "mine_dewatering_4"
    run_dewatering_example(example_dir)

