#!/usr/bin/env python3

import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft


@dataclass
class WellConfig:
    optimize: bool
    minimize: bool
    mcmc: bool
    predict_E: bool
    ies: bool
    obs_filename: str
    q_filename: str
    r: float
    screen_intvl: float
    d_to_water: float
    d_total_well: float
    pipe_D: float
    pump_E: float
    hazen: float
    por: float
    pump_duration: float
    pump_hr_year: float
    elec_cost: float


def load_bounds(well_dir="."):
    """Dynamically load bounds.py from a well directory."""
    bounds_path = Path(well_dir) / "bounds.py"
    spec = importlib.util.spec_from_file_location("bounds", bounds_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_well_config(path="well_data.txt"):
    with open(path, "r") as f:
        store_data = [line.split()[0] for line in f if line.strip()]

    to_bool = [item.lower().capitalize() == "True" for item in store_data[0:5]]
    optimize, minimize, mcmc, predict_E, ies = to_bool

    obs_filename = store_data[5]
    q_filename = store_data[6]
    to_real = [eval(item) for item in store_data[7:]]
    [
        r,
        screen_intvl,
        d_to_water,
        d_total_well,
        pipe_D,
        pump_E,
        hazen,
        por,
        pump_duration,
        pump_hr_year,
        elec_cost,
    ] = to_real

    return WellConfig(
        optimize,
        minimize,
        mcmc,
        predict_E,
        ies,
        obs_filename,
        q_filename,
        r,
        screen_intvl,
        d_to_water,
        d_total_well,
        pipe_D,
        pump_E,
        hazen,
        por,
        pump_duration,
        pump_hr_year,
        elec_cost,
    )


def make_tvec(S, T, tmax, r):
    extrat = 1.2
    tmin = 0.0
    tpeak = r * r * S / 4.0 / T
    dt = tpeak / 4.0
    ntpoints = 1 + int(np.ceil((extrat * tmax - tmin) / dt))
    tvec = np.linspace(tmin, extrat * tmax, ntpoints)
    tvec[0] = 1e-30
    return tvec


def makeQ_of_t(tvec, q_data):
    Q = np.zeros_like(tvec)
    for k in range(0, len(q_data[:, 0])):
        qnow = q_data[k, 1]
        tnow = q_data[k, 0]
        Q[tvec > tnow] = qnow
    return Q


def make_Q_at_data(obs_time, q_data):
    Q_at_data = np.zeros_like(obs_time)
    for k in range(0, len(q_data[:, 0])):
        qnow = q_data[k, 1]
        tnow = q_data[k, 0]
        Q_at_data[obs_time > tnow] = qnow
    return Q_at_data


def make_ds_dt(obs_time, obs_dd):
    ds_dt_at_data = obs_dd.copy()
    ds_dt_at_data[1:-1] = (obs_dd[1:-1] - obs_dd[0:-2]) / (obs_time[1:-1] - obs_time[0:-2])
    ds_dt_at_data[0] = (obs_dd[1] - obs_dd[0]) / (obs_time[1] - obs_time[0])
    return ds_dt_at_data


def fun_pump(params, Q, Q_at_data, tvec, ds_dt_at_data, obs_time, r):
    S = params[0]
    T = params[1]
    C = np.exp(params[2])
    p = params[3]
    Q_corrector = 1.0
    dt = tvec[2] - tvec[1]
    green = (np.exp(-(r * r * S / 4.0 / T / tvec))) / tvec
    green[0] = 0.0
    lwant2 = fft.next_fast_len(2 * len(green) + 1, real=True)
    s1 = fft.irfft(fft.rfft(green, lwant2) * fft.rfft(Q, lwant2))
    s1 = dt * s1[0 : len(tvec)]
    s1_at_data = (1 / (4 * np.pi * T)) * np.interp(obs_time, tvec, s1)
    Q_corr = np.maximum(np.zeros_like(Q_at_data), np.sign(Q_at_data) * (Q_at_data - Q_corrector * np.pi * r * r * ds_dt_at_data))
    model_dd = s1_at_data + C * np.sign(Q_at_data) * Q_corr**p
    efficiency = s1_at_data / model_dd
    return model_dd, efficiency


def run_model(params, obs_time, obs_dd, obs_err, q_data, q_at_data, ds_dt_at_data, r):
    S, T = params[0:2]
    tvec = make_tvec(S, T, max(obs_time), r)
    Q = makeQ_of_t(tvec, q_data)
    model_dd, _ = fun_pump(params, Q, q_at_data, tvec, ds_dt_at_data, obs_time, r)
    weights = 1 / obs_err
    resid = weights * (model_dd - obs_dd)
    return model_dd, resid


def log_likelihood(params, obs_time, obs_dd, obs_err, q_data, q_at_data, ds_dt_at_data, r):
    dd_params = params[0:4]
    log_K_mag = params[-1]
    model_dd, _ = run_model(dd_params, obs_time, obs_dd, obs_err, q_data, q_at_data, ds_dt_at_data, r)
    n = len(model_dd)
    sigma2 = (np.exp(log_K_mag) * obs_err) ** 2
    l_hood = -0.5 * (n * np.log(2 * np.pi) + np.sum((model_dd - obs_dd) ** 2 / sigma2 + np.log(sigma2)))
    return l_hood


class ProbabilityFunction:
    def __init__(self, lower_bound, upper_bound, obs_data, obs_err, q_at_data, ds_dt_at_data, r, q_data, inclusive_prior=True):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.obs_data = obs_data
        self.obs_err = obs_err
        self.q_at_data = q_at_data
        self.ds_dt_at_data = ds_dt_at_data
        self.r = r
        self.obs_dd = self.obs_data[:, 1]
        self.obs_time = self.obs_data[:, 0]
        self.q_data = q_data
        self.inclusive_prior = inclusive_prior

    def __call__(self, params):
        return self.log_probability(params)

    def log_prior(self, params):
        S, T, C, p, log_K_mag = params
        lS, lT, lC, lp, lK = self.lower_bound
        uS, uT, uC, up, uK = self.upper_bound
        if self.inclusive_prior:
            is_inside = lS <= S <= uS and lT <= T <= uT and lC <= C <= uC and lp <= p <= up and lK <= log_K_mag <= uK
        else:
            is_inside = lS < S < uS and lT < T < uT and lC < C < uC and lp < p < up and lK < log_K_mag < uK
        if is_inside:
            return 0.0
        return -np.inf

    def log_probability(self, params):
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)

    def log_likelihood(self, params):
        return log_likelihood(
            params,
            self.obs_time,
            self.obs_dd,
            self.obs_err,
            self.q_data,
            self.q_at_data,
            self.ds_dt_at_data,
            self.r,
        )

    def run_model(self, params):
        return run_model(
            params,
            self.obs_time,
            self.obs_dd,
            self.obs_err,
            self.q_data,
            self.q_at_data,
            self.ds_dt_at_data,
            self.r,
        )



def resid_factory(obs_time, obs_dd, obs_err, q_data, q_at_data, ds_dt_at_data, r, with_leading_space=False):
    prefix = " Classical RMSE = " if with_leading_space else "Classical RMSE = "

    def resid(params):
        tvec = make_tvec(params[0], params[1], max(obs_time), r)
        Q = makeQ_of_t(tvec, q_data)
        model_dd, _ = fun_pump(params, Q, q_at_data, tvec, ds_dt_at_data, obs_time, r)
        residual = (1 / obs_err) * (model_dd - obs_dd)
        print(
            prefix
            + str(np.sqrt(np.mean((model_dd - obs_dd) ** 2)))
            + " Weighted RMSE = "
            + str(np.sqrt(np.mean(residual * residual))),
            end="\r",
        )
        return residual

    return resid


def get_an_initial(initial, percent_dev, lower_bound, upper_bound, nwalkers, ndim):
    p0 = np.zeros([nwalkers, len(initial)])
    for i in range(nwalkers):
        p0[i, :] = initial + percent_dev * 0.01 * np.abs(initial) * np.random.randn(ndim)
        p0[i, :] = np.maximum(p0[i, :], lower_bound)
        p0[i, :] = np.minimum(p0[i, :], upper_bound)
    return p0


def sample_walkers(log_prob_fn, nsamples, flattened_chain):
    models = []
    draw = np.floor(np.random.uniform(0, len(flattened_chain), size=nsamples)).astype(int)
    thetas = flattened_chain[draw]
    for theta in thetas:
        dd_params = theta[0:-1]
        mod, _ = log_prob_fn.run_model(dd_params)
        models.append(mod)
    spread = np.std(models, axis=0)
    med_model = np.median(models, axis=0)
    return med_model, spread


def energy_calc(d_total_well, d_to_water, model_dd, pipe_D, pump_E, efficiency, hazen, q_predict):
    h_f = (4.73 * d_total_well * (q_predict / (86400 * hazen)) ** 1.85) / (pipe_D**4.87)
    TDH = d_to_water + model_dd + h_f
    TDH_Theis = d_to_water + efficiency * model_dd + h_f
    kW = (0.001 / pump_E) * (q_predict / 86400) * TDH * (3.3 ** (-4)) * 9810.0
    kW_Theis = (0.001 / pump_E) * (q_predict / 86400) * TDH_Theis * (1 / 3.3**4) * 9810.0
    return kW, kW_Theis


def finalize_mcmc_plots(obs_time, obs_dd, new_best_fit_model, med_model, spread, samples, initial):
    import corner

    labels = ["S", "T", "ln(C)", "p", "ln(k_mag)"]
    corner.corner(samples, labels=labels, truths=[initial[0], initial[1], initial[2], initial[3], initial[4]])
    plt.savefig("corner.pdf")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(obs_time, obs_dd, ".k", label="Observed Drawdown")
    plt.plot(obs_time, new_best_fit_model, label="Highest Likelihood Model")
    plt.fill_between(obs_time, med_model - 2 * spread, med_model + 2 * spread, color="grey", alpha=0.5, label=r"$2\sigma$ Posterior Spread")
    plt.legend()
    plt.ylabel("Drawdown (ft)")
    plt.xlabel("Time (d)")

    plt.subplot(1, 2, 2)
    plt.semilogy(obs_time, obs_dd, ".k", label="Observed Drawdown")
    plt.semilogy(obs_time, new_best_fit_model, label="Highest Likelihood Model")
    plt.fill_between(obs_time, med_model - 2 * spread, med_model + 2 * spread, color="grey", alpha=0.5, label=r"$2\sigma$ Posterior Spread")
    plt.legend()
    plt.ylabel("Drawdown (ft)")
    plt.xlabel("Time (d)")

    plt.savefig("model_spread.pdf")
    plt.show()


def sample_annual_power_energy_costs(flat_samples, r, norm_pump_rate, cfg, n_samp=400, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    obs_time = np.logspace(np.log10(0.001), np.log10(cfg.pump_duration), 40)
    obs_dd = np.ones_like(obs_time)
    q_data = np.array([[0.0, norm_pump_rate]])
    ds_dt_at_data = np.zeros_like(obs_dd)
    q_at_data = make_Q_at_data(obs_time, q_data)

    power_kw = np.zeros(n_samp)
    inds = rng.integers(0, len(flat_samples), size=n_samp)

    for i, ind in enumerate(inds):
        sample_now = flat_samples[ind]
        S, T = sample_now[0:2]
        tvec = make_tvec(S, T, max(obs_time), r)
        Q = makeQ_of_t(tvec, q_data)
        model_dd, efficiency = fun_pump(sample_now[0:4], Q, q_at_data, tvec, ds_dt_at_data, obs_time, r)
        noise_dd = np.exp(sample_now[-1]) * model_dd * rng.normal(size=np.size(model_dd))
        energy_kW, _ = energy_calc(
            cfg.d_total_well,
            cfg.d_to_water,
            (model_dd[-1] + noise_dd[-1]),
            cfg.pipe_D,
            cfg.pump_E,
            efficiency[-1],
            cfg.hazen,
            q_at_data[-1],
        )
        power_kw[i] = energy_kW

    annual_energy_kwh = power_kw * cfg.pump_hr_year
    annual_total_costs = annual_energy_kwh * cfg.elec_cost
    return power_kw, annual_energy_kwh, annual_total_costs


def sample_annual_total_costs(flat_samples, r, norm_pump_rate, cfg, n_samp=400, rng=None):
    _, _, annual_total_costs = sample_annual_power_energy_costs(flat_samples, r, norm_pump_rate, cfg, n_samp=n_samp, rng=rng)

    return annual_total_costs


def run_prediction_block(flat_samples, lower_bound, upper_bound, obs_err, r, norm_pump_rate, cfg):
    obs_time = np.logspace(np.log10(0.001), np.log10(cfg.pump_duration), 40)
    obs_dd = np.ones_like(obs_time)
    q_data = np.array([[0.0, norm_pump_rate]])
    ds_dt_at_data = np.zeros_like(obs_dd)
    q_at_data = make_Q_at_data(obs_time, q_data)

    print("flat samples shape: ", flat_samples.shape)
    n_samp = 400
    Eff_save = np.zeros(n_samp)
    kW_save = np.zeros(n_samp)
    kW_Theis_save = np.zeros(n_samp)

    inds = np.random.randint(len(flat_samples), size=n_samp)
    print("Running this many forward models: ", inds.shape)

    n_ind = 0
    for ind in inds:
        sample_now = flat_samples[ind]
        S, T, C, p = sample_now[0:4]
        tvec = make_tvec(S, T, max(obs_time), r)
        Q = makeQ_of_t(tvec, q_data)
        model_dd, efficiency = fun_pump(sample_now[0:4], Q, q_at_data, tvec, ds_dt_at_data, obs_time, r)
        noise_dd = np.exp(sample_now[-1]) * model_dd * np.random.randn(np.size(model_dd))
        energy_kW, energy_kW_Theis = energy_calc(
            cfg.d_total_well,
            cfg.d_to_water,
            (model_dd[-1] + noise_dd[-1]),
            cfg.pipe_D,
            cfg.pump_E,
            efficiency[-1],
            cfg.hazen,
            q_at_data[-1],
        )
        Eff_save[n_ind] = 100 * efficiency[-1]
        kW_save[n_ind] = energy_kW
        kW_Theis_save[n_ind] = energy_kW_Theis
        n_ind = n_ind + 1
        plt.plot(obs_time, model_dd, "C1", alpha=0.05)

    plt.ylabel("Drawdown (ft)")
    plt.xlabel("Time (d)")

    dollars_NL = cfg.pump_hr_year * (kW_save - kW_Theis_save) * cfg.elec_cost
    dollars_total = cfg.pump_hr_year * cfg.elec_cost * kW_save
    dollars_Theis = cfg.pump_hr_year * cfg.elec_cost * kW_Theis_save

    quants = [0.1, 0.5, 0.9]
    print("[0.1 0.5 0.9] quantiles of total cost per cubic foot :", np.quantile((dollars_total / (norm_pump_rate * cfg.pump_hr_year / 24)), quants))
    print("[0.1 0.5 0.9] quantiles of Theis cost per cubic foot :", np.quantile((dollars_Theis / (norm_pump_rate * cfg.pump_hr_year / 24)), quants))
    print("[0.1 0.5 0.9] quantiles of nonlinear cost per cubic foot :", np.quantile((dollars_NL / (norm_pump_rate * cfg.pump_hr_year / 24)), quants))
    print("[0.1 0.5 0.9] quantiles of efficiency:", np.quantile(Eff_save, quants))

    bins = np.linspace(50, 100, 26)
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.hist(Eff_save, bins, density=True)
    plt.xlabel("Well Efficiency (%)")
    plt.ylabel("pdf")
    plt.xlim(0, 100)
    plt.ylim(0, 0.5)
    plt.subplot(1, 4, 2)
    plt.hist(cfg.pump_hr_year * cfg.elec_cost * kW_save, density=True)
    plt.xlabel("Total Pumping Cost ($/yr)")
    plt.subplot(1, 4, 3)
    plt.hist(cfg.pump_hr_year * cfg.elec_cost * kW_Theis_save, density=True)
    plt.xlabel("Theis Pumping Cost ($/yr)")
    plt.subplot(1, 4, 4)
    plt.hist(dollars_NL, density=True)
    plt.xlabel("Nonlinear Drawdown Cost ($/yr)")
    plt.savefig("efficiency.svg")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(dollars_NL, density=True)
    plt.xlabel("Nonlinear Drawdown Cost ($/yr)")
    plt.ylabel("pdf")
    plt.xlim(0, 5000)
    plt.ylim(0, 0.025)
    plt.subplot(1, 2, 2)
    plt.hist(dollars_NL + 270 * cfg.screen_intvl / 50, density=True)
    plt.xlabel("Nonlinear Drawdown Cost + Filter Pack Cost ($/yr)")
    plt.ylabel("pdf")
    plt.xlim(0, 7000)
    plt.ylim(0, 0.04)
    plt.savefig("blah.svg")
    plt.show()


def print_execution_time(start_time, use_process_time):
    if use_process_time:
        elapsed_time = time.process_time() - start_time
        print("Execution time (h:m:s):", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    else:
        elapsed_time = time.perf_counter() - start_time
        print(f"Execution time: {elapsed_time:.4f} seconds")

