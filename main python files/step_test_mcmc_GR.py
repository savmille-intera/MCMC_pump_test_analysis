#!/usr/bin/env python3

# This will make the optimization slower, but speed up MCMC somewhat:
import os

os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing
import time

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize

from mcmc_core import (
    ProbabilityFunction,
    finalize_mcmc_plots,
    get_an_initial,
    load_bounds,
    load_well_config,
    log_likelihood,
    make_Q_at_data,
    make_ds_dt,
    print_execution_time,
    resid_factory,
    run_model,
    run_prediction_block,
    sample_walkers,
)


def gelman_rubin_convergence(withinchainvar, meanchain, chain_length, N, ndim):
    meanall = np.mean(meanchain[0:N, :], axis=0)
    print("meanall = ", meanall)
    mean_wcv = np.mean(withinchainvar[0:N, :], axis=0)
    print("mean_wcv = ", mean_wcv)
    vom = np.zeros(ndim, dtype=float)
    if N == 1:
        return 999 * np.ones_like(meanall)

    for jj in range(0, N):
        vom += (meanall - meanchain[jj]) ** 2 / (N - 1.0)
    B = vom - mean_wcv / chain_length
    return np.sqrt(1.0 + B / mean_wcv)


def run_optimization(params, lower_bound, upper_bound, obs_time, obs_dd, obs_err, q_data, r):
    print("--------------------------------------------------------------------------")
    print("Running classical Marquardt-Levenberg optimization of parameters ...")
    start_time = time.perf_counter()

    ST = params[0:4]
    ds_dt_at_data = make_ds_dt(obs_time, obs_dd)
    q_at_data = make_Q_at_data(obs_time, q_data)
    lower = lower_bound[0:-1]
    upper = upper_bound[0:-1]

    res_lsq = least_squares(
        resid_factory(obs_time, obs_dd, obs_err, q_data, q_at_data, ds_dt_at_data, r, with_leading_space=False),
        ST,
        bounds=(lower, upper),
        args=(),
    )

    final_params = res_lsq.x
    np.savetxt("final_params.txt", final_params)

    X = np.diag(obs_err) @ res_lsq.jac
    residuals = obs_err * res_lsq.fun
    degrees_freedom = X.shape[0] - X.shape[1]
    COV = (np.sum(residuals * residuals) / degrees_freedom) * (np.linalg.inv(X.transpose() @ X))
    sd = np.sqrt(np.diag(COV))
    np.savetxt("COV.txt", COV)
    np.savetxt("SD.txt", sd)

    print(" ")
    print('Final Parameters                 (saving to "final_params.txt") ')
    print("[    S,          T,          log(C),           p    ]")
    print(final_params)
    print(" ")
    print("Covariance matrix:")
    print(COV)
    print(" ")
    print("Linear standard dev. est.:")
    print(sd)
    print(" ")
    print("Correlation matrix:")
    print(COV / (np.outer(sd, sd)))
    print(" ")

    final_dd, final_res = run_model(final_params, obs_time, obs_dd, obs_err, q_data, q_at_data, ds_dt_at_data, r)
    np.savetxt("final_dd.txt", final_dd)
    np.savetxt("final_residual.txt", final_res)

    print_execution_time(start_time, use_process_time=False)

    tplotmin = 0.9 * min(obs_time)
    tplotmax = 1.1 * max(obs_time)
    splotmin = 0.9 * min(obs_dd)
    splotmax = 1.1 * max(obs_dd)

    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.loglog(obs_time, final_dd, "r-", label="M-L best-fit model")
    ax3.loglog(obs_time, obs_dd, "o", markersize=5, fillstyle="none", label="data")
    ax3.set_xlim(tplotmin, tplotmax)
    ax3.set_ylim(splotmin, splotmax)
    ax3.set_xlabel("Time (d)")
    ax3.set_ylabel("Drawdown (ft)")
    ax3.legend(loc="upper left")

    fig4, ax4 = plt.subplots(figsize=(5, 5))
    ax4.plot(obs_time, final_dd, "r-", label="M-L best-fit model")
    ax4.plot(obs_time, obs_dd, "o", markersize=3, fillstyle="none", label="data")
    ax4.set_xlim(tplotmin, tplotmax)
    ax4.set_ylim(splotmin, splotmax)
    ax4.set_xlabel("Time (d)")
    ax4.set_ylabel("Drawdown (ft)")
    ax4.legend()

    plt.show(block=False)


def run_minimization(params, lower_bound, upper_bound, obs_time, obs_dd, obs_err, q_data, r):
    print("--------------------------------------------------------------------------")
    print("Running minimization including error magnitude using log likelihood fn ...")

    initial = np.loadtxt("final_params.txt")
    initial = np.append(initial, params[-1])

    ds_dt_at_data = make_ds_dt(obs_time, obs_dd)
    q_at_data = make_Q_at_data(obs_time, q_data)

    bnds = (
        (lower_bound[0], upper_bound[0]),
        (lower_bound[1], upper_bound[1]),
        (lower_bound[2], upper_bound[2]),
        (lower_bound[3], upper_bound[3]),
        (lower_bound[4], upper_bound[4]),
    )

    nll = lambda *args: -log_likelihood(*args)
    soln = minimize(nll, initial, args=(obs_time, obs_dd, obs_err, q_data, q_at_data, ds_dt_at_data, r), bounds=bnds)
    final_params = soln.x

    print("")
    print('Final Parameters                 (saving to "final_params_min.txt") ')
    print("[    S,          T,          log(C),           p,        log(k)  ]")
    print(final_params)
    np.savetxt("final_params_min.txt", final_params)


def run_mcmc_gr(lower_bound, upper_bound, obs_data, obs_time, obs_dd, obs_err, q_data, r):
    multiprocessing.set_start_method("spawn")
    from multiprocessing import Pool, cpu_count

    print("--------------------------------------------------------------------------")
    print("Setting up MCMC after optimization/minimization ...")

    mcmc_filename = "mcmc_save.h5"
    backend = emcee.backends.HDFBackend(mcmc_filename)

    final_params = np.loadtxt("final_params_min.txt")
    nwalkers = 15
    niter = 2000
    initial = final_params.copy()
    ndim = len(initial)
    Nchains = 6
    percent_dev = 5
    print(nwalkers, " walkers running on", cpu_count(), " cores")

    ds_dt_at_data = make_ds_dt(obs_time, obs_dd)
    q_at_data = make_Q_at_data(obs_time, q_data)
    log_prob_fn = ProbabilityFunction(lower_bound, upper_bound, obs_data, obs_err, q_at_data, ds_dt_at_data, r, q_data, inclusive_prior=False)

    backend.reset(nwalkers, ndim)

    withinchainvar = np.zeros((Nchains, ndim), dtype=float)
    meanchain = np.zeros((Nchains, ndim), dtype=float)
    loopcriteria = True

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_prob_fn,
            moves=[
                (emcee.moves.StretchMove(), 0.1),
                (emcee.moves.DEMove(), 0.6),
                (emcee.moves.DESnookerMove(), 0.3),
            ],
            backend=backend,
            pool=pool,
        )

        jj = 0
        while loopcriteria:
            sampler.reset()

            print("running chain ", jj + 1, "  of maximum of ", Nchains)
            p0 = get_an_initial(initial, percent_dev, lower_bound, upper_bound, nwalkers, ndim)

            sampler.run_mcmc(p0, niter, progress=True)
            chain_length_per_walker = int(sampler.chain.shape[1])
            print("chain length per walker: ", chain_length_per_walker)
            chainsamples_now = sampler.chain[:, int(niter / 2) :, :]

            if jj == 0:
                chainstore = chainsamples_now
            else:
                chainstore = np.concatenate((chainstore, chainsamples_now), axis=1)
                print("shape of stored chains so far is", np.shape(chainstore))

            chainsamples = np.reshape(chainsamples_now, (-1, ndim))
            chain_length = len(chainsamples)
            withinchainvar[jj] = np.var(chainsamples, axis=0)
            meanchain[jj] = np.mean(chainsamples, axis=0)
            print("Within chain means:")
            print(meanchain)
            print("Within chain vars:")
            print(withinchainvar)
            R = gelman_rubin_convergence(withinchainvar, meanchain, chain_length, jj + 1, ndim)
            print("R = ", R)
            jj = jj + 1
            loopcriteria = not (all(np.absolute(1 - R) < 0.04) and (jj > 2))
            print("Convergence so far is ", not loopcriteria)
            if jj == Nchains:
                break

    tau = sampler.get_autocorr_time(tol=0)
    print(" ")
    print("Autocorrelation times tau = ", tau)
    print("Mean autocorrelation time = ", np.mean(tau))
    print("Should run for approx.", int(100 * np.mean(tau)))

    oldshape = np.shape(chainstore)
    samples = np.reshape(chainstore, (oldshape[0] * oldshape[1], oldshape[2]))
    print("samples shape now", np.shape(samples))
    new_theta_max = samples[np.argmax(sampler.flatlnprobability)]
    new_best_fit_model, _ = log_prob_fn.run_model(new_theta_max[0:-1])
    med_model, spread = sample_walkers(log_prob_fn, 50, samples)
    finalize_mcmc_plots(obs_time, obs_dd, new_best_fit_model, med_model, spread, samples, initial)


def run_predictions(cfg, lower_bound, upper_bound, obs_err):
    print("--------------------------------------------------------------------------")
    print("Setting up MC predictions ...")

    reader = emcee.backends.HDFBackend("mcmc_save.h5")
    tau = reader.get_autocorr_time(tol=0)
    burnin = int(1 * np.max(tau))
    thin = int(1 * np.min(tau))
    flat_samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

    print("Tau is still: ", tau)
    norm_pump_rate = 1.0 * 2 * 3.14 * cfg.r * cfg.screen_intvl * cfg.por * 1440 / 7.48
    run_prediction_block(flat_samples, lower_bound, upper_bound, obs_err, cfg.r, norm_pump_rate, cfg)


def main(well_dir="."):
    well_dir = str(well_dir)
    original_dir = os.getcwd()
    os.chdir(well_dir)
    try:
        bounds = load_bounds(well_dir)
        cfg = load_well_config("well_data.txt")
        print("prediction pump rate :", 1.0 * 2 * 3.14 * cfg.r * cfg.screen_intvl * cfg.por * 1440 / 7.48)

        params = bounds.STCPK_guess
        lower_bound = bounds.lower_bound
        upper_bound = bounds.upper_bound

        obs_data = np.loadtxt(cfg.obs_filename)
        q_data = np.loadtxt(cfg.q_filename)
        obs_time = obs_data[:, 0]
        obs_dd = obs_data[:, 1]
        obs_err = np.where(obs_dd < 1, 1, obs_dd)

        if cfg.optimize:
            run_optimization(params, lower_bound, upper_bound, obs_time, obs_dd, obs_err, q_data, cfg.r)

        if cfg.minimize:
            run_minimization(params, lower_bound, upper_bound, obs_time, obs_dd, obs_err, q_data, cfg.r)

        if cfg.mcmc:
            run_mcmc_gr(lower_bound, upper_bound, obs_data, obs_time, obs_dd, obs_err, q_data, cfg.r)

        if cfg.predict_E:
            run_predictions(cfg, lower_bound, upper_bound, obs_err)

        if cfg.ies:
            print("Done!")
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()

