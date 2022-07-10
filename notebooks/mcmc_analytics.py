#
# mcmc_analytics.py
# Baysis
#
# Created by Vladimir Sotskov on 10/06/2022, 13:18.
# Copyright © 2022 Vladimir Sotskov. All rights reserved.
#

import h5py
import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from matplotlib.ticker import MultipleLocator, NullLocator
from scipy.stats import norm, ttest_1samp, kstest, gaussian_kde
from statsmodels.tsa.stattools import acovf
from statsmodels.graphics.tsaplots import _prepare_data_corr_plot, _plot_corr
from cppbridge import MCMCsession, OUTPUTS_PATH


def ztest(series, mu, cov):
    var = np.stack([np.diag(c.squeeze()) for c in cov], axis=0)
    z = np.abs(np.mean(series, axis=0) - mu) / np.sqrt(var / series.shape[0])
    return 2 * norm.sf(z)


def ttest(series, mu):
    return ttest_1samp(series, mu)[1]


def norm_test(samples, means, cov):
    t, d = samples.shape[1:]
    retval = np.empty((t, d))

    for i in range(t):
        for j in range(d):
            retval[i, j] = kstest(samples[:, i, j], 'norm', (means[i, j], np.sqrt(cov[i].squeeze()[j, j])))[-1]

    return retval


def getEPSR(samples, burnin):
    y_dotc = []
    W = 0
    S = 0

    for _, c in samples.items():
        S = c[burnin:].shape[0]
        y_dotc.append(np.mean(c[burnin:], axis=0))
        W += np.var(c[burnin:], axis=0, ddof=1)

    y_dotc = np.stack(y_dotc, axis=0)
    B = np.var(y_dotc, axis=0, ddof=1)
    W /= len(samples)
    V_hat = (B / W) + (S - 1) / S
    return np.sqrt(V_hat)


def _get_acf_for(x, nlags, adjusted):
    avf = acovf(x, adjusted=adjusted, demean=False, fft=True, nlag=nlags)
    return avf / avf[0]


def getACF(chains, allsamples, lags=None, adjusted=False):
    nobs = list(chains.values())[0].shape[0]
    nlags = min(int(10 * np.log10(nobs)), nobs - 1) if lags is None else lags
    gammas = np.zeros((allsamples.shape[-1], allsamples.shape[-2], nlags + 1))
    xbar = np.mean(allsamples, axis=0, keepdims=True)

    for s, c in chains.items():
        chain = c - xbar
        for i in range(chain.shape[1]):
            for j in range(chain.shape[2]):
                gammas[j, i, :] += _get_acf_for(chain[:, i, j], nlags, adjusted)

    gammas /= len(chains)
    return gammas


def getHDR(sample, bins=50, kde=False):
    bars = np.histogram(sample, bins=bins, density=True)
    if kde:
        p = gaussian_kde(sample)(bars[1])
    else:
        p = bars[0]
    Z = np.sum(p)
    spx = np.flip(np.sort(p)) / Z
    crit = spx[np.flatnonzero(np.cumsum(spx) >= 0.95)[0]] * Z
    xx = np.flatnonzero(p >= crit)
    ranges = []
    for k, g in groupby(enumerate(xx), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), g))
        ranges.append((group[0], group[-1]))
    return [tuple(bars[1][list(r)]) for r in ranges]


#####################
# Summaries
#####################
def getPerformanceSummary(runs):
    output = defaultdict(dict)
    for run in runs:
        mcmc = MCMCsession(run)
        print(f"Loading experiment {run}")
        mcmc.loadResults()
        with h5py.File(OUTPUTS_PATH / f"{run}_specs.h5", "r") as f:
            reverse = f['simulation'].attrs['reverse']
            niters = f['simulation'].attrs['numiter']
            T = f['model'].attrs['length']
            pool_sz = f['sampler'].attrs['pool_size']
            nupd = f['sampler'].attrs['num_param_updates']
        print("Calculating performance metrics...", end='\t')
        nsamples = (1 + reverse) * (niters + 1)
        burnin = int(0.1 * nsamples)
        aclags = nsamples - burnin
        samples = mcmc.getSamples(burnin)
        psamples = np.concatenate([s[burnin:, np.newaxis] for _, s in mcmc.param_samples.items()])
        ac_ehmm = getACF(mcmc.samples, samples, lags=aclags, adjusted=True)
        ac_params = getACF({k: v[:, np.newaxis]
                            for k, v in mcmc.param_samples.items()}, psamples, lags=aclags, adjusted=True)
        taus_ehmm = 1 + 2 * np.sum(ac_ehmm, axis=2)
        meantaus_ehmm = np.mean(taus_ehmm)
        taus_params = 1 + 2 * np.sum(ac_params, axis=2)
        meantaus_params = np.mean(taus_params)
        epsr_ehmm = getEPSR(mcmc.samples, burnin).mean()
        epsr_params = getEPSR(mcmc.param_samples, burnin).mean()
        ehmm_met_acc = 100 * np.mean(
            [acc[:T].mean() / (niters * pool_sz * (1 + reverse)) for _, acc in mcmc.acceptances.items()])
        ehmm_shift_acc = 100 * np.mean(
            [acc[T + 1:].mean() / (niters * pool_sz * (1 + reverse)) for _, acc in
             mcmc.acceptances.items()])
        ehmm_partrm_acc = 100 * np.mean(
            [acc['trm'] / (niters * nupd * (1 + reverse)) for _, acc in mcmc.param_acceptances.items()])
        ehmm_parobm_acc = 100 * np.mean(
            [acc['obsm'] / (niters * nupd * (1 + reverse)) for _, acc in mcmc.param_acceptances.items()])
        tps = np.mean(list(mcmc.durations.values())) / nsamples
        summary = {('Simulation', 'Num. iterations'): niters,
                   ('Simulation', 'Num. seeds'): len(mcmc.samples),
                   ('Simulation', 'Time per sample, ms'): tps,
                   ('Acceptance rates, %', 'Autoregressive updates'): ehmm_met_acc,
                   ('Acceptance rates, %', 'Shift updates'): ehmm_shift_acc,
                   ('Acceptance rates, %', 'Transition model parameters'): ehmm_partrm_acc,
                   ('Acceptance rates, %', 'Observation model parameters'): ehmm_parobm_acc,
                   ('Average autocorrelation time, ms', 'States'): meantaus_ehmm * tps,
                   ('Average autocorrelation time, ms', 'Parameters'): meantaus_params * tps,
                   ('Average EPSR', 'States'): epsr_ehmm,
                   ('Average EPSR', 'Parameters'): epsr_params}
        output[run].update(summary)
        print("Done\n")

    return pd.DataFrame.from_dict(output)


def getInferenceSummary(runs, paramsdict):
    output = defaultdict(dict)
    for run in runs:
        mcmc = MCMCsession(run)
        print(f"Loading experiment {run}")
        mcmc.loadResults()
        params = paramsdict[run]
        output[run].update({('True values', k): v for k, v in params})
        datadict = {}
        print("Retrieving sampling information...", end='\t')
        with h5py.File(OUTPUTS_PATH / f"{run}_specs.h5", "r") as f:
            reverse = f['simulation'].attrs['reverse']
            niters = f['simulation'].attrs['numiter']
            scales = f['simulation'].attrs['scaling']
            scales = scales[2:] if len(scales) > 2 else np.ones(2)
            scales = scales if len(scales) > 1 else np.hstack([scales, scales])
            for i, k, v in enumerate(f['model/transition'].items()):
                try:
                    datadict.setdefault(('Transition model', 'priors'), {}).update({list(params.keys())[i]: v[4:]})
                except IndexError:
                    break
                datadict.setdefault(('Transition model', 'xi'), []).append(np.sqrt(v[3]) * scales[0])
            for i, k, v in enumerate(f['model/observation'].items()):
                try:
                    datadict.setdefault(('Observation model', 'priors'), {}).update({list(params.keys())[i]: v[4:]})
                except IndexError:
                    break
                datadict.setdefault(('Observation model', 'xi'), []).append(np.sqrt(v[3]) * scales[1])

        nsamples = (1 + reverse) * (niters + 1)
        burnin = int(0.1 * nsamples)
        psamples = mcmc.getParamSamples(burnin, params.keys())
        means = psamples.mean(axis=0).to_dict()
        stds = psamples.std(axis=0).to_dict()
        for pn, ps in psamples.items():
            datadict[("Inference", f"{pn} mean")] = means[pn]
            datadict[("Inference", f"{pn} std")] = stds[pn]
            datadict[("Inference", f"{pn} HDR")] = getHDR(ps, bins=1000, kde=True)

        output[run].update(datadict)
        print("Done\n")

    return pd.DataFrame.from_dict(output)


#####################
# Plotting functions
#####################
def plotStatTest(test, alpha=0.05, save=None):
    perd = 100 * np.sum(test < alpha, axis=0) / (test.shape[0])
    pert = 100 * np.sum(test < alpha, axis=1) / (test.shape[1])
    for name, series in {"time": pert, "dimension": perd}.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        if name == 'dimension':
            n = perd.shape[0]
            clr = [f'C{i}' for i in range(n)]
            loc = MultipleLocator(1)
        else:
            clr = 'C9'
            loc = MultipleLocator(50)
        ax.bar([i for i in range(series.shape[0])], series, color=clr, alpha=0.75)
        ax.xaxis.set_major_locator(loc)
        ax.set_xlabel(name)
        ax.set_ylabel("% rejected")
        plt.tight_layout()
        if isinstance(save, str):
            plt.savefig(OUTPUTS_PATH / f"{save}_nonequal_{name}s.png", dpi=300, format='png')
        plt.show()
    return {"avg over time": np.mean(pert), "avg over dims": np.mean(perd)}


def plotTrace(samples, t, d, fitnorm=False, kalman=None, save=None, hbins=50):
    fig = plt.figure(constrained_layout=False, figsize=(18, 6))
    gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.2, width_ratios=[3, 2])
    xs = np.array(range(samples[:, t, d].shape[0]))
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax0.scatter(xs, samples[:, t, d], s=0.1)
    ax1.hist(samples[:, t, d], bins=hbins, density=True)
    retval = {}

    if kalman is not None:
        mu = kalman.means[t, d]
        sigma = np.sqrt(kalman.covariances[t].squeeze()[d, d])
        ax1.plot(np.sort(samples[:, t, d]), norm.pdf(np.sort(samples[:, t, d]), loc=mu, scale=sigma),
                 lw=2, label="Kalman")
        retval["exact"] = {"mean": mu, "st.dev.": sigma}
    if fitnorm:
        mu = np.mean(samples, axis=0)[t, d]
        sigma = np.std(samples[:, t, d])
        ax1.plot(np.sort(samples[:, t, d]), norm.pdf(np.sort(samples[:, t, d]), loc=mu, scale=sigma),
                 lw=2, label="fitted normal")
        retval["fitted"] = {"mean": mu, "st.dev.": sigma}
    if fitnorm or kalman:
        ax1.legend()
    ax0.set_xmargin(0.01)
    ax0.set_ylabel(f"$x_{{{t},{d}}}$")
    ax0.set_xlabel("sample index")
    ax1.set_ylabel("density")
    ax1.set_xlabel(f"$x_{{{t},{d}}}$")
    if isinstance(save, str):
        plt.savefig(OUTPUTS_PATH / f"{save}_traceplot_{t}-{d}.png", dpi=300, format='png')
    plt.show()

    return retval


def plotMixing(samples_iter, t, d, save=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, sample in enumerate(samples_iter):
        if isinstance(sample, np.ndarray):
            xs = np.array(range(sample.shape[0]))
            ax.scatter(xs, sample[:, t, d], s=0.1)
        else:
            xs = np.array(range(samples_iter[sample].shape[0]))
            ax.scatter(xs, samples_iter[sample][:, t, d], s=0.1, label=sample)

    ax.set_xmargin(0.01)
    ax.set_ylabel(f"$x_{{{t},{d}}}$")
    ax.set_xlabel("sample index")
    if isinstance(save, str):
        plt.savefig(OUTPUTS_PATH / f"{save}_plotmix_{t}-{d}.png", dpi=300, format='png')
    plt.show()


def plotScatterMatrix(samples, t, save=None):
    n = samples.shape[2]
    df = pd.DataFrame(samples[:, t, :], columns=[f"$x_{{{t},{d}}}$" for d in range(n)])
    fig, ax = plt.subplots(figsize=(10, 10))
    mtrx = pd.plotting.scatter_matrix(df, ax=ax, alpha=0.2, diagonal="hist", hist_kwds={"bins": 50})
    for row in mtrx:
        for m in row:
            m.xaxis.set_major_locator(NullLocator())
            m.yaxis.set_major_locator(NullLocator())
    if isinstance(save, str):
        plt.savefig(OUTPUTS_PATH / f"{save}_scatterm_time{t}.png", dpi=300, format='png')
    plt.show()


def plotACF(acf, t, d, ax, nlags=None):
    npoints = len(acf[d, t]) - 1 if nlags is None else nlags
    lags, _, irregular = _prepare_data_corr_plot(None, npoints, True)
    # title = f"Autocorrelation function for $x_{{{t},{d}}}$"
    _plot_corr(ax, None, acf[d, t, :npoints+1], None, lags, irregular, True, {})

###
# Helper JIT functions
###
# @nb.njit("float64[:,:,:](float64[:,:,:])", cache=True, nogil=True, parallel=True)
# def _get_cov(samples):
#     s, t, n = samples.shape
#     retval = np.empty((t, n, n))
#     for t in nb.prange(samples.shape[1]):
#         retval[t] = (samples[:, t, :].T @ samples[:, t, :]) / (s - 1)
#     return retval
#
#
# # pre-compiling numba functions
# s = np.array(range(24), dtype=float).reshape((2, 3, 4))
# _get_cov(s)
