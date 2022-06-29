#
# mcmc_analytics.py
# Baysis
#
# Created by Vladimir Sotskov on 10/06/2022, 13:18.
# Copyright © 2022 Vladimir Sotskov. All rights reserved.
#

import numpy as np
import numba as nb
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
from scipy.stats import norm, ttest_1samp, kstest
from statsmodels.tsa.stattools import acovf
from statsmodels.graphics.tsaplots import _prepare_data_corr_plot, _plot_corr
from cppbridge import OUTPUTS_PATH


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


def mvn_test(samples, t, alpha=0.05):
    return pg.multivariate_normality(samples[:, t, :], alpha)


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


def _get_acf_for(x, nlags):
    avf = acovf(x, demean=False, fft=True)
    return avf[: nlags + 1] / avf[0]


def getACF(chains, allsamples):
    nobs = list(chains.values())[0].shape[0]
    nlags = min(int(10 * np.log10(nobs)), nobs - 1)
    gammas = np.zeros((allsamples.shape[-1], allsamples.shape[-2], nlags + 1))
    xbar = np.mean(allsamples, axis=0, keepdims=True)

    for s, c in chains.items():
        chain = c - xbar
        for i in range(chain.shape[1]):
            for j in range(chain.shape[2]):
                gammas[j, i, :] += _get_acf_for(chain[:, i, j], nlags)

    gammas /= len(chains)
    return gammas


#####################
# Plotting functions
#####################
def plot_stattest(test, alpha=0.05, save=None):
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


def plot_trace(samples, t, d, fitnorm=False, kalman=None, save=None, hbins=50):
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


def plot_mixing(samples_iter, t, d, save=None):
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


def plot_scatter_matrix(samples, t, save=None):
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


def plotACF(acf, t, d, ax):
    lags, nlags, irregular = _prepare_data_corr_plot(None, len(acf[d, t]) - 1, True)
    # title = f"Autocorrelation function for $x_{{{t},{d}}}$"
    _plot_corr(ax, None, acf[d, t], None, lags, irregular, True, {})


###
# Helper JIT functions
###
@nb.njit("float64[:,:,:](float64[:,:,:])", cache=True, nogil=True, parallel=True)
def _get_cov(samples):
    s, t, n = samples.shape
    retval = np.empty((t, n, n))
    for t in nb.prange(samples.shape[1]):
        retval[t] = (samples[:, t, :].T @ samples[:, t, :]) / (s - 1)
    return retval


# pre-compiling numba functions
s = np.array(range(24), dtype=float).reshape((2, 3, 4))
_get_cov(s)
