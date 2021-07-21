import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from .constants import plabels
from .models import w07, chisq


def fit_w07(logT, logF, logTerr=None, logFerr=None, p0=[None, None, 1.5, 0], bounds=None, return_guess=False, **kwargs):

    # handle automatic guessing if no guess is given
    Tguess, Fguess, alphaguess, tguess = p0
    if not (Tguess or Fguess):
        # idx_at_Fmean = np.abs(y - np.mean(y)).argmin()
        idx_at_75percent = int(0.75 * (len(logT) - 1))
        Tguess = logT[idx_at_75percent]
        Fguess = logF[idx_at_75percent]
    if alphaguess is None:
        alphaguess = 1.5
    if tguess is None:
        tguess = 0

    # reasonable curve_fit bounds
    if bounds is None:
        Tmin, Fmin, amin, tmin = 0.1, -50, 0, 0
        Tmax, Fmax, amax, tmax = 10, -1, 5, 50_000
    else:
        (Tmin, Fmin, amin, tmin), (Tmax, Fmax, amax, tmax) = bounds

    # deal with sigma.
    # sigma = yerr(xerr) if xerr(yerr) is None. otherwise, it's (xerr**2 + yerr**2)**(0.5)
    sigma = np.sqrt(np.sum([err ** 2 for err in [logTerr, logFerr] if err is not None], axis=0))
    if isinstance(sigma, int):
        sigma = None

    # run the fit
    print(logT, logF, p0, sigma)
    p, cov = curve_fit(
        w07,
        logT,
        logF,
        p0=[Tguess, Fguess, alphaguess, tguess],
        bounds=[(Tmin, Fmin, amin, tmin), (Tmax, Fmax, amax, tmax)],
        sigma=sigma,
        absolute_sigma=True if sigma is not None else False,
        method="trf",
        **kwargs,
    )

    if return_guess:
        return p, cov, [Tguess, Fguess, alphaguess, tguess]
    else:
        return p, cov


def plot_w07_fit(logT, logF, p, logTerr=None, logFerr=None, guess=None):
    fig, ax = plt.subplots(1)
    plotx = np.linspace(logT[0], logT[-1], 100)
    ax.errorbar(logT, logF, xerr=logTerr, yerr=logFerr, fmt=".", zorder=0)
    ax.plot(
        plotx,
        w07(plotx, *p),
        c="k",
        label="\n".join([f"{c} = {a:.1f}" for c, a in zip(plabels, p)]),
        zorder=-10,
    )
    T, F, *__ = p
    ax.scatter(T, F, c="tab:red", zorder=200, s=200, label="Fitted")
    if guess is not None:
        Tguess, Fguess, *__ = guess
        ax.scatter(Tguess, Fguess, c="tab:grey", zorder=200, s=200, label="Guess")
    ax.legend(framealpha=0.0)
    plt.show()
    plt.close()
    print(p)
    print(guess)
    fig, ax = None, None


def plot_w07_toy_fit(logT, logF, pfit, ptrue, logTerr=None, logFerr=None):
    fig, ax = plt.subplots(1, figsize=(8, 5))
    plotT = np.linspace(logT[0], logT[-1], 100)
    ax.errorbar(logT, logF, xerr=logTerr, yerr=logFerr, c="k", fmt="x", label="data", zorder=0)
    ax.plot(plotT, w07(plotT, *pfit), c="tab:red", label="fit", zorder=-10)
    ax.plot(plotT, w07(plotT, *ptrue), c="tab:blue", ls=":", label="truth", zorder=-10)
    Tfit, Ffit, *__ = pfit
    Ttrue, Ftrue, *__ = ptrue
    ax.scatter(Tfit, Ffit, c="tab:red", label="fit", s=80, zorder=200)
    ax.scatter(Ttrue, Ftrue, c="tab:blue", label="true", s=80, zorder=200)
    ax.legend(framealpha=0.0)
    plt.show()
    plt.close()
    fig, ax = None, None


def plot_chisq(x, y, p, pcov, fineness=0.1):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    T, F, alpha, t = p
    perr = np.sqrt(np.diag(pcov))

    multiplier = np.arange(-4, 4, fineness)
    paramspace = np.array([p + m * perr for m in multiplier])  # shape is (len(multiplier), 4)
    for idx in range(3):
        chisq_params = np.tile(p, (len(multiplier), 1))
        chisq_params[:, idx] = paramspace[:, idx]
        delta_chisq = [chisq(x, y, perr[idx], w07, *chisq_param) for chisq_param in chisq_params]
        ax[idx].plot(chisq_params[:, idx], delta_chisq, label=plabels[idx] + f"={p[idx]:.3f}")
        ax[idx].legend(framealpha=0.0)
        ax[idx].set_xlabel(r"$\sigma$ multiplier")
        ax[idx].set_ylabel(r"$\Delta\chi^2$")
        ax[idx].set_title(plabels[idx])
    plt.show()
    plt.close()


def correlation2D(X, Y, xlabel=None, ylabel=None):
    # Plots correlation between two variables -- to be eventually used dtl with LaTa

    def scatter_hist(x, y, ax, ax_histx, ax_histy, xlabel=None, ylabel=None):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        # now determine nice limits by hand:
        binwidth = 0.25
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax / binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, bins=bins, orientation="horizontal")

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    scatter_hist(X, Y, ax, ax_histx, ax_histy, xlabel, ylabel)

    plt.show()
