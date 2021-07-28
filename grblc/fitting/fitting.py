import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from .constants import plabels
from .models import w07, chisq
from convert.convert import get_dir, set_dir


def fit_w07(
    logT, logF, logTerr=None, logFerr=None, p0=[None, None, 1.5, 0], tt=0, bounds=None, return_guess=False, **kwargs
):
    mask = np.asarray(logT) >= tt
    logT = np.asarray(logT)[mask]
    logF = np.asarray(logF)[mask]

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
        Tmin, Fmin, amin, tmin = tt, -50, 0, -np.inf
        Tmax, Fmax, amax, tmax = 10, -1, 5, np.inf
    else:
        (Tmin, Fmin, amin, tmin), (Tmax, Fmax, amax, tmax) = bounds

    # deal with sigma.
    # sigma = yerr(xerr) if xerr(yerr) is None. otherwise, it's (xerr**2 + yerr**2)**(0.5)
    if logTerr is not None:
        logTerr = np.asarray(logTerr)[mask]
    if logFerr is not None:
        logFerr = np.asarray(logFerr)[mask]

    sigma = np.sqrt(np.sum([err ** 2 for err in [logTerr, logFerr] if err is not None], axis=0))
    if isinstance(sigma, int):
        sigma = None

    # run the fit
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

    if p[-1] < 0:
        p, cov = curve_fit(
            lambda x, T, F, alpha: w07(x, T, F, alpha, 0),
            logT,
            logF,
            p0=[Tguess, Fguess, alphaguess],
            bounds=((Tmin, Fmin, amin), (Tmax, Fmax, amax)),
            sigma=sigma,
            absolute_sigma=True if sigma is not None else False,
            method="trf",
            **kwargs,
        )
        p = np.append(p, [0])
        cov = np.pad(cov, ((0, 1), (0, 1)), "constant")

    if return_guess:
        return p, cov, [Tguess, Fguess, alphaguess, tguess]
    else:
        return p, cov


def plot_w07_fit(logT, logF, p, tt=0, logTerr=None, logFerr=None, p0=None, ax=None, show=True):
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.axvline(tt, c="k", ls=":", label=f"tt = {tt}", alpha=0.3, zorder=-999999)
    logT = np.asarray(logT)
    logF = np.asarray(logF)
    if logTerr is not None:
        logTerr = np.asarray(logTerr)
    if logFerr is not None:
        logFerr = np.asarray(logFerr)

    mask = logT >= tt
    plotx = np.linspace(logT[0] - 0.2, logT[-1] + 0.2, 100)
    ax.errorbar(
        logT[mask],
        logF[mask],
        xerr=logTerr[mask] if logTerr is not None else logTerr,
        yerr=logFerr[mask],
        fmt=".",
        zorder=0,
    )
    ax.errorbar(
        logT[~mask],
        logF[~mask],
        xerr=logTerr[~mask] if logTerr is not None else logTerr,
        yerr=logFerr[~mask],
        color="grey",
        fmt=".",
        alpha=0.4,
        zorder=0,
    )
    ax.plot(
        plotx,
        w07(plotx, *p),
        c="k",
        label="Fit",
        zorder=-10,
    )
    T, F, *__ = p
    ax.scatter(T, F, c="tab:red", zorder=200, s=200, label="Fitted")
    if p0 is not None:
        Tguess, Fguess, *__ = p0
        ax.scatter(Tguess, Fguess, c="tab:grey", zorder=200, s=200, label="Guess")
    ax.legend(framealpha=0.0)
    ax.set_xlim(logT[0] - 0.2, logT[-1] + 0.2)
    ax.set_xlabel("log T (sec)")
    ax.set_ylabel("log F (erg cm$^{-2}$ s$^{-1}$)")
    ax.set_title("Fitted Data")

    if show:
        plt.show()
        plt.close()


def plot_w07_toy_fit(logT, logF, pfit, ptrue, logTerr=None, logFerr=None, ax=None):
    if not ax:
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


def plot_chisq(x, y, yerr, p, perr, tt, fineness=0.1, ax=None, show=True):
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    multiplier = np.arange(-4, 4, fineness)
    paramspace = np.array([p + m * perr for m in multiplier])  # shape is (len(multiplier), 4)
    for idx, ax_ in enumerate(list(ax)):
        chisq_params = np.tile(p, (len(multiplier), 1))
        chisq_params[:, idx] = paramspace[:, idx]
        delta_chisq = [chisq(x, y, yerr, w07, tt, *chisq_param) for chisq_param in chisq_params]

        ax_.plot(multiplier, delta_chisq, label=plabels[idx] + f"={p[idx]:.3f} $\pm$ {perr[idx]:.3f}")
        ax_.legend(framealpha=0.0)
        ax_.set_xlabel(r"$\sigma$ multiplier")
        ax_.set_ylabel(r"$\Delta\chi^2$")
        ax_.set_title(plabels[idx])

    if show:
        plt.show()
        plt.close()


def plot_2d_chisq(x, y, yerr, p, pcov, fineness=0.1, xlabel="X", ylabel="Y", **kwargs):
    plt.figure(figsize=(7, 5))
    _, _, *other_ps = p
    perr = np.sqrt(np.diag(pcov))

    multiplier = np.arange(-6, 6, fineness)
    p1_, p2_ = np.array([p[:2] + m * perr[:2] for m in multiplier]).T
    p1, p2 = np.meshgrid(p1_, p2_)

    res = []
    for pp1, pp2 in zip(p1, p2):
        res.append([chisq(x, y, yerr, w07, a, b, *other_ps) for a, b in zip(pp1, pp2)])

    plt.xlabel(xlabel)
    plt.ylabel(xlabel)
    plt.contour(p1, p2, res, 50, **kwargs)
    plt.scatter(*p[:2], color="r", label="Best fit")
    plt.title("$\Chi^2$")
    plt.legend()
    plt.colorbar()
    plt.show()


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
