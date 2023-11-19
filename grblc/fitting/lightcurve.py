import re
import sys
import warnings
from copy import deepcopy
from functools import reduce
from os import path
from os import makedirs
from typing import Dict

import lmfit as lf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from collections import OrderedDict as odict

from . import io
from ..util import get_dir
from .model import chisq
from .model import Model, Parameter

__all__ = ["Lightcurve"]

class Lightcurve:
    _name_placeholder = "unknown grb"
    _flux_fixed_inplace = False

    def __init__(
        self,
        filename: str = None,
        xdata=None,
        ydata=None,
        xerr=None,
        yerr=None,
        data_space: str = "log",
        name: str = None,
        model: Model = None,
        fix_flux: bool = False,
        attrs: Dict[str, np.ndarray] = {},
    ):
        """The main module for fitting lightcurves.

        .. warning::
            Data stored in :py:class:`Lightcurve` objects are always in logarithmic
            space; the parameter ``data_space`` is only used to convert data to log space
            if it is not already in such. If your data is in linear space [i.e., your
            time data is sec, and not \(log\)(sec)], then you should set ``data_space``
            to ``lin``.


        Parameters
        ----------
        filename : str, optional
            Name of file containing light curve data, by default None
        xdata : array_like, optional
            X values, length (n,), by default None
        ydata : array_like, optional
            Y values, by default None
        xerr : array_like, optional
            X error, by default None
        yerr : array_like, optional
            Y error, by default None
        data_space : str, {log, lin}, optional
            Whether the data inputted is in log or linear space, by default 'log'
        name : str, optional
            Name of the GRB, by default :py:class:`Model` name, or ``unknown grb`` if not
            provided.
        model : :py:class:`Model`, optional
            :py:class:`Model` to use in lightcurve fitting, by default None
        fix_flux : bool, optional
            In some fits, the actual flux at the end of the plateau is not the fitted value
            F. This applies the necessary corrections to the fitted values to account for
            this discrepancy. Note this only works for the :py:meth:`Model.W07` and
            :py:meth:`Model.SMOOTH_BPL` models.
        attrs : dict, optional
            A :py:class:`dict` of array_like objects with length (n,) with
            any additional attributes (e.g., band) for each datapoint, by default {}.
        """
        assert bool(filename) ^ (
            xdata is not None and ydata is not None
        ), "Either provide a filename or xdata, ydata."

        self.attrs = attrs
        if model is not None and not hasattr(model, "name"):
            model = model()

        if name:
            self.name = name
        elif model is not None:
            self.name = model.name
        else:
            self.name = self._name_placeholder

        if isinstance(filename, str):
            self.filename = filename
            self.set_data(*self.read_data(filename), data_space=data_space)
        else:
            self.filename = reduce(
                path.join,
                [
                    get_dir(),
                    "lightcurves",
                    "{}_flux.txt".format(self.name.replace(" ", "_").replace(".", "p")),
                ],
            )
            self.set_data(xdata, ydata, xerr, yerr, data_space=data_space)

        if model is not None:
            self.set_model(model)

        self.fix_flux = fix_flux

    def set_bounds(
        self, bounds=None, xmin=-np.inf, xmax=np.inf, ymin=-np.inf, ymax=np.inf
    ):
        """Sets the bounds on the xdata and ydata to (1) plot and (2) fit with. Either
            provide bounds or xmin, xmax, ymin, ymax. Assumes data is already in log
            space. If :py:meth:`Lightcurve.set_data` has been called, then the data
            has already been converted to log space.

        Parameters
        ----------
        bounds : array_like of length 4, optional
            Bounds on inputted x and y-data, by default None
        xmin : float, optional
            Minimum x, by default -np.inf
        xmax : float, optional
            Maximum x, by default np.inf
        ymin : float, optional
            Minimum y, by default -np.inf
        ymax : float, optional
            Maximum y, by default np.inf
        """
        # assert that either bounds or any of xmin, xmax, ymin, ymax is not None,
        # but prohibiting both to be true
        assert (bounds is not None) ^ any(
            np.isfinite(x) for x in [xmin, xmax, ymin, ymax]
        ), "Must provide bounds or xmin, xmax, ymin, ymax."

        if bounds is not None:
            xmin, xmax, ymin, ymax = self.model.bounds = bounds
        else:
            for i, x in enumerate([xmin, xmax, ymin, ymax]):
                if np.isfinite(x):
                    self.model.bounds[i] = x

        xmask = (xmin <= self.xdata) & (self.xdata <= xmax)
        ymask = (ymin <= self.ydata) & (self.ydata <= ymax)
        self.mask = xmask & ymask

        self.set_data(self.xdata, self.ydata, self.xerr, self.yerr, data_space="log")

    def set_data(self, xdata, ydata, xerr=None, yerr=None, data_space="log"):
        """Set the `xdata` and `ydata`, and optionally `xerr` and `yerr` of the lightcurve.

        .. warning::
            Data stored in :py:class:`Lightcurve` objects are always in logarithmic
            space; the parameter ``data_space`` is only used to convert data to log space
            if it is not already in such. If your data is in linear space [i.e., your
            time data is sec, and not log(sec)], then you should set ``data_space``
            to ``lin``.

        Parameters
        ----------
        xdata : array_like
            X data
        ydata : array_like
            Y data
        xerr : array_like, optional
            X error, by default None
        yerr : array_like, optional
            Y error, by default None
        data_space : str, {log, lin}, optional
            Whether the data inputted is in logarithmic or linear space, by default 'log'.
        """
        if not hasattr(self, "mask"):
            self.mask = np.ones(len(xdata), dtype=bool)

        def convert_data(data):
            if data_space == "lin":
                d = np.log10(data)
            elif data_space == "log":
                d = data
            else:
                raise ValueError("data_space must be 'log' or 'lin'")

            return np.asarray(d)

        def convert_err(data, err):
            if data_space == "lin":
                eps = err / (data * np.log(10))
            elif data_space == "log":
                eps = err
            else:
                raise ValueError("data_space must be 'log' or 'lin'")
            return np.asarray(eps)

        self.orig_xdata = convert_data(xdata)
        self.orig_ydata = convert_data(ydata)
        self.xdata = self.orig_xdata[self.mask]
        self.ydata = self.orig_ydata[self.mask]
        self.orig_xerr = convert_err(xdata, xerr) if xerr is not None else None
        self.orig_yerr = convert_err(ydata, yerr) if yerr is not None else None
        self.xerr = self.orig_xerr[self.mask] if xerr is not None else None
        self.yerr = self.orig_yerr[self.mask] if yerr is not None else None

    def set_model(self, model: Model):
        """Sets the lightcurve model to use.

        Parameters
        ----------
        model : :py:class:`Model`
            :py:class:`Model` to use in lightcurve fitting
        """
        self.res = None
        self.figs = {}

        self.model = model
        if self.name == self._name_placeholder:
            self.name = model.name

        self.set_bounds(self.model.bounds)

        self._flux_fixed_inplace = False

    def exclude_range(self, xs=(), ys=(), data_space="log"):
        """
        `exclude_range` takes a range of x and y values and excludes them from the data
        of the current lightcurve.

        Parameters
        ----------
        xs : tuple of form (xmin, xmax), optional
            Range along the x-axis to exclude.
        ys : tuple of form (ymin, ymax), optional
            Range along the y-axis to exclude.
        data_space : str, {log, lin}, optional
            Whether you'd like to exclude in logarithmic or linear space, by default 'log'.
        """
        if xs == [] and ys == []:
            return
        if xs == ():
            xs = (-np.inf, np.inf)
        if ys == ():
            ys = (-np.inf, np.inf)
        assert len(xs) == 2 and len(ys) == 2, "xs and ys must be tuples of length 2"

        xmin, xmax = xs
        ymin, ymax = ys
        xmask = (xmin <= self.orig_xdata) & (self.orig_xdata <= xmax)
        self.mask &= ~xmask
        ymask = (ymin <= self.orig_ydata) & (self.orig_ydata <= ymax)
        self.mask &= ~ymask
        self.set_data(self.xdata, self.ydata, self.xerr, self.yerr, data_space=data_space)

    def read_data(self, filename: str):
        """
            Reads in data from a file. The data must be in the correct format.
            See the :py:meth:`io.read_data` for more information.

        Parameters
        ----------
        filename : str

        Returns
        ----------
        xdata, ydata, xerr, yerr : array_like
        """
        df = io.read_data(filename)

        xdata = df["time_sec"].to_numpy()
        ydata = df["flux"].to_numpy()
        xerr = None
        yerr = df["flux_err"].to_numpy()

        return xdata, ydata, xerr, yerr

    def show_data(self, save=False, fig_kwargs={}, save_kwargs={}):
        """
            Plots the lightcurve data. If no fit has been ran, :py:meth:`Lightcurve.show` will call
            this function.

            .. note:: This doesn't plot any fit results. Use :py:meth:`Lightcurve.show_fit` to do so.

            Example:

            .. jupyter-execute::

                import numpy as np
                from grblc.fitting import Model, Lightcurve

                model = Model.W07(vary_t=False)
                xdata = np.linspace(0, 10, 15)
                yerr = np.random.normal(0, 0.5, len(xdata))
                ydata = model(xdata, 5, -12, 1.5, 0) + yerr
                lc = Lightcurve(xdata=xdata, ydata=ydata, yerr=yerr, model=model)
                lc.show_data()


        Parameters
        ----------
        fig_kwargs : dict, optional
            Arguments to pass to ``plt.figure()``, by default {}.
        """

        fig_dict = dict(
            figsize=[
                plt.rcParams["figure.figsize"][0],
                plt.rcParams["figure.figsize"][0],
            ]
        )
        if bool(fig_kwargs):
            fig_dict.update(fig_kwargs)
        plot_fig = plt.figure(**fig_dict)
        ax = plot_fig.add_subplot(1, 1, 1)

        xmin, xmax, ymin, ymax = self.model.bounds
        logT = self.orig_xdata
        logF = self.orig_ydata
        logTerr = self.orig_xerr
        logFerr = self.orig_yerr

        mask = (logT >= xmin) & (logT <= xmax) & (logF >= ymin) & (logF <= ymax)
        mask_min=np.where(mask ==True)[0][0]
        if np.isfinite(xmax) and np.isfinite(ymin):
            mask_max=xmax
            ax.errorbar(
            logT[mask],
            logF[mask],
            xerr=logTerr[mask] if logTerr is not None else logTerr,
            yerr=logFerr[mask] if logFerr is not None else logFerr,
            color="k",
            fmt=".",
            ms=10,
            zorder=0,
            )
            print(ymin, logF[-1])
        else:
            mask_max=np.where(mask==True)[0][-1]
        # plot all data points inside xmin and xmax in black
            ax.errorbar(logT[mask_min:mask_max], logF[mask_min:mask_max], logFerr[mask_min:mask_max], fmt=".", ms=10, color="k", zorder=0,)
            ax.errorbar(logT[mask_max], logF[mask_max], logFerr[mask_max], fmt=".", ms=10, color="k", zorder=0)
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()

        # plot all data points outside of xmin and xmax in grey
        if sum(~mask) > 0:
            ax.errorbar(
                logT[~mask],
                logF[~mask],
                xerr=logTerr[~mask] if logTerr is not None else logTerr,
                yerr=logFerr[~mask] if logFerr is not None else logFerr,
                color="grey",
                fmt=".",
                ms=10,
                alpha=0.2,
                zorder=0,
            )

        ax.set_xlim(ax_xlim)
        ax.set_ylim(ax_ylim)
        ax.set_xlabel("log T (sec)")
        ax.set_ylabel("log F (erg cm$^{-2}$ s$^{-1}$)")
        ax.set_title(self.name)

        # if save:
        #     savefig_kwargs = dict(
        #         fname = self.slug
        #     )

        plt.show()

    def _res(self, params):
        p = params.valuesdict().values()

        return (self.model.func(self.xdata, *p) - self.ydata) / self.sigma

    def fit(
        self,
        p0,
        run_mcmc=True,
        show=False,
        minimize_kwargs={},
        emcee_kwargs={},
    ):
        """
            Fits the lightcurve data to the model. There are two steps in this process:

                #. Minimize the residuals using `Nelder-Mead` with
                    :scipydoc:`optimize.minimize`

                #. Probe the posterior distribution using `emcee <https://emcee.readthedocs.io/en/stable/>`_, a
                    Markov-Chain Monte Carlo Python package, using the best-fit parameters from step 1 as the starting
                    point. This is optional (via the ``run_mcmc`` parameter), but recommended, as it gives a better view
                    of the errors on the best-fit parameters.

        Parameters
        ----------
        p0 : array_list, length of number of parameters
            Initial guess for the parameters.
        run_mcmc : bool, optional
            Whether to run the optional MCMC step, by default True
        show : bool, optional
            [description], by default False
        minimize_kwargs : dict, optional
            Keyword arguments to pass to
            :scipydoc:`optimize.minimize`,
            by default {}
        emcee_kwargs : dict, optional
            Keyword arguments to pass to
            `lmfit.Minimizer.emcee <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.emcee>`_, by default {}

        Returns
        -------
        `lmfit.minimizer.MinimizerResult`
            See `lmfit.Minimizer.MinimizerResult <https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.MinimizerResult>`_ for more information.
        """

        assert self.model is not None, "No model set."
        assert self.xdata is not None, "xdata not supplied"
        assert self.ydata is not None, "ydata not supplied"
        assert np.shape(self.xdata) == np.shape(
            self.ydata
        ), "xdata and ydata not the same shape"
        if getattr(self, "yerr", None) is not None:
            assert np.shape(self.yerr) == np.shape(
                self.ydata
            ), "yerr not the same shape as input data"
        if getattr(self, "xerr", None) is not None:
            assert np.shape(self.xerr) == np.shape(
                self.xdata
            ), "xerr not the same shape as input data"
        assert len(p0) == len(
            self.model
        ), f"Initial guess not the same length as the number of arguments to {self.model.name}"
        p0 = np.copy(p0)
        for i, p in enumerate(self.model):
            if not(p0[i] > self.model[p].min and p0[i] < self.model[p].max):
                warnings.warn(f"Initial guess {p}={p0[i]} may not be on or outside the " \
                              f"bounds ({self.model[p].min} < {p0[i]} ({p}) < {self.model[p].max}). Moving " \
                              f"'{p}' slightly inwards by 1e-5.", UserWarning, stacklevel=2)
                if np.isinf([self.model[p].min, self.model[p].max]).any():
                    if np.isinf(self.model[p].min):
                        p0[i] = self.model[p].max - 1e-5
                    elif np.isinf(self.model[p].max):
                        p0[i] = self.model[p].min + 1e-5
                    elif abs(self.model[p].min - p0[i]) < abs(self.model[p].max - p0[i]):
                        p0[i] = self.model[p].min + 1e-5
                    else:
                        p0[i] = self.model[p].max - 1e-5
        self._flux_fixed_inplace = False

        self.sigma = np.sqrt(
            np.sum(
                [err ** 2 for err in [self.xerr, self.yerr] if err is not None],
                axis=0,
            )
        )
        if isinstance(self.sigma, (int, float)):
            self.sigma = 1

        self.params = lf.Parameters()
        param_details = [
            (
                p,
                p0[i],
                self.model[p].vary,
                self.model[p].min,
                self.model[p].max,
            )
            for i, p in enumerate(self.model)
        ]
        self.params.add_many(*param_details)

        minimizer = lf.Minimizer(self._res, self.params, nan_policy="propagate")
            
        # solve first with robust Nelder-Mead
        self.res = mi1 = minimizer.minimize(method="leastsq", **minimize_kwargs)
        self.params = self.res.params

        if run_mcmc:
            if "burn" not in emcee_kwargs:
                emcee_kwargs["burn"] = 300
            if "steps" not in emcee_kwargs:
                emcee_kwargs["steps"] = 5000
            if "thin" not in emcee_kwargs:
                emcee_kwargs["thin"] = 20

            # use emcee to probe posterior distribution
            # and find best errors
            if "progress" in emcee_kwargs:
                print("Running MCMC...")

            self.res = minimizer.minimize(
                method="emcee",
                params=mi1.params,
                is_weighted=not isinstance(self.sigma, int),
                **emcee_kwargs,
            )

            self.params = self.res.params

            # find solution to MLE & set it in self.params
            highest_prob = np.argmax(self.res.lnprob)
            highest_prob_loc = np.unravel_index(highest_prob, self.res.lnprob.shape)
            mle_soln = self.res.chain[highest_prob_loc]
            
            i = 0
            for name in self.params:
                if self.params[name].vary:
                    self.params[name].value = mle_soln[i]

                    # also, calculate standard errors from emcee
                    # & set in self.params
                    quantiles = np.percentile(
                        self.res.flatchain[name], [15.865, 50, 84.135]
                    )
                    median = quantiles[1]
                    errp = abs(median - quantiles[2])
                    errm = abs(median - quantiles[0])

                    self.params[name].stderr = np.mean([errp, errm])
                    
                    i+=1

        if show:
            self.show_fit()

        res = deepcopy(self.res)
        if self.fix_flux:
            newF, newFerr = self.apply_flux_corr()
            res.params["F"].value = newF
            res.params["F"].stderr = newFerr

        return res

    def show_fit(
        self,
        detailed=False,
        print_res=True,
        show_plot=True,
        show_corner=False,
        show_chisq=False,
        save_plots=None,
        show=True,
        fix_flux=None,
        model_name=None,
        corner_kwargs={},
        chisq_kwargs={},
        fig_kwargs={},
        residual_ax_kwargs={},
        fit_ax_kwargs={},
        data_kwargs={},
        fit_kwargs={},
    ):
        r"""
            Shows the fit to the data. If a fit has been ran, :py:meth:`Lightcurve.show`
            will call this function.

            This function can:

                * Print the best-fit parameters and their errors. (`print_res`)

                * Show the fit to the data. (`show_plot`)

                * Show the corner plot of the posterior distribution of the parameters. (`show_corner`)

                * Show the \(\Delta\chi^2\) confidence intervals of the fit. (`show_chisq`)

        Parameters
        ----------
        detailed : bool, optional
            Whether to use all plotting and printing capabilities available
            to show the fit, by default False
        print_res : bool, optional
            Prints the fit result parameters and their errors, by default True
        show_plot : bool, optional
            Shows the lightcurve and fitted model, as well as residuals, by default True
        show_corner : bool, optional
            Whether to show the corner plot. Can only be used when `use_mcmc` was set
            to True when calling :py:meth:`Lightcurve.fit`, by default False
        show_chisq : bool, optional
            Whether to show \(\Delta\chi^2\) confidence intervals, by default False
        save_plots : bool or str, optional
            If `bool`, whether to save the plots or not in a folder in the current
            directory called `plots`. If `str`, the directory and filename to save plots
            (e.g., ``../../fit/grb010222.pdf``), by default None
        show : bool, optional
            Whether you want `plt.show()` to be ran. If not true, the figures will be returned
            as a dictionary, by default True
        fix_flux : bool, optional
            If provided, will apply the flux correction to the flux at the end of the plateau.
            If not provided, will default to what user set in the constructor of the Lightcurve class.
        model_name : str
            If provided, will create directory to save information.
        corner_kwargs : dict, optional
            Additional arguments to pass to :py:meth:`corner.corner`, by default {}
        chisq_kwargs : dict, optional
            Additional arguments to pass to ``plt.plot`` for the \(\Delta\chi^2\)
            confidence interval plots, by default {}
        fig_kwargs : dict, optional
            Additional arguments to pass to ``plt.figure`` when showing the
            fit plot, by default {}
        residual_ax_kwargs : dict, optional
            Additional arguments to pass to the residual axes subplot, by default {}
        fit_ax_kwargs : dict, optional
            Additional arguments to pass to the fit axes subplot, by default {}
        data_kwargs : dict, optional
            Additional arguments to pass to data plotting, by default {}
        fit_kwargs : dict, optional
            Additional arguments to pass to the fitted model plotting, by default {}

        Returns
        -------
        dict
            Dictionary of figures. Depending on the options chosen, the keys are `fit`,
            `corner`, `chisq`.


        Example:

        .. jupyter-execute::

            import numpy as np
            from grblc.fitting import Model, Lightcurve

            model = Model.W07()
            xdata = np.linspace(0, 10, 15)
            yerr = np.random.normal(0, 0.5, len(xdata))
            ydata = model(xdata, 5, -12, 1.5, 0) + yerr
            lc = Lightcurve(xdata=xdata, ydata=ydata, yerr=yerr, model=model)
            lc.fit(p0=[4.5, -12.5, 1, 0])
            lc.show_fit(detailed=True)

        """
        assert getattr(self, "res", None) is not None, "No fit results found to show."
        if getattr(self, "_flux_fixed_inplace", False):
            warnings.warn("Flux corrections have been applied inplace, meaning that "
                          "these fit results will not be accurately displayed.")
        if fix_flux is None:
            fix_flux = self.fix_flux

        if show_plot or detailed:
            # create figure
            fig_dict = dict(figsize=[plt.rcParams["figure.figsize"][0]] * 2)
            if bool(fig_kwargs):
                fig_dict.update(fig_kwargs)
            plot_fig = plt.figure(**fig_dict)
            gridspec = plt.GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])

            # create axes
            ax_residual = plot_fig.add_subplot(gridspec[1], **residual_ax_kwargs)
            ax_fit = plot_fig.add_subplot(
                gridspec[0], sharex=ax_residual, **fit_ax_kwargs
            )
            plot_fig.subplots_adjust(hspace=0)
            
            x_uplim=max(self.xdata)

            # plot fit (within bounds first)
            if not isinstance(self.sigma, (int, float)):
                ax_fit.errorbar(
                    self.xdata, self.ydata, self.yerr, fmt="o", ms=5, color="k", **fit_kwargs
                )
                for i in range(len(self.xdata)):
                    ax_fit.errorbar(self.xdata[i], self.ydata[i], self.yerr[i], fmt="o", ms=5, color="k", **fit_kwargs)
            else:
                ax_fit.errorbar(
                    self.xdata, self.ydata, self.yerr, fmt="o", ms=5, color="k", **fit_kwargs
                )
            x_vals = np.linspace(0.8 * self.xdata.min(), 1.1 * self.xdata.max(), 100)
            y_vals = self.model(x_vals, *self.params.valuesdict().values())
            ax_fit.plot(x_vals, y_vals, ls="-", color="r", label="fit")

            # ! IMPORTANT! If using the willingale, the t factor
            #              changes the true location of the end of the plateau
            #              by subtracting 10^(t-T)/log(10) from F. This comes naturally
            #              from f(t=T) = F - 10^(t-T)/log(10)
            if self.model.slug == "w07":
                T, F, a, t = self.params.values()
                ax_fit.scatter(
                    T,
                    self.apply_flux_corr()[0] if fix_flux else F,
                    c="red",
                    zorder=-998,
                    s=150,
                    label="Fitted T, F",
                )

            # ! IMPORTANT! If using the smooth bpl, the smooth factor
            #              changes the true location of the end of the plateau
            #              by subtracting log10(2)/S from F. This comes naturally
            #              from f(t=T) = F - log(2)/S
            elif self.model.slug == "smooth_bpl":
                T, F, a1, a2, S = self.params.values()
                ax_fit.scatter(
                    T,
                    self.apply_flux_corr()[0] if fix_flux else F,
                    c="red",
                    zorder=-999,
                    s=150,
                    label="Fitted T, F",
                )

            fit_xlim = ax_fit.get_xlim()
            fit_ylim = ax_fit.get_ylim()

            # plot fit (outside bounds)
            if sum(~self.mask) > 0:
                if not isinstance(self.sigma, int):
                    ax_fit.errorbar(
                        self.orig_xdata[~self.mask],
                        self.orig_ydata[~self.mask],
                        self.orig_yerr[~self.mask],
                        fmt="o",
                        color="grey",
                        ms=5,
                        alpha=0.2,
                        **data_kwargs,
                    )
                else:
                    ax_fit.scatter(
                        self.orig_xdata[~self.mask],
                        self.orig_ydata[~self.mask],
                        color="grey",
                        s=5,
                        alpha=0.2,
                        **data_kwargs,
                    )

            ax_fit.set_xlim(fit_xlim)
            ax_fit.set_ylim(fit_ylim)
            ax_fit.legend(frameon=False)
            ax_fit.set_title(f"{self.name} Fit")
            ax_fit.set_ylabel("log Flux (erg cm$^{-2}$ s$^{-1})$")

            # plot residuals
            ax_residual.axhline(
                0, color="k", linewidth=plt.rcParams["axes.linewidth"], **fit_kwargs
            )
            residuals = self.ydata - self.model(self.xdata, *self.params.values())
            full_residuals = self.orig_ydata - self.model(
                self.orig_xdata, *self.params.values()
            )

            if not isinstance(self.sigma, int):
                ax_residual.errorbar(
                    self.xdata, residuals, self.yerr, ms=5, fmt="o", color="k", **fit_kwargs
                )
            else:
                ax_residual.scatter(self.xdata, residuals, s=5, color="k", **fit_kwargs)

            residual_ylim = ax_residual.get_ylim()

            if sum(~self.mask) > 0:

                if not isinstance(self.sigma, int):
                    ax_residual.errorbar(
                        self.orig_xdata[~self.mask],
                        full_residuals[~self.mask],
                        yerr=self.orig_yerr[~self.mask],
                        fmt="o",
                        color="grey",
                        ms=5,
                        alpha=0.2,
                        **data_kwargs,
                    )
                else:
                    ax_residual.scatter(
                        self.orig_xdata[~self.mask],
                        full_residuals[~self.mask],
                        color="grey",
                        s=5,
                        alpha=0.2,
                        **data_kwargs,
                    )

                # ax_residual.set_xlim(0.8 * self.xdata.min(), 1.1 * self.xdata.max())
                # ax_residual.set_ylim(residual_ylim)

            ax_residual.set_xlim(fit_xlim)
            ax_residual.set_ylim(residual_ylim)
            ax_residual.set_xlabel("log T (sec)")
            ax_residual.set_ylabel("residuals")
            plt.setp(ax_fit.get_xticklabels(), visible=False)
            self.figs["fit"] = plot_fig

            if show:
                plt.show()
            else:
                plt.close()
                
        if (show_corner or detailed) and (getattr(self.res, "flatchain")) is not None:
            import corner

            corner_fig = corner.corner(
                self.res.flatchain,
                labels=[
                    self.model[p].plot_fmt for p in self.model if self.model[p].vary
                ]
                + (["__lnsigma"] if isinstance(self.sigma, (int, float)) else []),
                truths=list(
                    self.params[p].value for p in self.params if self.params[p].vary
                ),
                **corner_kwargs,
            )

            self.figs["corner"] = corner_fig

            if show:
                plt.show()
            else:
                plt.close()
                
        if show_chisq or detailed:  # and getattr(self.res, "flatchain") is not None:

            num_varied = sum(self.params[p].vary for p in self.params)
            fig_chisq, ax_chisq = plt.subplots(
                1, num_varied, figsize=(5 * num_varied, 5)
            )

            fineness = chisq_kwargs.pop("fineness", 0.1)
            multiplier = np.arange(-2, 2, fineness)
            p, perr = np.array(
                [
                    [
                        self.params[p].value,
                        self.params[p].stderr
                        if self.params[p].stderr is not None
                        else 0,
                    ]
                    for p in self.params
                ]
            ).T

            paramspace = np.array(
                [p + m * perr for m in multiplier]
            )  # shape is (len(multiplier), len(params))

            best_chisq = chisq(self.xdata, self.ydata, self.sigma, self.model, p)

            idx = 0
            for ax_ in list(ax_chisq):
                if not self.params[list(self.params.keys())[idx]].vary:
                    idx += 1

                chisq_params = np.tile(p, (len(multiplier), 1))
                chisq_params[:, idx] = paramspace[:, idx]
                delta_chisq = [
                    chisq(self.xdata, self.ydata, self.sigma, self.model, chisq_param)
                    - best_chisq
                    for chisq_param in chisq_params
                ]
                curr_param_fmt = self.model[list(self.params.keys())[idx]].plot_fmt
                ax_.plot(
                    multiplier,
                    delta_chisq,
                    label=curr_param_fmt + fr"={p[idx]:.3f} $\pm$ {perr[idx]:.3f}",
                    color="k",
                    **chisq_kwargs,
                )
                ax_.legend(framealpha=0.0)
                ax_.set_xlabel(r"$\sigma$ multiplier")
                ax_.set_ylabel(r"$\Delta\chi^2$")
                ax_.set_title(curr_param_fmt)
                ax_.set_ylim(0, None)
                ax_.set_xlim(-2, 2)
                ax_.axvline(
                    x=-1,
                    c="k",
                    ls=":",
                    alpha=0.2,
                    linewidth=plt.rcParams["axes.linewidth"],
                    zorder=-999,
                )
                ax_.axvline(
                    x=1,
                    c="k",
                    ls=":",
                    alpha=0.2,
                    linewidth=plt.rcParams["axes.linewidth"],
                    zorder=-999,
                )
                idx += 1

            self.figs["chisq"] = fig_chisq

            if show:
                plt.show()
            else:
                plt.close()
                
        if print_res or detailed:
            self.print_fit(detailed, fix_flux)
        if bool(save_plots):
            savekwargs = dict(
                filename=save_plots if isinstance(save_plots, str) else None
            )
            for fig in self.figs:
                self._savefig(self.figs[fig], suffix=fig, **savekwargs, model_name=model_name)

        if not show:
            return self.figs

    def show(self, *args, **kwargs):
        """
        Calls :py:meth:`Lightcurve.show_fit` if a fit has been done,
        :py:meth:`Lightcurve.show_data` otherwise


        Returns
        -------
        *args : optional
            Positional arguments to pass to :py:meth:`Lightcurve.show_data` or :py:meth:`Lightcurve.show_fit`

        **kwargs : optional
            Keyword arguments to pass to :py:meth:`Lightcurve.show_data` or :py:meth:`Lightcurve.show_fit`
        """

        if getattr(self, "res", None) is not None:
            func = self.show_fit
        else:
            func = self.show_data

        return func(*args, **kwargs)

    def apply_flux_corr(self, inplace=False):
        '''
        Applies a correction to the best-fit flux at the end of
        the plateau.

        Parameters
        ----------
        res
            the result of the fit
        inplace, optional
            If True, the correction will be update the flux and flux error _inside_ the Lightcurve object.
            Note that this may make Lightcurve.show_fit() give incorrect results.
        '''
        assert getattr(self, "res", None) is not None, "No fit has been done"

        if self.model.slug not in ["w07", "smooth_bpl"]:
            warnings.warn(f"Lightcurve.fix_flux is set to true, but model '{self.model.slug}' is not 'W07' or 'smooth_bpl.'" \
                           "Skipping...", UserWarning, stacklevel=2)
            return self.res.params["F"].value, self.res.params["F"].stderr

        errF = self.res.params["F"].stderr
        F = self.res.params["F"].value
        if self.model.slug == "smooth_bpl":
            errS = 10**self.res.params["S"].stderr * np.log(10) if self.res.params["S"].stderr is not None else 0
            S = 10**self.res.params["S"].value
            newF = F - np.log(2)/S
            newerrF = np.sqrt(errF*errF + errS*errS*np.log(2)/S/S)
        elif self.model.slug == "w07":
            t = self.res.params["t"].value
            T = self.res.params["T"].value
            errT = self.res.params["T"].stderr
            errt = self.res.params["t"].stderr if self.res.params["t"].vary else 0
            newerrF = np.sqrt(errF*errF + (errT*errT + errt*errt)*10**(2*(t-T))/np.log(10)**2)
            newF = F - 10**(t-T)/np.log(10)

        if inplace:
            assert getattr(self, "_flux_fixed", False), "Flux already fixed in place."
            self.res.params["F"].value = newF
            self.res.params["F"].stderr = newerrF
            newF = self.res.params["F"].value
            newerrF = self.res.params["F"].stderr
            self._flux_fixed_inplace = True
            self.fix_flux = False
        return newF, newerrF

    def print_fit(self, detailed=False, fix_flux=None):
        """
            Print a fit report to `stdout`.

        Parameters
        ----------
        detailed : bool, optional
            Whether you'd like the full-on fit report, or a simplified version with
            the necessaries, by default False
        fix_flux : bool, optional
            If provided, will apply the flux correction to the flux at the end of the plateau.
            If not provided, will default to what user set in the constructor of the Lightcurve class.

        Example:

        .. jupyter-execute::

            import numpy as np
            from grblc.fitting import Model, Lightcurve

            model = Model.W07(vary_t=False)
            xdata = np.linspace(0, 10, 15)
            yerr = np.random.normal(0, 0.5, len(xdata))
            ydata = model(xdata, 5, -12, 1.5, 0) + yerr
            lc = Lightcurve(xdata=xdata, ydata=ydata, yerr=yerr, model=model)
            lc.fit(p0=[4.5, -12.5, 1, 0], run_mcmc=False)
            for detailed in [False, True]:
                print("="*10 + f"detailed={detailed}" + "="*10)
                lc.print_fit(detailed=detailed)
        """
        if fix_flux is None:
            fix_flux = self.fix_flux

        res = deepcopy(self.res)
        params = deepcopy(self.params)
        if fix_flux:
            newF, newFerr = self.apply_flux_corr()
            res.params["F"].value = params["F"].value = newF
            res.params["F"].stderr = params["F"].stderr = newFerr
        if detailed:
            print(lf.fit_report(res, show_correl=False))
        else:
            print(
                "\n".join(
                    [
                        "\t".join(
                            map(str, [x, params[x].value, params[x].stderr])
                        )
                        for x in params
                    ]
                )
            )
            
    
    def save_fitting(self, detailed=False, model_name=None, suffix=None):
        FILE_DIR = path.join(path.abspath(get_dir()), "plots/"+ self.name + "/" + model_name)
        if not path.exists(FILE_DIR):
            makedirs(FILE_DIR)
        suffix = "_" + suffix if suffix is not None else ""
        filename = path.join(FILE_DIR, self.name + "_" + model_name + suffix + ".txt")
        res = deepcopy(self.res)
        params = deepcopy(self.params)
        with open(filename, "w") as f:
            print(self.name, model_name, "\n", file=f)
            print(lf.fit_report(res, show_correl=False), file=f)
        

    def _savefig(self, fig, model_name=None, filename=None, suffix=None, format="pdf", **kwargs):
        assert isinstance(fig, Figure), "figs must be a matplotlib Figure."

        suffix = "_" + suffix if suffix is not None else ""

        if filename is None:
            FILE_DIR = path.join(path.abspath(get_dir()), "plots/" + self.name + "/" + model_name)

            if not path.exists(FILE_DIR):
                makedirs(FILE_DIR)

            filename = path.join(
                FILE_DIR,
                "{}{}.{}".format(
                    self.name.replace(" ", "_").replace(".", "p"), suffix, format
                ),
            )
        else:
            FILE_DIR = path.join(path.abspath(get_dir()), "plots/"+ self.name + "/" + model_name)

            if not path.exists(FILE_DIR):
                makedirs(FILE_DIR)
            *fn, extension = filename.split(".")
            filename = path.join(FILE_DIR, self.name + "_" + model_name +  ".".join(fn) + suffix + "." + extension)

        savefig_kwargs = dict(
            fname=filename,
            dpi=plt.rcParams["savefig.dpi"],
            metadata={"Creator": f"grbLC v{__version__}"},
        )
        if bool(kwargs):
            savefig_kwargs.update(kwargs)

        fig.savefig(**savefig_kwargs)

    def _check_dir(self):
        if not path.exists(self.dir):
            best_params = self.model.func_args
            best_err = [param + "_err" for param in best_params]
            best_guesses = [param + "_guess" for param in best_params]

            header = []
            header += ["GRB", "tt", "tf"]
            for param, err in zip(best_params, best_err):
                header += [param, err]
            header += best_guesses
            header += ["chisq"]

            with open(self.dir, "w") as f:
                f.write("\t".join(header) + "\n")

    def save_fit(self, filename=None):
        """
            Saves fit values to either a specified file or a default file.

        Parameters
        ----------
        filename : str, optional
            File name to save fit values to, by default None
        """
        assert getattr(self, "res", None) is not None, "No fit results found."

        io.save_fit(self.res, filename=filename)

    def __repr__(self):
        return f"<grbLC> {self.__class__.__name__}({self.name})"

    def to_dict(self, data_space="lin"):
        """Function that returns a dictionary of the lightcurve data.

            You can specify the data to be in either logarithmic (`log`) or linear
            (`lin`) space.

            Each dictionary contains the following keys:

                #. `time_sec` : The time of the datapoint in seconds (xdata)

                #. `flux` : The flux of the datapoint in erg cm\(^{-2}\) s\(^{-1}\) (ydata)

                #. `flux_err` : The flux error of the datapoint in erg cm\(^{-2}\) s\(^{-1}\)
                (yerr)

                #. `**attrs` : Any additional attributes of the datapoint as given in the
                instantiation of the :py:class:`Lightcurve` object.

        Parameters
        ----------
        data_space : str, {"log", "lin"}, optional
            Whether the data returned will be in logarithmic or linear space,
            by default "lin"

        Returns
        -------
        data : dict

        Raises
        ------
        ValueError
            If any "space" other than "log" and "lin" are specified.
        """
        assert hasattr(self, "mask")

        # assumes data is already in log space
        def loglin_data(data):
            if data_space == "log":
                log_data = data
                return log_data
            elif data_space == "lin":
                lin_data = np.power(10, data)
                return lin_data
            else:
                raise ValueError("data_space must be 'log' or 'lin'")

        # assumes data is already in log space
        def loglin_err(data, err):
            if data_space == "log":
                log_err = err
                return log_err
            elif data_space == "lin":
                log_err = err
                log_data = data
                lin_err = log_err * np.power(10, log_data) * np.log(10)
                return lin_err
            else:
                raise ValueError("data_space must be 'log' or 'lin'")

        data_attrs = {k: v[self.mask] for k, v in self.attrs.items()}

        data_dict = dict(
            time_sec=loglin_data(self.xdata),
            flux=loglin_data(self.ydata),
            flux_err=loglin_err(self.ydata, self.yerr),
            **data_attrs,
        )

        return data_dict

    def to_df(self, data_space="lin"):
        """Function that returns a Pandas DataFrames of the lightcurve data.

            You can specify the data to be returned in either logarithmic (`log`) or
            linear (`lin`) space.

            Each DataFrame contains the following columns:

                #. `time_sec` : The time of the datapoint in seconds (xdata)
                #. `flux` : The flux of the datapoint in erg cm\(^{-2}\) s\(^{-1}\) (ydata)
                #. `flux_err` : The flux error of the datapoint in erg cm\(^{-2}\) s\(^{-1}\) (yerr)
                #. **attrs : Any additional attributes of the datapoint as given in the
                             instantiation of the :py:class:`Lightcurve` object.


        Parameters
        ----------
        data_space : str, {"log", "lin"}, optional
            Whether the data returned will be in logarithmic or linear space,
            by default "log"

        Returns
        -------
        data : pd.DataFrame

        Raises
        ------
        ValueError
            If any "space" other than "log" and "lin" are specified.
        """
        return pd.DataFrame(self.to_dict(data_space=data_space))

    def prompt(self):
        """
        A pipeline for fitting a light curve from user inputs. The data is first shown,
        bounds are set, parameter priors are set, and the fit is run, and saved if desired.
        """
        self.show_data()

        auto_guess = input("want to fit? (y/[n])").lower()
        if auto_guess in ["y"]:

            if not hasattr(self.model, "func"):
                model_name = input(
                    "model to use? (i.e., W07, SIMPLE_BPL, or SMOOTH_BPL)"
                ).lower()
                self.model = getattr(Model, model_name).__init__()

            xmin = input("xmin : [-inf]")
            xmin = float(xmin) if xmin != "" else -np.inf
            xmax = input("xmax : [inf]")
            xmax = float(xmax) if xmax != "" else np.inf
            ymin = input("ymin : [-inf]")
            ymin = float(ymin) if ymin != "" else -np.inf
            ymax = input("ymax : [inf]")
            ymax = float(ymax) if ymax != "" else np.inf

            self.set_bounds(bounds=[xmin, xmax, ymin, ymax])

            param_guesses = []
            for param in self.model:
                param_guesses.append(float(input(f"init {param} : ")))

            self.fit(p0=param_guesses, run_mcmc=True)
            self.show_fit()

            if str(input("save? ([y]/n): ")) in ["", "y"]:
                self.show_fit(show=False, save_plots=True)

        else:
            from IPython import clear_output

            clear_output()
            return


major, *__ = sys.version_info
readfile_kwargs = {"encoding": "utf-8"} if major >= 3 else {}


def _readfile(filename):
    with open(filename, **readfile_kwargs) as fp:
        contents = fp.read()
    return contents


version_regex = re.compile('__version__ = "(.*?)"')
contents = _readfile(
    path.join(
        path.dirname(path.dirname(path.abspath(__file__))), "__init__.py"
    )
)
__version__ = version_regex.findall(contents)[0]

__directory__ = get_dir()
