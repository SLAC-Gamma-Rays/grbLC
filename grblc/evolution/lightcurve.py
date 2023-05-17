# standard libs
import os
import re
import sys
from functools import reduce

# third party libs
import numpy as np
import pandas as pd
import lmfit as lf
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# custom modules
from grblc.util import get_dir
from . import io


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
    ):
        """The main module for fitting lightcurves.

        .. warning::
            Data stored in :py:class:`Lightcurve` objects are always in logarithmic
            space; the parameter ``data_space`` is only used to convert data to log space
            if it is not already in such. If your data is in linear space [i.e., your
            time data is sec, and not $log$(sec)], then you should set ``data_space``
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
        """
        assert bool(filename) ^ (
            xdata is not None and ydata is not None
        ), "Either provide a filename or xdata, ydata."

        if name:
            self.name = name
        else:
            self.name = self._name_placeholder

        if isinstance(filename, str):
            self.filename = filename
            self.set_data(*self.read_data(filename), data_space=data_space)
        else:
            self.filename = reduce(
                os.path.join,
                [
                    get_dir(),
                    "lightcurves",
                    "{}_flux.txt".format(self.name.replace(" ", "_").replace(".", "p")),
                ],
            )
            self.set_data(xdata, ydata, xerr, yerr, data_space=data_space)


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
        ydata = df["mag"].to_numpy()
        xerr = None
        yerr = df["mag_err"].to_numpy()

        return xdata, ydata, xerr, yerr

    def show_data(self, save=False, fig_kwargs={}, save_kwargs={}):
        """
            Plots the lightcurve data. If no fit has been ran, :py:meth:`Lightcurve.show` will call
            this function.

            .. note:: This doesn't plot any fit results. Use :py:meth:`Lightcurve.show_fit` to do so.

            Example:

            .. jupyter-execute::

                import numpy as np
                import grblc

                model = grblc.Model.W07(vary_t=False)
                xdata = np.linspace(0, 10, 15)
                yerr = np.random.normal(0, 0.5, len(xdata))
                ydata = model(xdata, 5, -12, 1.5, 0) + yerr
                lc = grblc.Lightcurve(xdata=xdata, ydata=ydata, yerr=yerr, model=model)
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
        logt = self.orig_xdata
        mag = self.orig_ydata
        logterr = self.orig_xerr
        magerr = self.orig_yerr

        mask = (logt >= xmin) & (logt <= xmax) & (mag >= ymin) & (mag <= ymax)

        # plot all data points inside xmin and xmax in black
        ax.errorbar(
            logt[mask],
            mag[mask],
            xerr=logterr[mask] if logterr is not None else logterr,
            yerr=magerr[mask] if magerr is not None else magerr,
            color="k",
            fmt=".",
            ms=10,
            zorder=0,
        )

        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()

        # plot all data points outside of xmin and xmax in grey
        if sum(~mask) > 0:
            ax.errorbar(
                logt[~mask],
                mag[~mask],
                xerr=logterr[~mask] if logterr is not None else logterr,
                yerr=magerr[~mask] if magerr is not None else magerr,
                color="grey",
                fmt=".",
                ms=10,
                alpha=0.2,
                zorder=0,
            )

        ax.set_xlim(ax_xlim)
        ax.set_ylim(ax_ylim)
        ax.set_xlabel("log time (sec)")
        ax.set_ylabel("AB magnitude")
        ax.set_title(self.name)

        plt.show()

    def _res(self, params):
        p = params.valuesdict().values()

        return (self.model.func(self.xdata, *p) - self.ydata) / self.sigma


major, *__ = sys.version_info
readfile_kwargs = {"encoding": "utf-8"} if major >= 3 else {}


def _readfile(filename):
    with open(filename, **readfile_kwargs) as fp:
        contents = fp.read()
    return contents


version_regex = re.compile('__version__ = "(.*?)"')
contents = _readfile(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "__init__.py"
    )
)
__version__ = version_regex.findall(contents)[0]

__directory__ = get_dir()
