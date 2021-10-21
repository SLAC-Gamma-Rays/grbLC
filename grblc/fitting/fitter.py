import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize, leastsq
from .constants import *
from .models import smooth_bpl, w07, sharp_bpl, chisq, probability
from .fitting import plot_chisq, plot_fit, plot_data
import convert.convert as convert
from functools import reduce
import inspect, os
from IPython.display import clear_output
import time

W07 = 0
SMOOTH_BPL = 1
SHARP_BPL = 2

funclist = {W07: w07, SMOOTH_BPL: smooth_bpl, SHARP_BPL: sharp_bpl}
funcspecs = {
    W07: inspect.getargspec(w07).args,
    SMOOTH_BPL: inspect.getargspec(smooth_bpl).args,
    SHARP_BPL: inspect.getargspec(sharp_bpl).args,
}
plabellist = {
    W07: ["T", "F", r"$\alpha$", "t"],
    SMOOTH_BPL: ["T", "F", r"$\alpha_{1}$", r"$\alpha_{2}$", "S"],
    SHARP_BPL: ["T", "F", r"$\alpha_{1}$", r"$\alpha_{2}$"],
}

funcpriors = {
    W07: [[0, -50, 0, 0], [np.inf, -1, 5, np.inf]],
    SMOOTH_BPL: [[1e-5, -50, -20, -20, -np.inf], [np.inf, -1, 20, 20, np.inf]],
    SHARP_BPL: [[1e-5, -50, -5, 0], [np.inf, -1, 5, 20]],
}


class Fitter:
    def __init__(
        self,
        model,
        xdata,
        ydata,
        xerr=None,
        yerr=None,
        bounds=(-np.inf, np.inf, -np.inf, np.inf),
        priors=None,
        save_dir=None,
        GRBname=None,
    ):

        if isinstance(model, int):
            self.func = funclist[model]
            # self.func_args = ["x", "T", "F", "alpha1", "alpha2"]  # hardcoding sbpl rn
            self.func_args = funcspecs[model][1:]
            self.plabels = plabellist[model]
            self.priors = funcpriors[model]
            tmin, tmax, fmin, fmax = bounds
            self.priors[0][0] = tmin
            self.priors[0][1] = fmin
            self.priors[1][0] = tmax
            self.priors[1][1] = fmax

        else:
            raise Exception("other functions aren't fully ready for use yet")
            self.func = model
            self.func_args = list(inspect.getargspec(self.func)[0])
            self.plabels = self.func_args
            self.priors = priors

        self.xdata = np.asarray(xdata)
        self.ydata = np.asarray(ydata)
        self.xerr = np.asarray(xerr) if xerr is not None else None
        self.yerr = np.asarray(yerr) if yerr is not None else None
        self.set_bounds(bounds)
        self.bounds = bounds  # in the form of (xmin, xmax, ymin, ymax)

        self.GRBname = GRBname
        self.fit_vals = None

        if self.priors is not None:
            assert np.shape(self.priors) == (2, len(self.func_args)), "Incorrect shape for prior bounds."

        if save_dir is not None:
            self.dir = save_dir
        else:
            self.dir = convert.get_dir()

        self._check_dir()

    def _check_dir(self):
        self.FIT_VALS_DIR = os.path.join(self.dir, "fit_vals.txt")
        if not os.path.exists(self.FIT_VALS_DIR):
            best_params = self.func_args
            best_err = [param + "_err" for param in best_params]
            best_guesses = [param + "_guess" for param in best_params]

            header = []
            header += ["GRB", "tt", "tf"]
            for param, err in zip(best_params, best_err):
                header += [param, err]
            header += best_guesses
            header += ["chisq"]

            with open(self.FIT_VALS_DIR, "w") as f:
                f.write("\t".join(header) + "\n")

    def set_dir(self, dirname):
        self.dir = convert.set_dir(dirname)
        self._check_dir()

    def set_bounds(self, bounds):
        xmin, xmax, ymin, ymax = self.bounds = bounds
        xmask = (xmin <= self.xdata) & (self.xdata <= xmax)
        ymask = (ymin <= self.ydata) & (self.ydata <= ymax)
        self.mask = xmask & ymask

        self.set_data(self.xdata, self.ydata, self.xerr, self.yerr)

    def set_data(self, xdata, ydata, xerr=None, yerr=None):
        self.orig_xdata = np.asarray(xdata)
        self.orig_ydata = np.asarray(ydata)
        self.xdata = np.asarray(xdata)[self.mask]
        self.ydata = np.asarray(ydata)[self.mask]
        self.orig_xerr = np.asarray(xerr) if xerr is not None else None
        self.orig_yerr = np.asarray(yerr) if yerr is not None else None

        self.xerr = np.asarray(xerr)[self.mask] if xerr is not None else None
        self.yerr = np.asarray(yerr)[self.mask] if yerr is not None else None

        self.sigma = np.sqrt(np.sum([err ** 2 for err in [self.xerr, self.yerr] if err is not None], axis=0))
        if isinstance(self.sigma, int):
            self.sigma = None

    def fit(self, p0, return_guess=False, **kwargs):
        assert np.shape(self.xdata) == np.shape(self.ydata), "xdata and ydata not the same shape"
        assert np.shape(self.sigma) == np.shape(self.ydata), "err not the same shape as input data"
        assert self.xdata is not None, "xdata not supplied"
        assert self.ydata is not None, "ydata not supplied"

        p, cov = curve_fit(
            self.func,
            self.xdata,
            self.ydata,
            p0=p0,
            bounds=self.priors,
            sigma=self.sigma,
            absolute_sigma=self.sigma is not None,
            method="trf",
            **kwargs,
        )

        self.fit_vals = p, cov, p0
        self.chisq = chisq(self.xdata, self.ydata, self.sigma, self.func, *p)

        if return_guess:
            return p, cov, p0
        else:
            return p, cov

    def show(self):
        tt, tf, *__ = self.bounds
        plot_data(
            self.func,
            self.orig_xdata,
            self.orig_ydata,
            tt,
            tf,
            self.orig_xerr,
            self.orig_yerr,
            title=self.GRBname,
            show=True,
        )
        plt.show()

    def show_fit(self, save=False):

        p, cov, p0 = self.fit_vals
        free_params = self.func_args
        fit_length = int(0.75 * len(free_params))
        empty_length = len(free_params) - fit_length
        figlength = 10 / 3 * len(free_params)  # normalize to 10 when there are 3 parameters

        ax = plt.figure(constrained_layout=True, figsize=(figlength, 7)).subplot_mosaic(
            [
                ["fit"] * fit_length + ["EMPTY"] * empty_length,
                list(free_params),
            ],
            empty_sentinel="EMPTY",
        )

        tt, tf, *__ = self.bounds

        # read in fitted vals
        perr = np.sqrt(np.diag(cov))
        plot_fit(
            self.func,
            self.orig_xdata,
            self.orig_ydata,
            p,
            tt=tt,
            tf=tf,
            xerr=self.orig_xerr,
            yerr=self.orig_yerr,
            p0=p0,
            ax=ax["fit"],
            show=False,
        )
        plot_chisq(
            self.func,
            self.xdata,
            self.ydata,
            self.sigma,
            p,
            perr,
            labels=self.plabels,
            ax=[ax[param] for param in self.func_args],
            show=False,
            fineness=0.05,
        )

        chisquared = chisq(self.xdata, self.ydata, self.sigma, self.func, *p)
        reduced_nu = len(self.xdata) - 3
        reduced_nu = 1 if reduced_nu == 0 else reduced_nu
        reduced = chisquared / reduced_nu
        nu = len(self.xdata)
        prob = probability(reduced, nu)

        textx = fit_length / (len(free_params) + 1) * (1.2)
        plt.figtext(
            x=textx,
            y=0.6,
            s="""
            GRB %s

            $\\chi^2$: %.3f
            
            $\\chi_{\\nu}^2$: %.3f
            
            $\\alpha$ : %.3e
            """
            % (self.GRBname, chisquared, reduced, prob),
            size=18,
        )
        if save:
            plt.savefig(f"{self.GRBname}.pdf", dpi=300)

        plt.show()

    def save(self):

        assert self.fit_vals is not None, "Called save but no fit has been done."

        with open(self.FIT_VALS_DIR, "a") as f:

            p, cov, p0 = self.fit_vals
            perr = np.sqrt(np.diag(cov))
            tt, tf, *__ = self.bounds

            row = f"{self.GRBname}\t"  # GRBname
            row += f"{tt}\t{tf}\t"  # tt and tf
            row += "\t".join([f"{val}\t{valerr}" for val, valerr in zip(p, perr)])  # params and their errors
            row += "\t".join([f"{valguess}" for valguess in p0])  # parameter guesses
            row += f"\t{self.chisq}"
            row += "\n"

            f.write(row)

    def prompt(self):

        self.show()

        auto_guess = input("do you want to fit? (y/[n])").lower()
        if auto_guess in ["y"]:

            tt = float(input("tt : "))
            tf = float(input("tf : "))
            self.set_bounds([tt, tf, -np.inf, np.inf])

            param_guesses = []
            for param in self.func_args:
                param_guesses.append(float(input(f"{param} : ")))

            self.fit(p0=param_guesses)
            self.show_fit()
            plt.show()
            time.sleep(0.1)

            if str(input("save? ([y]/n): ")) in ["", "y"]:
                self.save()

        else:
            clear_output()
            return
