import inspect
from copy import deepcopy
from functools import reduce
from typing import Callable
from typing import Dict
from typing import List

import numpy as np



def chisq(x, y, sigma, model, p, return_reduced=False):
    r"""A function to calculate the chi-squared value of a given proposed solution:
    .. math:: \chi^2 = \sum_{i=1}^N \frac{(y_i - f(x_i))^2}{\sigma_i^2}
    The reduced :math:`\chi^2` value, :math:`\chi^2_\nu`, can also be returned, and is calculated as:
    .. math:: \chi^2_\nu = \frac{\chi^2}{{\rm \# ~data~ points} - {\rm \# ~free~ params}}
    Parameters
    ----------
    x, y : array_like
        The x and y values of the data points.
    sigma : array_like
        Standard error of the data points.
    model : callable
        The model to be fit to the data. Should take the form of a function that takes `x`, parameters `p`, and returns `y` in the form of ``y = model(x, *p)``.
    p : array_like
        List of parameter values to be used in the model.
    return_reduced : bool, optional
        Determines whether the reduced :math:`\chi^2` will be returned as well, by default False
    Returns
    -------
    numpy.ndarray
        :math:`\chi^2` for each point in the dataset, along with the reduced $\chi^2$ value (if return_reduced=True)
    """

    x = np.asarray(x)
    y = np.asarray(y)
    sigma = np.asarray(sigma)
    r = y - model(x, *p)
    _chisq = np.sum(r ** 2 / sigma ** 2)

    if return_reduced:
        return _chisq, _chisq / (len(x) - len(p))
    else:
        return _chisq


# The famous Willingale et. al 2007 model
# modified so T and F are logarithmic inputs
# to avoid numerical overflow issues
def _w07(x, T, F, alpha, t):
    if t > T:
        tmp=t
        t=T
        T=tmp
    before = lambda x: (
        -10 ** (t-x) + alpha - alpha * 10 ** (x - T) + F * np.log(10)
    ) / np.log(10)

    after = lambda x: F + T * alpha - x * alpha - 10 ** (t-x) / np.log(10)

    vals = np.piecewise(x, [x < T, x >= T], [before, after])

    return vals


# Simple broken power law
# modified and simplified so T and F are logarithmic inputs
# to avoid numerical overflow issues.
def _bpl(x, T, F, alpha1, alpha2):

    before = lambda x: F - alpha1*(x-T)
    after = lambda x: F - alpha2*(x-T)

    vals = np.piecewise(x, [x < T, x >= T], [before, after])

    return vals

# Smooth broken power law
# modified and simplified so T and F are logarithmic inputs
# to avoid numerical overflow issues.
def _sbpl(x, T, F, alpha1, alpha2, S):

    return F - alpha1*(x-T) - 1/(10**S)*np.log10(1 + 10**((10**S)*(alpha2-alpha1)*(x-T)))


# Power law
def _pl(x, F, alpha):

    return F - alpha*x


# Combination functions
# Prompt as Willingale 2007

def _w07_bpl(x, F_p, alpha_p, alpha_1, alpha_2, t0, t1, t2, t):
    if t0 > t1:
        tmp=t0
        t0=t1
        t1=tmp
    if t1 > t2:
        tmp=t1
        t1=t2
        t2=tmp
    if t0 > t1:
        tmp=t0
        t0=t1
        t1=tmp

    F_a = F_p + t0 * alpha_p - t1 * alpha_p - 10**(t-t1) / np.log(10)
    F_d = F_a - alpha_1 * (t2 - t1)
    before = lambda x: (
        -10 ** (t-x) + alpha_p - alpha_p * 10 ** (x - t0) + F_p * np.log(10)
    ) / np.log(10)

    middle = lambda x: F_p + t0 * alpha_p - x * alpha_p - 10 ** (t-x) / np.log(10)
    middle2 = lambda x: F_a - alpha_1 * (x - t1)
    after = lambda x: F_d - alpha_2 * (x - t2)

    cond = [(x < t0), (x >= t0) * (x < t1), (x >= t1) * (x < t2), (x >= t2)]
    func = [before, middle, middle2, after]
    vals = np.piecewise(x, cond, func)

    return vals


def _w07_sbpl(x, T_p, F_p, alpha_p, t_p, T_a, F_a, alpha_a1, alpha_a2, S_a, tt):

    p1 = T_p, F_p, alpha_p, t_p
    p2 = T_a, F_a, alpha_a1, alpha_a2, S_a

    cond = [(x < tt), (x >= tt)]
    func = [_w07(x[x<tt],*p1), _sbpl(x[x>=tt],*p2)]
    vals = np.piecewise(x, cond, func)

    return vals


def _w07_pl(x, F_p, alpha_p, t, alpha_1, t0, t1):

    if t0 > t1:
        tmp=t0
        t0=t1
        t1=tmp

    F_a = F_p + t0 * alpha_p - t1 * alpha_p - 10**(t-t1) / np.log(10)
    before = lambda x: (
        -10 ** (t-x) + alpha_p - alpha_p * 10 ** (x - t0) + F_p * np.log(10)
    ) / np.log(10)

    middle = lambda x: F_p + t0 * alpha_p - x * alpha_p - 10 ** (t-x) / np.log(10)
    after = lambda x: F_a - alpha_1 * (x - t1)

    cond = [(x < t0), (x >= t0) * (x < t1), (x >= t1)]
    func = [before, middle, after]
    vals = np.piecewise(x, cond, func)

    return vals


#  Prompt as power law

def _pl_w07(x, F_p, alpha_p, T_a, F_a, alpha_a, tt):
    
    p1 = F_p, alpha_p
    p2 = T_a, F_a, alpha_a, tt

    cond = [(x < tt), (x >= tt)]
    func = [_pl(x[x<tt],*p1), _w07(x[x>=tt],*p2)]
    vals = np.piecewise(x, cond, func)

    return vals
    
# power law + broken power law
def _pl_bpl(x, F_p, alpha_p, alpha_a1, alpha_a2, t0, t1):
    if t0 > t1:
      tmp = t0
      t0 = t1
      t1 = tmp
    F_a = F_p - alpha_p * t0 - alpha_a1 * (t1 - t0)
    before = lambda x: F_p - alpha_p*x
    middle = lambda x: F_a - alpha_a1 * (x-t1)
    after = lambda x: F_a - alpha_a2 * (x - t1)
    vals = np.piecewise(x, [x < t0, (x >= t0) * (x < t1), x >= t1 ], [before, middle, after])
    return vals

    
def _pl_sbpl(x, F_p, alpha_p, t1, F_a, alpha_a1, alpha_a2, S_a, t0):
    p1 = F_p, alpha_p
    p2 = t1, F_a, alpha_a1, alpha_a2, S_a

    cond = [(x < t0), (x >= t0)]
    func = [_pl(x[x<t0],*p1), _sbpl(x[x>=t0],*p2) + _pl(t0, *p1) - _sbpl(t0, *p2)]
    vals = np.piecewise(x, cond, func)
    vals=_pl(x, *p1) + _sbpl(x, *p2)

    return vals
    

def _w07_bpl_pl(x, F_p, alpha_p, alpha_1, alpha_2, alpha_3, t0, t1, t2, t3, t):
    if t0 > t1:
        tmp=t0
        t0=t1
        t1=tmp
    if t1 > t2:
        tmp=t1
        t1=t2
        t2=tmp
    if t2 > t3:
        tmp=t2
        t2=t3
        t3=tmp
    if t0 > t1:
        tmp=t0
        t0=t1
        t1=tmp
    if t1 > t2:
        tmp=t1
        t1=t2
        t2=tmp
    if t0 > t1:
        tmp=t0
        t0=t1
        t1=tmp
        

    F_a = F_p + t0 * alpha_p - t1 * alpha_p - 10**(t-t1) / np.log(10)
    F_d = F_a - alpha_1 * (t2 - t1)
    F_d2=F_d - alpha_2 * (t3-t2)
    before = lambda x: (
        -10 ** (t-x) + alpha_p - alpha_p * 10 ** (x - t0) + F_p * np.log(10)
    ) / np.log(10)

    middle = lambda x: F_p + t0 * alpha_p - x * alpha_p - 10 ** (t-x) / np.log(10)
    middle2 = lambda x: F_a - alpha_1 * (x - t1)
    middle3 = lambda x: F_d - alpha_2 * (x - t2)
    after = lambda x: F_d2 - alpha_3 * (x - t3)

    cond = [(x < t0), (x >= t0) * (x < t1), (x >= t1) * (x < t2), (x >= t2)*(x<t3), (x >= t3)]
    func = [before, middle, middle2, middle3, after]
    vals = np.piecewise(x, cond, func)

    return vals


def _pl_bpl_pl(x, F_p, alpha_p, alpha_a1, alpha_a2, alpha_a3, t0, t1, t2):
    if t0 > t1:
        tmp=t0
        t0=t1
        t1=tmp
    if t1 > t2:
        tmp=t1
        t1=t2
        t2=tmp
    if t0 > t1:
        tmp=t0
        t0=t1
        t1=tmp
    F_a = F_p - alpha_p * t0 - alpha_a1 * (t1-t0)
    F_d = F_a - alpha_a2 * (t2 - t1)
    before = lambda x: F_p - alpha_p*x
    middle = lambda x: F_a - alpha_a1 * (x-t1)
    middle2 = lambda x: F_a - alpha_a2 * (x - t1)
    after = lambda x: F_d - alpha_a3 * (x - t2)
    vals = np.piecewise(x, [x < t0, (x >= t0) * (x < t1), (x >= t1) * (x < t2), x >= t2], [before, middle, middle2, after])
    return vals



class Parameter:
    def __init__(
        self,
        name: str,
        description: str = None,
        min: float = -np.inf,
        max: float = np.inf,
        vary: bool = True,
        plot_fmt: str = None,
    ):
        """
        Parameter class for use with the :class:`Model` class.
                This class is used to store the information about a parameter in a model.
                Information includes the name, description, parameter priors, and whether
                the parameter is to be varied in fitting.
        Parameters
        ----------
        name : str
            Parameter name.
        description : str, optional
            Description of the parameter, by default None
        min : float, optional
            Minimum possible value of the parameter, by default ``-np.inf``
        max : float, optional
            Maximum possible value of the parameter, by default ``np.inf``
        vary : bool, optional
            Controls whether the variable will be allowed to vary in fitting, by default True
        plot_fmt : str, optional
            LaTeX form of the parameter name to be plotted, by default the same as `name`.
        """
        self.name = name
        self.description = (
            description
            if description is not None
            else "Autogenerated argument from input function."
        )
        self.plot_fmt = plot_fmt if plot_fmt is not None else name
        self.min = min
        self.max = max
        self.vary = vary

    def __repr__(self):
        return f"<grblc Parameter> {self.name}=[{self.min}, {self.max}], vary={self.vary}"


class Model:
    def __init__(
        self,
        func: Callable,
        name: str = "",
        slug: str = "",
        func_args: List[Parameter] = None,
        bounds: list = None,
    ):
        """Model class for use with the :class:`Lightcurve` class.
                This class is a wrapper around a function that can be used to fit a lightcurve.
        Parameters
        ----------
        func : Callable
            Function to fit to.
        name : str, optional
            Name to the function, by default the variable name of `func`
        slug : str, optional
            The shortened and simplified name of the function, by default `name`
        func_args : List[Parameter], optional
            Function arguments in the form of a list of :class:`Parameter`, by default None
        bounds : list, optional
            Bounds by which `x` may be varied in fitting, by default ``[-np.inf, np.inf, -np.inf, np.inf]``
        Raises
        ------
        ValueError
            Makes sure all parameter names in `func_args` are actual
            parameters to the function.
        """
        self.__func = func

        self.name = name if name else func.__name__
        self.slug = slug if slug else self.name
        func_argspec = np.asarray(inspect.getfullargspec(func).args[1:], dtype=str)
        if func_args is not None:
            # make sure that each value of in func_args is in func_argspec
            for p in func_args:
                if p.name not in func_argspec:
                    raise ValueError(
                        f"'{p.name}' is not a valid argument for the {self.name} model. Expected one of {func_argspec}."
                    )

            self.__func_args = {p.name: p for p in func_args}

        else:
            # we assume 1 independent variable, and give blanket bounds to each param
            self.__func_args = {
                name: Parameter(name) for name in inspect.getargspec(func).args[1:]
            }

        # xy bounds
        if bounds is not None:
            assert len(bounds) == 4, "bounds must be a list of length 4"
            self.bounds = bounds
        else:
            self.bounds = [-np.inf, np.inf, -np.inf, np.inf]  # blanket bounds

    def __call__(self, x: np.ndarray, *p, **kwargs):
        return self.func(x, *p[: len(self)], **kwargs)

    def __getitem__(self, key):
        return self.func_args[key]

    def __iter__(self):
        return iter(self.func_args)

    def __len__(self):
        return len(self.func_args)

    def __repr__(self) -> str:
        return f"<grbLC> Model({self.name})"

    @property
    def func(self) -> Callable:
        return self.__func

    @property
    def func_args(self) -> Dict[str, Parameter]:
        return self.__func_args


    @classmethod
    def W07(cls, vary_t=True):
        r"""Willingale et al. (2007) model
            This is a phenomenological model for GRB lightcurve afterglows
            popularized in the paper by Willingale et. al, (2007). [#w07]_
            Taken from his paper, it is as follows:
            $$f(t) = \left \{ \begin{array}{ll}\displaystyle{F_i \exp{\left ( \alpha_i \left( 1 - \frac{t}{T_i} \right)
            \right )} \exp{\left (- \frac{t_i}{t} \right )}} & {\rm for} \ \ t < T_i \\ ~ & ~ \\
            \displaystyle{F_i \left ( \frac{t}{T_i} \right )^{-\alpha_i} \exp{\left ( - \frac{t_i}{t} \right )}} &
            {\rm for} \ \ t \ge T_i, \\\end{array} \right .$$
            where the transition from the exponential to the power law occurs at the
            point ($T_i$, $F_i$), $\alpha$ determines the temporal decay index of the
            power law, and $t_i$ is the time of the initial rise of the lightcurve.
            As implemented, log space is used for the time (sec) and flux
            (erg cm$^{-2}$ s$^{-1}$). This means that for a light curve in which the
            afterglow plateau phase ends at 10,000 seconds corresponds to a $T_i$ of 5.
            Pre-defined priors on these parameters are:
                * $T_i$ : Uniform(1e-10, 10)
                * $F_i$ : Uniform(-20, 2)
                * $\alpha$ : Uniform(0, 5)
                * $t$ : Uniform(0, inf)
        Parameters
        ----------
        vary_t : bool, optional
            The fourth parameter to this :py:class:`Model`, `t`, often does not vary
            the lightcurve in any way and thus is sometimes set to zero. This allows
            the user to make the fitter not vary it. Otherwise, you can set the vary
            parameter to zero via ``Model[Parameter.name].vary = False``. By default True.
        Returns
        -------
        :class:`Model`
            The Willingale et al. (2007) model.
        An example lightcurve is shown below:
        .. jupyter-execute::
            import matplotlib.pyplot as plt
            import numpy as np
            import grblc
            %matplotlib inline
            w07 = grblc.Model.W07()
            x = np.linspace(2, 8, 100)
            T, F, alpha, t = 5, -12, 1.5, 1
            y = w07(x, T, F, alpha, t)
            plt.plot(x, y)
            plt.title(w07.name)
            plt.xlabel("log Time (s)")
            plt.ylabel("log Flux (erg cm$^{-2}$ s$^{-1}$)")
            plt.show()
        .. [#w07] https://arxiv.org/abs/astro-ph/0612031
        """
        return cls(
            name="Willingale 2007",
            slug="w07",
            func=_w07,
            func_args=[
                Parameter(
                    "T",
                    "log time at end of plateau (log sec)",
                    min=0,
                    max=10,
                ),
                Parameter(
                    "F",
                    "log flux at end of plateau (log erg/cm^2/s)",
                    min=-18,
                    max=-8,
                ),
                Parameter(
                    "alpha",
                    "temporal decay index of power law",
                    plot_fmt=r"$\alpha$",
                    min=0,
                    max=7,
                ),
                Parameter(
                    "t",
                    "log time at peak (log sec)",
                    min=-1e-8,
                    max=5,
                    vary=vary_t,
                )
            ]
        )


    @classmethod
    def SIMPLE_BPL(cls):
        r"""Simple broken power law model
            This is an empirical piece-wise model for GRB lightcurve afterglows.
            The function is as follows:
            $$f(t) = \left \{ \begin{array}{ll} \displaystyle{F_i \left (\frac{t}{T_i} \right)^{-\alpha_1} } & {\rm for} \ \ t < T_i \\ \displaystyle{F_i \left ( \frac{t}{T_i} \right )^{-\alpha_2} } & {\rm for} \ \ t \ge T_i, \\ \end{array} \right . $$
            where the transition from the exponential to the power law occurs at the point
            ($T_i$, $F_i$), $\alpha_1$ determines the temporal decay index of the initial
            power law, and $\alpha_2$ is the temporal decay index of the final power law.
            As implemented, log space is used for the time (sec) and flux
            (erg cm$^{-2}$ s$^{-1}$). This means that for a light curve in which the
            afterglow plateau phase ends at 10,000 seconds corresponds to a $T_i$ of 5.
            Pre-defined priors on these parameters are:
                * T : Uniform(1e-10, 10)
                * F : Uniform(-20, 2)
                * $\alpha_1$ : Uniform(-5, 5)
                * $\alpha_2$ : Uniform(-5, 5)
        Returns
        -------
        :class:`Model`
            The simple broken power law model.
        An example lightcurve is shown below:
        .. jupyter-execute::
            import matplotlib.pyplot as plt
            import numpy as np
            import grblc
            %matplotlib inline
            sbpl = grblc.Model.SIMPLE_BPL()
            x = np.linspace(2, 8, 100)
            T, F, alpha1, alpha2 = p = 5, -12, -0.1, 1.5
            y = sbpl(x, *p)
            plt.plot(x, y)
            plt.title(sbpl.name)
            plt.xlabel("log Time (s)")
            plt.ylabel("log Flux (erg cm$^{-2}$ s$^{-1}$)")
            plt.show()
        """
        return cls(
            name="Simple broken power law",
            slug="bpl",
            func=_bpl,
            func_args=[
                Parameter(
                    "T",
                    "log time at end of plateau (log sec)",
                    min=0,
                    max=10,
                ),
                Parameter(
                    "F",
                    "log flux at end of plateau  (log erg cm^-2 s^-1)",
                    min=-18,
                    max=-8,
                ),
                Parameter(
                    "alpha1",
                    "temporal decay index of initial power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_1$",
                ),
                Parameter(
                    "alpha2",
                    "temporal decay index of end power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_2$",
                )
            ]
        )


    @classmethod
    def SMOOTH_BPL(cls):
        r"""Smooth broken power law model
            This is an empirical piece-wise model for GRB lightcurve afterglows.
            The function is as follows:
            $$f(t) = F_i \left (\left (\frac{t}{T_i} \right )^{S\alpha_1} + \left (\frac{t}{T_i} \right )^{S \alpha_2} \right )^{-\frac{1}{S}}$$
            where the transition from the exponential to the power law occurs at the
            point ($T_i$, $F_i$), $\alpha_1$ determines the temporal decay index of
            the initial power law, and $\alpha_2$ is the temporal decay index of the
            final power law, and $S$ is the smoothing factor.
            As implemented, log space is used for the time (sec) and flux
            (erg cm$^{-2}$ s$^{-1}$). This means that for a light curve in which the
            afterglow plateau phase ends at 10,000 seconds corresponds to a $T_i$ of 5.
            Pre-defined priors on these parameters are::
                * $T_i$ : Uniform(1e-10, 10)
                * $F_i$ : Uniform(-20, 2)
                * $\alpha_1$ : Uniform(-5, 5)
                * $\alpha_2$ : Uniform(-5, 5)
                * $S$ : Uniform(-10, 2)
        Returns
        -------
        :class:`Model`
            The simple broken power law model.
        An example lightcurve is shown below:
        .. jupyter-execute::
            import matplotlib.pyplot as plt
            import numpy as np
            import grblc
            %matplotlib inline
            sbpl = grblc.Model.SMOOTH_BPL()
            x = np.linspace(2, 8, 100)
            T, F, alpha1, alpha2, S = p = 5, -12, -0.1, 1.5, 0.5
            y = sbpl(x, *p)
            plt.plot(x, y)
            plt.title(sbpl.name)
            plt.xlabel("log Time (s)")
            plt.ylabel("log Flux (erg cm$^{-2}$ s$^{-1}$)")
            plt.show()
        """
        return cls(
            name="Smooth broken power law",
            slug="sbpl",
            func=_sbpl,
            func_args=[
                Parameter(
                    "T",
                    "log time at end of plateau (log sec)",
                    min=0,
                    max=10,
                ),
                Parameter(
                    "F",
                    "log flux at end of plateau  (log erg cm^-2 s^-1)",
                    min=-18,
                    max=-8,
                ),
                Parameter(
                    "alpha1",
                    "temporal decay index of initial power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_1$",
                ),
                Parameter(
                    "alpha2",
                    "temporal decay index of end power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_2$",
                ),
                Parameter(
                    "S",
                    "smoothing factor in log scale",
                    min=0,
                    max=5,
                )
            ]
        )
    

    @classmethod
    def POWER_LAW(cls):
        r"""Power law model
            This is the simplest model for GRB lightcurve afterglows.
            The function is as follows:
            $$f(t) = t^(\alpha)$$
            where $\alpha$ is the temporal decay index of the power law.
            As implemented, log space is used for the time (sec) and flux
            (erg cm$^{-2}$ s$^{-1}$).
            Pre-defined priors on these parameters are:
                * T : Uniform(1e-10, 10)
                * F : Uniform(-20, 2)
                * $\alpha$ : Uniform(-5, 5)
        Returns
        -------
        :class:`Model`
            Power law model.
        An example lightcurve is shown below:
        .. jupyter-execute::
            import matplotlib.pyplot as plt
            import numpy as np
            import grblc
            %matplotlib inline
            pl = grblc.Model.POWER_LAW()
            x = np.linspace(0, 8, 100)
            T, F, alpha = p = 5, -12, 1
            y = pl(x, *p)
            plt.plot(x, y)
            plt.title(pl.name)
            plt.xlabel("log Time (s)")
            plt.ylabel("log Flux (erg cm$^{-2}$ s$^{-1}$)")
            plt.show()
        """
        return cls(
            name="Power law",
            slug="pl",
            func=_pl,
            func_args=[
                Parameter(
                    "F",
                    "log flux at the peak of prompt  (log erg cm^-2 s^-1)",
                    min=-18,
                    max=-3,
                ),
                Parameter(
                    "alpha",
                    "temporal decay index of the power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha$",
                )
            ]
        )


    @classmethod
    def W07_SIMPLE_BPL(cls, vary_t=True):
        return cls(
            name="Willingale 2007 + Simple broken power law",
            slug="w07+bpl",
            func=_w07_bpl,
            func_args=[
                Parameter(
                    "F_p",
                    "log flux at peak of prompt (log erg/cm^2/s)",
                    min=-18,
                    max=-2,
                    plot_fmt=r"$F_p$",
                ),
                Parameter(
                    "alpha_p",
                    "temporal decay index of power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_p$",
                ),
                Parameter(
                    "alpha_1",
                    "temporal decay index of initial power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_(a1)$",
                ),
                Parameter(
                    "alpha_2",
                    "temporal decay index of end power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_(a2)$",
                ),
                Parameter(
                    "t0",
                    "log time at prompt (log sec)",
                    min=0,
                    max=10,
                    plot_fmt=r"$t_0$",
                ),
                Parameter(
                    "t1",
                    "log time at begininng of plateau (log sec)",
                    min=0,
                    max=10,
                    plot_fmt=r"$t_1$",
                ),
                Parameter(
                    "t2",
                    "log time at end of plateau (log sec)",
                    min=0,
                    max=10,
                ),
                Parameter(
                    "t",
                    "log time at peak (log sec)",
                    min=-1e-8,
                    max=5,
                    vary=vary_t,
                )
            ]
        )


    @classmethod
    def W07_SMOOTH_BPL(cls, vary_t=True):
        return cls(
            name="Willingale 2007 + Smooth broken power law",
            slug="w07+sbpl",
            func=_w07_sbpl,
            func_args=[
                Parameter(
                    "T_p",
                    "log time at peak of prompt (log sec)",
                    min=0,
                    max=10,
                    plot_fmt=r"$T_p$",
                ),
                Parameter(
                    "F_p",
                    "log flux at peak of prompt (log erg/cm^2/s)",
                    min=-18,
                    max=-8,
                    plot_fmt=r"F_p$",
                ),
                Parameter(
                    "alpha_p",
                    "temporal decay index of power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_p$",
                ),
                Parameter(
                    "t_p",
                    "log time at peak (log sec)",
                    min=-1,
                    max=10,
                    vary=vary_t,
                    plot_fmt=r"$t_p$",
                ),
                Parameter(
                    "T_a",
                    "log time at end of plateau (log sec)",
                    min=0,
                    max=10,
                    plot_fmt=r"$T_a$",
                ),
                Parameter(
                    "F_a",
                    "log flux at end of plateau  (log erg cm^-2 s^-1)",
                    min=-18,
                    max=-8,
                    plot_fmt=r"$F_a$",
                ),
                Parameter(
                    "alpha_a1",
                    "temporal decay index of initial power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_(a1)$",
                ),
                Parameter(
                    "alpha_a2",
                    "temporal decay index of end power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_(a2)$",
                ),
                Parameter(
                    "S_a",
                    "smoothing factor in log scale",
                    min=0,
                    max=5,
                    plot_fmt=r"$S_a$",
                ),
                Parameter(
                    "tt",
                    "beginning of plateau",
                    min=-1e-8,
                    max=10,
                )
            ]
        )


    @classmethod
    def W07_POWER_LAW(cls, vary_t=True):
        return cls(
            name="Willingale 2007 + Simple broken power law",
            slug="w07+pl",
            func=_w07_pl,
            func_args=[
                Parameter(
                    "F_p",
                    "log flux at peak of prompt (log erg/cm^2/s)",
                    min=-18,
                    max=-2,
                    plot_fmt=r"$F_p$",
                ),
                Parameter(
                    "alpha_p",
                    "temporal decay index of power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_p$",
                ),
                Parameter(
                    "t",
                    "log time at peak (log sec)",
                    min=-1e-8,
                    max=5,
                    vary=vary_t,
                    plot_fmt=r"$t$",
                ),
                Parameter(
                    "alpha_1",
                    "temporal decay index of initial power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_(a1)$",
                ),
                Parameter(
                    "t0",
                    "beginning of plateau",
                    min=0,
                    max=10,
                ),
                Parameter(
                    "t1",
                    "beginning of plateau",
                    min=0,
                    max=5,
                )
            ]
        )


    @classmethod
    def POWER_LAW_W07(cls, vary_t=True):
        return cls(
            name="Power law + Williingale 2007",
            slug="pl+w07",
            func=_pl_w07,
            func_args=[
                Parameter(
                    "F_p",
                    "log flux at the peak of prompt  (log erg cm^-2 s^-1)",
                    min=-18,
                    max=-8,
                    plot_fmt=r"F_p$",
                ),
                Parameter(
                    "alpha_p",
                    "temporal decay index of the power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha$",
                ),
                Parameter(
                    "T_a",
                    "log time at peak of prompt (log sec)",
                    min=0,
                    max=10,
                    plot_fmt=r"$T_a$",
                ),
                Parameter(
                    "F_a",
                    "log flux at peak of prompt (log erg/cm^2/s)",
                    min=-18,
                    max=-8,
                    plot_fmt=r"$F_a$",
                ),
                Parameter(
                    "alpha_a",
                    "temporal decay index of power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_a$",
                ),
                Parameter(
                    "tt",
                    "log time at peak (log sec)",
                    min=-1e-8,
                    max=10,
                    vary=vary_t,
                    plot_fmt=r"$tt$",
                )
            ]
        )


    @classmethod
    def POWER_LAW_SIMPLE_BPL(cls):
        return cls(
            name="Power law + Simple broken power law",
            slug="pl+bpl",
            func=_pl_bpl,
            func_args=[
                Parameter(
                    "F_p",
                    "log flux at peak of prompt (log erg/cm^2/s)",
                    min=-18,
                    max=-2,
                    plot_fmt=r"$F_p$",
                ),
                Parameter(
                    "alpha_p",
                    "temporal decay index of power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_p$",
                ),
                Parameter(
                    "alpha_a1",
                    "temporal decay index of initial power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_(a1)$",
                ),
                Parameter(
                    "alpha_a2",
                    "temporal decay index of end power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_(a2)$",
                ),
                Parameter(
                    "t0",
                    "log time at end of prompt (log sec)",
                    min=0,
                    max=10,
                    plot_fmt=r"$t0$",
                ),
                Parameter(
                    "t1",
                    "log time at end of plateau (log sec)",
                    min=0,
                    max=10,
                    plot_fmt=r"$t1$",
                ),
            ]
        )


    @classmethod
    def POWER_LAW_SMOOTH_BPL(cls):
        return cls(
            name="Power law + Smooth broken power law",
            slug="pl+sbpl",
            func=_pl_sbpl,
            func_args=[
                Parameter(
                    "F_p",
                    "log flux at peak of prompt (log erg/cm^2/s)",
                    min=-18,
                    max=-2,
                    plot_fmt=r"$F_p$",
                ),
                Parameter(
                    "alpha_p",
                    "temporal decay index of power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_p$",
                ),
                Parameter(
                    "t1",
                    "log time at end of plateau (log sec)",
                    min=0,
                    max=10,
                    plot_fmt=r"$T_a$",
                ),
                Parameter(
                    "F_a",
                    "log flux at end of plateau  (log erg cm^-2 s^-1)",
                    min=-18,
                    max=-2,
                    plot_fmt=r"$F_a$",
                ),
                Parameter(
                    "alpha_a1",
                    "temporal decay index of initial power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_(a1)$",
                ),
                Parameter(
                    "alpha_a2",
                    "temporal decay index of end power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_(a2)$",
                ),
                Parameter(
                    "S_a",
                    "smoothing factor in log scale",
                    min=0,
                    max=5,
                    plot_fmt=r"$S_a$",
                ),
                Parameter(
                    "t0",
                    "beginning of plateau",
                    min=0,
                    max=10,
                )
            ]
        )

    
    @classmethod
    def DOUBLE_POWER_LAW(cls):
        return cls(
            name="Double power law",
            slug="double_pl",
            func=_double_pl,
            func_args=[
                Parameter(
                    "F_p",
                    "log flux at peak of prompt (log erg/cm^2/s)",
                    min=-18,
                    max=-8,
                    plot_fmt=r"$F_p$",
                ),
                Parameter(
                    "alpha_p",
                    "temporal decay index of power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_p$",
                ),
                Parameter(
                    "F_a",
                    "log flux at end of plateau  (log erg cm^-2 s^-1)",
                    min=-18,
                    max=-8,
                    plot_fmt=r"$F_a$",
                ),
                Parameter(
                    "alpha_a",
                    "temporal decay index of initial power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_a$",
                ),
                Parameter(
                    "tt",
                    "beginning of plateau",
                    min=-1,
                    max=10,
                )
            ]
        )


    @classmethod
    def W07_SIMPLE_BPL_POWER_LAW(cls, vary_t=True):
        return cls(
            name="Willingale 2007 + Simple broken power law + power law",
            slug="w07+bpl+pl",
            func=_w07_bpl_pl,
            func_args=[
                Parameter(
                    "F_p",
                    "log flux at peak of prompt (log erg/cm^2/s)",
                    min=-18,
                    max=-2,
                    plot_fmt=r"$F_p$",
                ),
                Parameter(
                    "alpha_p",
                    "temporal decay index of power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_p$",
                ),
                Parameter(
                    "alpha_1",
                    "temporal decay index of initial power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_(a1)$",
                ),
                Parameter(
                    "alpha_2",
                    "temporal decay index of end power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_(a2)$",
                ),
                Parameter(
                    "alpha_3",
                    "temporal decay index of end power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_(a3)$",
                ),
                Parameter(
                    "t0",
                    "log time at prompt (log sec)",
                    min=0,
                    max=10,
                    plot_fmt=r"$t_0$",
                ),
                Parameter(
                    "t1",
                    "log time at begininng of plateau (log sec)",
                    min=0,
                    max=10,
                    plot_fmt=r"$t_1$",
                ),
                Parameter(
                    "t2",
                    "log time at end of plateau (log sec)",
                    min=0,
                    max=10,
                ),
                Parameter(
                    "t3",
                    "log time at end of plateau (log sec)",
                    min=0,
                    max=10,
                ),
                Parameter(
                    "t",
                    "log time at peak (log sec)",
                    min=-1e-8,
                    max=5,
                    vary=vary_t
                )
            ]
        )


    @classmethod
    def POWER_LAW_SIMPLE_BPL_POWER_LAW(cls):
        return cls(
            name="Power law + Simple broken power law",
            slug="pl+bpl+pl",
            func=_pl_bpl_pl,
            func_args=[
                Parameter(
                    "F_p",
                    "log flux at peak of prompt (log erg/cm^2/s)",
                    min=-18,
                    max=-2,
                    plot_fmt=r"$F_p$",
                ),
                Parameter(
                    "alpha_p",
                    "temporal decay index of power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_p$",
                ),
                Parameter(
                    "alpha_a1",
                    "temporal decay index of initial power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_(a1)$",
                ),
                Parameter(
                    "alpha_a2",
                    "temporal decay index of middle power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_(a2)$",
                ),
                Parameter(
                    "alpha_a3",
                    "temporal decay index of end power law",
                    min=0,
                    max=7,
                    plot_fmt=r"$\alpha_(a3)$",
                ),
                Parameter(
                    "t0",
                    "end of prompt",
                    min=0,
                    max=10,
                ),
                Parameter(
                    "t1",
                    "begining of plateau",
                    min=0,
                    max=10,
                ),
                Parameter(
                    "t2",
                    "end of plateau",
                    min=0,
                    max=10,
                )
            ]
        )



class Models:
    r"""Collection of models to fit together."""

    def __init__(self, models:List[Model]):
        assert len(models) > 0, "Must have at least one model."
        assert all(isinstance(m, Model) for m in models), "All elements must be models."

        self.models = [deepcopy(m) for m in models]

        arg_names, model_idx = np.concatenate([[list(model.func_args.keys()), [idx]*len(model)] for idx,model in enumerate(self.models)], axis=1)
        arg_names = list(arg_names)
        model_idx = list(map(int, model_idx))
        # make sure no collisions in model parameters
        if len(set(arg_names)) != len(arg_names):
            for idx, (name, model_id) in enumerate(zip(arg_names, model_idx)):
                if arg_names.count(name) > 1:
                    # warnings.warn(f"{name} is used in multiple models. Renaming parameters in ascending order.")
                    new_name = name + str(arg_names[:idx].count(name) + 1)

                    self.models[model_id].func_args[new_name] = self.models[model_id].func_args.pop(name)
                    self.models[model_id].func_args[new_name].name = new_name

        merge_dict = lambda d1, d2: {**d1, **d2}
        self.__func_args = reduce(merge_dict, [m.func_args for m in self.models])
        xmin, ymin = [min([models[i].bounds[j] for i in range(len(models))]) for j in [0, 2]]
        xmax, ymax = [max([models[i].bounds[j] for i in range(len(models))]) for j in [1, 3]]
        self.bounds = [xmin, xmax, ymin, ymax]

        self.name = " + ".join(model.name for model in self.models)
        self.slug = "+".join([model.slug for model in self.models])


    def __call__(self, x: np.ndarray, *p, **kwargs):
        targs = 0
        ans = np.zeros_like(x, dtype=float)
        p = np.ravel(p)
        for model in self.models:
            nargs = len(model)
            ans += model(x, *p[targs:targs+nargs], **kwargs)
            targs += nargs

        return ans

    __func = __call__

    @property
    def func_args(self) -> Dict[str, Parameter]:
        return self.__func_args

    def __getitem__(self, key):
        return self.func_args[key]

    def __iter__(self):
        return iter(self.func_args)

    def __len__(self):
        return len(self.func_args)

    def __repr__(self) -> str:
        return f"<grbLC> Models({self.name})"

    @property
    def func(self) -> Callable:
        return self.__func

    @property
    def func_args(self) -> Dict[str, Parameter]:
        return self.__func_args