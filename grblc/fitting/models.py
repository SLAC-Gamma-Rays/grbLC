import numpy as np


# The famous Willingale et. al 2007 model
def w07(x, T, F, alpha, t):

    linT, linF, lint = np.power(10, [T, F, t], dtype=object)
    linX = np.power(10, x)
    before = lambda x: np.log10(linF * np.exp(alpha - alpha * (x / linT)) * np.exp(-lint / x))
    after = lambda x: np.log10(linF * (x / linT) ** (-alpha) * np.exp(-lint / x))
    vals = np.piecewise(linX, [linX < linT, linX >= linT], [before, after])
    return vals


def sharp_bpl(x, T, F, alpha1, alpha2):

    linT, linF = np.power(10, [T, F])
    linX = np.power(10, x)

    before = lambda x: linF * (x / linT) ** (-alpha1)
    after = lambda x: linF * (x / linT) ** (-alpha2)
    vals = np.piecewise(linX, [linX < linT, linX >= linT], [before, after])

    return np.log10(vals)


def smooth_bpl(x, T, F, alpha1, alpha2, S):

    linT, linF = np.power(10, [T, F])
    linX = np.power(10, x)
    return np.log10(linF * ((linX / linT) ** (S * alpha1) + (linX / linT) ** (S * alpha2)) ** (-1 / S))


def chisq(x, y, sigma, model, *p):
    """
    Calculate chisq for a given proposed solution

    chisq = Sum_i (Y_i - model(X_i, F, T, alpha, t))^2 / sigma_i^2,

    """

    x = np.asarray(x)
    y = np.asarray(y)
    sigma = np.asarray(sigma)
    r = y - model(x, *p)
    return np.sum((r / sigma) ** 2)


def probability(reduced_chisq, nu, tt=0):
    import scipy.integrate as si
    from scipy.special import gamma

    integrand = lambda x: (2 ** (-nu / 2) / gamma(nu / 2)) * np.exp(-x / 2) * x ** (-1 + nu / 2)

    y, yerr = si.quad(integrand, reduced_chisq * nu, np.inf)

    return y
