import numpy as np


# The famous Willingale et. al 2007 model
def w07(x, T, F, alpha, t):
    before = lambda x: np.log10(
        np.power(10, F) * np.exp(alpha - alpha * (np.power(10, x) / np.power(10, T))) * np.exp(-t / np.power(10, x))
    )
    after = lambda x: np.log10(
        np.power(10, F) * np.power((np.power(10, x) / np.power(10, T)), (-alpha)) * np.exp(-t / np.power(10, x))
    )
    vals = np.piecewise(x, [np.power(10, x) < np.power(10, T), np.power(10, x) >= np.power(10, T)], [before, after])
    return vals


def chisq(x, y, yerr, model, tt=0, *p):
    """
    Calculate chisq for a given proposed solution

    chisq = Sum_i (Y_i - W07_fit(X_i, F, T, alpha, t))^2 / Y_err_i^2,

    """

    x = np.asarray(x)
    mask = x >= tt
    y = np.asarray(y)
    yerr = np.asarray(yerr)
    return np.sum(np.square(y[mask] - model(x[mask], *p)) / np.power(yerr[mask], 2))


def reduced_chisq(x, y, yerr, model, p, tt=0, correction=0):
    """reduced_chisq
    chi squared per degree of freedom
    """
    return chisq(x, y, yerr, model, tt, *p) / dof(x, p, tt, correction)


def dof(x, p, tt=0, correction=0):
    x = np.asarray(x)
    return len(x[x >= tt]) - len(p) + correction


def probability(x, y, yerr, model, p, tt=0, correction=0):
    import scipy.integrate as si
    from scipy.special import gamma

    reducedChiSquared = reduced_chisq(x, y, yerr, model, p, tt, correction)
    nu = dof(x, p, tt, correction)

    integrand = lambda x: (2 ** (-nu / 2) / gamma(nu / 2)) * np.exp(-x / 2) * x ** (-1 + nu / 2)

    y, yerr = si.quad(integrand, reducedChiSquared * nu, np.inf)

    return y
