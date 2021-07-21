import numpy as np


# The famous Willingale et. al 2007 model
def w07(x, T, F, alpha, t):
    before = lambda x: np.log10(10 ** F * np.exp(alpha - alpha * (10 ** x / 10 ** T)) * np.exp(-t / 10 ** x))
    after = lambda x: np.log10(10 ** F * (10 ** x / 10 ** T) ** (-alpha) * np.exp(-t / 10 ** x))
    vals = np.piecewise(x, [10 ** x < 10 ** T, 10 ** x >= 10 ** T], [before, after])
    return vals


def chisq(x, y, yerr, model, *p):
    """
    Calculate chisq for a given proposed solution

    chisq = Sum_i (Y_i - W07_fit(X_i, F, T, alpha, t))^2 / Y_err_i^2,

    """
    chisq = np.sum(np.square(y - model(x, *p)) / yerr ** 2)

    return chisq
