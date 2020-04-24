from scipy.optimize import curve_fit
import numpy as np
import numpy

def gauss(x, amp, mu, sigma):
    return amp/(2*numpy.pi)**.5/sigma * numpy.exp(-0.5*(x-mu)**2./sigma**2.)

def exp(x, a0, tau):
    return a0 * numpy.exp(x/tau) 

def expgauss(x, a0, tau, amp, mu, sigma):
    return a0 * numpy.exp(x/tau) + amp/(2*numpy.pi)**.5/sigma * numpy.exp(-0.5*(x-mu)**2./sigma**2.)

def get_errors(cov):
    """
    Find errors from covariance matrix
    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix of the fit parameters.
    Returns
    -------
    err : 1-dim np.ndarray
        Errors asociated to the fit parameters.
    """
    return numpy.sqrt(numpy.diag(cov))

def fit(func, x, y, seed=(), fit_range=None, **kwargs):
    if fit_range is not None:
        sel = (fit_range[0] <= x) & (x < fit_range[1])
        x, y = x[sel], y[sel]
        
    vals, cov = curve_fit(func, x, y, seed, **kwargs)
    
    fitf = lambda x: func(x, *vals)
    
    return fitf, vals, get_errors(cov)