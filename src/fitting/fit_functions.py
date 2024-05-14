from scipy.optimize import curve_fit
import numpy as np
import numpy

# from src.fitting.fit_functions import expgauss
from scipy.optimize import curve_fit
from scipy.optimize import curve_fit
import numpy as np
import numpy

# from src.fitting.fit_functions import expgauss
from scipy.optimize import curve_fit


def gauss(x, exp_norm, exp_tau, gauss_norm, gauss_mu, gauss_sigma):
    return gauss_norm/((gauss_sigma * 2*numpy.pi)**.5) * numpy.exp(-0.5*(x-gauss_mu)**2./gauss_sigma**2.)

def exp(x, exp_norm, exp_tau, gauss_norm, gauss_mu, gauss_sigma):
    return exp_norm * numpy.exp(- x/exp_tau) 

def expgauss(x, exp_norm, exp_tau, gauss_norm, gauss_mu, gauss_sigma):
    return exp(x, exp_norm, exp_tau, gauss_norm, gauss_mu, gauss_sigma) \
        + gauss(x, exp_norm, exp_tau, gauss_norm, gauss_mu, gauss_sigma)
    # return exp_norm * numpy.exp(x/exp_tau) + gauss_norm/(2*numpy.pi)**.5/gauss_sigma * numpy.exp(-0.5*(x-gauss_mu)**2./gauss_sigma**2.)

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


def fit(func, x, y, seed : dict = {}, fit_range=None, **kwargs):
    if fit_range is not None:
        sel = (fit_range[0] <= x) & (x < fit_range[1])
        x, y = x[sel], y[sel]
        
    # Coerce the seed (which is a dict) into a tuple of properly ordered args
    seed = (
        seed["exp_norm"],
        seed["exp_tau"],
        seed["gauss_norm"],
        seed["gauss_mu"],
        seed["gauss_sigma"],
    )

    print("Seed: ", seed)

    vals, cov = curve_fit(func, x, y, seed, **kwargs)
    
    
    # Repack the vals into a dict to make them easier to comprehend:

    vals = {
        "exp_norm"    : vals[0],
        "exp_tau"     : vals[1],
        "gauss_norm"  : vals[2],
        "gauss_mu"    : vals[3],
        "gauss_sigma" : vals[4],
    }

    fitf = lambda x: func(x, **vals)


    return fitf, vals, get_errors(cov)

def fit_energies_with_mask(energies, fit_params, mask = None, override_seed = None):
    """Perform the fit, with optional mask on energies.

    Returns the fit function 

    Args:
        energies (_type_): _description_
        fit_params (_type_): _description_
        mask (_type_, optional): _description_. Defaults to None.
    """

    # First, bin the energies:
    bins = numpy.arange(fit_params.min_e, fit_params.max_e, fit_params.bin_e)
    
    if mask is not None:
        energies = energies[mask]

    

    binned_energies, edges = numpy.histogram(energies, bins=bins)
    bin_centers = 0.5*(edges[1:] + edges[:-1])
    bin_widths = edges[1:] - edges[:-1]


    fit_seed = fit_params.seed if override_seed is None else override_seed

    # Do the fit:
    fit_f, vals, errors = fit(expgauss, bin_centers, binned_energies, seed = fit_seed)

    return fit_f, vals, errors