import pathlib

import pandas
import numpy
from scipy.optimize import curve_fit
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt


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


def delta_z_correction(df):
    Z_corr_factor = 2.76e-4 

    df.loc[:,'evt_energy_z'] = df.evt_energy/(1. - Z_corr_factor*(df.evt_z_max-df.evt_z_min))

    return df

def basic_cuts(df):
    df = df.query("evt_r_max < 180")
    df = df.query("evt_z_min >= 10")
    df = df.query("evt_z_max <= 510")
    df = df.query("evt_ntrks == 1")

    return df
    # fiducial (R < 180, 10 <= Z <= 510) and energy cut (1.2 < E < 2.0), and pass the 1 track cut (n_trks==1)

def measure_fits(fit_values):

    a0, tau, amp, mu, sigma = fit_values

    N = 500
    integral_start = 1.6478
    integral_end   = 1.6918
    integral_step  = (integral_end - integral_start ) / N

    points = numpy.linspace(start=integral_start, stop=integral_end, num=N)

    gaussian    = gauss(points, amp, mu, sigma)
    exponential = exp(points, a0, tau)

    gauss_integral = numpy.sum(gaussian*integral_step)
    exp_integral   = numpy.sum(exponential*integral_step)

    return gauss_integral, exp_integral



def create_fom(file_name, plot_folder):

    plot_folder = pathlib.Path(plot_folder)

    plot_folder.mkdir(exist_ok=True)

    # Open the dataframe:
    df = pandas.read_hdf(file_name)


    # Clear NaN energy entries:
    df = df.dropna()

    df = delta_z_correction(df)

    df = basic_cuts(df)

    energy_bins = numpy.arange(1.4, 1.8, 0.005)
    bin_centers = 0.5*(energy_bins[1:] + energy_bins[:-1])
    bin_widths  = energy_bins[1:] - energy_bins[:-1]

    plotting_bins    = numpy.arange(1.4, 1.8, 0.0005)
    plotting_centers = 0.5*(plotting_bins[1:] + plotting_bins[:-1])
    plotting_widths  = plotting_bins[1:] - plotting_bins[:-1]

    data_events, _ = numpy.histogram(df['evt_energy_z'], energy_bins)

    # a0, tau, amp, mu, sigma
    seed = (100.0, -1.0, 400.0, 1.65, 0.05)

    # First, fit the raw data with gauss + exp:
    raw_fitf, raw_values, raw_errors = fit(expgauss, bin_centers, data_events, seed)
    raw_fitf, raw_values, raw_errors = fit(expgauss, bin_centers, data_events, raw_values)

    # Compute the baseline selected signal, background:
    n_signal_baseline, n_background_baseline = measure_fits(raw_values)


    cut_values = []
    fom_values = []
    previous_values = raw_values

    for cut in numpy.linspace(0.0, 0.999, num = 200):    
        print(cut)
        cut_values.append(cut)

        selected_events = df.query(f"score_signal >= {cut}")

        sig_events, _ = numpy.histogram(df.query(f"score_signal >= {cut}")['evt_energy_z'], energy_bins)
        # bkg_events, _ = numpy.histogram(df.query(f"score_signal <  {cut}")['evt_energy_z'], energy_bins)

        sig_fit, sig_values, sig_errors = fit(expgauss, bin_centers, sig_events, previous_values, maxfev=5000)

        n_signal_selected, n_background_selected = measure_fits(sig_values)


        full_values = sig_fit(plotting_centers)

        FOM = (n_signal_selected / n_signal_baseline) / numpy.sqrt(n_background_selected / n_background_baseline)

        fom_values.append(FOM)

        previous_values = sig_values
        # Save the plot:

        a0, tau, amp, mu, sigma = sig_values
        fit_string = f"${a0:.1f}e^{{{tau:.1f} x}} + {amp:.1f} e^{{\\frac{{-(x - {mu:.1f})^2}}{{{sigma:.1}}} }}$"

        plt.bar(bin_centers, sig_events, width=bin_widths, alpha=0.5, zorder=4, label="Selected Events")
        plt.plot(plotting_centers, full_values, zorder=4, label=fit_string)
        plt.title(f"Cut @ {cut:.3f}, " + r"$f(x) = A e^{-\lambda x } + B e^{\frac{-(x - \mu)^2}{\sigma}}$")
        plt.legend()
        plt.grid(zorder=0)
        plt.ylabel("Events")
        plt.xlabel("Energy [MeV]")
        plt.savefig(plot_folder / pathlib.Path(f"DataCategorization_cut{cut:.2f}.pdf"))
        plt.close()
        # plt.show()

    a0, tau, amp, mu, sigma = sig_values
    fit_string = f"${a0:.1f}e^{{{tau:.1f} x}} + {amp:.1f} e^{{\\frac{{-(x - {mu:.1f})^2}}{{{sigma:.1}}} }}$"

    plt.plot(cut_values, fom_values, zorder=4, label="FOM")
    plt.legend()
    plt.grid(zorder=0)
    plt.ylabel("Figure of Merit [$\epsilon_{Sig.} / \sqrt{\epsilon_{Bkg.}}$]")
    plt.xlabel("Sparse NN Cut [MeV]")
    plt.savefig(plot_folder / pathlib.Path("FOM.pdf"))
    # plt.show()






if __name__ == "__main__":
    create_fom("/Users/corey.adams/data/NEXT/mmkekic_second_production/final_inference.h5", plot_folder = "./plots/")