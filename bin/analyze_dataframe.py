import pathlib
import argparse

import pandas
import numpy
from scipy.optimize import curve_fit
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt


def gauss(x, amp, mu, sigma):
    return amp/(2*numpy.pi)**.5/sigma * numpy.exp(-0.5*(x-mu)**2./sigma**2.)

def exp(x, a0, tau):
    return a0 * numpy.exp(x*tau) 

def expgauss(x, a0, tau, amp, mu, sigma):
    return a0 * numpy.exp(x*tau) + amp/(2*numpy.pi)**.5/sigma * numpy.exp(-0.5*(x-mu)**2./sigma**2.)

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

def measure_fits(fit_values, is_mc):

    a0, tau, amp, mu, sigma = fit_values

    N = 500
    if is_mc:
        integral_start = 1.57
        integral_end   = 1.63
    else:
        integral_start = 1.6478
        integral_end   = 1.6918
    integral_step  = (integral_end - integral_start ) / N

    points = numpy.linspace(start=integral_start, stop=integral_end, num=N)

    gaussian    = gauss(points, amp, mu, sigma)
    exponential = exp(points, a0, tau)

    gauss_integral = numpy.sum(gaussian*integral_step)
    exp_integral   = numpy.sum(exponential*integral_step)

    return gauss_integral, exp_integral



def create_fom(file_name, plot_folder, is_mc):

    plot_folder = pathlib.Path(plot_folder)

    plot_folder.mkdir(exist_ok=True)

    # Open the dataframe:
    df = pandas.read_hdf(file_name)


    # Clear NaN energy entries:
    df = df.dropna()

    if not is_mc:
        df = delta_z_correction(df)
        energy_key = 'evt_energy_z'
    else:
        energy_key = 'evt_energy'

    df = basic_cuts(df)

    energy_bins = numpy.arange(1.4, 1.8, 0.005)
    bin_centers = 0.5*(energy_bins[1:] + energy_bins[:-1])
    bin_widths  = energy_bins[1:] - energy_bins[:-1]

    plotting_bins    = numpy.arange(1.4, 1.8, 0.0005)
    plotting_centers = 0.5*(plotting_bins[1:] + plotting_bins[:-1])
    plotting_widths  = plotting_bins[1:] - plotting_bins[:-1]

    data_events, _ = numpy.histogram(df[energy_key], energy_bins)

    # a0, tau, amp, mu, sigma
    seed = (100.0, -1.0, 400.0, 1.65, 0.05)

    # First, fit the raw data with gauss + exp:
    raw_fitf, raw_values, raw_errors = fit(expgauss, bin_centers, data_events, seed)
    raw_fitf, raw_values, raw_errors = fit(expgauss, bin_centers, data_events, raw_values)

    # Compute the baseline selected signal, background:
    n_signal_baseline, n_background_baseline = measure_fits(raw_values, is_mc)
    
    plot_peak = numpy.max(raw_fitf(plotting_centers))
    print(plot_peak)

    cut_values = []
    fom_values = []
    eff_sig    = []
    eff_bkg    = []

    if is_mc:
        n_sig_true = len(df.query("label==1"))
        n_bkg_true = len(df.query("label==0"))

        true_eff_sig = []
        true_eff_bkg = []

    previous_values = raw_values

    for i, cut in enumerate(numpy.linspace(0.0, 0.99, num = 50)):
        cut_values.append(cut)

        selected_events = df.query(f"score_signal >= {cut}")

        sig_events, _ = numpy.histogram(df.query(f"score_signal >= {cut}")[energy_key], energy_bins)
        # bkg_events, _ = numpy.histogram(df.query(f"score_signal <  {cut}")[energy_key], energy_bins)

        sig_fit, sig_values, sig_errors = fit(expgauss, bin_centers, sig_events, previous_values, maxfev=5000)

        n_signal_selected, n_background_selected = measure_fits(sig_values, is_mc)


        full_values = sig_fit(plotting_centers)


        FOM = (n_signal_selected / n_signal_baseline) / numpy.sqrt(n_background_selected / n_background_baseline)
        fom_values.append(FOM)

        eff_sig.append(n_signal_selected     / n_signal_baseline)
        eff_bkg.append(n_background_selected / n_background_baseline)




        previous_values = sig_values
        # Save the plot:

        a0, tau, amp, mu, sigma = sig_values
        fit_string = f"${a0:.1f}e^{{{tau:.1f} x}} + {amp:.1f} e^{{\\frac{{-(x - {mu:.1f})^2}}{{{sigma:.1}}} }}$"

        if not is_mc:
            plt.bar(bin_centers, sig_events, width=bin_widths, alpha=0.5, zorder=4, label="Selected Events")
        else:
            selection = df.query(f"score_signal >= {cut}")
            sig_events, _ = numpy.histogram(selection.query("label==1")[energy_key], energy_bins)
            bkg_events, _ = numpy.histogram(selection.query("label==0")[energy_key], energy_bins)

            plt.bar(bin_centers, bkg_events, width=bin_widths, alpha=0.5, zorder=4, label="Selected Background")
            plt.bar(bin_centers, sig_events, width=bin_widths, bottom=bkg_events, alpha=0.5, zorder=4, label="Selected Signal")

            this_sig_true = len(selection.query("label==1"))
            this_bkg_true = len(selection.query("label==0"))
            true_eff_sig.append(this_sig_true / n_sig_true)
            true_eff_bkg.append(this_bkg_true / n_bkg_true)
            

        plt.plot(plotting_centers, full_values, zorder=4, label=fit_string)
        plt.title(f"Cut @ {cut:.3f}, " + r"$f(x) = A e^{-\lambda x } + B e^{\frac{-(x - \mu)^2}{\sigma}}$")
        plt.legend()
        plt.grid(zorder=0)
        plt.ylabel("Events")
        plt.xlabel("Energy [MeV]")
        plt.tight_layout()
        plt.ylim(0,1.5*plot_peak)

        if is_mc:
            name = "Sim"
        else:
            name = "Data"

        plt.savefig(plot_folder / pathlib.Path(name + f"Categorization_cut{cut:.2f}.png"))
        plt.close()
        # plt.show()

    a0, tau, amp, mu, sigma = sig_values
    fit_string = f"${a0:.1f}e^{{{tau:.1f} x}} + {amp:.1f} e^{{\\frac{{-(x - {mu:.1f})^2}}{{{sigma:.1}}} }}$"

    plt.plot(cut_values, fom_values, zorder=4, label="FOM")
    plt.legend()
    plt.grid(zorder=0)
    plt.ylabel("Figure of Merit [$\epsilon_{Sig.} / \sqrt{\epsilon_{Bkg.}}$]")
    plt.xlabel("Sparse NN Cut [MeV]")
    plt.tight_layout()
    plt.savefig(plot_folder / pathlib.Path("FOM.png"))
    plt.close()
    # plt.show()


    d = {
        "cut_values" : cut_values,
        "fom_values" : fom_values,
        "eff_sig" : eff_sig,
        "eff_bkg" : eff_bkg,
    }
    if is_mc:
        d["true_eff_sig"] = true_eff_sig
        d["true_eff_bkg"] = true_eff_bkg

    return d


def analyze_eff(sim_d, plot_folder):

    # Here, we want to plot the signal and background efficiency as a function of cut.


    plt.plot(sim_d['cut_values'], sim_d['eff_sig'], color='blue', ms=10,
            ls="none", marker="+", zorder=4, label="Fit Sig. Eff.")
    plt.plot(sim_d['cut_values'], sim_d['true_eff_sig'], color='blue',
            zorder=4, label="True Sig. Eff.")
    plt.plot(sim_d['cut_values'], sim_d['eff_bkg'], color='red', ms=10,
            ls="none", marker="+", zorder=4, label="Fit Bkg. Eff.")
    plt.plot(sim_d['cut_values'], sim_d['true_eff_bkg'], color='red',
            zorder=4, label="True Bkg. Eff.")

    plt.legend()
    plt.grid(zorder=0)
    plt.xlabel("Neural Network Cut")
    plt.ylabel("Efficiency")
    plt.tight_layout()
    plt.savefig(plot_folder / pathlib.Path("eff.png"))
    plt.close()

def plot_roc(sim_d, data_d, plot_folder):
    # This creates a 2D plot of signal vs bkg efficiency

    plt.plot(1.0 - numpy.asarray(sim_d['eff_bkg']), sim_d['eff_sig'], label="Sim., Fit")
    plt.plot(1.0 - numpy.asarray(sim_d['true_eff_bkg']), sim_d['true_eff_sig'], label="Sim., True")
    plt.plot(1.0 - numpy.asarray(data_d['eff_bkg']), data_d['eff_sig'], label="Data, Fit")

    plt.legend()
    plt.grid(zorder=0)
    plt.xlabel("Background Efficiency")
    plt.ylabel("Signal Efficiency")
    plt.tight_layout()
    plt.savefig(plot_folder / pathlib.Path("ROC.png"))
    plt.close()

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description     = 'Run Network Training',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--simulation-file",
        type    = str,
        default = "none",
        help    = "Merged simulation results")

    parser.add_argument("--data-file",
        type    = str,
        default = "none",
        help    = "Merged data results")

    parser.add_argument("--plot-location",
        type    = pathlib.Path,
        default = pathlib.Path("./plots/"),
        help    = "Location for plots")



    args = parser.parse_args()

    args.plot_location.mkdir(exist_ok=True)


    if args.simulation_file != "none":
        sim_d  = create_fom(
            file_name   = args.simulation_file, 
            plot_folder = args.plot_location / pathlib.Path("sim"), 
            is_mc       = True)
    
        analyze_eff(sim_d, plot_folder = args.plot_location)


    if args.data_file != "none":

        data_d = create_fom(
            file_name   = args.data_file, 
            plot_folder = args.plot_location / pathlib.Path("data"), 
            is_mc       = False)


    if args.simulation_file != "none" and args.data_file != "none":

        plot_roc(sim_d, data_d, plot_folder = args.plot_location)





