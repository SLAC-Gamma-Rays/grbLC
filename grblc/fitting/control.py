import pandas as pd
import pandas.errors
import numpy as np
import os, re
from fitting.fitting import fit_w07, plot_w07_fit, plot_chisq, plot_fit_and_chisq
from fitting.models import chisq, probability, w07
from convert import get_dir, set_dir
import glob2
import matplotlib.pyplot as plt
from functools import reduce


def run_fit(filepaths):
    from IPython.display import clear_output

    for filepath in filepaths:

        grb = re.search("(\d{6}[A-Z]?)", filepath)[0]
        filepath = os.path.join(os.path.split(filepath)[0], f"{grb}_converted_flux_accepted.txt")

        try:
            data = pd.read_csv(filepath, sep="\t", header=0)
            if (data["time_sec"] < 0).sum() > 0:
                import warnings

                warnings.warn("Warning: time values < 0 found. Removing...")
                data.drop(data[data["time_sec"] < 0], axis=0)

            if len(data.index) > 0:
                plot_data(filepath)
                auto_guess = input("Do you want to fit? (y/[n])")
                if auto_guess in ["y"]:
                    tt = float(input("tt : "))
                    T = float(input("T : "))
                    F = float(input("F : "))
                    alpha = float(input("α : "))
                    t = float(input("t : "))
                    fit_vals = [T, F, alpha, t, tt]

                    p, pcov = fit_routine(filepath, guess=fit_vals, return_fit=True)
                    T, F, alpha, t = p
                    T_err, F_err, alpha_err, t_err = np.sqrt(np.diag(pcov))
                    if str(input("save? ([y]/n): ")) in ["", "y"]:
                        fit_data = _try_import_fit_data()
                        if isinstance(fit_data, dict):
                            fit_df = pd.DataFrame(fit_data, columns=list(fit_data.keys()))
                            fit_df.set_index("GRB", inplace=True)
                        else:
                            fit_df = fit_data
                        fit_df.loc[grb] = [tt, T, T_err, F, F_err, alpha, alpha_err, t, t_err, *fit_vals[:-1]]
                        savepath = os.path.join(get_dir(), "fit_vals.txt")
                        fit_df.to_csv(savepath, sep="\t", index=True)
                        clear_output()

                elif auto_guess in ["", "n"]:
                    clear_output()
                    continue

                elif auto_guess in "q":
                    clear_output()
                    return

            else:
                print(f"No accepted datapoints found for {grb}.")

        except FileNotFoundError as e:
            print(e)
            continue


def fit_routine(filepath, guess=[None, None, None, None, 0], return_fit=False):
    failct = 0
    df = pd.read_csv(filepath, delimiter=r"\t+|\s+", engine="python", header=0)

    if sum(df["time_sec"] <= 0) == 0:
        flux = df["flux"]
        flux_err = df["flux_err"]
        time = df["time_sec"]

        xdata = np.array(np.log10(time))
        ydata = np.array(np.log10(flux))
        yerr = flux_err / (flux * np.log(10))

        try:
            p, pcov = fit_w07(
                xdata,
                ydata,
                p0=guess[:-1],
                tt=guess[-1],
                logTerr=None,
                logFerr=yerr,
                return_guess=False,
                maxfev=10000,
            )
            tt = guess[-1]
            p0 = guess[:-1]
            plot_fit_and_chisq(filepath, p, pcov, p0, tt)
            # plot_w07_fit(xdata, ydata, p, tt=guess[-1], logTerr=None, logFerr=yerr, p0=guess[:-1])
            # plot_chisq(xdata, ydata, yerr, p, np.sqrt(np.diag(pcov)), tt=guess[-1])
            # chisquared = chisq(xdata, ydata, yerr, w07, guess[-1], *p)
            # reduced = chisquared / (len(xdata[xdata >= guess[-1]]) - 3)
            # nu = len(xdata[xdata >= guess[-1]])
            # prob = probability(xdata, reduced, nu, tt=guess[-1])

            # print("GUESS:         ", guess[:-1])
            # print("FIT:           ", p)
            # print("FIT ERR:       ", np.sqrt(np.diag(pcov)))
            # print("CHISQ:         ", chisquared)
            # print("REDUCED CHISQ: ", reduced)
            # print("PROBABILITY α: ", prob)

            if return_fit:
                return p, pcov
        except RuntimeError:
            failct += 1
            print("Fitting does not work :(")


def plot_data(filepath):

    import plotly.express as px

    grb = re.search("(\d{6}[A-Z]?)", filepath)[0]
    df = pd.read_csv(filepath, delimiter=r"\t+|\s+", engine="python", header=0)

    t_dupes = set(
        (df["time_sec"][df["time_sec"].duplicated(keep=False)]).apply(lambda x: round(np.log10(x), 4)).astype(str)
    )

    if len(t_dupes) > 0:
        dupe_string = ", ".join(t_dupes)
        print(f"Some duplicate times found at T = [{dupe_string}]. Did you correctly go through Stage 2?")

    fig = px.scatter(
        df,
        x=np.log10(df["time_sec"]),
        y=np.log10(df["flux"]),
        error_y=df["flux_err"] / (df["flux"] * np.log(10)),
        color="band",
        width=700,
        height=400,
    )

    fig.update_layout(
        title=grb,
        xaxis_title=r"logT (sec)",
        yaxis_title=r"logF (erg cm-xw2 s-1)",
        legend_title="Band",
        yaxis_zeroline=True,
        xaxis_zeroline=True,
    )

    fig.show()


def LC_summary(filepaths):
    lc_data = {}
    fig, ax = None, None
    for filepath in filepaths:
        try:
            grb = os.path.split(filepath)[-1].rstrip("_converted_flux_accepted.txt")
        except:
            grb = os.path.split(filepath)[-1].rstrip("_converted_flux.txt")
        df = pd.read_csv(filepath, delimiter=r"\t+|\s+", engine="python", header=0)
        num_rows = len(df.index)
        bands = ",".join(list(df["band"]))  # because lists aren't hashable >:(
        lc_data[grb] = [num_rows, bands]

    return {grb: l for grb, (l, _) in lc_data.items()}


def _try_import_fit_data():
    empty_dict = {
        k: []
        for k in [
            "GRB",
            "tt",
            "T",
            "T_err",
            "F",
            "F_err",
            "alpha",
            "alpha_err",
            "t",
            "t_err",
            "T_guess",
            "F_guess",
            "alpha_guess",
            "t_guess",
        ]
    }
    try:
        filepath = os.path.join(get_dir(), f"fit_vals.txt")
        data = pd.read_csv(filepath, delimiter=r"\t+|\s+", engine="python", header=0, index_col="GRB")
        if len(data.index) > 0:
            return data
        else:
            return empty_dict
    except FileNotFoundError:
        return empty_dict
    except pandas.errors.EmptyDataError:
        return empty_dict


def check_fits(save=True):

    main_dir = os.path.join(get_dir(), "accepted")
    accepted_paths = np.array(glob2.glob(os.path.join(main_dir, "*.txt")))
    accepted = np.array([os.path.split(path)[1].rstrip("_converted_flux_accepted.txt") for path in accepted_paths])

    # import fit_vals.txt to cross-reference with accepted_grbs.
    # if we have a match in both, that means we can plot!
    fitted_paths = glob2.glob(os.path.join(get_dir(), "fit_vals_*.txt"))
    fits = 0
    for path in fitted_paths:
        fitted = pd.read_csv(path, sep="\t", header=0, index_col=0)

        # find intersection between GRBs with accepted pts and fitted grbs
        intersection = list(set(accepted) & set(fitted.index))

        for GRB in intersection:
            fits += 1
            # set up figure
            ax = plt.figure(constrained_layout=True, figsize=(10, 7)).subplot_mosaic(
                [["fit", "fit", "EMPTY"], ["T", "F", "alpha"]], empty_sentinel="EMPTY"
            )

            # read in fitted vals
            curr = fitted.loc[GRB]
            accepted_path, *__ = accepted_paths[accepted == GRB]
            acc = pd.read_csv(accepted_path, sep="\t", header=0)
            xdata = np.array(np.log10(acc.time_sec))
            ydata = np.array(np.log10(acc.flux))
            yerr = acc.flux_err / (acc.flux * np.log(10))
            p = np.array([curr["T"], curr.F, curr.alpha, curr.t])
            perr = np.array([curr.T_err, curr.F_err, curr.alpha_err, curr.t_err])
            tt = curr.tt
            p0 = np.array([curr.T_guess, curr.F_guess, curr.alpha_guess, curr.t_guess])
            plot_w07_fit(xdata, ydata, p, tt=tt, logTerr=None, logFerr=yerr, p0=p0, ax=ax["fit"], show=False)
            plot_chisq(xdata, ydata, yerr, p, perr, tt=tt, ax=[ax["T"], ax["F"], ax["alpha"]], show=False)

            chisquared = chisq(xdata, ydata, yerr, w07, tt, *p)
            reduced_nu = len(xdata[xdata >= tt]) - 3
            reduced_nu = 1 if reduced_nu == 0 else reduced_nu
            reduced = chisquared / reduced_nu
            nu = len(xdata[xdata >= tt])
            prob = probability(xdata, reduced, nu, tt=tt)

            plt.figtext(
                x=0.63,
                y=0.6,
                s="""
                GRB %s
                
                $\\chi^2$: %.3f
                
                $\\chi_{\\nu}^2$: %.3f
                
                $\\alpha$ : %.3e
                """
                % (GRB, chisquared, reduced, prob),
                size=18,
            )

            if save:
                plt.savefig(reduce(os.path.join, [get_dir(), "fits", f"{GRB}_fitted.pdf"]))
                plt.close()
            else:
                plt.show()

    print("Plotted", fits, "fits.")


def copy_accepted():
    from shutil import copyfile

    # fitted_paths = glob2.glob(os.path.join(get_dir(), "fit_vals_*.txt"))
    copies = 0
    # for path in fitted_paths:
    # fitted = pd.read_csv(path, sep="\t", header=0, index_col=0)
    accepted_filespaths = glob2.glob(reduce(os.path.join, [get_dir(), "*flux", "*accepted.txt"]))
    accepted_dir = [reduce(os.path.join, [get_dir(), "accepted", os.path.split(f)[1]]) for f in accepted_filespaths]
    for src, dst in zip(accepted_filespaths, accepted_dir):
        # if os.path.split(src)[1].rstrip("_converted_flux_accepted.txt") in fitted.index:
        copies += 1
        copyfile(src, dst)

    print("Copied", copies, "accepted and previously fitted GRBs")


def copy_rejected():
    from shutil import copyfile

    copies = 0
    rejected_filenames = glob2.glob(reduce(os.path.join, [get_dir(), "*flux", "*rejected.txt"]))
    rejected_dir = [reduce(os.path.join, [get_dir(), "rejected", os.path.split(f)[1]]) for f in rejected_filenames]
    for src, dst in zip(rejected_filenames, rejected_dir):
        copies += 1
        copyfile(src, dst)

    print("Copied", copies, "accepted and previously fitted GRBs")
