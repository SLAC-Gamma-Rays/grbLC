import pandas as pd
import pandas.errors
import numpy as np
import os, re
from fitting import fit_w07, plot_w07_fit, plot_chisq
from convert import get_dir


def run_fit(filepaths):
    from IPython.display import clear_output

    for filepath in filepaths:

        grb = re.search("(\d{6}[A-Z]?)", filepath)[0]
        filepath = os.path.join(os.path.split(filepath)[0], f"{grb}_converted_flux_accepted.txt")

        try:
            if len(pd.read_csv(filepath, header=0).index) > 0:
                plot_data(filepath)
                auto_guess = input("Do you want to fit? (y/[n])")
                if auto_guess in ["y"]:
                    tt = float(input("tt : "))
                    T = float(input("T : "))
                    F = float(input("F : "))
                    alpha = float(input("alpha : "))
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

        xdata = list(np.log10(time))
        ydata = list(np.log10(flux))
        known_yerr = flux_err / (flux * np.log(10))

        try:
            p, pcov = fit_w07(
                xdata,
                ydata,
                p0=guess[:-1],
                tt=guess[-1],
                logTerr=None,
                logFerr=known_yerr,
                return_guess=False,
                maxfev=10000,
            )
            plot_w07_fit(xdata, ydata, p, tt=guess[-1], logTerr=None, logFerr=known_yerr, p0=guess[:-1])
            plot_chisq(xdata, ydata, p, pcov)

            print("GUESS:", guess[:-1])
            print("FIT:  ", p)
            print("ERR:  ", np.sqrt(np.diag(pcov)))

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
        (df["time_sec"][df["time_sec"].duplicated(keep=False)]).apply(lambda x: round(np.log(x), 4)).astype(str)
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
