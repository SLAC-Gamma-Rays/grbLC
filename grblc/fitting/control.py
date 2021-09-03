import pandas as pd
import pandas.errors
import numpy as np
import os, re
from fitting.fitting import fit_w07, plot_w07_fit, plot_chisq, plot_fit_and_chisq
from fitting.models import chisq, probability, w07
from .assignments import locate
from convert import get_dir, set_dir
import glob2
import matplotlib.pyplot as plt
from functools import reduce
from .outlier import OutlierPlot, check_all_
from PyPDF2 import PdfFileMerger


def run_fit(filepaths):
    from IPython.display import clear_output

    for filepath in filepaths:

        grb = re.search("(\d{6}[A-Z]?)", filepath)[0]
        filepath = os.path.join(os.path.dirname(filepath), f"{grb}_converted_flux_accepted.txt")

        try:
            data = pd.read_csv(filepath, sep="\t", header=0)
            if (data["time_sec"] < 0).sum() > 0:
                import warnings

                warnings.warn("Warning: Negative time values found. Removing...")
                data.drop(data[data["time_sec"] < 0], axis=0)

            if len(data.index) > 0:
                plot_data(filepath)
                auto_guess = input("Do you want to fit? (y/[n])")
                if auto_guess in ["y"]:
                    tt = float(input("tt : "))
                    tf = float(input("tf : "))
                    T = float(input("T : "))
                    F = float(input("F : "))
                    alpha = float(input("α : "))
                    t = float(input("t : "))
                    fit_vals = [T, F, alpha, t, tt, tf]

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
                        fit_df.loc[grb] = [
                            tt,
                            T,
                            T_err,
                            F,
                            F_err,
                            alpha,
                            alpha_err,
                            t,
                            t_err,
                            *fit_vals[:-2],
                            tf,
                        ]
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


def fit_routine(filepath, guess=[None, None, None, None, 0, np.inf], return_fit=False, plot=True):
    failct = 0
    df = pd.read_csv(filepath, delimiter=r"\t+|\s+", engine="python", header=0)

    if sum(df["time_sec"] <= 0) == 0:
        flux = df["flux"]
        flux_err = df["flux_err"]
        time = df["time_sec"]

        xdata = np.array(np.log10(time))
        ydata = np.array(np.log10(flux))
        yerr = flux_err / (flux * np.log(10))

        *p0, tt, tf = guess

        try:
            p, pcov = fit_w07(
                xdata,
                ydata,
                p0=p0,
                tt=tt,
                tf=tf,
                logTerr=None,
                logFerr=yerr,
                return_guess=False,
                maxfev=10000,
            )
            if plot:
                plot_fit_and_chisq(filepath, p, pcov, p0, tt, tf)

            if return_fit:
                return p, pcov
        except RuntimeError:
            failct += 1
            print("Fitting does not work :(")


def plot_data(filepath, return_plot=False):
    import plotly.express as px

    grb = re.search("(\d{6}[A-Z]?)", filepath)[0]
    df = pd.read_csv(filepath, delimiter=r"\t+|\s+", engine="python", header=0)

    t_dupes = set(
        (df["time_sec"][df["time_sec"].duplicated(keep=False)]).apply(lambda x: round(np.log10(x), 4)).astype(str)
    )

    # if len(t_dupes) > 0:
    #     dupe_string = ", ".join(t_dupes)
    #     print(f"Some duplicate times found at T = [{dupe_string}]. Did you correctly go through Stage 2?")

    try:
        df["source"]
        kwargs = {"hover_data": ["source"]}
    except:
        kwargs = {}

    fig = px.scatter(
        df,
        x=np.log10(df["time_sec"]),
        y=np.log10(df["flux"]),
        error_y=df["flux_err"] / (df["flux"] * np.log(10)),
        color="band",
        width=700,
        height=400,
        **kwargs,
    )

    fig.update_layout(
        title=grb,
        xaxis_title=r"logT (sec)",
        yaxis_title=r"logF (erg cm-2 s-1)",
        legend_title="Band",
        yaxis_zeroline=True,
        xaxis_zeroline=True,
    )

    if not return_plot:
        fig.show()
    else:
        return fig


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


def _try_import_fit_data(filename="fit_vals.txt"):
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
            "tf",
            "chisq",
        ]
    }
    try:
        filepath = os.path.join(get_dir(), filename)
        data = pd.read_csv(filepath, delimiter=r"\t+|\s+", engine="python", header=0, index_col="GRB")
        if len(data.index) > 0:
            return data
        else:
            return empty_dict
    except FileNotFoundError:
        return empty_dict
    except pandas.errors.EmptyDataError:
        return empty_dict


def check_lc(save=False):

    main_dir = os.path.join(get_dir(), "accepted")
    lc_dir = os.path.join(get_dir(), "lightcurve")
    os.makedirs(lc_dir, exist_ok=True)
    accepted_paths = np.array(glob2.glob(os.path.join(main_dir, "*.txt")))
    pdfs = []

    for path in accepted_paths:
        grb = re.search("(\d{6}[A-Z]?)", path)[0]
        fig1 = OutlierPlot(path, plot=False).plot(return_display=True)
        fig1.write_image("temp1.pdf")
        fig2 = plot_data(path, return_plot=True)
        fig2.write_image("temp2.pdf")
        merger = PdfFileMerger()

        temp_pdfs = ["temp1.pdf", "temp2.pdf"]
        for pdf in temp_pdfs:
            merger.append(pdf)

        pdf_path = f"{lc_dir}{os.sep}{grb}.pdf"
        merger.write(pdf_path)
        pdfs.append(pdf_path)
        merger.close()

    os.remove("temp1.pdf")
    os.remove("temp2.pdf")

    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)

    pdf_path = f"{lc_dir}{os.sep}all.pdf"
    merger.write(pdf_path)
    merger.close()


def check_fits(save=True):

    main_dir = os.path.join(get_dir(), "accepted")
    accepted_paths = np.array(glob2.glob(os.path.join(main_dir, "*accepted_ext_corr*.txt")))
    accepted = np.array(
        [os.path.split(path)[1].rstrip("_converted_flux_accepted_ext_corr.txt") for path in accepted_paths]
    )

    # import fit_vals.txt to cross-reference with accepted_grbs.
    # if we have a match in both, that means we can plot!
    fitted_paths = glob2.glob(os.path.join(get_dir(), "fit_vals_approved.txt"))
    fits = 0
    for path in fitted_paths:
        try:
            fitted = pd.read_csv(path, sep=r"\t+|\s+", header=0, index_col=0, engine="python")
        except Exception as e:
            print(path)
            raise e

        # find intersection between GRBs with accepted pts and fitted grbs
        intersection = list(set(accepted) & set(fitted.index))

        for GRB in intersection:
            try:
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
                tf = curr.tf
                p0 = np.array([curr.T_guess, curr.F_guess, curr.alpha_guess, curr.t_guess])
                plot_w07_fit(xdata, ydata, p, tt=tt, tf=tf, logTerr=None, logFerr=yerr, p0=p0, ax=ax["fit"], show=False)
                mask = (xdata >= tt) & (xdata <= tf)
                plot_chisq(
                    xdata[mask], ydata[mask], yerr[mask], p, perr, ax=[ax["T"], ax["F"], ax["alpha"]], show=False
                )

                chisquared = chisq(xdata[mask], ydata[mask], yerr[mask], w07, *p)
                reduced_nu = len(xdata[xdata >= tt]) - 3
                reduced_nu = 1 if reduced_nu == 0 else reduced_nu
                reduced = chisquared / reduced_nu
                nu = len(xdata[mask])
                prob = probability(reduced, nu)

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
                    plt.savefig(reduce(os.path.join, [get_dir(), "fits_approved", f"{GRB}_fitted_approved.pdf"]))
                    plt.close()
                    print("saved to", reduce(os.path.join, [get_dir(), "fits_approved", f"{GRB}_fitted_approved.pdf"]))
                else:
                    plt.show()
            except Exception as e:
                print(GRB)
                raise e

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

    print("Copied", copies, "rejected and previously fitted GRBs")


def prepare_fit_data():

    """
    1 - Name
    2 - Redshift
    3 - Fbest
    4 - T_abest
    5 - Alpha_best
    6 - F_min
    7 - F_max
    8 - T_amin
    9 - T_amax
    10 - Alpha_min
    11 - Alpha_max
    12 - Beta
    13 - beta_errp
    14 - beta_errm
    15 - chi sq
    16 - tt
    17 - tf
    18 - class
    19 - T90
    """

    fit_val_paths = glob2.glob(os.path.join(get_dir(), "fit_vals_approved.txt"))
    dataframes = [pd.read_csv(f, sep="\t+|\s+", header=0, engine="python") for f in fit_val_paths]
    result = pd.concat(dataframes)
    result.drop_duplicates(subset="GRB", keep="last", inplace=True)
    result.set_index("GRB", inplace=True)
    grbs = result.index
    # GRB    tt     T   T_err    F  F_err     alpha     alpha_err t t_err   T_guess  F_guess  alpha_guess  t_guess    grbs = result.index
    trigs_and_specs = pd.read_csv(
        os.path.join(get_dir(), "trigs_and_specs.txt"), sep="\t+|\s+", header=0, engine="python"
    )
    # GRB         photon_index   photon_index_err     trigger_date   trigger_time   z              T90
    trigs_and_specs["spectral_index"] = trigs_and_specs["photon_index"] - 1
    trigs_and_specs["spectral_index_err"] = trigs_and_specs["photon_index_err"]
    trigs_and_specs.set_index("GRB", inplace=True)

    final = []
    for GRB in grbs:
        name = GRB
        try:
            z = trigs_and_specs.loc[GRB, "z"]
        except:
            print(f"grb {name} not found in trigs and specs. trying a different name")
            if "A" in GRB:
                GRB = GRB[:-1]
            elif GRB.isnumeric():
                GRB = GRB + "A"

            try:
                z = trigs_and_specs.loc[GRB, "z"]
            except:
                print(f"grb {name} is definitely not in trigs and specs :(")
                return
        F = result.loc[GRB, "F"]
        T = result.loc[GRB, "T"]
        alpha = result.loc[GRB, "alpha"]
        F_err = result.loc[GRB, "F_err"]
        F_min = F - F_err
        F_max = F + F_err
        T_err = result.loc[GRB, "T_err"]
        T_min = T - T_err
        T_max = T + T_err
        alpha_err = result.loc[GRB, "alpha_err"]
        alpha_min = alpha - alpha_err
        alpha_max = alpha + alpha_err
        beta = trigs_and_specs.loc[GRB, "spectral_index"]
        beta_errp = beta_errm = trigs_and_specs.loc[GRB, "spectral_index_err"]
        chisq = result.loc[GRB, "chisq"]  # unneeded
        tt = result.loc[GRB, "tt"]
        tf = result.loc[GRB, "tf"]
        cls = "n/a"  # needed dtl
        auth = "n/a"
        T90 = trigs_and_specs.loc[GRB, "T90"]

        final.append(
            [
                name,
                z,
                F,
                T,
                alpha,
                F_min,
                F_max,
                T_min,
                T_max,
                alpha_min,
                alpha_max,
                beta,
                beta_errp,
                beta_errm,
                chisq,
                tt,
                tf,
                cls,
                T90,
                auth,
            ]
        )

    final_df = pd.DataFrame(
        final,
        columns=[
            "GRB",
            "z",
            "F",
            "T",
            "alpha",
            "F_min",
            "F_max",
            "T_min",
            "T_max",
            "alpha_min",
            "alpha_max",
            "beta",
            "beta_errp",
            "beta_errm",
            "chisq",
            "tt",
            "tf",
            "class",
            "T90",
            "author",
        ],
    )

    totaltable131 = pd.read_csv(
        "/Users/youngsam/Code/SULI/Totaltable_131_14_06-2021.txt", sep="\t+", engine="python", header=0
    )
    totaltable131.columns = pd.Index(
        [
            "GRB",
            "z",
            "T90",
            "class",
            "F",
            "F_err",
            "T",
            "T_err",
            "alpha",
            "alpha_err",
            "beta",
            "beta_err",
            "ktotal",
            "ktotal_err",
            "T_rest",
            "T_rest_err",
            "L",
            "L_err",
            "source",
            "tt",
            "4pt",
        ]
    )
    totaltable131["F_min"] = totaltable131["F"] - totaltable131["F_err"]
    totaltable131["F_max"] = totaltable131["F"] + totaltable131["F_err"]
    totaltable131["T_min"] = totaltable131["T"] - totaltable131["T_err"]
    totaltable131["T_max"] = totaltable131["T"] + totaltable131["T_err"]
    totaltable131["alpha_min"] = totaltable131["alpha"] - totaltable131["alpha_err"]
    totaltable131["alpha_max"] = totaltable131["alpha"] + totaltable131["alpha_err"]
    totaltable131["beta_errp"] = totaltable131["beta_errm"] = totaltable131["beta_err"]
    totaltable131["chisq"] = np.nan
    totaltable131["tf"] = np.nan
    totaltable131.drop(
        [
            "ktotal",
            "ktotal_err",
            "T_rest",
            "T_rest_err",
            "L",
            "L_err",
            "beta_err",
            "T_err",
            "F_err",
            "alpha_err",
            "source",
            "4pt",
        ],
        axis=1,
        inplace=True,
    )
    final = pd.concat([final_df, totaltable131])

    final_df.to_csv(
        os.path.join(get_dir(), f"for_mathematica_{len(final_df.index)}.txt"), sep="\t", index=False, na_rep="n/a"
    )
    print(f"Successfully prepared {len(final_df.index)} GRBs for Mathematica")

    final.to_csv(
        os.path.join(get_dir(), f"for_mathematica_{len(final.index)}.txt"), sep="\t", index=False, na_rep="n/a"
    )
    print(f"Successfully prepared {len(final.index)} GRBs for Mathematica")


def check_one(grb):

    purpose = input("What do you want to do? [a] Show everything [b] Checking outliers [c] Fitting")
    path = locate(grb)

    if not path:
        print(f"There is no data for GRB {grb}")
        return

    if purpose == "a":
        op = OutlierPlot(path[0], plot=True)
        name = input("What is the <name> of the source txt: ")
        source = glob2.glob(os.path.join(get_dir(), f"fit_vals_{name}.txt"))

        if not source:
            print("No txt file for the name")
            return

        plot_data(path[0])
        with open(source[0], "r") as file:
            df = pd.read_csv(file, delimiter=r"\t+|\s+", engine="python", header=0)

        df = df[df["GRB"] == grb]
        if df.empty:
            print("No GRB data found in the txt file.")
            return

        data = df.iloc[0].to_dict()
        keys = ["T_guess", "F_guess", "alpha_guess", "t_guess", "tt", "tf"]
        guess = [float(data[k]) for k in keys]
        filepath = os.path.join(os.path.dirname(path), f"{grb}_converted_flux_accepted_ext_corr.txt")
        fit_routine(filepath, guess=guess)

    if purpose == "b":
        check_all_(path, save=True)
        run_fit(path)

    if purpose == "c":
        OutlierPlot(path[0], plot=True)
        run_fit(path)


def plot_for_gold_sample(filepath, grb, tt, tf, return_plot=True):
    fig = plot_data(filepath, return_plot=True)
    fig.add_vline(x=tt, line_width=1, opacity=0.4, line_dash="dash", line_color="black", annotation_text=f"tt = {tt}")
    if tf != 999:
        fig.add_vline(
            x=tf, line_width=1, opacity=0.4, line_dash="dash", line_color="black", annotation_text=f"tt = {tf}"
        )
    fig.show()


def refit_all(guess="original", save=True):

    main_dir = os.path.join(get_dir(), "accepted")
    accepted_paths = np.array(glob2.glob(os.path.join(main_dir, "*ext_corr.txt")))
    accepted = np.array(
        [os.path.split(path)[1].rstrip("_converted_flux_accepted_ext_corr.txt") for path in accepted_paths]
    )
    # import fit_vals.txt to cross-reference with accepted_grbs.
    # if we have a match in both, that means we can plot!
    fitted_paths = glob2.glob(os.path.join(get_dir(), "fit_vals_approved.txt"))
    fits = 0
    for path in fitted_paths:
        try:
            fitted = pd.read_csv(path, sep=r"\t+|\s+", header=0, index_col=0, engine="python")
        except Exception as e:
            print(path)
            raise e

        # find intersection between GRBs with accepted pts and fitted grbs
        intersection = list(set(accepted) & set(fitted.index))

        for GRB in intersection:
            try:
                # set up figure
                ax = plt.figure(constrained_layout=True, figsize=(10, 7)).subplot_mosaic(
                    [["fit", "fit", "EMPTY"], ["T", "F", "alpha"]], empty_sentinel="EMPTY"
                )

                # read in fitted vals
                curr = fitted.loc[GRB]
                # print(curr.index)
                accepted_path, *__ = accepted_paths[accepted == GRB]
                acc = pd.read_csv(accepted_path, sep="\t", header=0)
                xdata = np.array(np.log10(acc.time_sec))
                ydata = np.array(np.log10(acc.flux))
                yerr = acc.flux_err / (acc.flux * np.log(10))

                if guess == "original":
                    p0 = np.array(
                        [curr["T_guess"], curr["F_guess"], curr["alpha_guess"], curr["t_guess"], curr.tt, curr.tf]
                    )
                elif guess == "fitted":
                    p0 = np.array([curr["T"], curr.F, curr.alpha, curr.t, curr.tt, curr.tf])
                else:
                    p0 = [None, None, None, None, 0, np.inf]

                # refit using the previously fitted values as our guess
                p, pcov = fit_routine(accepted_path, guess=p0, return_fit=True, plot=False)
                fits += 1
                perr = np.sqrt(np.diag(pcov))
                tt = curr.tt
                tf = curr.tf
                mask = (xdata >= tt) & (xdata <= tf)
                plot_w07_fit(xdata, ydata, p, tt=tt, tf=tf, logTerr=None, logFerr=yerr, p0=p0, ax=ax["fit"], show=False)
                plot_chisq(
                    xdata[mask], ydata[mask], yerr[mask], p, perr, ax=[ax["T"], ax["F"], ax["alpha"]], show=False
                )

                chisquared = chisq(xdata[mask], ydata[mask], yerr[mask], w07, *p)
                reduced_nu = len(xdata[mask]) - 3
                reduced_nu = 1 if reduced_nu == 0 else reduced_nu
                reduced = chisquared / reduced_nu
                nu = len(xdata[mask])
                prob = probability(reduced, nu)

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

                FITS_NAME = "fit_vals_ext_corr_w_chisq.txt"
                fit_data = _try_import_fit_data(FITS_NAME)
                if isinstance(fit_data, dict):
                    fit_df = pd.DataFrame(fit_data, columns=list(fit_data.keys()))
                    fit_df.set_index("GRB", inplace=True)
                else:
                    fit_df = fit_data

                T, F, alpha, t = p
                T_err, F_err, alpha_err, t_err = np.sqrt(np.diag(pcov))
                fit_df.loc[GRB] = [tt, T, T_err, F, F_err, alpha, alpha_err, t, t_err, *p0[:-2], tf, reduced]
                savepath = os.path.join(get_dir(), FITS_NAME)
                fit_df.to_csv(savepath, sep="\t", index=True)

                if save:
                    plt.savefig(
                        reduce(
                            os.path.join,
                            [get_dir(), "fits_approved_ext_corr_old_guess", f"{GRB}_fitted_approved_ext_corr.pdf"],
                        )
                    )
                    plt.close()
                    print(
                        "saved to",
                        reduce(
                            os.path.join,
                            [get_dir(), "fits_approved_ext_corr_old_guess", f"{GRB}_fitted_approved_ext_corr.pdf"],
                        ),
                    )

                else:
                    plt.show()
            except Exception as e:
                print(GRB, e)
                # raise e
                continue

    print("Plotted", fits, "new fits with extinction.")
