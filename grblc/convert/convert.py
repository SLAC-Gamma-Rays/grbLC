from astropy import units as u, constants as const
from astropy.time import Time
import pandas as pd
import numpy as np
from .constants import flux_densities
from .time import grb_to_date
import os.path
import requests, re, glob2
from functools import reduce


def angstromToHz(ang: float):
    return (const.c / (ang * u.angstrom).to(u.m)).to(u.Hz).value


def toFlux(
    mag: float, band: str, magerr: float = 0, index: float = 0, index_type: str = "spectral", index_err: float = 0
):

    band = band.strip("'").strip("_").strip("\\")  # TODO: regex substitute instead of a bunch of strips
    band = band if band != "v" else "V"

    # determine index type for conversion of f_nu to R band
    if index_type in ["spectral", "gamma"]:
        # transformation from gamma to alpha (alpha = Gamma - 1)
        power = index - 1
    elif index_type in ["photon", "alpha"]:
        power = index

    try:
        lambda_R, _ = flux_densities["R"]  # lambda_R in angstrom
        lambda_x, f_x = flux_densities[band]
    except KeyError:
        # TODO: fix the weird ""'s being printed when the error is raised.
        raise KeyError(f"Band '{band}' is not currently supported. Please fix the band or contact Nicole/Sam!")

    # convert from flux density in another band to R!
    f_R = f_x * (lambda_x / lambda_R) ** (-power)

    f_lam_or_nu = f_R

    if band.lower() in ["uvw1", "uvw2", "uvm2", "white"]:
        # If flux density is given as f_lambda (erg / cm2 / s / Å)
        lam_or_nu = lambda_R * (u.angstrom)
        f_lam_or_nu = f_lam_or_nu * (u.erg / u.cm ** 2 / u.s / u.angstrom)
    else:
        # If flux density is given as f_nu (erg / cm2 / s / Hz)
        lam_or_nu = angstromToHz(lambda_R) * (u.Hz)
        f_lam_or_nu = f_lam_or_nu * (u.erg / u.cm ** 2 / u.s / u.Hz)

    flux = (lam_or_nu * f_lam_or_nu * 10 ** (-mag / 2.5)).value

    # see https://youngsam.me/files/error_prop.pdf for derivation
    fluxerr = abs(flux) * np.sqrt((magerr * np.log(10 ** (0.4))) ** 2 + (index_err * np.log(lambda_x / lambda_R)) ** 2)

    assert flux >= 0, "Error computing flux."
    assert fluxerr >= 0, "Error computing flux error."
    return flux, fluxerr


# main conversion function to call
def convertGRB(
    GRB: str, battime: str = "", index: float = 0, index_type: str = "", use_nick: bool = False, debug: bool = False
):
    # assign column names and datatypes before importing
    dtype = {
        "date": str,
        "time": str,
        "exp": str,
        "mag": np.float64,
        "mag_err": np.float64,
        "band": str,
    }
    names = list(dtype.keys())
    if use_nick:
        names.insert(0, "nickname")  # add nickname column
        dtype["nickname"] = str  # add nickname type

    """ will import data using the following headers
    IF: use_nick = False
    | date | time | exp | mag | mag_err | band |
    OR
    IF: use_nick = True
    | nickname | date | time | exp | mag | mag_err | band |
    """

    # try to import mag_table to convert
    try:
        global directory
        glob_path = reduce(os.path.join, (directory, "**", f"{GRB}.txt"))
        filename, *__ = glob2.glob(glob_path)
        mag_table = pd.read_csv(
            filename,
            delimiter=r"\t+|\s+",
            names=names,
            dtype=dtype,
            skiprows=1,
            engine="python",
        )

    except ValueError as error:
        raise error
    except IndexError as error:
        raise ImportError(message=f"Couldn't find GRB table at {filename}.")

    # grab index and trigger time
    if battime and index:
        starttime = Time(battime)
        index = index
        index_type = "spectral"
    else:
        try:
            bat_spec_df = pd.read_csv(
                os.path.join(directory, "trigs_and_specs.txt"), delimiter="\t", index_col=0, header=0, engine="c"
            )
            indices = bat_spec_df.loc[GRB, ["photon_index", "spectral_index"]]
            na_indices = indices.isna()

            # if theres any NaN, we'll pick the non-NaN
            if sum(na_indices) > 0:
                index, *__ = indices[~na_indices]
                index_type, *__ = np.array(["photon", "spectral"])[~na_indices]
            else:
                # otherwise, default to spectral index
                index = indices[1]
                index_type = "spectral"

            battime = list(bat_spec_df.loc[GRB, ["trigger_date", "trigger_time"]])
            battime = " ".join(battime)
            starttime = Time(battime)

        except KeyError as e:
            raise ImportError(
                f"{GRB} isn't currently supported and it's trigger time and spectral/photon index must be manually provided. :("
            )

    converted = {k: [] for k in ("time_sec", "flux", "flux_err", "band")}
    if debug:
        converted_debug = {k: [] for k in ("time_sec", "flux", "flux_err", "band", "logF", "logT")}

    for __, row in mag_table.iterrows():
        # strip band string of any whitespaces
        band = row["band"]
        magnitude = row["mag"]
        mag_err = row["mag_err"]

        # attempt to convert a single mag to flux given band, mag_err, and spectral index
        try:
            flux, flux_err = toFlux(magnitude, band, mag_err, index=index, index_type=index_type)
        except KeyError as error:
            print(error)
            continue

        date_UT = row["date"]
        time_UT = row["time"]
        time_UT = f"{date_UT} {time_UT}"
        astrotime = Time(time_UT)  # using astropy Time package
        dt = astrotime - starttime  # for all other times, subtract start time
        time_sec = round(dt.sec, 5)  # convert delta time to seconds

        converted["time_sec"].append(time_sec)
        converted["flux"].append(flux)
        converted["flux_err"].append(flux_err)
        converted["band"].append(band)

        # VERBOSE
        if debug:
            logF = np.log10(flux)
            logT = np.log10(time_sec)
            converted_debug["time_sec"].append(time_sec)
            converted_debug["flux"].append(flux)
            converted_debug["flux_err"].append(flux_err)
            converted_debug["band"].append(band)
            converted_debug["logF"].append(logF)
            converted_debug["logT"].append(logT)

    # after converting everything, go from dictionary -> DataFrame -> csv!
    save_path = os.path.join(os.path.split(filename)[0], f"{GRB}_flux.txt")
    pd.DataFrame.from_dict(converted).to_csv(save_path, sep="\t", index=False)
    if debug:
        save_path = os.path.join(os.path.split(filename)[0], f"{GRB}_flux_DEBUG.txt")
        pd.DataFrame.from_dict(converted_debug).to_csv(save_path, sep="\t", index=False)

    return


# Saves GRB spectral and photon indices to a text file from the Swift website to a file called
# trigs_and_specs.txt in `directory`. If `directory` hasn't been set it'll save to the current
# working directory (e.g., the folder in which your notebook is)
def save_convert_params(save_dir=None, return_df=False):
    if not save_dir:
        global directory
        save_dir = directory

    # requests a table link from the Swift website. this html can be huge as the entire formatted
    # table is returned here, so we'll request it as byte stream and parse it by line to save
    # the user and the NASA site bandwith.
    q = requests.get(
        "https://swift.gsfc.nasa.gov/archive/grb_table/table.php?obs=All+Observatories&year=All+Years&restrict=none&grb_time=1&bat_photon_index=1&xrt_gamma=1",
        stream=True,
    )
    lines = q.iter_lines(decode_unicode=True)
    regsearch = re.compile(r"(?<=\/)(?:grb_table_\d+\.txt)")
    filename = None
    while not filename:
        filename = regsearch.search(next(lines))

    # grab the actual plaintext table
    df = pd.read_csv(
        f"https://swift.gsfc.nasa.gov/archive/grb_table/tmp/{filename[0]}",
        header=0,
        index_col=0,
        delimiter="\t",
        engine="c",
        names=["GRB", "trigger_time", "photon_index", "spectral_index"],
        na_values="n/a",
    )
    # clean photon index columns from extra information
    df["photon_index"] = df["photon_index"].str.strip(",()~ CPL \n\t")
    df["photon_index"] = df["photon_index"].str.rstrip(".")
    df["photon_index"] = df["photon_index"].str.replace(r"(?<=\d)[^.0-9\n]+(?=\d{0,7})", ".", regex=True)
    df["photon_index"] = df["photon_index"].astype(np.float64)

    # delete columns where we have NaN for both photon_index & spectral_index
    df.dropna(axis=0, how="all", subset=["photon_index", "spectral_index"], inplace=True)

    # insert a new column containing dates for each grb trigger date
    df.insert(0, "trigger_date", list(map(grb_to_date, df.index)))

    # save!
    df.to_csv(os.path.join(save_dir, "trigs_and_specs.txt"), sep="\t", na_rep="n/a")

    if return_df:
        return df


# small setter to set the main conversion directory
def set_dir(dir):
    global directory
    directory = os.path.abspath(dir)
    return directory


def get_dir():
    global directory
    return directory


# sets directory to the current working directory, or whatever folder you're currently in
directory = os.getcwd()
