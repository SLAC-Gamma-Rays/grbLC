from astropy import units as u, constants as const
from astropy.time import Time
from numpy.lib.index_tricks import IndexExpression
import pandas as pd
import numpy as np
from .constants import flux_densities

""" TO USE:
Import this file into a directory to be used in your command line interfance (e.g., terminal on mac, cmd.exe on windows)
Make sure you are in the directory, and then start Jupyter notebook or Jupyter lab / any other Python instance from there
and import as any other module.
"""


def dec_to_UT(decimal: float) -> str:

    assert isinstance(decimal, float) or isinstance(decimal, int), "decimal must be of type float or int!"

    decimal -= int(decimal)

    hours = decimal * 24
    hours_decimal = hours - int(hours)
    hours = int(hours)

    minutes = hours_decimal * 60
    minutes_decimal = minutes - int(minutes)
    minutes = int(minutes)

    seconds = minutes_decimal * 60
    seconds_str = "{:.3f}".format(seconds)

    leading_hours = f"{hours}".zfill(2)
    leading_minutes = f"{minutes}".zfill(2)

    # hardcoding leading zero for seconds because of the need for 3 decimal places
    if len(seconds_str) == 5:
        leading_seconds = "0" + seconds_str
    elif len(seconds_str) == 6:
        leading_seconds = seconds_str
    else:
        raise Exception("Error with converting seconds string!")

    return f"{leading_hours}:{leading_minutes}:{leading_seconds}"


def UT_to_dec(yr_time: str) -> str:
    # format: YYYY-MM-DD ##:##:##.### #
    try:
        float(yr_time.split(" ")[1].split(":")[0])
    except:
        raise Exception("Input string must be in the format: YYYY-MM-DD ##:##:##.###")

    (date, time) = yr_time.split(" ")
    (year, month, day) = date.split("-")
    (hours, minutes, seconds) = [float(num) for num in time.split(":")]
    day = str(int(day) + (((hours * 60 * 60) + (minutes * 60) + seconds) / (24 * 60 * 60)))

    return f"{year}:{month.zfill(2)}:{day}"


def angstromToHz(ang):
    return (const.c / (ang * u.angstrom).to(u.m)).to(u.Hz).value


def magToFlux(mag, band, spectral_index=0):
    band = "V" if band == "v" else band

    try:
        lambda_R, _ = flux_densities["R"]  # lambda_R in angstrom
        lambda_x, F_x = flux_densities[band]
    except KeyError:
        raise KeyError(f"Band '{band}' is not currently supported. Please fix the band or contact Sam!")

    # Conversion
    F_R = F_x * (lambda_x / lambda_R) ** (-spectral_index)

    f_lam_or_nu = F_R
    lam = lambda_x

    if any(band == swift_band for swift_band in ["uvw1", "uvw2", "uvm2", "white"]):
        # If flux density is given as f_lambda (erg / cm2 / s / Å)
        lam_or_nu = lam * (u.angstrom)
        f_lam_or_nu = f_lam_or_nu * (u.erg / u.cm ** 2 / u.s / u.angstrom)
    else:
        # If flux density is given as f_nu (erg / cm2 / s / Hz)
        lam_or_nu = angstromToHz(lam) * (u.Hz)
        f_lam_or_nu = f_lam_or_nu * (u.erg / u.cm ** 2 / u.s / u.Hz)

    return (lam_or_nu * f_lam_or_nu * 10 ** (mag / -2.5)).value


def magErrToFluxErr(mag, magerr, band):

    flux = magToFlux(mag, band)

    fluxerr = magerr * flux * np.log(10 ** (2.0 / 5))
    assert fluxerr > 0, "Error computing flux error."
    return fluxerr


def getFlux(magnitude, magerr, band, spectral_index):
    return magToFlux(magnitude, band, spectral_index=spectral_index), magErrToFluxErr(magnitude, magerr, band)


def convertGRB(GRB: str, battime: str, spectral_index: float = 0, use_nick=False, verbose=False):
    global directory

    names = ["date", "time", "exp", "mag", "mag_err", "band"]
    dtype = "U10,U12,U6,f8,f8,U5"
    if use_nick:
        names.insert(0, "nickname")  # add nickname column
        dtype = "U10," + dtype  # add nickname type

    """ will import data using the following headers
    | date | time | exp | mag | mag_err | band |
    IF: use_nick = False
    OR
    | nickname | date | time | exp | mag | mag_err | band |
    IF: use_nick = True
    """
    Input_Data = pd.read_csv(directory + GRB + ".txt", delimiter="\t+|\s+", names=names, skiprows=1, engine="python")

    # setting band
    band_ID = Input_Data["band"][0].strip()
    print(directory + GRB)
    f = open(directory + GRB + "_" + "flux.txt", "w")
    f.write(str("time_sec") + str("\t") + str("flux") + str("\t") + str("flux_err") + str("\t") + str("band") + "\n")
    f.close()

    # VERBOSE -- use if debugging odd datapoints from Mathematica plots
    if verbose:
        f = open(directory + GRB + "_" + "VERBOSE_flux.txt", "w")
        f.write("\t".join(["nickname", "time_sec", "flux", "flux_err", "band", "logF", "logTime"] + "\n"))
        f.close()

    starttime = Time(battime)

    for idx, row in Input_Data.iterrows():
        # strip band string of any whitespaces
        band = row["band"].strip()

        magnitude = row["mag"]
        mag_err = row["mag_err"]

        try:
            Flux, Flux_err = getFlux(magnitude, mag_err, band, spectral_index=spectral_index)
        except KeyError as e:
            print(e)
            continue

        date_UT = row["date"]
        time_UT = row["time"]
        time_UT = date_UT + " " + time_UT
        astrotime = Time(time_UT)  # using astropy Time package
        dt = astrotime - starttime  # for all other times, subtract start time
        time_sec = round(dt.sec, 5)  # convert delta time to seconds

        logF = np.log10(Flux)
        logTime = np.log10(time_sec)

        f = open(directory + GRB + "_" + "flux.txt", "a")
        f.write(
            str(time_sec) + str("\t") + str(Flux) + str("\t") + str(Flux_err) + str("\t") + str(row["band"]) + "\n"
        )
        f.close()

        # VERBOSE
        if verbose:
            f = open(directory + GRB + "_" + "VERBOSE_flux_" + band_ID + ".txt", "a")
            f.write("\t".join([row["nickname"], time_sec, Flux, Flux_err, row["band"], logF, logTime]) + "\n")

            f.close()


def set_dir(dir):
    global directory
    directory = dir


directory = "./"
