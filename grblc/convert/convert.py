import os.path
from functools import reduce
import re

import astropy.units as u
import numpy as np
from astropy.time import Time
from pandas import DataFrame
from pandas import read_csv

#from .constants import *
from .sfd import data_dir

path1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "filters.txt")
path2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "reddening.txt")
filters = read_csv(path1, sep='\t', header=0, index_col=None)
factors = read_csv(path2, sep='\t', header=0, index_col=0)


def count(str1, str2): 
    
    str1 = re.sub(r"[\u02BB\u02BC\u066C\u2018\u201A\u275B\u275C\u0027\u02B9\u02BB\u02BC\u02BE\u02C8\u02EE\u0301\u0313\u0315\u055A\u05F3\u07F4\u07F5\u1FBF\u2018\u2019\u2032\uA78C\uFF07]", "p", str1)
    str2 = re.sub(r"[\u02BB\u02BC\u066C\u2018\u201A\u275B\u275C\u0027\u02B9\u02BB\u02BC\u02BE\u02C8\u02EE\u0301\u0313\u0315\u055A\u05F3\u07F4\u07F5\u1FBF\u2018\u2019\u2032\uA78C\uFF07]", "p", str2)
    
    c, j = 0,0
    for i in str1:    
        if str2.find(i)>= 0 and j == str1.find(i): 
            c += 1
        j+=1
    
    if len(str1)>0:
        match = c/len(str1)*100
    else:
        match = 0

    return match

def stripcount(str1, str2): 

    str1 = ''.join(i for i in str1 if i.isalnum()).lower() ## removes special characters
    str2 = ''.join(i for i in str2 if i.isalnum()).lower()
    
    c, j = 0,0
    for i in str1:    
        if str2.find(i)>= 0 and j == str1.find(i): 
            c += 1
        j+=1
    
    if len(str1)>0:
        match = c/len(str1)*100
    else:
        match = 0

    return match

def calibration(band: str, system: str, telescope: str):

    ## initialising

    sep = ['-','_','.',',']

    j = 0
    for i in sep:
        if i in band:
            bandpass, filter = band.split(i)
            if len(filter) > len(bandpass):
                filter, bandpass = band.split(i)
            j += 1
    
    if j == 0:
        filter = band
        bandpass = 'Johnson/Cousins/SDSS/UKIDSS/MKO/Swift'
        
    if filter == 'CV': ## CV clear callibrated as V
        filter ='V'
    elif filter == 'CR': ## CR clear callibrated as R
        filter ='R'

    ## match filters then bandpass then telescope

    for m, n in zip(filters.index, filters['filter_id']):

        grblc_obs, grblc_tel, grblc_pass, grblc_fil = str(n).split(".")

        if grblc_pass == "":
            grblc_pass = grblc_obs

        match1 = count(grblc_fil, filter)
        match2 = stripcount(grblc_pass, bandpass)
        match3 = stripcount(grblc_obs+grblc_tel, telescope)

        filters.loc[m, 'match1'] = match1
        filters.loc[m, 'match2'] = match2
        filters.loc[m, 'match3'] = match3

        if len(filters[filters['match3'] > 0]) == 0:

            telescope = 'Gen/GCPD/Cat'

            match1 = count(grblc_fil, filter)
            match2 = stripcount(grblc_pass, bandpass)
            match3 = stripcount(grblc_obs+grblc_tel, telescope)

            filters.loc[m, 'match1'] = match1
            filters.loc[m, 'match2'] = match2
            filters.loc[m, 'match3'] = match3

            if len(filters[filters['match3'] > 0]) == 0:
                raise KeyError(
                    f"No matching telescopes."
                )

        filters.loc[m, 'match'] = (match1/2) + (match2/4) + (match3/4)
    
    matched = filters[filters['match1'] == 100]

    if len(matched) == 0:
        matched = filters[filters['match1'] >= 50]
        if len(matched) == 0:
            raise KeyError(
                f"No matching filters."
            )
    
    index = matched.index[matched['match'] == max(matched['match'])].to_list()[0]

    filter_id = filters.loc[index,'filter_id']
    lam = filters.loc[index,'lambda_eff']
    lam_round = round(lam, -1)

    if "AB" in system or "SDSS" in system:
            zp = 3631e-23
    else:
            zp = np.float64(filters.loc[index,'zp_v'])*10**-23

    # this factor is A_b / E(B-V)
    R_v = factors.loc[lam_round, 'Rv']

    return lam, zp, R_v, filter_id


def ebv(grb: str, ra="", dec=""):
    r"""A function that returns the galactic extinction correction
       at a given position for a given band.

                            This takes data from Schlegel, Finkbeiner & Davis (1998) in the form
                            of the SFD dust map, and is queried using the dustmaps python package.
                            Updated coefficient conversion values for the SFD is taken from Schlafly & Finkbeiner (2011)
                            and is found in SF11_conversions.txt.

    Parameters
    ----------
    grb : str
        Gamma ray burst name
    bandpass : str
        One of the 94 bandpasses supported. See SF11_conversion.txt for these bandpasses.
    ra : str, optional
        Right ascension, by default None
    dec : str, optional
        Declination, by default None

    Returns
    -------
    float
        Galactic extinction correction in magnitude ($A_\nu$).

    Raises
    ------
    astroquery.exceptions.RemoteServiceError
        If the GRB position cannot be found with `astroquery`, then
        the user is prompted to enter the RA and DEC manually.
    """

    from astropy.coordinates import SkyCoord

    from .sfd import SFDQuery

    sfd = SFDQuery()

    if not (ra or dec):
        import astroquery.exceptions
        from astroquery.simbad import Simbad

        try:
            obj = Simbad.query_object(f"GRB {grb}")
            skycoord = SkyCoord(
                "{} {}".format(obj["RA"][0], obj["DEC"][0]), unit=(u.hourangle, u.deg)
            )
        except astroquery.exceptions.RemoteServiceError:
            raise astroquery.exceptions.RemoteServiceError(
                f"Couldn't find the position of GRB {grb}. Please supply RA and DEC manually."
            )
    else:
        skycoord = SkyCoord(f"{ra} {dec}", frame="icrs", unit=(u.hourangle, u.deg))

    # this grabs the degree of reddening E(B-V) at the given position in the sky.
    # see https://astronomy.swin.edu.au/cosmos/i/interstellar+reddening for an explanation of what this is
    ebv = sfd(skycoord)

    return ebv


@np.vectorize
def toMag(
    system: str,
    fnu: float,
    fnu_err: float = 0
):

    if system == 'AB':
        c = -48.6
    elif system == 'ST':
        c = 8.9
    else: # Anyother is assumed to be Johnson
        c = 0.03

    mag = -2.5 * np.log10(fnu) + c 
    mag_err = 2.5 * fnu_err / (fnu * np.log(10))
    
    return mag, mag_err


@np.vectorize
def toFlux(
    band: str,
    system: str,
    telescope: str,
    extcorr: float,
    mag: float,
    mag_err: float = 0,
    beta: float = 1,
    beta_err: float = 0,
    grb: str = None,
    ra: str = None,
    dec: str = None
):
    r"""
        A function that converts a given magnitude to flux (erg cm$^{-2}$ s$^{-1}$),
        normalized to the R photometric band. This is done by converting the
        zero-point flux densities of a given band to the R band via assuming that
        the spectral energy distribution follows a simple power law.

        The conversion is as follows:

        $${\rm Flux~}= \lambda_R F_x \left(\frac{\lambda_X}{\lambda_R}\right)^{-\beta} \left(10\right)^{-m/2.5},$$

        where $\lambda_R$ is the R band wavelength, $\lambda_X$ is the bandpass
        wavelength, $\beta$ is the photon index ($\beta = \Gamma - 1$), and $m$ is the dust-corrected magnitude.

        Dust extinction corrections are taken from Schlafly & Finkbeiner (2011). Error propagation
        for the flux, assuming there is error on the magnitude and photon index and no covariance
        between the two values, is as follows:

        $$\sigma = \left|{\rm Flux}\right| \sqrt{(\frac{\sigma_m}{2.5} \log10)^2 +
        \left(\sigma_\beta\log{\left(\frac{\lambda_X}{\lambda_R}\right)}\right)^2},$$

        where $\sigma_i$ denotes the standard error on $i$.

        Supported bands:

        .. jupyter-execute::
            :hide-code:

            import grblc
            from grblc.convert.constants import photometry
            print(", ".join(list(photometry.keys())))


    Parameters
    ----------
    mag : float
        Magnitude to convert to flux.
    band : str
        Photometric bandpass of the given magnitude. Must be one of the bandpasses
        from :py:data:`grblc.constants.photometry`.
    grb : str
        GRB name.
    mag_err : float, optional
        Error on the magnitude, by default 0
    photon_index : float, optional
        Photon index $\Gamma$ ($\Gamma = \beta + 1$), by default 1
    photon_index_err : float, optional
        Error on the photon index, by default 0
    A_b : float, optional
        Galactic extinction to add onto the magnitude. If not provided, extinction values
        will be looked up and added automatically to the final flux.
    ra : str, optional
        Right ascension, for use in grabbing dust extinction values, by default None
    dec : str, optional
        Declination, for use in grabbing dust extinction values, by default None

    Returns
    -------
    float, float
        flux and flux error in erg cm$^{-2}$ s$^{-1}$ normalized to the R band.

    Raises
    ------
    KeyError
        If a bandpass is not found in :py:data:`grblc.constants.photometry`.
    """
    assert bool(extcorr != 0) ^ bool(grb) ^ bool(ra and dec), "Must provide either extcorr or grb or ra, dec"
    _check_dust_maps()

    try:
        lambda_R, *__ = calibration("Johnson.R", system, "Gen") # lambda_R in angstrom # telescope shpuld be standardised
        lambda_x, F_x, R_v, filter_id = calibration(band, system, telescope)
    except KeyError:
        raise KeyError(f"Band '{band}' of telescope '{telescope}' is not currently supported.")

    # get correction for galactic extinction to be added to magnitude if not already supplied
    
    if extcorr == 'y' or True:
        A_b = 0
    if extcorr == 'n' or 'nan' or 'None' or False:
        A_b = R_v * ebv(grb, ra, dec)

    # convert from flux density in another band to R!
    F_R = F_x * (lambda_x / lambda_R) ** (-beta)
    F_nu = F_R * (u.erg / u.cm ** 2 / u.s / u.Hz)

    # If flux density is given as F_nu (erg / cm2 / s / Hz)
    nu = (lambda_R * u.AA).to(u.Hz, equivalencies=u.spectral())

    if "Johnson".lower() in system.lower() and "V" in band:
        mag += 0.03

    flux = (nu * F_nu * 10 ** (-(mag + A_b) / 2.5)).value

    if mag_err == 0:
        flux_err = 0
    else:
        flux_err = (flux) * np.sqrt(
            (mag_err * np.log(10 ** (0.4))) ** 2 +
            (beta_err * np.log(lambda_x / lambda_R)) ** 2
        ) ## see https://youngsam.me/files/error_prop.pdf for derivation

    assert np.all(flux >= 0), "Error computing flux."
    assert np.all(flux_err >= 0), "Error computing flux error."
    return flux, flux_err, filter_id


@np.vectorize
def toFlux_R(
    band: str,
    system: str,
    telescope: str,
    flux_x: float,
    flux_err_x: float = 0,
    beta: float = 1,
    beta_err: float = 0
):

    try:
        lambda_R, *__ = calibration("R", system, telescope) # lambda_R in angstrom
        lambda_x, *__ = calibration(band, system, telescope)
    except KeyError:
        raise KeyError(f"Band '{band}' is not currently supported.")

    # convert flux in another band to R
    flux_R = flux_x * (lambda_x / lambda_R) ** (-beta)

    flux_err_R = flux_R * np.sqrt(
        (flux_err_x / flux_x) ** 2 +
        (beta_err * np.log(lambda_x / lambda_R)) ** 2
    )

    assert np.all(flux_R >= 0), "Error computing flux."
    assert np.all(flux_err_R >= 0), "Error computing flux error."
    return flux_R, flux_err_R


# main conversion function to call
def convertGRB(
    GRB: str,
    filename: str = None,
    author: str = "",
    ra: str = "",
    dec: str = "",
    beta: float = 0,
    beta_err: float = 0,
    ftol = None,
    debug: bool = False,
):
    # make sure we have dust maps downloaded for calculating galactic extinction
    _check_dust_maps()

    # assign column names and datatypes before importing
    dtype = {
        "time_sec": str,
        "mag": np.float64,
        "mag_err": np.float64,
        "band": str,
        "system": str,
        "telescope": str,
        "extcorr": str,
        "source": str,
    }
    names = list(dtype.keys())
    #if use_nick:
    #    names.insert(0, "nickname")  # add nickname column
    #    dtype["nickname"] = str  # add nickname type

    """ will import data using the following headers
    IF: use_nick = False
    | date | time_sec | mag | mag_err | band |
    OR
    IF: use_nick = True
    | nickname | date_sec | exp | mag | mag_err | band |
    """

    # try to import magnitude table to convert
    try:
        global directory
        mag_table = read_csv(
            filename,
            delimiter=r"\t+|\s+",
            names=names,
            dtype=dtype,
            index_col=None,
            header=0,
            engine="python",
            encoding="ISO-8859-1"
        )
    except ValueError as error:
        raise error
    except IndexError:
        raise ImportError(message=f"Couldn't find GRB table at {filename}.")

    converted = {k: [] for k in ("time_sec", "flux", "flux_err", "band_init", "band_norm", "source")}

    if debug:
        converted_debug = {
            k: []
            for k in (
                "time_sec",
                "flux",
                "flux_err",
                #"logF",
                #"logT",
                #"mag",
                #"mag_err",
                "band_norm",
                "band_init",
                "band_match",
                "telescope",
                "system",
                "extcorr",
                "source"
            )
        }

    for __, row in mag_table.iterrows():

        time_sec = row["time_sec"]
        mag = row["mag"]
        mag_err = row["mag_err"]
        band = row["band"]
        system = row["system"]
        telescope = row["telescope"]
        extcorr = row["extcorr"]
        source = row["source"]

        # attempt to convert a single magnitude to flux given a band, position in the sky, mag_err, and photon index
        try:
            flux, flux_err, filter_id = toFlux(
                band=band,
                system=system,
                telescope=telescope,
                extcorr=extcorr,
                mag=mag,
                mag_err=mag_err,
                beta=beta,
                beta_err=beta_err,
                grb=GRB,
                ra=ra,
                dec=dec
            )
        except KeyError as error:
            print(error)
            continue

        if flux_err == 0 or (ftol is not None and flux_err/flux > ftol):
            continue

        converted["time_sec"].append(time_sec)
        converted["flux"].append(flux)
        converted["flux_err"].append(flux_err)
        converted["band_init"].append(band)
        converted["band_norm"].append('R')
        converted["source"].append(source)

        # verbosity if you want it
        if debug:
            #logF = np.log10(flux)
            #logT = np.log10(time_sec)
            converted_debug["time_sec"].append(time_sec)
            converted_debug["flux"].append(flux)
            converted_debug["flux_err"].append(flux_err)
            #converted_debug["logF"].append(logF)
            #converted_debug["logT"].append(logT)
            #converted_debug["mag"].append(mag)
            #converted_debug["mag_err"].append(mag_err)
            converted_debug["band_norm"].append('R')
            converted_debug["band_init"].append(band)
            converted_debug["band_match"].append(filter_id)
            converted_debug["telescope"].append(telescope)
            converted_debug["system"].append(system)
            converted_debug["extcorr"].append(extcorr)
            converted_debug["source"].append(source)

    # after converting everything, go from dictionary -> DataFrame -> csv!
    if not debug:
        save_path = os.path.join(os.path.dirname(filename), f"{GRB}_{author}_converted_flux.txt")
        DataFrame.from_dict(converted).to_csv(save_path, sep="\t", index=False)
    else:
        save_path = os.path.join(
            os.path.dirname(filename), f"{GRB}_{author}_converted_flux_DEBUG.txt"
        )
        DataFrame.from_dict(converted_debug).to_csv(save_path, sep="\t", index=False)


def normaliseGRB_to_R(
    GRB: str,
    author: str = "",
    filename: str = None,
    beta: float = 0,
    beta_err: float = 0,
    ftol = None,
):

    # assign column names and datatypes before importing
    dtype = {
        "time_sec": str,
        "flux_x": np.float64,
        "flux_err_x": np.float64,
        "band": str,
        "source": str,
    }
    names = list(dtype.keys())

    # try to import magnitude table to convert
    try:
        global directory
        flux_table = read_csv(
            filename,
            delimiter=r"\t+|\s+",
            names=names,
            dtype=dtype,
            index_col=None,
            header=None,
            engine="python",
        )
    except ValueError as error:
        raise error
    except IndexError:
        raise ImportError(message=f"Couldn't find GRB table at {filename}.")

    converted = {k: [] for k in ("time_sec", "flux_R", "flux_err_R", "band_init", "band_norm", "source")}

    for __, row in flux_table.iterrows():

        time_sec = row["time_sec"]
        flux_x = row["flux_x"]
        flux_err_x = row["flux_err_x"]
        band = row["band"]
        source = row["source"]

        # attempt to convert a single magnitude to flux given a band, position in the sky, mag_err, and photon index
        try:
            flux_R, flux_err_R = toFlux_R(
                band,
                flux_x,
                flux_err_x,
                beta,
                beta_err
            )
        except KeyError as error:
            print(error)
            continue

        if ftol is not None and flux_err_R/flux_R > ftol:
            continue

        converted["time_sec"].append(time_sec)
        converted["flux"].append(flux_R)
        converted["flux_err"].append(flux_err_R)
        converted["band_init"].append(band)
        converted["band_norm"].append('R')
        converted["source"].append(source)

        save_path = os.path.join(
            os.path.dirname(filename), f"{GRB}_{author}_normalised_flux.txt"
        )
        DataFrame.from_dict(converted).to_csv(save_path, sep="\t", index=False)


def convertGRB_f2F(
    GRB: str,
    filename: str,
    author: str = "",
    scale: str = "mu",
    ra: str = "",
    dec: str = "",
    beta: float = 0,
    beta_err: float = 0,
    extcorr: bool = False,
    ftol = None,
    debug: bool = False,
):

    # assign column names and datatypes before importing
    dtype = {
        "time_sec": str,
        "fnu_x": np.float64,
        "fnu_err_x": np.float64,
        "band": str,
        "source": str,
    }
    names = list(dtype.keys())

    # try to import magnitude table to convert
    try:
        global directory
        fnu_table = read_csv(
            filename,
            delimiter=r"\t+|\s+",
            names=names,
            dtype=dtype,
            index_col=None,
            header=None,
            engine="python",
        )
    except ValueError as error:
        raise error
    except IndexError:
        raise ImportError(message=f"Couldn't find GRB table at {filename}.")

    if extcorr == True:
        ext = 0
    if extcorr == False:
        ext = None

    converted = {k: [] for k in ("time_sec", "flux", "flux_err", "band_init", "band_norm", "source")}

    for __, row in fnu_table.iterrows():

        time_sec = row["time_sec"]
        fnu_x = row["fnu_x"]
        fnu_err_x = row["fnu_err_x"]
        band = row["band"]
        source = row["source"]

        # attempt to convert a single magnitude to flux given a band, position in the sky, mag_err, and photon index
        try:
            mag, mag_err = toMag(
                fnu_x,
                fnu_err_x,
            )
            flux, flux_err = toFlux(
                band,
                mag,
                mag_err,
                beta,
                beta_err,
                A_b=ext,
                grb=GRB,
                ra=ra,
                dec=dec,
            )
        except KeyError as error:
            print(error)
            continue

        if ftol is not None and flux_err/flux> ftol:
            continue

        converted["time_sec"].append(time_sec)
        converted["flux"].append(flux)
        converted["flux_err"].append(flux_err)
        converted["band_init"].append(band)
        converted["band_norm"].append('R')
        converted["source"].append(source)

        save_path = os.path.join(
            os.path.dirname(filename), f"{GRB}_{author}_converted_flux.txt"
        )
        DataFrame.from_dict(converted).to_csv(save_path, sep="\t", index=False)


# simple checker that downloads the SFD dust map if it's not already there
def _check_dust_maps():
    if not os.path.exists(os.path.join(data_dir(), "sfd")):
        from .sfd import sfd
        sfd.fetch()


# sets directory to the current working directory, or whatever folder you're currently in
directory = os.getcwd()
