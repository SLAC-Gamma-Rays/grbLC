from astropy import units as u, constants as const
import numpy as np


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


def magToFlux(mag, band):
    # convert u', g', r', i', z' to u, g, r, i, z
    band = band.strip("'")
    flux_densities = {
        # from Bessell et al. (1998) // http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
        # * in angstrom (Å) | erg cm-2 s-1 Hz-1
        "U": [3600, 1.79e-20],
        "B": [4380, 4.063e-20],
        "V": [5450, 3.636e-20],
        "R": [6410, 3.064e-20],
        "I": [7980, 2.416e-20],
        "J": [12200, 1.589e-20],
        "H": [16300, 1.021e-20],
        "K": [21900, 0.64e-20],
        # SDSS filters on the AB system
        # from Fukugita et al. (1996) // http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
        # * in angstrom (Å) | erg cm-2 s-1 Hz-1
        "u": [3560, 3631e-23],
        "g": [4830, 3631e-23],
        "r": [6260, 3631e-23],
        "i": [7670, 3631e-23],
        "z": [9100, 3631e-23],
        # Swift UVOT filters
        # from Poole et al. (2008) // https://academic.oup.com/mnras/article/383/2/627/993537
        # * in angstrom | erg cm-2 s-1 Å-1
        "uvw1": [2634, 4.00e-16],
        "uvw2": [2030, 6.2e-16],
        "uvm2": [2231, 8.5e-16],
        "white": [3471, 3.7e-17],
        # Additional various bands
        # from https://coolwiki.ipac.caltech.edu/index.php/Central_wavelengths_and_zero_points
        # * in angstrom (Å) | erg cm-2 s-1 Hz-1
        "Rc": [6550, 3080e-23],  # Cousins R, not Johnson R!
        "Ic": [7996, 2432.84e-23],  # Cousins R, not Johnson R!
        "Ks": [16620, 666.7e-23],  # K sharp, not Johnson K!
        "Z": [8817, 2232e-23],
        "Y": [10305, 2026e-23],
    }

    lam, f_lam_or_nu = flux_densities[band]
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
    fluxerr = magerr * flux * np.log(10 ** (2.0 / 5)) * (10 ** (-0.4 * mag))
    assert fluxerr > 0, "Error computing flux error."
    return fluxerr
