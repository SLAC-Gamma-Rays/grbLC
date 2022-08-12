import os

import pandas as pd

photometry = {
    # The usual Landolt UBVRI & UKIRT JHK system
    # from Bessell et al. (1998)http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
    # * in angstrom (Å) | erg cm-2 s-1 Hz-1 | host extinction band
    "U": [3600, 1.79e-20, "Landolt U"],
    "B": [4380, 4.063e-20, "Landolt B"],
    "V": [5450, 3.636e-20, "Landolt V"],
    "R": [6410, 3.064e-20, "Landolt R"],
    "I": [7980, 2.416e-20, "Landolt I"],
    "J": [12200, 1.589e-20, "UKIRT J"],
    "H": [16300, 1.021e-20, "UKIRT H"],
    "K": [21900, 0.64e-20, "UKIRT K"],
    # SDSS filters on the AB system
    # from Fukugita et al. (1996) // http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
    "u": [3560, 3631e-23, "SDSS u"],
    "g": [4830, 3631e-23, "SDSS g"],
    "r": [6260, 3631e-23, "SDSS r"],
    "i": [7670, 3631e-23, "SDSS i"],
    "z": [9100, 3631e-23, "SDSS z"],
    # Swift UVOT filters
    # from SVO Filter Profile Service (Rodrigo, C., Solano, E., 2020) //
    # http://svo2.cab.inta-csic.es/theory/fps3/pavosa.php?oby=id&fid=Swift/UVOT.UVM2#Swift/UVOT.UVM2
    "u_swift": [3520, 1480e-23, "Swift u"],
    "b_swift": [4346, 4060e-23, "Swift b"],
    "v_swift": [5411, 3636e-23, "Swift v"],
    "uvw1_swift": [2684, 981e-23, "Swift uvw1"],
    "uvw2_swift": [2086, 760e-23, "Swift uvw2"],
    "uvm2_swift": [2246, 770e-23, "Swift uvm2"],
    # Additional various bands
    # from https://coolwiki.ipac.caltech.edu/index.php/Central_wavelengths_and_zero_points
    "Rc": [6550, 3080e-23, "CTIO R"],  # Cousins R, not Johnson R!
    "Ic": [7996, 2432.84e-23, "CTIO I"],  # Cousins R, not Johnson R!
    "Ks": [16620, 666.7e-23, "UKIRT K"],  # K sharp, not Johnson K!
    "Y": [10305, 2026e-23, "DES Y"],
    # GROND/SDSS primed (air) wavelengths
    # from https://articles.adsabs.harvard.edu/pdf/1996AJ....111.1748F #corrected for atmospheric extinction
    "u'": [3580, 3631e-23, "SDSS u"],
    "g'": [4754, 3631e-23, "SDSS g"],
    "r'": [6204, 3631e-23, "SDSS r"],
    "i'": [7698, 3631e-23, "SDSS i"],
    "z'": [9665, 3631e-23, "SDSS z"],
    # Not supported: v, h, y, q, Z, clear, n/a, none, unfiltered white points, HST
    "v": [5450, 3.636e-20, "Landolt V"],
    "h": [16300, 1.021e-20, "UKIRT H"],
    "y":[10305, 2026e-23, "DES Y"],
    "Z": [9665, 3631e-23, "SDSS z"],
}

table_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "SF11_conversions.txt"
)
ebv2A_b_df = pd.read_table(table_path, comment="#", index_col=0)

__all__ = ["photometry", "ebv2A_b_df"]
