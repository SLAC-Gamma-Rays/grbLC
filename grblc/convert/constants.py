import pandas as pd
import os

flux_densities = {
    # from Bessell et al. (1998) // http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
    # * in angstrom (Å) | erg cm-2 s-1 Hz-1 | host extinction band
    "U": [3600, 1.79e-20, "Landolt U"],
    "B": [4380, 4.063e-20, "Landolt B"],
    "V": [5450, 3.636e-20, "Landolt V"],
    "R": [6410, 3.064e-20, "Landolt R"],
    "I": [7980, 2.416e-20, "Landolt I"],
    "J": [12200, 1.589e-20, "UKIRT J"],  # | need to check this bc I'm getting bad vals
    "H": [16300, 1.021e-20, "UKIRT H"],  # |
    "K": [21900, 0.64e-20, "UKIRT K"],  #  |
    # SDSS filters on the AB system
    # from Fukugita et al. (1996) // http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
    # * in angstrom (Å) | erg- cm-2 s-1 Hz-1
    "u": [3560, 3631e-23, "SDSS u"],  # maybe also need to check these out
    "g": [4830, 3631e-23, "SDSS g"],
    "r": [6260, 3631e-23, "SDSS r"],
    "i": [7670, 3631e-23, "SDSS i"],
    "z": [9100, 3631e-23, "SDSS z"],
    # Swift UVOT filters
    # from Poole et al. (2008) // https://academic.oup.com/mnras/article/383/2/627/993537
    # * in angstrom | erg cm-2 s-1 Å-1
    "u_swift": [3465, 1.5e-16, "Swift u"],
    "b_swift": [4392, 1.32e-16, "Swift b"],
    "v_swift": [5468, 2.61e-16, "Swift v"],
    "uvw1_swift": [2600, None, "Swift uvw1"],
    "uvw2_swift": [1928, None, "Swift uvw2"],
    "uvm2_swift": [2246, None, "Swift uvm2"],
    # "uvw1": [2634, 4.00e-16],
    # "uvw2": [2030, 6.2e-16],
    # "uvm2": [2231, 8.5e-16],
    # "white": [3471, 3.7e-17], // skip white!
    # Additional various bands
    # from https://coolwiki.ipac.caltech.edu/index.php/Central_wavelengths_and_zero_points
    # * in angstrom (Å) | erg cm-2 s-1 Hz-1
    "Rc": [6550, 3080e-23, "CTIO R"],  # Cousins R, not Johnson R!
    "Ic": [7996, 2432.84e-23, "CTIO I"],  # Cousins R, not Johnson R!
    "Ks": [16620, 666.7e-23, "UKIRT K"],  # K sharp, not Johnson K!
    "Z": [8817, 2232e-23, "SDSS z"],
    "Y": [10305, 2026e-23, "DES Y"],
}
table_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "SF11_conversions.txt")
ebv2A_b_df = pd.read_table(table_path, comment="#", index_col=0)
