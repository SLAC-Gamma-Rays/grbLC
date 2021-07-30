flux_densities = {
    # from Bessell et al. (1998) // http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
    # * in angstrom (Å) | erg cm-2 s-1 Hz-1
    "U": [3600, 1.79e-20],
    "B": [4380, 4.063e-20],
    "V": [5450, 3.636e-20],
    "R": [6410, 3.064e-20],
    "I": [7980, 2.416e-20],
    "J": [12200, 1.589e-20],  # | need to check this bc I'm getting bad vals
    "H": [16300, 1.021e-20],  # |
    "K": [21900, 0.64e-20],  #  |
    # SDSS filters on the AB system
    # from Fukugita et al. (1996) // http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
    # * in angstrom (Å) | erg- cm-2 s-1 Hz-1
    "u": [3560, 3631e-23],  # maybe also need to check these out
    "g": [4830, 3631e-23],
    "r": [6260, 3631e-23],
    "i": [7670, 3631e-23],
    "z": [9100, 3631e-23],
    # Swift UVOT filters
    # from Poole et al. (2008) // https://academic.oup.com/mnras/article/383/2/627/993537
    # * in angstrom | erg cm-2 s-1 Å-1
    "u_swift": [3465, None],
    "b_swift": [4392, None],
    "v_swift": [5468, None],
    "uvw1_swift": [2600, None],
    "uvw2_swift": [1928, None],
    "uvm2_swift": [2246, None],
    "uvw1": [2634, 4.00e-16],
    "uvw2": [2030, 6.2e-16],
    "uvm2": [2231, 8.5e-16],
    # "white": [3471, 3.7e-17], // skip white!
    # Additional various bands
    # from https://coolwiki.ipac.caltech.edu/index.php/Central_wavelengths_and_zero_points
    # * in angstrom (Å) | erg cm-2 s-1 Hz-1
    "Rc": [6550, 3080e-23],  # Cousins R, not Johnson R!
    "Ic": [7996, 2432.84e-23],  # Cousins R, not Johnson R!
    "Ks": [16620, 666.7e-23],  # K sharp, not Johnson K!
    "Z": [8817, 2232e-23],
    "Y": [10305, 2026e-23],
}
