from numpy import loadtxt,log,hstack
from scipy.interpolate import interp1d

import os

def ebv(grb: str, ra="", dec=""):
    r"""A function that returns the galactic extinction correction
       at a given position for a given band.

                            This takes data from Schlegel, Finkbeiner & Davis (1998) in the form
                            of the SFD dust map, and is queried using the dustmaps python package.
                            Updated coefficient conversion values for the SFD is taken from Schlafly & Finkbeiner (2011)
                            and is found in SF11_conversions.txt.

        Author: Sam Young

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

pfile=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pei_extinct.txt')
l1,x1,l2,x2,l3,x3 = loadtxt(pfile,unpack=True)

def pei_av(lam,A_V=1.0,gal=3,R_V=0.0):
    """
      lam in units of Angstroms

      # Author: Nathaniel R. Butler
    """
    if (gal==1):
        # Milky Way
        if (R_V==0): R_V=3.08
        ll=1.*l1[::-1]
        xx=1.*x1[::-1]
    elif (gal==2):
        # LMC
        if (R_V==0): R_V=3.16
        ll=1.*l2[::-1]
        xx=1.*x2[::-1]
    else:
        # SMC, gal=3
        if (R_V==0): R_V=2.93
        ll=1.*l3[::-1]
        xx=1.*x3[::-1]


    ll_minus = 1.e4
    xx_minus = (xx[1]-xx[0])/log(ll[1]/ll[0])*log(ll_minus/ll[0]) + xx[0]
    ll_plus = 0.1
    xx_plus = (xx[-1]-xx[-2])/log(ll[-1]/ll[-2])*log(ll_plus/ll[-2]) + xx[-2]

    xx = hstack((xx_minus,xx,xx_plus))
    ll = hstack((ll_minus,ll,ll_plus))

    # in angstroms
    lambda0 = 1.e4/ll

    A_lam = A_V*( 1+xx/R_V )
    res = interp1d(log(lambda0),A_lam,bounds_error=False,fill_value=0)

    return res(log(lam))