# Author: Nathaniel R. Butler


from numpy import loadtxt,log,hstack
from scipy.interpolate import interp1d

import os

pfile=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pei_extinct.txt')
l1,x1,l2,x2,l3,x3 = loadtxt(pfile,unpack=True)

def pei_av(lam,A_V=1.0,gal=3,R_V=0.0):
    """
      lam in units of Angstroms
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