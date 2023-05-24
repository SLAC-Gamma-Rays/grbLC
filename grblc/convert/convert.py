import os
import re

import numpy as np
import pandas as pd

from .constants import *
from .sfd import data_dir

path1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "filters.txt")
path2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "reddening.txt")
path3 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SF11_conversions.txt")


# count functions

def count(str1, str2): 
    '''
    Counts how much percentage two strings match. 
    
    Input:
    ------
    str1, str2: strings
    
    Output:
    ------
    match: percentage match
    '''
    
    str1 = re.sub(r"[\u02BB\u02BC\u066C\u2018\u201A\u275B\u275C\u0027\u02B9\u02BB\u02BC\u02BE\u02C8\u02EE\u0301\u0313\u0315\u055A\u05F3\u07F4\u07F5\u1FBF\u2018\u2019\u2032\uA78C\uFF07]", "p", str1)
    str2 = re.sub(r"[\u02BB\u02BC\u066C\u2018\u201A\u275B\u275C\u0027\u02B9\u02BB\u02BC\u02BE\u02C8\u02EE\u0301\u0313\u0315\u055A\u05F3\u07F4\u07F5\u1FBF\u2018\u2019\u2032\uA78C\uFF07]", "p", str2)
    
    diff = len(str1) - len(str2)
    if diff < 0:
      temp = str1
      str1 = str2
      str2 = temp
    else:
      pass

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
    '''
    Counts how much percentage two strings match excluding special characters 
    
    Input:
    ------
    str1, str2: strings
    
    Output:
    ------
    match: percentage match excluding special characters
    '''

    str1 = ''.join(i for i in str1 if i.isalnum()).lower() ## removes special characters
    str2 = ''.join(i for i in str2 if i.isalnum()).lower()

    diff = len(str1) - len(str2)
    if diff < 0:
      str1 = str1 + ("-"*diff)
    elif diff > 0:
      str2 = str2 + ("-"*diff)
    else:
      pass
    
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


def count_hst(str1, str2):
    '''
    count() and stripcount() are not sequential leading to strings like F606W and F066W having 100% match. 
    This function place emphasis on the numerical order which is important for HST-like filters.
    
    Input:
    ------
    str1, str2: strings
    
    Output:
    ------
    match: percentage match excluding special characters
    '''
    wave1 = re.findall("\d+", str1)
    wave2 = re.findall("\d+", str2)
    if wave1 == wave2:
        str1="".join(re.findall("[a-zA-Z]+", str1))
        str2="".join(re.findall("[a-zA-Z]+", str2))
        match = count(str1, str2)
    else:
        match = 0

    return match


# function to match filter and telescope info to get band wavelength, zeropoint and extinction correction

def calibration(band: str, telescope: str):

    ## Step 1: initialising data input

    ## finding the bandpass and filter in data
    
    sep = ['-','_','.',',']

    j = 0
    for i in sep:
        if i in band:
            bandpass, filter = band.split(i)
            if filter.lower in ['unfiltered', 'clear']:
                filter = 'clear'
            else:
                if len(filter) > len(bandpass):
                    filter, bandpass = band.split(i)
            j += 1
    if j == 0:
        filter = band
        bandpass = ''

    ## assumptions

    assume_R = ['-', '—', '|', '\\', '/', '35', '145', 'P-', 'P—', 'P|', 'P\\', 'P/', 'P', 'polarised', 'polarized', 'unfiltered', 'clear', 'CR', 'lum', 'N', 'IR-cut', 'TR-rgb', 'RM'] # q

    for i in assume_R:
      if filter.casefold() == i:
        filter = 'Rc' ## Gendre's suggestion
    
    if filter == 'CV': ## CV clear calibrated as V
      filter ='V'
    
    if filter == 'BJ':
      filter = 'B'

    if filter == 'VJ':
      filter = 'V'        
        
    if filter == 'UJ':
      filter = 'U'
    
    if filter == 'BM':
      filter = 'B'    
    
    if filter == 'UM':
      filter = 'U'  

    ## formatting
    
    if filter == 'KS':
      filter = 'Ks'
    
    if filter == 'IC':
      filter = 'Ic'
    
    if filter == 'RC':
      filter = 'Rc'
    
    if filter == 'UB':
      filter = 'U'
    
    if "'" in filter:
        filter = filter.replace("'","p")
        
    if "*" in filter:
        filter = filter.replace("*","p")
    
    if "+" in telescope:
      telescope=telescope.split("+")[0]
    
    if "'" in filter:
        filter = filter.replace("'","p")
        
    if "*" in filter:
        filter = filter.replace("*","p")
    
    if "+" in telescope:
      telescope=telescope.split("+")[0]

    ## finding the observatory, telescope, instrument in data

    observatory, telescope, instrument = telescope.split('/') 

    if instrument == 'CCD':
      instrument = 'None'

    if "." in telescope:
      telescope=telescope.replace(".", ",")


    ## Step 2: checking if the data filter exists in grblc filters

    for id in filters.index:       
      grblc_fil = str(id).split(".")[-1]
      if len(filter) >= 5:
        filters.loc[id, 'match_fil'] = count_hst(grblc_fil, filter)
      else:
        if grblc_fil == filter:
          filters.loc[id, 'match_fil'] = 100
        else:
          filters.loc[id, 'match_fil'] = count(grblc_fil, filter)

    matched_fil = filters.loc[filters['match_fil'] == 100]
    if len(matched_fil) == 0 and len(filter) <= 2:
      matched_fil = filters.loc[filters['match_fil'] >= 50]
      if len(matched_fil) == 0:
        raise KeyError(
            f"No matching filters.")
    elif len(matched_fil) == 0 and len(filter) > 2:
       raise KeyError(
            f"No matching filters.")
      
    ## Step 3: finding exact matches in observatory, telescope, instrument

    probablefilters = []
    
    for id in matched_fil.index:
      
      grblc_obs, grblc_tel, grblc_ins, *__ = str(id).split(".")
        
      if grblc_obs.casefold() == observatory.casefold():

        matched_fil.loc[id, 'match_obs'] = 'found'

        match_tel =  count(grblc_tel.casefold(), telescope.casefold())
        match_ins =  count(grblc_ins.casefold(), instrument.casefold())

        if match_tel == 100:
          matched_fil.loc[id, 'match_tel'] = 1
          if match_ins == 100:
            matched_fil.loc[id, 'match_ins'] = match_ins
          elif match_ins >= 50:
            matched_fil.loc[id, 'match_ins'] = match_ins
          else:
            matched_fil.loc[id, 'match_ins'] = match_ins

        elif match_tel >= 50:
            matched_fil.loc[id, 'match_tel'] = 2
            matched_fil.loc[id, 'match_ins'] = match_ins

        else:
          matched_fil.loc[id, 'match_tel'] = 3
          matched_fil.loc[id, 'match_ins'] = None

      else:
         matched_fil.loc[id, 'match_obs'] = None
         matched_fil.loc[id, 'match_tel'] = None
         matched_fil.loc[id, 'match_ins'] = None

    matched_obs =  matched_fil.loc[matched_fil['match_obs'] == 'found']
    if len(matched_obs) != 0:
      matched_tel = matched_obs.loc[matched_obs['match_tel'] == np.max(matched_obs['match_tel'])]
      matched_tel =  matched_tel.sort_values(by=['match_ins'])
      probablefilters = list(matched_tel.index)

    ## Step 4: in case of no match, resort to generics

    #standard = ['Johnson', 'Cousins', 'Bessel', 'Special', 'Tyson', 'SDSS', 'SuperSDSS', 'Stromgren', 'MKO', 'UKIRT', 'UKIDSS', 'PS1']

    if len(probablefilters) == 0:
      for id in matched_fil.index:

        grblc_obs = str(id).split(".")[0]

        if grblc_obs.casefold()=='average':
          matched_fil.loc[id, 'match_status'] = 1

        elif grblc_obs.casefold()=='generic':
          matched_fil.loc[id, 'match_status'] = 2
          
        elif grblc_obs.casefold()=='gcpd':
          matched_fil.loc[id, 'match_status'] = 3

        elif grblc_obs.casefold()=='catalog':
          matched_fil.loc[id, 'match_status'] = 4

        else:
          matched_fil.loc[id, 'match_status'] = None

      matched_gen =  matched_fil.sort_values(by=['match_status'], na_position='last')
      probablefilters = list(matched_gen.index)

    correctfilter = probablefilters[0]
    
    try:
      lam = float(matched_fil.loc[correctfilter,'lambda_eff'])
    except TypeError:
      lam = float(matched_fil.loc[correctfilter,'lambda_eff'][0])
    
    lam_round = round(lam, -1)

    shift_toAB = matched_fil.loc[correctfilter,'mag_fromVega_toAB']

    try:
        coeff = schafly.loc[lam_round, '3.1']
        coeff_source = "Schafly+11"
    except KeyError:
        coeff = adps.loc[lam_round, 'coeff']
        coeff_source = "APDS+02"
    
    return lam, shift_toAB, coeff, correctfilter, coeff_source


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


# conversion for mag in AB and corrected for extinction
@np.vectorize
def toAB(grb: str,
         ra: str,
         dec: str,
         band: str,
         system: str,
         extcorr: str,
         telescope: str,
         mag: float):

    try:
        lambda_x, shift_toAB, coeff, filter_id, coeff_source = calibration(band, telescope)
    except KeyError:
        raise KeyError(f"Band '{band}' of telescope '{telescope}' is not currently supported.")

    # get correction for galactic extinction to be added to magnitude if not already supplied
    if extcorr == 'y':
        mag_corr = mag    
    else:
        A_x = coeff * ebv(grb, ra, dec)
        mag_corr = mag - A_x

    if 'ab' or 'sdss' or 'panstarrs' in system.casefold:
        mag_corr = mag_corr
    else:
        if shift_toAB != 'notfound':
            mag_corr = mag_corr + float(shift_toAB)
                    
        else:
            for i in defaultshift_toAB.keys():
                if band==i:
                    shiftAB=defaultshift_toAB[i]
                    break
            mag_corr = mag_corr + float(shiftAB)

    return mag_corr, filter_id, coeff_source


# main conversion function to call
def correctGRB(
    grb: str,
    ra: str,
    dec: str,
    filename: str = None,
    save_in_folder: str = None,
    debug: bool = False,
):

    # assign column names and datatypes before importing
    dtype = {
        'time_sec': np.float64,
        'mag': np.float64,
        'mag_err': np.float64,
        'band': str,
        'system': str,
        'telescope': str,
        'extcorr': str,
        'source': str,
    }
    names = list(dtype.keys())

    # try to import magnitude table to convert
    try:
        global directory
        mag_table = pd.read_csv(
            filename,
            delimiter=r'\t+|\s+',
            names=names,
            dtype=dtype,
            index_col=None,
            header=0,
            engine='python',
            encoding='ISO-8859-1'
        )
    except ValueError as error:
        raise error
    except IndexError:
        raise ImportError(message=f"Couldn't find grb table at {filename}.")

    converted = {k: [] for k in ('time_sec', 'mag', 'mag_err', 'band', 'system', 'telescope', 'extcorr', 'source')}

    if debug:
        converted_debug = {
            k: []
            for k in (
                'time_sec',
                'mag_corr',
                'mag_init',
                'mag_err',
                'band',
                'band_match',
                'system_init',
                'system_final',
                'telescope',
                'extcorr',
                'coeff_source'
                'mag_source'
            )
        }

    for __, row in mag_table.iterrows():

        time_sec = row['time_sec']
        mag = row['mag']
        mag_err = row['mag_err']
        system = row['system']
        band = row['band']
        telescope = row['telescope']
        extcorr = row['extcorr']
        source = row['source']

        # attempt to convert a single magnitude to flux given a band, position in the sky, mag_err, and photon index
        try:
            mag_corr, filter_id, coeff_source = toAB(
                grb,
                ra,
                dec,
                band,
                system,
                extcorr,
                telescope,
                mag
                )
        except KeyError as error:
            print("KeyError")
            continue

        converted['time_sec'].append(time_sec)
        converted['mag'].append(mag_corr)
        converted['mag_err'].append(mag_err)
        converted['band'].append(band)
        converted['system'].append("AB")
        converted['telescope'].append(telescope)
        converted['extcorr'].append("y")
        converted['source'].append(source)

        # verbosity if you want it
        if debug:
            converted_debug['time_sec'].append(time_sec)
            converted_debug['mag_corr'].append(mag_corr)
            converted_debug['mag_init'].append(mag)
            converted_debug['mag_err'].append(mag_err)
            converted_debug['band_init'].append(band)
            converted_debug['band_match'].append(filter_id)
            converted_debug['system_init'].append(system)
            converted_debug['system_final'].append("AB")
            converted_debug['telescope'].append(system)
            converted_debug['extcorr'].append("y")
            converted_debug['coeff_source'].append(coeff_source)
            converted_debug['mag_source'].append(source)

    # after converting everything, go from dictionary -> DataFrame -> csv!
    if not debug:
        save_path = os.path.join(save_in_folder+'/', f"{grb}_magAB_extcorr.txt")
        pd.DataFrame.from_dict(converted).to_csv(save_path, sep='\t', index=False)
    else:
        save_path = os.path.join(save_in_folder+'/', f"{grb}_magAB_extcorr_DEBUG.txt")
        pd.DataFrame.from_dict(converted_debug).to_csv(save_path, sep='\t', index=False)


# simple checker that downloads the SFD dust map if it's not already there
def _check_dust_maps():
    if not os.path.exists(os.path.join(data_dir(), "sfd")):
        from .sfd import sfd
        sfd.fetch()


# sets directory to the current working directory, or whatever folder you're currently in
directory = os.getcwd()
