import re
import numpy as np

# Module imports files with telescopes and filter information
from .constants import *

# count functions
def _count(str1, str2): 
    """
    Counts how much percentage two strings match.

    """
    
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

# from collections import Counter

# def _count(str1, str2): 
#     """
#     Calculate the percentage of characters in str1 that are present in str2.

#     """
    
#     # ensure str1 is the longer or equal length string
#     if len(str2) > len(str1):
#         str1, str2 = str2, str1

#     # count character frequencies
#     count1 = Counter(str1)
#     count2 = Counter(str2)

#     match_count = sum(min(count1[char], count2[char]) for char in count2)

#     match_percent = (match_count / len(str1) * 100) if str1 else 0

#     return match_percent


def _stripcount(str1, str2): 
    """
    Counts how much percentage two strings match excluding special characters.

    """

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


def _count_hst(str1, str2):
    """
    count() and stripcount() are not sequential leading to strings like F606W and F066W having 100% match. 
    This function place emphasis on the numerical order which is important for HST-like filters.

    """
    
    wave1 = re.findall("\d+", str1)
    wave2 = re.findall("\d+", str2)
    if wave1 == wave2:
        str1="".join(re.findall("[a-zA-Z]+", str1))
        str2="".join(re.findall("[a-zA-Z]+", str2))
        match = _count(str1, str2)
    else:
        match = 0

    return match


def calibration(band: str, 
                telescope: str):
    """
    Parameters:
    -----------
    - band:
    - telescope: 

    Returns:
    --------
    - lam: float: Wavelength in units of Angstrom 
    - shift_toAB: float: Factor to shift to AB system
    - coeff: float: A_V for R_V = 3.1 from extinction laws
    - matched_filter: str: ID of filter of the telescope in 'filters.txt', 
    - coeff_source: source of A_V. Either Schlafly & Finkbeiner (2011) or ADPS (2003).

    Raises:
    -------
    - KeyError: If no matching filters found.

    """

    # function to match filter and telescope info to get band wavelength, zeropoint and extinction correction
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
        filters.loc[id, 'match_fil'] = _count_hst(grblc_fil, filter)
      else:
        if grblc_fil == filter:
          filters.loc[id, 'match_fil'] = 100
        else:
          filters.loc[id, 'match_fil'] = _count(grblc_fil, filter)

    matched_fil = filters[filters['match_fil'] == 100].copy()

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

        match_tel =  _count(grblc_tel.casefold(), telescope.casefold())
        match_ins =  _count(grblc_ins.casefold(), instrument.casefold())

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

    matched_filter = probablefilters[0]
    
    try:
      lam = float(matched_fil.loc[matched_filter,'lambda_eff'])
    except TypeError:
      lam = float(matched_fil.loc[matched_filter,'lambda_eff'][0])
    
    lam_round = round(lam, -1)

    shift_toAB = matched_fil.loc[matched_filter,'mag_fromVega_toAB']

    try:
        coeff = schafly.loc[lam_round, '3.1']
        coeff_source = "Schafly+11"
    except KeyError:
        coeff = adps.loc[lam_round, 'coeff']
        coeff_source = "APDS+02"
    
    return lam, shift_toAB, coeff, matched_filter, coeff_source