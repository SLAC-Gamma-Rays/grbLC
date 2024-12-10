# standard libs
import os

# third party libs
import numpy as np
import pandas as pd

# can be removed for pandas 3.0
pd.options.mode.copy_on_write = True


# Removed in version 0.2.0 since the format is unified

# def isfloat(value):
#     try:
#         float(value)
#         return True
#     except ValueError:
#         return False


# def check_header(path, n=None, debug=False, more_than_one_row=False):
#     """
#     Returns what line a header is at
#     """
#     with open(path) as f:
#         lines = f.readlines()

#     if isinstance(n, int) and n > len(lines):
#         raise Exception(f"Error in file ({path})! Something wrong "
#                          "is going on here and I can't find the "
#                          "header row for this file")

#     try:
#         # attempt importing the datafile with the header "n"
#         data = pd.read_csv(path, delimiter=r"\t+|\s+", header=n, engine="python")
#     except pd.errors.ParserError as pe:
#         if debug:
#             print("ParserError:", pe)

#         # if fail, recursively try again with the next row as the header
#         n = -1 if n is None else n
#         return check_header(path, n=n + 1, more_than_one_row=True)
#     except pd.errors.EmptyDataError:

#         if more_than_one_row:
#             return None
#         else:
#             print(os.path.split(path)[-1], "is empty?")
#             return -1

#     h = data.columns


#     # todo: if header is Int64Index, check the 2nd row (i.e. first row of data for the not isfloat)
#     # ... so maybe change the h in [not isfloat(x) for x in h] to the second row???
#     if isinstance(h, type(pd.Index([], dtype=int))) or sum(isfloat(x) for x in h) >= 0.3 * len(h) // 1:
#         if debug:
#             print("Some are floats...")

#         # recursively try again with the next row as the header
#         n = -1 if n is None else n
#         return check_header(path, n=n + 1, more_than_one_row=True)
#     else:
#         return n  # <-- the final stop in our recursion journey

# def readin(directory="."):
#     import glob2

#     return np.asarray(glob2.glob(directory + "/*.txt"))


def read_data(
    path: str = None, 
    df: pd.DataFrame = None,
    data_space: str ='lin',
    limiting_mags: bool = False
):
    """
    Reads data, sorts by time, excludes negative time, converts data_space.

    """

    assert path or df is not None, "Provide either the dataframe or the path"
    
    dtype = {
        "time_sec": np.float64,
        "mag": np.float64,
        "mag_err": np.float64,
        "band": str,
        "system": str,
        "telescope": str,
        "extcorr": str,
        "source": str,
        "flag": str
        }
    
    if path:
    
        try:
            df = pd.read_csv(path, 
                            sep='\t', 
                            dtype=dtype,
                            names=list(dtype.keys()),
                            header=0, 
                            index_col=None,
                            engine="python"
                            ).sort_values(by=['time_sec'])
            
        except:
            dtype = {
            "time_sec": np.float64,
            "mag": np.float64,
            "mag_err": np.float64,
            "band": str,
            "system": str,
            "telescope": str,
            "extcorr": str,
            "source": str
            }
            
            df = pd.read_csv(path, 
                            sep='\t', 
                            dtype=dtype,
                            names=list(dtype.keys()),
                            header=0, 
                            index_col=None,
                            engine="python"
                            ).sort_values(by=['time_sec'])

    # asserting those data points only which does not have limiting nagnitude
    if not limiting_mags:
        df = df[df['mag_err'] != 0] 
        assert len(df)!=0, "Only limiting magnitudes present."

    if data_space=='log':
        try:
            df['time_sec'] = np.log10(df['time_sec'])
        except Exception as e:
            print("Issue with logT calculation. Check if negative values are given for time.\nError:", e)
            
    elif data_space=='lin':
        pass

    else:
        raise Exception("Dataspace could be either 'log' or 'lin")

    df = df.reset_index(drop=True)

    df = df[df['time_sec']>0]

    return pd.DataFrame(df)


def _format_bands(data):
    """
    Function to approximate bands for conversion, matching it with the filters.txt
    """

    data = list(data) # reading the data as a list

    # here we format the bands to match filters.txt
    assume_R = ['-', '—', '|', '\\', '/', '35', '145',\
            'P-', 'P—', 'P|', 'P\\', 'P/', 'polarised', 'polarized',\
            'unfiltered', 'clear', 'CR', 'lum', 'IR-cut', 'TR-rgb', 'RM'] 
    # Excluded P, there is a P filter. Excluded N, present in list

    # Mapping replacements
    band_format = {
        'UJ': 'U',
        'BJ': 'B',
        'VJ': 'V',
        'CV': 'V',  # CV clear calibrated as V
        'UB': 'U',
        'UM': 'U',
        'BM': 'B',
        'RM': 'R',
        'KS': 'Ks',
        'IC': 'Ic',
        'RC': 'Rc'
    }

    sep = ['-','_','.',',']
      
    for i, band in enumerate(data):
        # Check for any separators in the band
        # in cases like Gunn-R, the bandpass needs to be separated from system
        #  by default bandpass is None
        bandpass = None

        for char in sep:
            if char in band:
                bandpass, band = band.split(char, 1)  # split only once
                if len(band) > len(bandpass):
                    if any(band.lower() == k.lower() for k in assume_R):
                        break
                else:
                    band, bandpass = band.split(char in band for char in sep) 
                    break
    
        # Check if band should be set to 'Rc'
        if any(band.lower() == k.lower() for k in assume_R):
            data[i] = 'Rc'

        else:
            if band in band_format:
                data[i] = band_format[band]
                
            if "'" in band:
                data[i] = band.replace("'", "p")

            if "*" in band:
                data[i] = band.replace("*", "p")

    return data


def _appx_bands(data):
    """
    Function to approximate bands for color evolution
    """
    
    data = list(data) # reading the data as a list

    # Define the mapping of bands
    band_appx = {
        "up": "u",
        "gp": "g",
        "rp": "r",
        "ip": "i",
        "zp": "z",
        "Js": "J",
        "Ks": "K",
        "Kp": "K"
    }

    for i, band in enumerate(data):
        if band in band_appx:
                data[i] = band_appx[band]

    return data
