# standard libs
import os

# third party libs
import numpy as np
import pandas as pd


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


# Removed in version 0.2.0 since the format is unified
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


def read_data(
    path: str, 
    data_space='lin'
):
    """
    Reads data, sorts by time, excludes negative time, converts data_space.

    """
    
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
    
    try:
        data = pd.read_csv(path, sep=r"\t+|\s+", 
                    dtype=dtype,
                    names=list(dtype.keys()),
                    header=0, 
                    index_col=None,
                    comment='#',
                    engine="python").sort_values(by=['time_sec'])
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

        data = pd.read_csv(path, sep=r"\t+|\s+", 
                    dtype=dtype,
                    names=list(dtype.keys()),
                    header=0, 
                    index_col=None,
                    engine="python").sort_values(by=['time_sec'])

    if data_space=='log':
        try:
            data['time_sec'] = np.log10(data['time_sec'])
        except Exception as e:
            print("Issue with logT calculation. Check if negative values are given for time.\nError:", e)
    elif data_space=='lin':
        pass
    else:
        raise Exception("Dataspace could be either 'log' or 'lin")

    data = data.reset_index(drop=True)

    data = data[data['time_sec']>0]

    return pd.DataFrame(data)


def readin(directory="."):
    import glob2

    return np.asarray(glob2.glob(directory + "/*.txt"))

# converting the data here in the required format for color evolution analysis
def convert_data(data):

    data = list(data) # reading the data as a list

    for i, band in enumerate(data):
        if band.lower() in ['clear', 'unfiltered', 'lum']:  # here it is checking for existence of the bands in lower case for three filters 'clear', 'unfiltered', 'lum'
            band == band.lower()  # here it passes the lower case bands

    #if appx_bands:  # here we reassigns the bands (reapproximation of the bands), e.g. u' reaasigned to u,.....
    for i, band in enumerate(data):
        if band=="u'":
            data[i]="u"
        if band=="g'":
            data[i]="g"
        if band=="r'":
            data[i]="r"
        if band=="i'":
            data[i]="i"
        if band=="z'":
            data[i]="z"
        if band.upper()=="BJ":
            data[i]="B"
        if band.upper()=="VJ":
            data[i]="V"
        if band.upper()=="UJ":
            data[i]="U"
        if band.upper()=="RM":
            data[i]="R"
        if band.upper()=="BM":
            data[i]="B"
        if band.upper()=="UM":
            data[i]="U"
        if band.upper()=="JS":
            data[i]="J"
        if band.upper()=="KS":
            data[i]="K"
        if band.upper()=="K'":
            data[i]="K"
        if band.upper()=="KP":
            data[i]="K"
        if band.upper()=="CR":
            data[i]="R"
        if band.upper()=="CLEAR":
            data[i]="Clear"
        if band.upper()=="N":
            data[i]="Unfiltered"
        if band.upper()=="UNFILTERED":
            data[i]="Unfiltered"

    bands = data

    return bands