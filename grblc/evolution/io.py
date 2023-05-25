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


def check_header(path, n=None, debug=False, more_than_one_row=False):
    """
    This monstrosity returns what line a header is at
    """
    with open(path) as f:
        lines = f.readlines()

    if isinstance(n, int) and n > len(lines):
        raise Exception(f"Error in file ({path})! Something wrong "
                         "is going on here and I can't find the "
                         "header row for this file")

    try:
        # attempt importing the datafile with the header "n"
        data = pd.read_csv(path, delimiter=r"\t+|\s+", header=n, engine="python")
    except pd.errors.ParserError as pe:
        if debug:
            print("ParserError:", pe)

        # if fail, recursively try again with the next row as the header
        n = -1 if n is None else n
        return check_header(path, n=n + 1, more_than_one_row=True)
    except pd.errors.EmptyDataError:

        if more_than_one_row:
            return None
        else:
            print(os.path.split(path)[-1], "is empty?")
            return -1

    h = data.columns


    # todo: if header is Int64Index, check the 2nd row (i.e. first row of data for the not isfloat)
    # ... so maybe change the h in [not isfloat(x) for x in h] to the second row???
    if isinstance(h, type(pd.Index([], dtype=int))) or sum(isfloat(x) for x in h) >= 0.3 * len(h) // 1:
        if debug:
            print("Some are floats...")

        # recursively try again with the next row as the header
        n = -1 if n is None else n
        return check_header(path, n=n + 1, more_than_one_row=True)
    else:
        return n  # <-- the final stop in our recursion journey


def read_data(path, header=-999, approximate_band=False):

    header = check_header(path) if header==-999 else header

    if header == -1:
        return
    
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
                  header=header, 
                  index_col=None,
                  engine="python").sort_values(by=['time_sec'])## correct

    try:
        data['time_sec'] = np.log10(data['time_sec'])
    except Exception as e:
        print("Issue with logT calculation. Check if negative values are given for time.\nError:", e)

    data = data.reset_index(drop=True)

    if approximate_band:
        for j, filter in zip(data.index, data.band):
            if filter=="u'":
                data.loc[j, filter]="u"
            if filter=="g'":
                data.loc[j, filter]="g"            
            if filter=="r'":
                data.loc[j, filter]="r"
            if filter=="i'":
                data.loc[j, filter]="i"            
            if filter=="z'":
                data.loc[j, filter]="z"            
            if filter=="BJ":
                data.loc[j, filter]="B"            
            if filter=="VJ":
                data.loc[j, filter]="V"
            if filter=="UJ":
                data.loc[j, filter]="U"            
            if filter=="RM":
                data.loc[j, filter]="R"             
            if filter=="BM":
                data.loc[j, filter]="B"
            if filter=="UM":
                data.loc[j, filter]="U"            
            if filter=="KS":
                data.loc[j, filter]="K"  
            if filter=="Ks":
                data.loc[j, filter]="K"     
            if filter=="K'":
                data.loc[j, filter]="K" 
            if filter=="Kp":
                data.loc[j, filter]="K" 

    data = data[data['time_sec']>0]

    return pd.DataFrame(data)


def readin(directory="."):
    import glob2

    return np.asarray(glob2.glob(directory + "/*.txt"))
