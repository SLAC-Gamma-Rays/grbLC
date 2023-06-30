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
    Returns what line a header is at
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


def read_data(path, header=-999, data_space='log'):
    """
    Reads data, sorts by time, excludes negative time, converts time to log
    """

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
                  engine="python").sort_values(by=['time_sec'])

    if data_space=='lin':
        try:
            data['time_sec'] = np.log10(data['time_sec'])
        except Exception as e:
            print("Issue with logT calculation. Check if negative values are given for time.\nError:", e)
    elif data_space=='log':
        pass
    else:
        raise Exception("Dataspace could be either 'log' or 'lin")

    data = data.reset_index(drop=True)

    data = data[data['time_sec']>0]

    return pd.DataFrame(data)


def readin(directory="."):
    import glob2

    return np.asarray(glob2.glob(directory + "/*.txt"))
