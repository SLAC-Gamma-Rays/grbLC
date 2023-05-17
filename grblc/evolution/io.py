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
        df = pd.read_csv(path, delimiter=r"\t+|\s+", header=n, engine="python")
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

    h = df.columns


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


def read_data(path, header=-999, debug=False):
    data = {}

    if debug:
        print("First 10 Lines:\n", "".join(open(path).readlines()[:10]))

    header = check_header(path) if header==-999 else header

    if header == -1:
        return

    df = pd.read_csv(path, delimiter=r"\t+|\s+", header=header, engine="python")
    header = h = df.columns

    time = df[h[0]]
    mag = df[h[1]]
    mag_err = df[h[2]]
    band = df[h[3]]
    system = df[h[4]]	
    telescope = df[h[5]]
    extcorr = df[h[6]]	
    source = df[h[7]]


    try:
        logtime = np.log10(time)
    except Exception as e:
        print("Issue with logT calculation:", time, e)

    if all(time > 0) and all(mag_err > 0):
        data["time_sec"] = logtime
        data["flux"] = mag
        data["flux_err"] = mag_err
        data["band"] = band
        data['system'] = system
        data['telescope'] = telescope
        data['extcorr'] = extcorr
        data['source'] = source
    else:
        raise ImportError("Some logT's are < 0... Ahh!")

    return pd.DataFrame(data)


def readin(directory="."):
    import glob2

    return np.asarray(glob2.glob(directory + "/*.txt"))
