import pandas as pd
import pandas.errors
import numpy as np
import re, os


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def check_header(path, n=None, debug=False):
    """
    This monstrosity returns what

    For example, a file named '*_Oates.txt' will be interpreted as data in the same format
    as Sam Oates' data.

    """
    try:
        # attempt importing the datafile with the header "n"
        df = pd.read_csv(path, delimiter=r"\t+|\s+", header=n, engine="python")
    except pd.errors.ParserError as pe:
        if debug:
            print("ParserError:", pe)

        # if fail, recursively try again with the next row as the header
        n = 0 if n is None else n
        return check_header(path, n=n + 1)
    except pd.errors.EmptyDataError:
        print(os.path.split(path)[-1].lower(), "is empty")
        return -1

    header = h = df.columns

    # check if the header is pd.Int64Index and if 30% or more of the data aren't floats
    # ! TODO: if header is Int64Index, check the 2nd row (i.e. first row of data for the not isfloat)
    # ... so maybe change the h in [not isfloat(x) for x in h] to the second row???
    if isinstance(h, pd.Int64Index) and sum([not isfloat(x) for x in h]) >= 0.3 * len(h) // 1:
        if debug:
            print("Some aren't floats...")

        # recursively try again with the next row as the header
        n = 0 if n is None else n
        return check_header(path, n=n + 1)
    else:
        return n  # <-- the final stop in our recursion journey


def check_datatype(filename):
    """
    Given a filename, try and guess what dataset the data comes from
    (e.g., Si, Kann, Oates, etc.)

    For example, a file named '*_Oates.txt' will be interpreted as data in the same format
    as Sam Oates' data.

    """
    check = lambda x, *args: any([f in filename for f in [x, *args]])

    if check("zaninoni"):
        datatype = "zaninoni"

    elif check("si", "gendre", "tarot"):
        datatype = "si"

    elif check("liang"):
        datatype = "liang"

    # regex here is for 'GRBid.txt' files
    elif check("kann") or re.search(r"(?<!.)\d+[A-Z]?\.txt", filename):
        datatype = "kann"

    elif re.search(r"combined(?!rest)", filename):
        datatype = "combined"

    elif check("combinedrest"):
        datatype = "combinedrest"

    elif check("oates"):
        datatype = "oates"

    elif check("wczytywanie") or check("block"):
        datatype = "wczytywanie"

    else:
        datatype = ""

    return datatype


def read_data(path, datatype="", debug=False):
    data = {}

    if check_header(path) == -1:
        return

    df = pd.read_csv(path, delimiter=r"\t+|\s+", header=check_header(path), engine="python")
    header = h = df.columns

    filename = os.path.split(path)[-1].lower()
    datatype = datatype.lower() if datatype else check_datatype(filename)

    if datatype in ["si", "liang", "combinedrest"]:

        time = df[h[0]]
        timeerr = None
        flux = df[h[1]]
        fluxerr = df[h[2]]

    elif datatype == "zaninoni":

        time = df[h[0]]
        timeerr = None
        flux = df[h[2]]
        fluxerr = df[h[3]]

    elif datatype == "kann":

        time = df[h[0]]
        flux = df[h[1]]
        timeerr = None
        posfluxerr, negfluxerr = df[h[2]], df[h[3]]
        fluxerr = (posfluxerr + negfluxerr) / 2
        try:
            beta = df[h[4]]
            z = df[h[5]]
        except:
            pass

    elif datatype == "oates":

        time = df[h[0]]
        timeerr = None
        flux = df[h[2]]
        maxflux = df[h[3]]
        minflux = df[h[4]]
        fluxerr = (maxflux - minflux) / (2 * 1.65)

    elif datatype == "combined":
        z = df[h[3]]
        beta = df[h[4]]
        time = df[h[0]] * (1 + z)
        timeerr = None
        flux = df[h[1]] * (1 + z) ** (1 - beta)
        fluxerr = df[h[2]] * (1 + z) ** (1 - beta)

    elif datatype == "wczytywanie":
        time = df[h[0]]
        maxtime = df[h[1]]
        mintime = df[h[2]]
        timeerr = (maxtime - mintime) / (2 * 1.65)
        flux = df[h[3]]
        maxflux = df[h[4]]
        minflux = df[h[5]]
        fluxerr = (maxflux - minflux) / (2 * 1.65)

    else:
        # if debug:
        # print('No datatype found. Assuming format:\n| time | flux | fluxerr |')
        # read_data(path, datatype='si', debug=debug)
        time = np.array([1])
        timeerr = None
        flux = np.array([1])
        fluxerr = np.array([0])

    try:
        logtime = np.log10(time)
    except Exception as e:
        print("Issue with logT calculation:", time, e)

    logflux = np.log10(flux)
    logfluxerr = fluxerr / (flux * np.log(10))
    logtimeerr = timeerr / (time * np.log(10)) if not isinstance(timeerr, type(None)) else None

    if all(logtime > 0):
        data["T"] = logtime
        data["F"] = logflux
        data["Ferr"] = logfluxerr

        if not isinstance(logtimeerr, type(None)):
            data["Terr"] = logtimeerr
    else:
        raise ImportError("Error importing data! Ahh!")

    return data


def readin(directory="."):
    import glob2

    return glob2.glob(directory + "/*.txt")
