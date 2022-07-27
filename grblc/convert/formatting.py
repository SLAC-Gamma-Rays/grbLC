import os
import re
from functools import reduce

import glob2

from ..util import get_dir


def fix_format():
    ff = []
    glob_path = reduce(os.path.join, [get_dir(), "**","*.txt"])
    grbsearch = re.compile(r"\d{4,7}[A-Z]?")
    filepaths = [g for g in glob2.glob(glob_path) if "_converted_flux.txt" not in g and grbsearch.search(g)]
    for line in filepaths:
        grbname = line.split("/")[-1].rstrip(".txt")
        if len(grbname) < 6:
            ff.append(line)
            grbname = "0" * (6 - len(grbname)) + grbname
            filename = os.path.join(os.path.split(line)[0], f"{grbname}.txt")
            os.rename(line, filename)
    if ff != []:
        print("The following files were renamed to add the leading zero: " + str(ff))

    # Correct \t
    filepaths = [g for g in glob2.glob(glob_path) if "_converted_flux.txt" not in g and grbsearch.search(g)]
    for line in filepaths:
        with open(line, "r") as f:
            txt = f.read()
        txt = txt.split("\n")
        txt = [re.sub(r"\t+|\s+", r"\t", t) for t in txt]
        txt = "\n".join(txt)
        with open(line, "w") as f:
            f.write(txt)
