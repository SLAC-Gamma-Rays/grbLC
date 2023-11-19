import os

import pandas as pd


defaultshift_toAB={
            "U":[0.800527],
            "B":[-0.107512],
            "BM":[-0.107512],
            "V":[0.006521],
            "VM":[0.006521],
            "R":[0.190278],
            "unfiltered":[0.190278],
            "clear":[0.190278],
            "N":[0.190278],
            "lum":[0.190278],
            "RM":[0.190278],
            "TR-rgb":[0.190278],
            "IR-cut":[0.190278],
            "CR":[0.190278],
            "CV":[0.006521],
            "I":[0.431372],
            "Rc":[0.117],
            "Ic":[0.342],
            "Bj":[1.344],
            "Vj":[0.006521],
            "Uj":[0.800527],
            "u":[0.9],
            "g":[-0.125],
            "g_Gunn":[-0.013],
            "r":[0.119],
            "r_Gunn":[-0.226], #https://lweb.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html
            "i":[0.332],
            "i_Gunn":[-0.296], #https://lweb.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html
            "z":[0.494],
            "z_Gunn":[0.494], #https://lweb.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html
            "up":[0], #p='
            "gp":[-0.125],
            "rp":[0.119],
            "ip":[0.332],
            "zp":[0.494],
            "H":[1.344],
            "J":[0.87],
            "K":[1.815],
            "Ks":[1.814],
            "Kp":[1.84],
            "K'":[1.815], 
            "Z":[0.489],
            "Y":[0.591],
            "q":[0.190278],
            "w":[0.8],
            "F220W":[1.683496],
            "F250W":[1.495635],
            "F330W":[1.097],
            "F344N":[1.151053],
            "F435W":[-0.129],
            "F475W":[-0.122],
            "F502N":[-0.083596],
            "F550M":[0.024283],
            "F555W":[-0.03],
            "F606W":[0.063],
            "F625W":[0.14],
            "F658N":[0.376572],
            "F660N":[0.278863],
            "F775W":[0.364],
            "F814W":[0.4],
            "F850LP":[0.494],
            "F892N":[0.487505]
}

path1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "filters.txt")
path2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "reddening.txt")
path3 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SF11_conversions.txt")

filters = pd.read_csv(path1, sep='\s+', header=0, index_col=0, engine='python', encoding='ISO-8859-1')
adps = pd.read_csv(path2, sep='\t', header=0, index_col=0, engine='python', encoding='ISO-8859-1')
schafly = pd.read_csv(path3, sep='\t', header=0, index_col='lambda_eff')

__all__ = ["filters", "adps", "schafly", "defaultshift_toAB"]
