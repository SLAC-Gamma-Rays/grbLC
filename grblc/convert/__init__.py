""" TO USE:
Import this file into a directory to be used in your command line interfance (e.g., terminal on mac, cmd.exe on windows)
Make sure you are in the directory, and then start Jupyter notebook or Jupyter lab / any other Python instance from there
and import as any other module.
"""


from .convert import convertGRB, set_dir, get_dir, convert_all
from .time import dec_to_UT, UT_to_dec, grb_to_date
