import os

def get_grb(grb: str,
            type: str = 'raw'
):
    """
    Function to load data directly from the package.
    """
    assert type != 'raw' or 'converted', "Can only be 'raw' or 'converted'."

    if type == 'raw':
        folder = 'mag_files/'
        suffix = '_mag.txt'

    else:
        folder = 'magAB_extcorr_files/'
        suffix = '_magAB_extcorr.txt'

    filename = folder+grb+suffix

    return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

