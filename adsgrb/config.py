from ads import config
from pathlib import Path
import os
from functools import reduce


def read_apikey():
    """
    Reads in the required API key for APS queries by setting the `ads` package's
    config.token to the string found in {HOME}/.ads/dev_key.

    :returns:
        No return.
    """
    try:
        global DEV_KEY_DIR
        with open(DEV_KEY_DIR, "r") as f:
            config.token = f.read()
    except:
        print(
            f"""API key not found in {DEV_KEY_DIR}. Either set adsgrb.config.token manually or consider
calling adsgrb.set_apikey() to save the API key onto your system, bypassing the need to set your API key after
each import. Your key can be found here: https://ui.adsabs.harvard.edu/user/settings/token."""
        )

    return config.token


def set_apikey(key):
    """
    Creates a file at {HOME}/.ads/dev_key containing the
    ADS API key.

    :param key:
        ADS API key
    :type key:
        :class:`str`
    :returns:
        No return, but calls _read_apikey() after setting.
    """
    try:
        global HOME
        os.mkdir(os.path.join(HOME, ".ads"))
    except:
        pass

    global DEV_KEY_DIR
    with open(DEV_KEY_DIR, "w") as f:
        f.write(key)

    read_apikey()


def reset_apikey():
    """
    Resets the user's ADS API key found in {HOME}/.ads/dev_key.

    :returns:
        No return, but calls _read_apikey() after reset.
    """
    try:
        global DEV_KEY_DIR
        os.remove(DEV_KEY_DIR)
    except:
        print(f"No key found in {os.path.split(DEV_KEY_DIR)[0]} to delete.")

    read_apikey()


HOME = Path.home()
DEV_KEY_DIR = reduce(os.path.join, (HOME, ".ads", "dev_key"))
