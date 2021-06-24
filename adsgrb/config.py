from ads import config
from pathlib import Path
import os


def read_apikey():
    """
    Reads in the required API key for APS queries by setting the `ads` package's
    config.token to the string found in {HOME}/.ads/dev_key.

    :returns:
        No return.
    """
    try:
        with open(f"{HOME}/.ads/dev_key", "r") as f:
            config.token = f.read()
    except:
        print(
            f"""API key not found in {HOME}/.ads/dev_key. Either set adsgrb.config.token manually or consider
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
        os.mkdir(f"{HOME}/.ads")
    except:
        pass

    with open(f"{HOME}/.ads/dev_key", "w") as f:
        f.write(key)

    read_apikey()


def reset_apikey():
    """
    Resets the user's ADS API key found in {HOME}/.ads/dev_key.

    :returns:
        No return, but calls _read_apikey() after reset.
    """
    try:
        os.remove(f"{HOME}/.ads/dev_key")
    except:
        print(f"No key found in {HOME}/.ads to delete.")

    read_apikey()


HOME = Path.home()
