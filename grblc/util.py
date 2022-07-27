import os

__all__ = ["set_dir", "get_dir"]

# small setter to set the main conversion directory
def set_dir(dir):
    global directory
    directory = os.path.abspath(dir)
    return directory


# getter to return conversion directory
def get_dir():
    global directory
    return directory

directory = os.getcwd()
