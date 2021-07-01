#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
import re, os, sys

with open("README.rst") as readme_file:
    readme = readme_file.read()

requirements = ["ads", "requests"]

test_requirements = []

# Thank you Andy Casey for this nice versioning method
major, minor1, minor2, release, serial = sys.version_info


readfile_kwargs = {"encoding": "utf-8"} if major >= 3 else {}


def readfile(filename):
    with open(filename, **readfile_kwargs) as fp:
        contents = fp.read()
    return contents


version_regex = re.compile('__version__ = "(.*?)"')
contents = readfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "adsgrb", "__init__.py"))

version = version_regex.findall(contents)[0]


setup(
    author="Sam Young",
    author_email="youngsam@sas.upenn.edu",
    python_requires=">=3.6",
    description="A simple way to scour the ADS for GRB data.",
    entry_points={
        "console_scripts": [
            "adsgrb=adsgrb.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="adsgrb",
    name="adsgrb",
    packages=find_packages(include=["adsgrb", "adsgrb.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/youngsm/adsgrb",
    version=version,
    zip_safe=False,
)
