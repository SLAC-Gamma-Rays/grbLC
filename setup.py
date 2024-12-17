#!/usr/bin/env python
"""The setup script."""
import os
import re
import sys
from functools import reduce
from setuptools import find_packages, setup

# Read the README file
with open("README.rst", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

# Read the requirements file
with open("requirements.txt", "r", encoding="utf-8") as requirements_file:
    requirements = [line.strip() for line in requirements_file.readlines()]

with open("requirements_dev.txt", "r", encoding="utf-8") as requirements_test_file:
    test_requirements = [line.strip() for line in requirements_test_file.readlines()]

# Version extraction method
version_regex = re.compile('__version__ = "(.*?)"')
with open(
    reduce(os.path.join, [os.path.dirname(os.path.abspath(__file__)), "grblc", "__init__.py"]),
    encoding="utf-8"
) as fp:
    contents = fp.read()
    version = version_regex.findall(contents)[0]

setup(
    author="Ridha Fathima Mohideen Malik",
    author_email="ridhafathima273@gmail.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        # "Programming Language :: Python :: 3.9",
        # "Programming Language :: Python :: 3.10",
    ],
    description="A Python package for GRB optical light curve studies.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type='text/x-rst',
    keywords="grblc",
    name="grblc",
    packages=find_packages(include=["grblc", "grblc.*"]),
    include_package_data=True,
    package_data={
        'grblc': ['grblc/data/*'],
        '': ['*.txt']
    },
    test_suite="tests",
    # tests_require=test_requirements,
    url="https://github.com/SLAC-Gamma-Rays/grbLC",
    version=version,
    zip_safe=False,
)
