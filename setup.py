#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["ads==0.12.3", "requests==2.25.1"]

test_requirements = []

setup(
    author="Sam Young",
    author_email="youngsam@sas.upenn.edu",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A simple way to scour the ADS for GRB data.",
    entry_points={
        "console_scripts": [
            "adsgrb=adsgrb.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="adsgrb",
    name="adsgrb",
    packages=find_packages(include=["adsgrb", "adsgrb.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/youngsm/adsgrb",
    version="0.0.0",
    zip_safe=False,
)
