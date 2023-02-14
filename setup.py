# Copyright IBM Corporation 2022.
# SPDX-License-Identifier: EPL-2.0

import setuptools
from setuptools import setup
import os

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

install_requires=[
        "pandas",
        "scikit-learn",
        "matplotlib",
        "scipy",
        "rdkit-pypi",
        "dask[distributed]",
        "IPython",
        "networkx",
        "numpy==1.21",
        "pillow",
        "networkx",
        "pubchempy"
    ]

setuptools.setup(
    name="ccsfp",
    version="0.0.1",
    author="James McDonagh, Stamatia Zavitsanou, Flaviu Cipcigan, Dmitry Zubarev and Alexander Harrison",
    author_email="james.mcdonagh@uk.ibm.com",
    description="Methods for carbon capture molecular informatics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    license="All rights reserved, Copyright IBM Corporation 2022. SPDX-License-Identifier: MIT",
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    install_requires=install_requires
)
