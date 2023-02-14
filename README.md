# Carbon-capture-fingerprint-generation
The code allows for the generation of a molecular representation for amines used in carbon capture, generation from molecular fragment combinations and analysis of the chemical space.

Copyright IBM Corporation 2022.
SPDX-License-Identifier: MIT

This toolkit is a collection of tools designed to assist in the design and prediction
of carbon capture molecular properties. The current capabilities are:
1. Chemical space plotting and analysis
1. Chemical fingerprint generation
1. Molecular dataset analysis
1. Duplication of molecular string data identification

# Contributing
Please make contributions to the `dev` branch and open PRs to merge in to `master` branch.
We use docstring and unit tests to help maintain the library these are called through the `unit_test.py` script.
Please make sure all tests pass and add new tests for new code.

# Installation

Once you have installed Anaconda, run the following commands

```
git clone  $THIS_REPO

# careful, removes previous environment with the same name
yes | conda create --name ccsfp python=3.8
conda activate ccsfp
python setup.py install
python unit_test.py
```

The notebooks directory has examples which can be run to check the code runs correctly.
