#!/usr/bin/env python
# Copyright IBM Corporation 2022.
# SPDX-License-Identifier: MIT

# https://www.rdkit.org/docs/GettingStartedInPython.html
# creative commons sa 4.0 tutorial used to learn rdkit methods
# https://creativecommons.org/licenses/by-sa/4.0/
# (C) 2007-2021 by Greg Landrum

from rdkit import Chem

import logging

def smiles_to_inchi(smiles : str) -> str :
    """ 
     Function to conveniently convert smiles to InChI
    :param smiles:
    :return: str InChI
    >>> smiles_to_inchi("O")
    'InChI=1S/H2O/h1H2'
    """

    log = logging.getLogger(__name__)

    s = Chem.CanonSmiles(smiles, useChiral=1)
    mol = Chem.MolFromSmiles(s)

    inchi = Chem.inchi.MolToInchi(mol)

    return inchi
