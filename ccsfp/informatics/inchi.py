#!/usr/bin/env python
# Copyright IBM Corporation 2022.
# SPDX-License-Identifier: MIT

# https://www.rdkit.org/docs/GettingStartedInPython.html
# creative commons sa 4.0 tutorial used to learn rdkit methods
# https://creativecommons.org/licenses/by-sa/4.0/
# (C) 2007-2021 by Greg Landrum

from rdkit import Chem
import logging

def inchi_to_smiles(inchi: str) -> str:
    """
    Function to convert inchi to smiles
    :param inchi: str - inchi molecule representation 
    :return: str smiles
    >>> inchi_to_smiles("InChI=1S/H2O/h1H2")
    'O'
    """
    if inchi[:6] != "InChI=":
        inchi = "InChI={}".format(inchi)
    
    mol = Chem.inchi.MolFromInchi(inchi)
    
    smiles = Chem.rdmolfiles.MolToSmiles(mol)
    
    return smiles

if __name__ == '__main__':
	import doctest
	doctest.testmod()
