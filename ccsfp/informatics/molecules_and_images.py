#!/usr/bin/env python
# Copyright IBM Corporation 2022.
# SPDX-License-Identifier: MIT

# https://www.rdkit.org/docs/GettingStartedInPython.html
# creative commons sa 4.0 tutorial used to learn rdkit methods
# https://creativecommons.org/licenses/by-sa/4.0/
# (C) 2007-2021 by Greg Landrum

#RDKit
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdCIPLabeler

# Logging
import logging

def smiles_to_molecule(s: str, addH: bool = True, canonicalize: bool = True, threed: bool = True,
                      add_stereo: bool = False, remove_stereo: bool = False, random_seed: int= 10459,
                      verbose: bool = False, test : bool = False) -> rdkit.Chem.rdchem.Mol:
    """
    A function to build a RDKit molecule from a smiles string
    :param s: str - smiles string
    :param addH: bool - Add Hydrogens or not
    :param canonicalize: bool - canonicalize molecule rep
    :param threed: bool - get 3D coordinates of the molecule from smiles
    :param add_stereo: bool - set stereo chemistry for the molecule
    :param remove_stereo: bool - remove stereo chemistry for the molecule
    :param random_seed: int - make the structure generation deterministic
    :param verbose: bool - provide verbose logging out
    :param test: True/False - for unit testing
    >>> smiles_to_molecule("O", test=True)
    3
    """

    log = logging.getLogger(__name__)

    mol = get_mol_from_smiles(s, canonicalize=canonicalize)
    Chem.rdmolops.Cleanup(mol)
    Chem.rdmolops.SanitizeMol(mol)
    
    if remove_stereo is True:
        non_isosmiles = Chem.rdmolfiles.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=False)
        mol = get_mol_from_smiles(non_isosmiles, canonicalize=canonicalize)
        Chem.rdmolops.Cleanup(mol)
        Chem.rdmolops.SanitizeMol(mol)
        
        if verbose is True:
            for atom in mol.GetAtoms():
                log.info("Atom {} {} in molecule from smiles {} tag will be cleared. "
                        "Set properties {}.".format(atom.GetIdx(), atom.GetSymbol(), s,
                                                    atom.GetPropsAsDict(includePrivate=True, includeComputed=True)))

    if addH is True:
        mol = Chem.rdmolops.AddHs(mol)

    if add_stereo is True:
        rdCIPLabeler.AssignCIPLabels(mol)


    if threed:
        AllChem.EmbedMolecule(mol, randomSeed=random_seed)

    if test is True:
        return mol.GetNumAtoms()

    return mol 
    
def get_mol_from_smiles(smiles: str, canonicalize: bool = True, test : bool = False) -> rdkit.Chem.rdchem.Mol:
    """ 
    Function to make a mol object based on smiles
    :param smiles: str - SMILES string
    :param canonicalize: True/False - use RDKit canonicalized smile or the input resprectively
    :param test: True/False - for unit testing
    >>> get_mol_from_smiles("O", test=True)
    1
    """

    log = logging.getLogger(__name__)

    if canonicalize is True:
        s = Chem.CanonSmiles(smiles, useChiral=1)
    else:
        s = smiles
    mol = Chem.MolFromSmiles(s)
    log.debug("Input smiles: {} RDKit Canonicalized smiles {} (Note RDKit does not use "
              "general canon smiles rules https://github.com/rdkit/rdkit/issues/2747)".format(smiles, s)) 
    Chem.rdmolops.SanitizeMol(mol)
    Chem.rdmolops.Cleanup(mol)

    if test is True:
        return mol.GetNumHeavyAtoms()

    return mol


def inchi_to_molecule(inchi: str, addH: bool = True, canonicalize: bool = True, threed: bool = True,
                       add_stereo: bool = False, remove_stereo: bool = False, random_seed: int = 10459,
                       verbose: bool = False, test : bool = False) -> rdkit.Chem.rdchem.Mol:
    """
    A function to build a RDKit molecule from an InChI string
    :param inchi: str - InChI string
    :param addH: bool - Add Hydrogens or not
    :param canonicalize: bool - canonicalize molecule rep
    :param threed: bool - get 3D coordinates of the molecule from smiles
    :param add_stereo: bool - set stereo chemistry for the molecule
    :param remove_stereo: bool - remove stereo chemistry for the molecule
    :param random_seed: int - make the structure generation deterministic
    :param verbose: bool - provide verbose logging out
    :param test: bool - for unit test
    >>> inchi_to_molecule("InChI=1S/H2O/h1H2", test=True)
    3
    """

    log = logging.getLogger(__name__)

    mol = get_mol_from_inchi(inchi)
    Chem.rdmolops.Cleanup(mol)
    Chem.rdmolops.SanitizeMol(mol)

    if remove_stereo is True:
        non_isosmiles = Chem.rdmolfiles.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=False)
        mol = get_mol_from_smiles(non_isosmiles, canonicalize=canonicalize)
        Chem.rdmolops.Cleanup(mol)
        Chem.rdmolops.SanitizeMol(mol)

        if verbose is True:
            for atom in mol.GetAtoms():
                log.info("Atom {} {} in molecule from InChI {} tag will be cleared. "
                         "Set properties {}.".format(atom.GetIdx(), atom.GetSymbol(), inchi,
                                                     atom.GetPropsAsDict(includePrivate=True, includeComputed=True)))

    if addH is True:
        mol = Chem.rdmolops.AddHs(mol)

    if add_stereo is True:
        rdCIPLabeler.AssignCIPLabels(mol)

    if threed:
        AllChem.EmbedMolecule(mol, randomSeed=random_seed)

    if test is True:
        return mol.GetNumAtoms()

    return mol


def get_mol_from_inchi(inchi: str, test : bool = False) -> rdkit.Chem.rdchem.Mol:
    """
    Function to make a mol object based on smiles
    :param inchi: str - SMILES string
    :param test: True/False - for unit testing
    >>> get_mol_from_inchi("InChI=1S/H2O/h1H2", test=True)
    1
    """

    log = logging.getLogger(__name__)

    mol = Chem.MolFromInchi(inchi)
    log.debug("Input inchi: {})".format(inchi))
    Chem.rdmolops.SanitizeMol(mol)
    Chem.rdmolops.Cleanup(mol)

    if test is True:
        return mol.GetNumHeavyAtoms()

    return mol

if __name__ == '__main__':
    import doctest
    doctest.testmod()
