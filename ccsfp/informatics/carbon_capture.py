#!/usr/bin/env python
# Copyright IBM Corporation 2022.
# SPDX-License-Identifier: MIT

# https://www.rdkit.org/docs/GettingStartedInPython.html
# creative commons sa 4.0 tutorial used to learn rdkit methods
# https://creativecommons.org/licenses/by-sa/4.0/
# (C) 2007-2021 by Greg Landrum

# Python packages and utilities
import pandas as pd
import numpy as np

#RDKit
import rdkit
from rdkit import Chem

# disp
from IPython.display import display

# Logging
import logging

# own modules
from ccsfp.informatics import molecules_and_images as mai

citation = "{ADD BIBTEX ENTRY}"

random_seed = 15791

def number_of_x_atoms(mol: rdkit.Chem.rdchem.Mol, x="N"):
    """
    Function to calculate the number of nitrogen atoms in a molecule
    :param mol: RDKit mol - rdkit.Chem.Mol instance
    :param x: str - element symbol to count in a RDKit molecule should be the same symbol RDKit uses
    :return: int
    >>> number_of_x_atoms(Chem.rdmolfiles.MolFromSmiles("O"), "O")
    1
    """

    log = logging.getLogger(__name__)

    n = 0
    for at in mol.GetAtoms():
        if at.GetSymbol() == x:
            n = n + 1

    return n


def number_of_n_atoms(mol: rdkit.Chem.rdchem.Mol) -> int:
    """
    Function to calculate the number of nitrogen atoms in a molecule
    :param mol: RDKit mol - rdkit.Chem.Mol instance
    :return: int
    >>> number_of_x_atoms(Chem.rdmolfiles.MolFromSmiles("CCN"))
    1
    """

    log = logging.getLogger(__name__)

    nN = 0
    for at in mol.GetAtoms():
        if at.GetSymbol() == "N":
            nN = nN + 1

    return nN

def count_amine_types(smi: str, primary: str = "[NX3;H2][CX4;!$(C=[#7,#8])]",
                      secondary: str = "[NX3;H1][CX4;!$(C=[#7,#8])][CX4;!$(C=[#7,#8])]",
                      tertiary: str = "[NX3]([CX4;!$(C=[#7,#8])])([CX4;!$(C=[#7,#8])])[CX4;!$(C=[#7,#8])]",
                      aromatic_sp2_n: str = "[$([nX3,X2](:[c,n,o,b,s]):[c,n,o,b,s])]",
                      show: bool = False, show_primary: bool = False, show_secondary: bool = False, show_tertiary: bool = False,
                      show_aromaticsp2: bool = False) -> (int, int, int, int):
    """
    Function to count the sub-structuer matches to the 4 amine types
    :param smi: str - smiles string
    :param primary: str - Smarts string for primary aliphatic (chain) amine identifying sub-structures
    :param secondary: str - Smarts string for identifying identifying secondary aliphatic (chain) amine sub-structures
    :param tertiary: str - Smarts string for identifying tertiary aliphatic (chain) amine sub-structures
    :param aromatic_sp2_n: str - Smarts string for identifying aromatic sp2 hybridized nitrogen atoms sub-structures
    :param show: True/False - boolean to plot the molecule graphs and overlaps
    :param show_primary: True/False - boolean to plot the molecule graphs and overlaps for primary amine matches
    :param show_secondary: True/False - boolean to plot the molecule graphs and overlaps for secondary amine matches
    :param show_tertiary: True/False - boolean to plot the molecule graphs and overlaps for tertiary amine matches
    :param show_aromaticsp2=False : True/False - boolean to plot the molecule graphs and overlaps for aromatic sp2 hybridized nitrogen atom matches
    :return: tuple(int, int, int, int)
    >>> count_amine_types("NCCN(CCc1ccncc1)CCNC")
    (1, 1, 1, 1)
    """

    log = logging.getLogger(__name__)

    primary_substructure = Chem.MolFromSmarts(primary)
    secondary_substructure = Chem.MolFromSmarts(secondary)
    tertiary_substructure = Chem.MolFromSmarts(tertiary)
    aromsp2_substructure = Chem.MolFromSmarts(aromatic_sp2_n)

    mol = mai.smiles_to_molecule(smi, threed=False)
    matches = mol.GetSubstructMatches(primary_substructure)
    n_primary = len(matches)
    if show is True or show_primary is True:
        log.info("\n----- Primary -----")
        if len(matches) > 0:
            log.info("{}".format(display(mol)))
        else:
            log.info("No matches")
        log.info("\nNumber of matches: {} Match atom indexes: {}".format(len(matches), matches))

    mol = mai.smiles_to_molecule(smi, threed=False)
    matches = mol.GetSubstructMatches(secondary_substructure)
    n_secondary = len(matches)
    if show is True or show_secondary is True:
        log.info("\n----- Secondary -----")
        if len(matches) > 0:
            log.info("{}".format(display(mol)))
        else:
            log.info("No matches")
        log.info("\nNumber of matches: {} Match atom indexes: {}".format(len(matches), matches))

    mol = mai.smiles_to_molecule(smi, threed=False)
    matches = mol.GetSubstructMatches(tertiary_substructure)
    n_tertiary = len(matches)
    if show is True or show_tertiary is True:
        log.info("\n----- Tertiary -----")
        if len(matches) > 0:
            log.info("{}".format(display(mol)))
        else:
            log.info("No matches")
        log.info("\nNumber of matches: {} Match atom indexes: {}".format(len(matches), matches))

    mol = mai.smiles_to_molecule(smi, threed=False)
    matches = mol.GetSubstructMatches(aromsp2_substructure)
    n_aromaticsp2 = len(matches)
    if show is True or show_aromaticsp2 is True:
        log.info("\n----- Atomatic sp2 hybridized nitrogen atoms -----")
        if len(matches) > 0:
            log.info("{}".format(display(mol)))
        else:
            log.info("No matches")
        log.info("\nNumber of matches: {} Match atom indexes: {}".format(len(matches), matches))

    return n_primary, n_secondary, n_tertiary, n_aromaticsp2


def capacity_classes(n_primary: list, n_secodnary: list, n_tertiary: list, n_aromatic_sp2: list,
                     capacity: list, units: str = "molco2_moln",
                     number_N_atoms: list = None, amines_mr: list = None, co2_mass: float = 44.009) -> list:
    """
    Function to output a suggested threshold for 'good' or 'bad' classification of amine molecules based on carbon capture capacity
    in the given units
    :param n_primary: list - number of primary amine groups in the molecule
    :param n_secondary: list - number of secondary amine groups in the molecule
    :param n_tertiary: list - number of tertiary amine groups in the molecule
    :param n_aromatic_sp2: list - number of aromatic sp2 hybridized nitrogen atoms in the molecule
    :param capacity: list - amine capacity values in the appropiate units
    :param units: str - Three accepted unit "molco2_moln", "molco2_molamine", "gco2_gamine" classes are consistent across tehse units
    :param number_N_atoms: list - The number of N atoms in the amine for each smiles
    :param amine_mr: list - The molar mass to each amine for each smiles
    :param co2_mass: float - mass of a co2 molecule
    :return: list
    >>> capacity_classes([1], [0], [0], [1], [0.55], amines_mr=[102.01])
    [1]
    """

    log = logging.getLogger(__name__)

    classes = []

    molar_ratios = [ent / co2_mass for ent in amines_mr]
    df = pd.DataFrame(data=np.array([n_primary, n_secodnary, n_tertiary, n_aromatic_sp2]).T,
                      columns=["primary_amine_counts", "secondary_amine_counts", "tertiary_amine_counts",
                               "aromatic_sp2_n"])

    for indx, row in df.iterrows():
        ret = classify(*row.values)

        # N molar capacity
        if units == "molco2_moln":

            if capacity[indx] < ret:
                classes.append(0)
            else:
                classes.append(1)
            log.info("{} N molar capacity (mol co2 / mol N) threshold {:.2f} capacity {:.2f} class {}".format(indx, ret,
                                                                                                              capacity[
                                                                                                                  indx],
                                                                                                              classes[
                                                                                                                  -1]))

        elif units == "molco2_molamine":
            # amine molar capacity
            if capacity[indx] < ret * number_N_atoms[indx]:
                classes.append(0)
            else:
                classes.append(1)
            log.info(
                "{} Amine molar capacity (mol co2 / mol amine) threshold {:.2f} capacity {:.2f} class {}".format(indx,
                                                                                                                 ret *
                                                                                                                 number_N_atoms[
                                                                                                                     indx],
                                                                                                                 capacity[
                                                                                                                     indx],
                                                                                                                 classes[
                                                                                                                     -1]))

        elif units == "gco2_gamine":
            # mass capacity
            if capacity[indx] < (ret * number_N_atoms[indx]) / molar_ratios[indx]:
                classes.append(0)
            else:
                classes.append(1)
            log.info("{} Mass capacity (co2 (g) / amine (g)) threshold {:.2f} capacity {:.2f} class {}\n----\n".format(
                indx, (ret * number_N_atoms[indx]) / molar_ratios[indx], capacity[indx], classes[-1]))

    return classes


def classify(n_primary: int, n_secodnary: int, n_tertiary: int, n_aromatic_sp2: int, primary_secondary_weight: float=0.5, tertiary_weight: float=1.0):
    """
    Function to output a suggested threshold for 'good' or 'bad' classification of amine molecules based on carbon capture capacity
    :param n_primary: int - number of primary amine groups in the molecule
    :param n_secondary: int - number of secondary amine groups in the molecule
    :param n_tertiary: int - number of tertiary amine groups in the molecule
    :param n_aromatic_sp2: int - number of aromatic sp2 hybridized nitrogen atoms in the molecule
    :param primary_secondary_weight: float - weighting per primary or secondary amine group
    :param tertiary_weight: float - weighting per tertiary amine group
    :return: float
    """

    if n_primary + n_secodnary > 0 and n_tertiary > 0:
        # d provides a tempering to the other wise ever increasing expectation of a polyamine of all amine types in which the tertiary is likely 
        # to play a small role comapred to kinetically favourable primary and secondary amine reactions.
        n = (primary_secondary_weight * (n_primary + n_secodnary)) + (tertiary_weight * n_tertiary)
        d = 2.0 * n_tertiary
        return n / d

    elif n_primary >= 1 and n_secodnary >= 1:
        return primary_secondary_weight * (n_primary + n_secodnary)

    elif n_primary > 1:
        return primary_secondary_weight * (n_primary)

    elif n_secodnary > 1:
        return primary_secondary_weight * (n_secodnary)

    elif n_primary == 1:
        return primary_secondary_weight

    elif n_secodnary == 1:
        return primary_secondary_weight

    elif n_tertiary > 0:
        # reacts as a catalytic molecule rather than a reactant
        return tertiary_weight

    elif n_primary == 0 and n_secodnary == 0 and n_tertiary == 0 and n_aromatic_sp2 != 0:
        # estimate as it is not clear on exactly what route these may take
        return primary_secondary_weight

    else:
        return primary_secondary_weight

if __name__ == "__main__":
    import doctest
    doctest.testmod()
