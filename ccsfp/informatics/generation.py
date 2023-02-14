#!/usr/bin/env python
# Copyright IBM Corporation 2022.
# SPDX-License-Identifier: MIT

# https://www.rdkit.org/docs/GettingStartedInPython.html
# creative commons sa 4.0 tutorial used to learn rdkit methods
# https://creativecommons.org/licenses/by-sa/4.0/
# (C) 2007-2021 by Greg Landrum

"""
This module is designed for molecular enumeration from core scaffolds and sidechains. It is not fool proof and
makes zero effort to identify whether the sturcture are valid moleucles or accesible synthetically or purchasable.

The idea is to provide a set of methods to produce plain text stores of cores and side chains then to read and combine
them over a limited number of generation to produce new potential molecules.

An imagined standard workflow is envisaged to be:
core_smarts = string_to_ext_murko(list[inchi and/or smiles])
sidechains_smarts = strings_to_sidechains(list[inchi and/or smiles])

ext_murko_to_scaffold_file(core_smarts)
smarts_to_sidechains_file(sidechains_smarts)

scaffolds, sidechains = load_cores_and_sidechains_from_csv_files("scaffolds.csv", "sidechains.csv")

generated_smis = list()
for scaffold in scaffolds:
 scaffold_mol = Chem.MolFromSmiles(scaffold)
 generated_smis.append(limited_combinatorial_generation(scaffold_mol, sidechains))

print(generated_smis)
"""


# Python packages and utilities
import os
import pandas as pd
import random
import re
import logging

# RDKit
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolHash, MolStandardize

def string_to_core_scaffolds(strings: list, return_mols=False, return_mol_and_smarts=False):
    """
    Function to read smiles and/or inchi and make extended Murko hashes for core scaffold possibilities in molecule
    generation
    :param strings: list - list of smiles and or inchi
    :param return_mols: bool - return the rdkit molecules rather than the smarts
    :param return_mols_and_smarts: bool - return both the rdkit molecules and smarts in this order
    :return: list smarts (default)
    """

    log = logging.getLogger(__name__)

    # ouput variable
    ext_murko_smarts = list()

    for s in strings:
        inchi = False
        if s[:6] == "InChI=":
            inchi = True

        if inchi is False:
            mol = Chem.MolFromSmiles(s)
            log.info("Molecule: {}".format(mol))
        else:
            mol = Chem.MolFromInchi(s)
            log.info("Molecule: {}".format(mol))

        if mol is None:
            log.warning("WARNING - could not generate extended murko from {} will skip".format(s))
            continue

        ext_murko_smarts_tmp = rdMolHash.MolHash(mol, rdMolHash.HashFunction.ExtendedMurcko)
        log.info("Extended Murko smarts: {}".format(ext_murko_smarts_tmp))

        ext_murko_smarts = ext_murko_smarts + [ext_murko_smarts_tmp]

    if return_mol_and_smarts is True:
        ext_murko_mols = [Chem.MolFromSmarts(sma) for sma in ext_murko_smarts]
        return ext_murko_mols, ext_murko_smarts
    elif return_mols is True:
        ext_murko_mols = [Chem.MolFromSmarts(sma) for sma in ext_murko_smarts]
        return ext_murko_mols
    else:
        return ext_murko_smarts


def strings_to_sidechains(strings: list, return_mols=False, return_mols_and_smarts=False):
    """
    Function to read smiles and/or inchi and make regio hashes for side chain possibilities in molecule generation
    :param strings: list - list of smiles and or inchi
    :param return_mols: bool - return the rdkit molecules rather than the smarts
    :param return_mols_and_smarts: bool - return both the rdkit molecules and smarts in this order
    :return: list smarts (default)
    """

    log = logging.getLogger(__name__)

    # ouput variable
    regio_fragment_smarts = list()

    for s in strings:
        inchi = False
        if s[:6] == "InChI=":
            inchi = True

        if inchi is False:
            mol = Chem.MolFromSmiles(s)
        else:
            mol = Chem.MolFromInchi(s)

        if mol is None:
            log.warning("WARNING - could not generate extended murko from {} will skip".format(s))
            continue

        regio_fragment_smarts_tmp = rdMolHash.MolHash(mol, rdMolHash.HashFunction.Regioisomer)
        regio_fragment_smarts = regio_fragment_smarts + list(set(regio_fragment_smarts_tmp.split(".")))

    if return_mols_and_smarts is True:
        regio_fragment_mols = [Chem.MolFromSmarts(sma) for sma in regio_fragment_smarts]
        return regio_fragment_mols, regio_fragment_smarts
    elif return_mols is True:
        regio_fragment_mols = [Chem.MolFromSmarts(sma) for sma in regio_fragment_smarts]
        return regio_fragment_mols
    else:
        return regio_fragment_smarts


def core_scaffolds_to_scaffold_file(ext_murko_mols:list, scaffold_file="scaffolds.csv", mode="a+"):
    """
    Function to take a list of smarts from extended murko hashes in rdkit and make a core scaffolds csv plain text file
    to be loaded for moleucle generation
    :param ext_murko_mols: list - list of smarts from extended murko hashing
    :param scaffold_file: str - file name to save core scaffold to
    :param mode: str - mode to open the file in
    :return: list of rdkit molecules
    """

    log = logging.getLogger(__name__)

    if isinstance(ext_murko_mols[0], rdkit.Chem.rdchem.Mol):
        ext_murko_mols = [MolStandardize.rdMolStandardize.Normalize(m) for m in ext_murko_mols]
    else:
        ext_murko_mols = [Chem.MolFromSmarts(m) for m in sorted(ext_murko_mols)]
        ext_murko_mols = [MolStandardize.rdMolStandardize.Normalize(m) for m in ext_murko_mols]

    log.debug("Extended murko moleucles: {}".format(ext_murko_mols))
    ext_murko_mols_list = list()
    for jth, ext_murko_mol in enumerate(ext_murko_mols):
        if ext_murko_mol is None:
            log.warning("Scaffold {} failed normalization and will not be included".format(jth))
            continue
        elif Chem.MolToSmiles(ext_murko_mol) == "":
            log.warning("Scaffold {} failed to regenerate smiles and will not be included".format(jth))
            continue
        log.info(Chem.MolToSmiles(ext_murko_mol))

        log.info("{} mol: {}".format(jth, ext_murko_mol))
        # In this step we loose information as we no longer have supersets of saturated and unstaurated for example
        # We collapse to one or the other. TODO: can we better control that collapse?
        log.info("Collapsing to {} from {}".format(Chem.MolToSmiles(ext_murko_mol), Chem.MolToSmarts(ext_murko_mol)))
        m = Chem.MolFromSmiles(Chem.MolToSmiles(ext_murko_mol))
        n_attachment_p = 0
        for ent in Chem.MolToSmiles(ext_murko_mol):
            if ent == "*":
                n_attachment_p = n_attachment_p + 1

        if os.path.isfile(scaffold_file):
            with open(scaffold_file, "r") as fin:
                content = fin.readlines()
                n = len(content) - 1
        else:
            with open(scaffold_file, "w") as fout:
                fout.write(
                    "id,molecular_formula,molecular_mass,numer_of_atoms,number_of_attachment_points,smiles\n")
            n = 0
        
        with open(scaffold_file, mode) as fout:
            fout.write("{},{},{},{},{},{},{}\n".format(n + 1,
                                                       Chem.rdMolDescriptors.CalcMolFormula(ext_murko_mol),
                                                       Chem.rdMolDescriptors.CalcExactMolWt(ext_murko_mol),
                                                       Chem.rdMolDescriptors.CalcNumAtoms(ext_murko_mol),
                                                       Chem.rdMolDescriptors.CalcNumRings(m),
                                                       n_attachment_p,
                                                       Chem.MolToSmiles(ext_murko_mol)))
        ext_murko_mols_list.append(ext_murko_mol)

    return ext_murko_mols_list


def smarts_to_sidechains_file(smarts: list, sidechain_file="sidechains.csv", mode="a+"):
    """
    Function to take a list of smarts from regio hashes in rdkit and make a side chains csv plain text file to be loaded
    for moleucles genration
    :param smarts: list - list of smarts from regio hashes
    :param sidechain_file: str - file name to save the side chains to
    :param mode: str - mode to open the file in
    :return: list of rdkit molecule objects
    """

    log = logging.getLogger(__name__)

    if os.path.isfile(sidechain_file):
        with open(sidechain_file, "r") as fin:
            content = fin.readlines()
            n = len(content) - 1
    else:
        with open(sidechain_file, "w") as fout:
            fout.write(
                "id,molecular_formula,molecular_mass,numer_of_atoms,number_of_rings,number_of_attachment_points,smiles\n")
            n = 0

    lines = []
    molecules = []
    for jth, sma in enumerate(sorted(smarts)):

        m = Chem.MolFromSmarts(sma)
        Chem.rdmolops.Cleanup(m)

        # In this step we loose information as we no longer have supersets of saturated and unstaurated for example
        # We collapse to one or the other. TODO: can we better control that collapse?
        log.info("Collapsing to {} from {}".format(Chem.MolToSmiles(m), sma))
        m = Chem.MolFromSmiles(Chem.MolToSmiles(m))

        if m is None:
            log.warning("Sidechain {} failed normalization and will not be included".format(jth))
            continue

        n_attachment_p = 0
        for ent in Chem.MolToSmiles(m):
            if ent == "*":
                n_attachment_p = n_attachment_p + 1

        lines.append("{},{},{},{},{},{},{}".format(n + 1,
                                                   Chem.rdMolDescriptors.CalcMolFormula(m),
                                                   Chem.rdMolDescriptors.CalcExactMolWt(m),
                                                   Chem.rdMolDescriptors.CalcNumAtoms(m),
                                                   Chem.rdMolDescriptors.CalcNumRings(m),
                                                   n_attachment_p,
                                                   Chem.MolToSmiles(m)))
        n = n + 1

        molecules.append(m)
        # log.info(lines)
    with open(sidechain_file, mode) as fout:
        fout.write("{}\n".format("\n".join(lines)))

    return molecules

def load_cores_and_sidechains_from_csv_files(core_file: str, sidechain_file: str, core_indexes: list = None,
                                             sidechain_indexes: list = None):
    """
    Function to load core and side chain csv plain text files
    :param core_file: str - file name and path
    :param sidechain_file: str - file name and path
    :param core_indexes: list - indexes in the files to keep and remove others
    :param sidechain_indexes: list - indexes in the files to keep and remove others
    :return: core list, sidechain list
    """

    log = logging.getLogger(__name__)

    cores = pd.read_csv(core_file)

    if core_indexes is not None:
        cores = cores.iloc[core_indexes]
    cores = cores["smiles"]
    sidechains = pd.read_csv(sidechain_file)
    if sidechain_indexes is not None:
        sidechains = sidechains.iloc[sidechain_indexes]
    sidechains = sidechains["smiles"]

    return cores.to_list(), sidechains.to_list()


def limited_combinatorial_generation(core: rdkit.Chem.rdchem.Mol, sidechains: list, generations: int = 5,
                                     return_mols: bool = False, random_seed=1, recursive_test=False,
                                     extra_random=False):
    """
    Function to carry out a combinatorial generation of molecules based on random sampling with replacement. A sample of
    side chains are chosen in each generation and appended to attachment points on the core scaffold. Note large numbers
    of structures can be generated if the generations is set even to a modest value, the expected number of structures is
    n_attachment_points^(generations) + 1, the actual number returned may differ based on filtering of equivalent
    structures and failures to produce rdkit parsable structures.

    The algorithim goes:
    for generation in range(generations):

       for attachment_point_x:

           choose randomly with replacement a side chain

           for scaffold in core_scaffolds:
               add side chain to all free attachmnet points and store the molecule

               add the side chain to each free attachment point sequentially and store the molecules

               if side chain adds attachment points run a recusive attachment over the side chains

           loop over the new generated cores until all attachment points are filled

    return list of lists

    :param core: rdkit.Chem.rdchem.Mol - core scaffold acting as the central structure
    :param sidechains: list - list of smiles side chains to append to the attachment points
    :param generations: int - number of generation of random choices of side chains
    :param return_mols: bool - return the rdkit mol objects of the built smiles
    :param random_seed: int - random seed initialization for pseudo-random choices of side chains
    :param recursive_test: bool - run a recusive test with a dixed initial side chain of *N(*)*
    :param extra_random: bool - change the side chain in replacement iteration to add more chemical randomness
    :return: list of lists
    """

    log = logging.getLogger(__name__)

    # set random seed
    random.seed(random_seed)

    # Number of places where attachments can be made to the scaffold
    n_replacements = len(core.GetSubstructMatches(Chem.MolFromSmarts('[#0]')))
    log.info("Number of core attachment points identified by '*' {}".format(n_replacements))
    if n_replacements == 0:
        log.info("No attachment points from '*' will use the first atom as the only connection point")

    # ouput variable
    products = list()

    # for information
    if generations == "all":
        use_all = True
        generations = len(sidechains)
        log.info("Will run {} generations as generations set to all and this is the number of side chains".format(
            generations))
    else:
        use_all = False
        log.info("Will run {} generations".format(generations))

    # The limited version comes from the use of generations this limits the size and space overall
    for generation in range(generations):
        log.info("Generation: {}".format(generation))
        # List of cores to iterate over
        cores = [core]

        for replacement_iteration in range(n_replacements):
            # Randomly choose a side chain with replacement each time
            if use_all is True and replacement_iteration == 0:
                sidechain = sidechains[generation]
            else:
                sidechain = random.choice(sidechains)
            log.info("\tReplacement iteration: {}\n\t\tSide chain: {}".format(replacement_iteration, sidechain))
            sidechain = re.sub("\*", "", sidechain, count=1)
            additional_attachment_points = len(re.findall("\*", sidechain))
            sidechain = Chem.MolFromSmarts(sidechain)
            log.info("\tAfter attachment point marker * removed {}".format(Chem.MolToSmiles(sidechain)))

            # Loop over all cores generated and make replacements
            tmp_cores = list()
            for scaffold in cores:

                if extra_random is True and round(random.random()) == 1:
                    sidechain = random.choice(sidechains)
                    log.info("\t\tExtra random sidechain switch: {}\n\t\tSide chain: {}".format(replacement_iteration,
                                                                                                sidechain))
                    sidechain = re.sub("\*", "", sidechain, count=1)
                    additional_attachment_points = len(re.findall("\*", sidechain))
                    sidechain = Chem.MolFromSmarts(sidechain)
                    log.info("\t\tExtra random sidechain switch after attachment point marker * removed {}".format(
                        Chem.MolToSmiles(sidechain)))

                # log.info(f"{scaffold}, {Chem.MolFromSmarts('[#0]')}, {sidechain}")
                log.debug(f"{type(scaffold)}, {type(Chem.MolFromSmarts('[#0]'))}, {type(sidechain)}")
                log.info("\t\tScaffold: {}".format(Chem.MolToSmiles(scaffold)))

                # Replace all attachment points with randomly chosen sidechain
                generated_all_replaced = Chem.ReplaceSubstructs(scaffold,
                                                                Chem.MolFromSmarts('[#0]'),
                                                                sidechain,
                                                                replaceAll=True,
                                                                useChirality=True)

                # Generate a replacement at each avaliable attachemnent point
                # for the side chain and leave the other vacant for another replacement_iteration
                generated_cores = Chem.ReplaceSubstructs(scaffold,
                                                         Chem.MolFromSmarts('[#0]'),
                                                         sidechain,
                                                         replaceAll=False,
                                                         replacementConnectionPoint=0,
                                                         useChirality=True)

                tmp_cores = tmp_cores + list(generated_all_replaced) + list(generated_cores)

            if additional_attachment_points > 0:
                cores = additional_replacements(tmp_cores, additional_attachment_points, sidechains,
                                                test=recursive_test, extra_random=extra_random, random_seed=random_seed)
            else:
                cores = tmp_cores.copy()

            log.info(
                "\tEnd of replacement iteration {} (number generated: {})".format(replacement_iteration, len(cores)))
            log.info("\tSMILES:\n\t{}".format("\n\t".join([Chem.MolToSmiles(m) for m in cores])))
        log.info("End of generation {} (number generated: {})\n".format(generation, len(cores)))
        products = products + cores

    if return_mols is True:
        return list(set(products))

    else:
        products_smis = [Chem.MolToSmiles(p) for p in products]
        products_smis = sorted(list(set(products_smis)))
        return products_smis


def additional_replacements(cores, n_replacements, sidechains, test=False, extra_random=False, random_seed=1):
    """
    A sub section of limited_combinatorial_generation to carry out recursive attachments if side chains add attachment points
    This is a combinatorial generation of molecules based on random sampling with replacement. A sample of
    side chains are chosen in each generation and appended to attachment points on the core scaffold. Note large numbers
    of structures can be generated if the generations is set even to a modest value depedning on the number of attachment points.
    :param cores: list of rdkit.Chem.rdchem.Mol - core scaffold acting as the central structure
    :param n_replacements: int - number of attachment_points in the core
    :param test: bool - to test the recusive function can set the first recursive runs sidechain struct
    :param sidechains: list - list of smiles side chains to append to the attachment points
    :param random_seed: int - random seed initialization for pseudo-random choices of side chains
    :param extra_random: bool - change the side chain in replacement iteration to add more chemical randomness
    :return: list of lists
    """

    log = logging.getLogger(__name__)

    log.info("\t\tSide chain added {} extra replacements".format(n_replacements))

    random.seed(random_seed)
    tmp_cores = []

    for replacement_iteration in range(n_replacements):
        # Randomly choose a side chain with replacement each time
        if test is True:
            sidechain = "*N*"  # random.choice(sidechains)
        else:
            sidechain = random.choice(sidechains)
        log.info("\tReplacement iteration: {}\n\t\tSide chain: {}".format(replacement_iteration, sidechain))
        sidechain = re.sub("\*", "", sidechain, count=1)
        additional_attachment_points = len(re.findall("\*", sidechain))
        sidechain = Chem.MolFromSmarts(sidechain)
        log.info("\tAfter attachment point marker * removed {}".format(Chem.MolToSmiles(sidechain)))

        # Loop over all cores generated and make replacements
        tmp_cores = list()
        for scaffold in cores:

            if extra_random is True and round(random.random()) == 1:
                sidechain = random.choice(sidechains)
                log.info("\t\tExtra random sidechain switch: {}\n\t\tSide chain: {}".format(replacement_iteration,
                                                                                            sidechain))
                sidechain = re.sub("\*", "", sidechain, count=1)
                additional_attachment_points = len(re.findall("\*", sidechain))
                sidechain = Chem.MolFromSmarts(sidechain)
                log.info("\t\tExtra random sidechain switch after attachment point marker * removed {}".format(
                    Chem.MolToSmiles(sidechain)))

            # log.info(f"{scaffold}, {Chem.MolFromSmarts('[#0]')}, {sidechain}")
            log.debug(f"{type(scaffold)}, {type(Chem.MolFromSmarts('[#0]'))}, {type(sidechain)}")
            log.info("\t\tScaffold: {}".format(Chem.MolToSmiles(scaffold)))
            # Replace all attachment points with randomly chosen sidechain
            generated_all_replaced = Chem.ReplaceSubstructs(scaffold,
                                                            Chem.MolFromSmarts('[#0]'),
                                                            sidechain,
                                                            replaceAll=True,
                                                            useChirality=True)

            # Generate a replacement at each avaliable attachemnent point
            # for the side chain and leave the other vacant for another replacement_iteration
            generated_cores = Chem.ReplaceSubstructs(scaffold,
                                                     Chem.MolFromSmarts('[#0]'),
                                                     sidechain,
                                                     replaceAll=False,
                                                     replacementConnectionPoint=0,
                                                     useChirality=True)

            tmp_cores = tmp_cores + list(generated_all_replaced) + list(generated_cores)

        if additional_attachment_points > 0:
            cores = additional_replacements(tmp_cores, additional_attachment_points, sidechains)
        else:
            cores = tmp_cores.copy()

    return cores

def limited_enumeration(core_molecule_strings: list, side_chain_moleucle_strings: list, generations:int=5,
                        extra_random:bool=False, scaffold_file:str="scaffolds.csv", sidechain_file:str="sidechains.csv",
                        save_all:bool=False, filename:str="generated_smiles.csv", return_all:bool=False,
                        overwrite:bool=False):
    """
    Function to run a standard limited enumeration process from two lists of smiles andor inchi and a generations limit.
    Warning this is not fool proof and make no attempt to check the molecules validity in reality.
    :param core_molecule_strings: list - list of smiles and/or inchi
    :param sidechain_moleucle_strings: list - list of smiles and/or inchi
    :param generations: int - number of generation in which a random set of side chains a chosen and attached
    :param extra_random: bool - change the side chain in replacement iteration to add more chemical randomness
    :param scaffold_file: str - file of core scaffolds to use and save to
    :param sidechain_file: str - file of sidechains to use and save to
    :param save_all: bool - save all generated structures will maintain duplicates
    :param filename: str - name of a file to save sturctures to
    :param return_all: bool return a list of all including duplicates and a list of none duplicated smiles
    :param overwrite: bool - overwrite previous scaffold and sidechain files
    :return: tuple(list, list) first list all generated smiles second list all unique generated strings
    >>> limited_enumeration(["Cc1ccccc1CCN"], ['c1ccccc1', '*CC'], extra_random=True, overwrite=True) #doctest: +NORMALIZE_WHITESPACE
    ['CCc1ccccc1-c1ccccc1', 'CCc1ccccc1CC', 'c1ccc(-c2ccccc2-c2ccccc2)cc1']
    """
    log = logging.getLogger(__name__)

    log.info("It is advisable to remove any previous scaffold or side chain files otherwise behaviour can be difficult"
             " to reproduce. Set overwrite to True to rename any old scaffold and sidechain files that are there"
             " with '.old' to deal with this")

    if overwrite is True:
        if os.path.isfile(scaffold_file):
            os.rename(scaffold_file, scaffold_file + ".old")

        if os.path.isfile(sidechain_file):
            os.rename(sidechain_file, sidechain_file + ".old")

    core_smarts = string_to_core_scaffolds(core_molecule_strings)
    sidechains_smarts = strings_to_sidechains(side_chain_moleucle_strings)

    core_scaffolds_to_scaffold_file(core_smarts, scaffold_file=scaffold_file)
    smarts_to_sidechains_file(sidechains_smarts, sidechain_file=sidechain_file)

    scaffolds, sidechains = load_cores_and_sidechains_from_csv_files(scaffold_file, sidechain_file)

    generated_smis = list()
    for scaffold in scaffolds:
        log.info("Generating using scaffold: {}".format(scaffold))
        scaffold_mol = Chem.MolFromSmiles(scaffold)
        generated_smis.append(limited_combinatorial_generation(scaffold_mol,
                                                               sidechains,
                                                               generations=generations,
                                                               extra_random=extra_random))

    log.info("SMILES produced:\n{}".format(generated_smis))

    gsmiles = [elm for ent in generated_smis for elm in ent]
    if save_all is True:
        df = pd.DataFrame(gsmiles, columns=["smiles"])
        df.to_csv("all_" + filename, index=False)

    df = pd.DataFrame(list(set(gsmiles)), columns=["smiles"])
    df.to_csv(filename, index=False)

    if return_all is False:
        return sorted(list(set(gsmiles)))
    else:
        return generated_smis, sorted(list(set(gsmiles)))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
