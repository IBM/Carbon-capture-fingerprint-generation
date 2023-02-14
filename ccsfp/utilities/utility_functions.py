#!/usr/bin/env python
# Copyright IBM Corporation 2022.
# SPDX-License-Identifier: MIT 

# https://www.rdkit.org/docs/GettingStartedInPython.html
# creative commons sa 4.0 tutorial used to learn rdkit methods
# https://creativecommons.org/licenses/by-sa/4.0/
# (C) 2007-2021 by Greg Landrum

"""
This module provides utilities which might be useful
"""

# Python packages and utilities
import pandas as pd
import numpy as np
import logging
from ccsfp.informatics.inchi import inchi_to_smiles
from ccsfp.informatics.smiles import smiles_to_inchi
from rdkit import Chem

def check_for_duplicates(data:pd.DataFrame = None, datafile: str = None, smiles: list = None, inchi: list = None,
                         inchi_key: list = None, smiles_column_name: str = "smiles",
                         inchi_column_name: str = "inchi", inchi_key_column_name: str = "inchikey",
                         compare_which: str = "all", return_reduced_data: bool = False):
    """
    Function to compare a dataset of molecules over multiple common identifiers to id duplicates
    :param data: pd.DataFrame - pandas data frame with columns of at least smiles, inchi and inchikey
    :param datafile: str - csv file name to load or dataframe
    :param smiles: list - smiles strings
    :param inchi: list - inchci strings
    :param inchi_key: list - inchi_key strings
    :param smiles_column_name: str - column header for smiles column
    :param inchi_column_name: str - column header for inchi column
    :param inchi_key_column_name: str - column header for inchi_key column
    :param compare_which: str - all or one of the identifiers smiles, inchi or inchi_keys
    :param return_reduced_data: bool - return a data frame with the duplicates removed if true
    :return: default indexes of duplicate rows maintaining the first instance of each entry. If
             return_reduced_data is True returns a dataframe with the duplicated data rows removed
             except the first instance of the row.
    """

    log = logging.getLogger(__name__)

    # Use inchi if they are given. If not use smiles and assume one of them is given
    if datafile is not None:
        log.info(f"Loading datafile to find duplicates in {datafile}")
        data = pd.read_csv(datafile)
        data.columns = [ent.strip() for ent in data.columns]
        log.info(f"data columns {data.columns}")

        if inchi_column_name in data.columns:
            inchi = data[inchi_column_name].to_list()
            inchi = [ent.strip() for ent in inchi]

            if smiles_column_name not in data.columns:
                smiles = [inchi_to_smiles(ent) for ent in inchi]
            else:
                smiles = data[smiles_column_name].to_list()
                smiles = [ent.strip() for ent in smiles]

            if inchi_key_column_name not in data.columns:
                inchi_key = [Chem.inchi.InchiToInchiKey(ent) for ent in inchi]
            else:
                inchi_key = data[inchi_key_column_name].to_list()
                inchi_key = [ent.strip() for ent in inchi_key]

        elif smiles_column_name in data.columns:
            smiles = data[smiles_column_name].to_list()
            smiles = [ent.strip() for ent in smiles]

            if inchi_column_name not in data.columns:
                inchi = [smiles_to_inchi(ent) for ent in smiles]
            else:
                inchi = data[inchi_column_name].to_list()
                inchi = [ent.strip() for ent in inchi]

            if inchi_key_column_name not in data.columns:
                inchi_key = [Chem.inchi.InchiToInchiKey(ent) for ent in inchi]
            else:
                inchi_key = data[inchi_key_column_name].to_list()
                inchi_key = [ent.strip() for ent in inchi_key]

        else:
            log.error("Neither inchi or smiles column keys in the data set.")
            raise RuntimeError("One of smiles or InChI must be specified")

    elif data is not None:
        log.info(f"data columns {data.columns}")
        data = data.copy(deep=True)

        if inchi_column_name in data.columns:
            inchi = data[inchi_column_name].to_list()
            inchi = [ent.strip() for ent in inchi]

            if smiles_column_name not in data.columns:
                smiles = [inchi_to_smiles(ent) for ent in inchi]
            else:
                smiles = data[smiles_column_name].to_list()
                smiles = [ent.strip() for ent in smiles]

            if inchi_key_column_name not in data.columns:
                inchi_key = [Chem.inchi.InchiToInchiKey(ent) for ent in inchi]
            else:
                inchi_key = data[inchi_key_column_name].to_list()
                inchi_key = [ent.strip() for ent in inchi_key]

        elif smiles_column_name in data.columns:
            smiles = data[smiles_column_name].to_list()
            smiles = [ent.strip() for ent in smiles]

            if inchi_column_name not in data.columns:
                inchi = [smiles_to_inchi(ent) for ent in smiles]
            else:
                inchi = data[inchi_column_name].to_list()
                inchi = [ent.strip() for ent in inchi]

            if inchi_key_column_name not in data.columns:
                inchi_key = [Chem.inchi.InchiToInchiKey(ent) for ent in inchi]
            else:
                inchi_key = data[inchi_key_column_name].to_list()
                inchi_key = [ent.strip() for ent in inchi_key]

        else:
            log.error("Neither inchi or smiles column keys in the data set.")
            raise RuntimeError("One of smiles or InChI must be specified")

    else:


        if inchi is not None:
            inchi = [ent.strip() for ent in inchi]
            if smiles is None:
                smiles = [inchi_to_smiles(ent) for ent in inchi]
            else:
                smiles = [ent.strip() for ent in smiles]

            if inchi_key is None:
                inchi_key = [Chem.inchi.InchiToInchiKey(ent) for ent in inchi]
            else:
                inchi_key = [ent.strip() for ent in inchi_key]

        elif smiles is not None:
            smiles = [ent.strip() for ent in smiles]

            if inchi is None:
                inchi = [smiles_to_inchi(ent) for ent in smiles]
            else:
                inchi = [ent.strip() for ent in inchi]

            if inchi_key is None:
                inchi_key = [Chem.inchi.InchiToInchiKey(ent) for ent in inchi]
            else:
                inchi_key = [ent.strip() for ent in inchi_key]

        else:
            log.error("Neither smiles or inchi given to call.")
            raise RuntimeError("One of smiles or InChI must be specified")

        data = pd.DataFrame(data=np.array([smiles, inchi, inchi_key]).T, columns=["smiles", "inchi", "inchikeys"])

    #smiles_column_name "smiles",
    #inchi_column_name"inchi",
    #inchi_key_column_name"inchikey"

    all_col_dup = data[data[[smiles_column_name, inchi_column_name, inchi_key_column_name]].duplicated(
        keep="first")].index.to_list()
    smiles_col_dup = data[data[[smiles_column_name, inchi_column_name, inchi_key_column_name]].duplicated(
        subset=[smiles_column_name], keep="first")].index.to_list()
    inchi_col_dup = data[data[[smiles_column_name, inchi_column_name, inchi_key_column_name]].duplicated(
        subset=[inchi_column_name], keep="first")].index.to_list()
    inchikeys_col_dup = data[data[[smiles_column_name, inchi_column_name, inchi_key_column_name]].duplicated(
        subset=[inchi_key_column_name], keep="first")].index.to_list()

    log.info("id columns = {}, {} and {}".format(smiles_column_name, inchi_column_name, inchi_key_column_name))
    log.info(f"all id columns number of rows which are duplicates: {len(all_col_dup)}\nIndexes: {all_col_dup}\n"
             f"smiles column number of rows which are duplicates: {len(smiles_col_dup)}\nIndexes: {smiles_col_dup}\n"
             f"inchi column number of rows which are duplicates: {len(inchi_col_dup)}\nIndexes: {inchi_col_dup}\n"
             f"inchi keys column number of rows which are duplicates: {len(inchikeys_col_dup)}\nIndexes: "
             f"{inchikeys_col_dup}\n"
             f"out of {len(smiles)} entries\n")

    if compare_which is None:
        return all_col_dup, smiles_col_dup, inchi_col_dup, inchikeys_col_dup
    elif compare_which == "all":
        if return_reduced_data is False:
            return all_col_dup
        else:
            return data.drop_duplicates(subset=[smiles_column_name, inchi_column_name, inchi_key_column_name],
                                        keep="first")
    elif compare_which == "smiles" or compare_which == "smile":
        if return_reduced_data is False:
            return smiles_col_dup
        else:
            return data.drop_duplicates(subset=[smiles_column_name], keep="first")
    elif compare_which == "inchis" or compare_which == "inchi":
        if return_reduced_data is False:
            return inchi_col_dup
        else:
            return data.drop_duplicates(subset=[inchi_column_name], keep="first")
    elif compare_which == "inchikeys" or compare_which == "inchikey":
        if return_reduced_data is False:
            return inchikeys_col_dup
        else:
            return data.drop_duplicates(subset=[inchi_key_column_name], keep="first")

if __name__ == "__main__":
    import doctest
    doctest.testmod()
