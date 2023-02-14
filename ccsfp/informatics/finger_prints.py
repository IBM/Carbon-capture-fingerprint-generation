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
from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs
from rdkit import SimDivFilters

# Logging
import logging

# Dask
import dask

# own modules
from . import molecules_and_images as mai

citation = """@article{mcdonagh2022chemical,
  title={Chemical space analysis and property prediction for carbon capture amine molecules},
  author={McDonagh, James and Zavitsanou, Stamatia and Harrison, Alexander and Zubarev, Dimitry and Wunsch, Benjamin and van Kessel, Theordore and Cipcigan, Flaviu},
  year={2022},
  url={https://chemrxiv.org/engage/chemrxiv/article-details/62e110cbadb01e653cae19f4}
}"""

random_seed = 15791

class ccus_fps(object):


    def __init__(self, fingerprint_version: int = 1, names: list = None, substructures: list = None,
                 log: logging.Logger = None, verbose: bool = True):
        """
        Initialise the class
        :param fingerprint_version: int - version number
        :param names: iterable - list of names of the substructure to use for the fingerprint
        :param substructures: iterable - list of substructure strings in smarts notation
        :param log: logging.Logger - logger object
        :param verbose: bool - print extra information is verbose
        """
        self.version_explanations = {
            1: "This is a filtered set which seems to perform well for modelling this finger print includes rarer groups like"
            "sulphur.",
            2: "This is a filtered set which seems to perform well for modelling but does not include rarer groups like"
            "sulphur containing hetrocycles."
            }

        self.version_names = {
            1: ["ammonia",
              "primary_amine",
              "secondary_amine",
              "tertiary_amine",
              "quaternary_N",
              "imine",
              "nitrogen_bonded_to_carbon",
              "aromatic_N_sp2",
              "carboxylic_acid",
              "primary_alcohol",
              "secondary_alcohol",
              "tertiary_alcohol",
              "t_butyl",
              "carbonyl",
              "halocarbon",
              "benezene_ring",
              "6_member_aromatic_c_and_n_ring",
              "6_member_c_and_o_ring",
              "5_c_ring",
              "5_member_aromatic_c_and_n_ring",
              "5_member_c_and_o_ring",
              "Cyclohexane",
              "Cyclohexylamine",
              "Aniline",
              "benzylamine",
              "piperidine",
              "pyridine",
              "pyrrole",
              "primary_amino_alcohol_two_carbon_separation",
              "secondary_amino_alcohol_two_carbon_separation",
              "tertiary_amino_alcohol_two_carbon_separation",
              "primary_amino_alcohol_three_carbon_separation",
              "secondary_amino_alcohol_three_carbon_separation",
              "tertiary_amino_alcohol_three_carbon_separation",
              "aliphatic_primary_amino_alcohol_two_carbon_separation",
              "aliphatic_secondary_amino_alcohol_two_carbon_separation",
              "aliphatic_tertiary_amino_alcohol_two_carbon_separation",
              "aliphatic_primary_amino_alcohol_three_carbon_separation",
              "aliphatic_secondary_amino_alcohol_three_carbon_separation",
              "aliphatic_tertiary_amino_alcohol_three_carbon_separation",
              "primary_amine_one_carbon_aromatic_group",
              "primary_amine_two_carbon_aromatic_group",
              "primary_amine_three_carbon_aromatic_group",
              "secondary_amine_one_carbon_aromatic_group",
              "secondary_amine_two_carbon_aromatic_group",
              "secondary_amine_three_carbon_aromatic_group",
              "tertiary_amine_one_carbon_aromatic_group",
              "tertiary_amine_two_carbon_aromatic_group",
              "tertiary_amine_three_carbon_aromatic_group",
              "methyl_branch_one_carbon_from_a_N_atom",
              "methyl_branch_two_carbon_from_a_N_atom",
              "methyl_branch_three_carbon_from_a_N_atom",
              "methyl_branch_four_carbon_from_a_N_atom",
              "methyl_branch_five_carbon_from_a_N_atom",
              "methyl_branch_six_carbon_from_a_N_atom",
              "ethyl_chain",
              "propyl_chain",
              "butyl_chain",
              "pentyl_chain",
              "hexyl_chain",
              "poly_primary_and_or_secondary_amine",
              "poly_primary_and_or_secondary_and_or_tertiary_amine",
              "poly_alcohol",
              "pyrazine_aliphatic_C_2_and_5_substitution",
              "pyridine_aliphatic_C_2_and_5_substitution",
              "pyridine_aliphatic_C_2_substitution",
              "Presence_of_Boron",
              "Presence_of_Silicon",
              "Presence_of_Phosphurus",
              "Presence_of_Sulphur",
              "positive_charge_group",
              "negative_charge_group"
              ],
            2:["ammonia", 
              "primary_amine", 
              "secondary_amine", 
              "tertiary_amine", 
              "quaternary_N",
              "aromatic_N_sp2", 
              "carboxylic_acid",
              "primary_alcohol",
              "secondary_alcohol",
              "tertiary_alcohol",
              "t_butyl",
              "carbonyl",
              "halocarbon",
              "benezene_ring",
              "6_member_aromatic_c_and_n_ring",
              "6_member_c_and_o_ring",
              "5_c_ring",
              "5_member_aromatic_c_and_n_ring",
              "5_member_c_and_o_ring",
              "Cyclohexane",
              "Cyclohexylamine",
              "Aniline",
              "benzylamine",
              "piperidine",
              "pyridine",
              "pyrrole",
              "primary_amino_alcohol_two_carbon_separation",
              "secondary_amino_alcohol_two_carbon_separation",
              "tertiary_amino_alcohol_two_carbon_separation",
              "primary_amino_alcohol_three_carbon_separation",
              "secondary_amino_alcohol_three_carbon_separation",
              "tertiary_amino_alcohol_three_carbon_separation",
              "aliphatic_primary_amino_alcohol_two_carbon_separation",
              "aliphatic_secondary_amino_alcohol_two_carbon_separation",
              "aliphatic_tertiary_amino_alcohol_two_carbon_separation",
              "aliphatic_primary_amino_alcohol_three_carbon_separation",
              "aliphatic_secondary_amino_alcohol_three_carbon_separation",
              "aliphatic_tertiary_amino_alcohol_three_carbon_separation",
              "primary_amine_one_carbon_aromatic_group",
              "primary_amine_two_carbon_aromatic_group",
              "primary_amine_three_carbon_aromatic_group",
              "secondary_amine_one_carbon_aromatic_group",
              "secondary_amine_two_carbon_aromatic_group",
              "secondary_amine_three_carbon_aromatic_group",
              "tertiary_amine_one_carbon_aromatic_group",
              "tertiary_amine_two_carbon_aromatic_group",
              "tertiary_amine_three_carbon_aromatic_group",
              "methyl_branch_one_carbon_from_a_N_atom",
              "methyl_branch_two_carbon_from_a_N_atom",
              "methyl_branch_three_carbon_from_a_N_atom",
              "methyl_branch_four_carbon_from_a_N_atom",
              "methyl_branch_five_carbon_from_a_N_atom",
              "methyl_branch_six_carbon_from_a_N_atom",
              "ethyl_chain",
              "propyl_chain",
              "butyl_chain",
              "pentyl_chain",
              "hexyl_chain",
              "poly_primary_and_or_secondary_amine",
              "poly_primary_and_or_secondary_and_or_tertiary_amine",
              "poly_alcohol",
              "pyrazine_aliphatic_C_2_and_5_substitution",
              "pyridine_aliphatic_C_2_and_5_substitution",
              "pyridine_aliphatic_C_2_substitution"
              ]
            }

        self.version_substructures = {
            1: ["[NH3]", # ammonia
                "[NX3;H2][C;!$(C=[#7,#8])]", # 1' amine
                "[NX3;H1][C;!$(C=[#7,#8])][C;!$(C=[#7,#8])]", # 2' amine
                "[NX3]([C;!$(C=[#7,#8])])([C;!$(C=[#7,#8])])[C;!$(C=[#7,#8])]", # 3' amine
                "[NX4+]", # ammonium
                "[N]=[C]", # imine,
                "[#6]~[#7]", # N bonded to C "[$([#6]~[#7]);!$([#6]-[#7])]", # nitrogen bonded to carbon with any bond other than a single bond
                "[a]:[nX3,X2]:[a]", # SP2 aromatic N
                "[CX3;$([#6]),$([O;H1])](=[OX1])[$([O])]", # carboxylic acid
                "[#6][#6;!$(C(=O)[OH])][OH]", # 1' alcohol
                "[#6][#6]([#6])[OH]", # 2' alcohol'
                "[#6][#6]([#6])([#6])[OH]", # 3' alcohol
                "[#6]C([CH3])([CH3])([CH3])", # t-butyl
                "[CX3]=[O;!$(O*)]", # Carbonyl
                "[#6]~[F,Cl,Br,I]", # halo carbon
                "c1ccccc1", # benzene
                "[c,n]1[c,n][c,n][c,n][c,n][c,n]1", # aromatic n or c 6 member hetrocycle
                "[#6,#8]1~[#6,#8]~[#6,#8]~[#6,#8]~[#6,#8]~1", # Any O and C 6 ring
                "[#6]1~[#6]~[#6]~[#6]~[#6]~1", # any C 5 ring
                "[c,n]1[c,n][c,n][c,n][c,n]1",  # aromatic n or c 5 member hetrocycle
                "[#6,#8]1~[#6,#8]~[#6,#8]~[#6,#8]~[#6,#8]~1",  # any O or C 5 member ring system
                "C1CCCCC1", # cyclohexane
                "[NX3;H2,H1][#6]1~[#6]~[#6]~[#6]~[#6]~[#6]~1", # amine bound to ring
                "[NH2]c1ccccc1", # 1' amine bound to benzene
                "c1ccccc1[CH2][NH2]", # benzyl NH2
                "C1N([#1])CCCC1", # H connected to N in an unsaturated ring
                "c1ncccc1", # Pyridine
                "c1n([H])ccc1", # pyrrole
                "[$([#6]([OH])[#6][#7H2]);!$([#6]([OH])(=O)[#6][#7H2])]", # see description
                "[$([#6]([OH])[#6][#7H]([#6]));!$([#6]([OH])(=O)[#6][#7H]([#6]))]", # see description
                "[$([#6]([OH])[#6][#7]([#6])([#6]));!$([#6]([OH])(=O)[#6][#7]([#6])([#6]))]", # see description
                "[$([#6]([OH])[#6][#6][#7H2]);!$([#6]([OH])(=O)[#6][#6][#7H2])]", # see description
                "[$([#6]([OH])[#6][#6][#7H]([#6]));!$([#6]([OH])(=O)[#6][#6][#7H]([#6]))]", # see description
                "[$([#6]([OH])[#6][#6][#7]([#6])([#6]));!$([#6]([OH])(=O)[#6][#6][#7]([#6])([#6]))]",
                "C([#6,#1])([#6,#1])([OH])C([#6,#1])([#6,#1])[#7H2]", # see description
                "C([#6,#1])([#6,#1])([OH])C([#6,#1])([#6,#1])[#7H]([CX4])", # see description
                "C([#6,#1])([#6,#1])([OH])C([#6,#1])([#6,#1])[#7]([CX4])([CX4])", # see description
                "C([#6,#1])([#6,#1])([OH])C([#6,#1])([#6,#1])C([#6,#1])([#6,#1])[NH2]", # see description
                "C([#6,#1])([#6,#1])([OH])C([#6,#1])([#6,#1])C([#6,#1])([#6,#1])[NH]([CX4])", # see description
                "C([#6,#1])([#6,#1])([OH])C([#6,#1])([#6,#1])C([#6,#1])([#6,#1])[N]([CX4])([CX4])", # see description
                "[a][C][#7H2]", # see description
                "[a][C][C][#7H2]", # see description
                "[a][C][C][C][#7H2]", # see description
                "[a][C][#7H]([#6])", # see description
                "[a][C][C][#7H]([#6])", # see description
                "[a][C][C][C][#7H]([#6])", # see description
                "[a][C][#7]([#6])([#6])", # see description
                "[a][C][C][#7]([#6])([#6])", # see description
                "[a][C][C][C][#7]([#6])([#6])", # see description
                "[NH2][CX4]([CH3])", # see description
                "[NH2][CX4][CX4]([CH3])", # see description
                "[NH2][CX4][CX4][CX4]([CH3])", # see description
                "[NH2][CX4][CX4][CX4][CX4]([CH3])", # see description
                "[NH2][CX4][CX4][CX4][CX4][CX4]([CH3])", # see description
                "[NH2][CX4][CX4][CX4][CX4][CX4][CX4]([CH3])", # see description
                "[CX4;H2][CX4;H2]", # see description
                "[CX4;H2][CX4;H2][CX4;H2]", # see description
                "[CX4;H2][CX4;H2][CX4;H2][CX4;H2]", # see description
                "[CX4;H2][CX4;H2][CX4;H2][CX4;H2][CX4;H2]", # see description
                "[CX4;H2][CX4;H2][CX4;H2][CX4;H2][CX4;H2][CX4;H2]", # see description
                "[$([#7X3;H2][C;!$(C=[#7,#8])]),$([#7X3;H1]([C;!$(C=[#7,#8])])[C;!$(C=[#7,#8])])].[$([#7X3;H2][C;!$(C=[#7,#8])])," \
                "$([#7X3;H1]([C;!$(C=[#7,#8])])[C;!$(C=[#7,#8])])]",  # poly 1' 2' amine
                "[$([#7X3;H2][C;!$(C=[#7,#8])]),$([#7X3;H1]([C;!$(C=[#7,#8])])[C;!$(C=[#7,#8])]),$([#7X3]([C;!$(C=[#7,#8])])"\
                "([C;!$(C=[#7,#8])])[C;!$(C=[#7,#8])])].[$([#7X3;H2][C;!$(C=[#7,#8])]),$([#7X3;H1]([C;!$(C=[#7,#8])])[C;!$(C=[#7,#8])]),"\
                "$([#7X3]([C;!$(C=[#7,#8])])([C;!$(C=[#7,#8])])[C;!$(C=[#7,#8])])]",  # poly 1' 2' or 3' amine
                "[#6][O;H1].[#6][O;H1]", # poly alcohol
                "n1c([CX4])cnc([CX4])c1", # pyrazine aliphatic C2 and C5 substitution
                "n1c([CX4])ccc([CX4])c1", # pyridine_aliphatic_C_2_and_5_substitution
                "n1cccc([CX4])c1", # pyridine_aliphatic_C_2_substitution
                "[#5]", # B
                "[#14]", # Si
                "[#15]", # P
                "[#16]", # S
                "[+]", # positive cahrged group
                "[-]"  # negative charge group 
                ],
            2: ["[NH3]", # ammonia
                "[NX3;H2][CX4;!$(C=[#7,#8])]", # 1' amine
                "[NX3;H1][CX4;!$(C=[#7,#8])][CX4;!$(C=[#7,#8])]", # 2' amine
                "[NX3]([CX4;!$(C=[#7,#8])])([CX4;!$(C=[#7,#8])])[CX4;!$(C=[#7,#8])]", # 3' amine
                "[NX4+]", # ammonium
                "[a]:[nX3,X2]:[a]", # SP2 aromatic N
                "[CX3;$([#6]),$([O;H1])](=[OX1])[$([O])]", # carboxylic acid
                "[#6][#6;!$(C(=O)[OH])][OH]", # 1' alcohol
                "[#6][#6]([#6])[OH]", # 2' alcohol'
                "[#6][#6]([#6])([#6])[OH]", # 3' alcohol
                "[#6]C([CH3])([CH3])([CH3])", # t-butyl
                "[CX3]=[O;!$(O*)]", # Carbonyl
                "[#6]~[F,Cl,Br,I]", # halo carbon
                "c1ccccc1", # benzene
                "[c,n]1[c,n][c,n][c,n][c,n][c,n]1", # aromatic n or c 6 member hetrocycle
                "[#6,#8]1~[#6,#8]~[#6,#8]~[#6,#8]~[#6,#8]~1", # Any O and C 6 ring
                "[#6]1~[#6]~[#6]~[#6]~[#6]~1", # any C 5 ring
                "[c,n]1[c,n][c,n][c,n][c,n]1",  # aromatic n or c 5 member hetrocycle
                "[#6,#8]1~[#6,#8]~[#6,#8]~[#6,#8]~[#6,#8]~1",  # any O or C 5 member ring system
                "C1CCCCC1", # cyclohexane
                "[NX3;H2,H1][#6]1~[#6]~[#6]~[#6]~[#6]~[#6]~1", # amine bound to ring
                "[NH2]c1ccccc1", # 1' amine bound to benzene
                "c1ccccc1[CH2][NH2]", # benzyl NH2
                "C1N([#1])CCCC1", # H connected to N in an unsaturated ring
                "c1ncccc1", # Pyridine
                "c1n([H])ccc1", # pyrrole
                "[$([#6]([OH])[#6][#7H2]);!$([#6]([OH])(=O)[#6][#7H2])]", # see description
                "[$([#6]([OH])[#6][#7H]([#6]));!$([#6]([OH])(=O)[#6][#7H]([#6]))]", # see description
                "[$([#6]([OH])[#6][#7]([#6])([#6]));!$([#6]([OH])(=O)[#6][#7]([#6])([#6]))]", # see description
                "[$([#6]([OH])[#6][#6][#7H2]);!$([#6]([OH])(=O)[#6][#6][#7H2])]", # see description
                "[$([#6]([OH])[#6][#6][#7H]([#6]));!$([#6]([OH])(=O)[#6][#6][#7H]([#6]))]", # see description
                "[$([#6]([OH])[#6][#6][#7]([#6])([#6]));!$([#6]([OH])(=O)[#6][#6][#7]([#6])([#6]))]",
                "C([#6,#1])([#6,#1])([OH])C([#6,#1])([#6,#1])[#7H2]", # see description
                "C([#6,#1])([#6,#1])([OH])C([#6,#1])([#6,#1])[#7H]([CX4])", # see description
                "C([#6,#1])([#6,#1])([OH])C([#6,#1])([#6,#1])[#7]([CX4])([CX4])", # see description
                "C([#6,#1])([#6,#1])([OH])C([#6,#1])([#6,#1])C([#6,#1])([#6,#1])[NH2]", # see description
                "C([#6,#1])([#6,#1])([OH])C([#6,#1])([#6,#1])C([#6,#1])([#6,#1])[NH]([CX4])", # see description
                "C([#6,#1])([#6,#1])([OH])C([#6,#1])([#6,#1])C([#6,#1])([#6,#1])[N]([CX4])([CX4])", # see description
                "[a][C][#7H2]", # see description
                "[a][C][C][#7H2]", # see description
                "[a][C][C][C][#7H2]", # see description
                "[a][C][#7H]([#6])", # see description
                "[a][C][C][#7H]([#6])", # see description
                "[a][C][C][C][#7H]([#6])", # see description
                "[a][C][#7]([#6])([#6])", # see description
                "[a][C][C][#7]([#6])([#6])", # see description
                "[a][C][C][C][#7]([#6])([#6])", # see description
                "[NH2][CX4]([CH3])", # see description
                "[NH2][CX4][CX4]([CH3])", # see description
                "[NH2][CX4][CX4][CX4]([CH3])", # see description
                "[NH2][CX4][CX4][CX4][CX4]([CH3])", # see description
                "[NH2][CX4][CX4][CX4][CX4][CX4]([CH3])", # see description
                "[NH2][CX4][CX4][CX4][CX4][CX4][CX4]([CH3])", # see description
                "[CX4;H2][CX4;H2]", # see description
                "[CX4;H2][CX4;H2][CX4;H2]", # see description
                "[CX4;H2][CX4;H2][CX4;H2][CX4;H2]", # see description
                "[CX4;H2][CX4;H2][CX4;H2][CX4;H2][CX4;H2]", # see description
                "[CX4;H2][CX4;H2][CX4;H2][CX4;H2][CX4;H2][CX4;H2]", # see description
                "[$([#7X3;H2][CX4;!$(C=[#7,#8])]),$([#7X3;H1]([CX4;!$(C=[#7,#8])])[CX4;!$(C=[#7,#8])])].[$([#7X3;H2][CX4;!$(C=[#7,#8])])," \
                "$([#7X3;H1]([CX4;!$(C=[#7,#8])])[CX4;!$(C=[#7,#8])])]",  # poly 1' 2' amine
                "[$([#7X3;H2][CX4;!$(C=[#7,#8])]),$([#7X3;H1]([CX4;!$(C=[#7,#8])])[CX4;!$(C=[#7,#8])]),$([#7X3]([CX4;!$(C=[#7,#8])])"\
                "([CX4;!$(C=[#7,#8])])[CX4;!$(C=[#7,#8])])].[$([#7X3;H2][CX4;!$(C=[#7,#8])]),$([#7X3;H1]([CX4;!$(C=[#7,#8])])[CX4;!$(C=[#7,#8])]),"\
                "$([#7X3]([CX4;!$(C=[#7,#8])])([CX4;!$(C=[#7,#8])])[CX4;!$(C=[#7,#8])])]",  # poly 1' 2' or 3' amine
                "[#6][O;H1].[#6][O;H1]", # poly alcohol
                "n1c([CX4])cnc([CX4])c1", # pyrazine aliphatic C2 and C5 substitution
                "n1c([CX4])ccc([CX4])c1", # pyridine_aliphatic_C_2_and_5_substitution
                "n1cccc([CX4])c1" # pyridine_aliphatic_C_2_substitution
                ]
            }

        try:
            log.info("\n")
        except Exception:
            log = logging.getLogger(__name__)
            log.info("\n")
        
        self.fingerprint_version = fingerprint_version

        if names is None:
            self.names = self.get_default_names(version=self.fingerprint_version)
        else:
            self.names = names

        
        if substructures is None:
            self.substructures = self.get_default_substructures(version=self.fingerprint_version)
        else:
            self.substructures = substructures
        
        if substructures is None and names is None :
            self.fingerprint_explanation = self.get_default_explanation(version=self.fingerprint_version)
            if verbose is True:
                log.info("Finger print is version {}\n{}".format(self.fingerprint_version, self.fingerprint_explanation))
        else:
            log.warning("No fingerprint explanation avaliable as custom substructures have been given, hence you know better than I do what they mean.")


        if len(self.names) != len(self.substructures):
            try:
                log.warning("WARNING - the number of names ({}) and the number of substructures ({}) is different, " \
                        "This will cause issues for defualt functions in this module. Names will be reset to indexes.".format(len(self.names), len(self.substructures)))
                self.names = [str(ith) for ith in enumerate(self.substructures)]
                log.warning("New names and substructures:\n{}".format("\n".join(["{} : {}".format(n, s) for n, s in zip(self.names, self.substructures)])))
            except NameError:
                print("WARNING - the number of names ({}) and the number of substructures ({}) is different, " \
                        "This will cause issues for defualt functions in this module.".format(len(self.names), len(self.substructures)))

        log.info("Please use the citation below for use of this code:\n{}".format(citation))

    def get_default_names(self, version: int = None) -> list:
        """
        Function to get the names descriptive names of the substructures we are looking for
        Essentially these substructures look for the amine environment and groups which can interact with it. The first
        elements identify specific groups. The later elements look for the motifs of closeness of certain functional
        groups to the amine groups.
        :param version: int - which version of the fingerprints to get name for 
        """

        if version is None :
            version = self.fingerprint_version

        return self.version_names[version]

    def get_default_substructures(self, version: int = None) -> list:
        """
        Function get the smarts to search for substructures. Essentially these substructures look for the amine
        environment and groups which can interact with it. The first elements identify specific groups. The
        later elements look for the motifs of closeness of certain functional groups to the amine groups.
        :param version: int - which version of the fingerprints to get substructures for 
        """

        if version is None :
            version = self.fingerprint_version

        return self.version_substructures[version]

    def get_default_explanation(self, version: int = None) -> str:
        """
        Function to get the description of the version you have picked  names of the substrictires we are looking for
        Essentially these substructures look for the amine environment and groups which can interact with it. The first
        elements identify specific groups. The later elements look for the motifs of closeness of certain
        functional groups to the amine groups.
        :param version: int - which version of the fingerprints to get explanations for 
        """
        
        if version is None :
            version = self.fingerprint_version

        return self.version_explanations[version]

    def get_fp_information(self, return_df: bool = False):
        """
        Print the infomration related to the current fingerprint instantiation
        """

        log = logging.getLogger(__name__)

        log.info("{:4} | {:59} | {:50}\n---------------------------------------------------------"
         "-------------------------------------".format("bit", "description", "smarts"))

        names = []
        for nam in self.names:
            nam = " ".join(nam.split("_"))
            nam = " ".join(nam.split("-"))
            names.append(nam)
            
        for ith, ds in enumerate(zip(names, self.substructures)):
            log.info("{:4} | {:59} | {:50}".format(ith, ds[0], ds[1]))
            
        if return_df is True:
            df_information = pd.DataFrame(np.array([names, self.substructures]).T, columns=["description", "smarts"])
            return df_information

############## END of class ############

def maccskeys_fingerprints(smiles: list) -> list:
    """
    Function to get MACCS fingerprints
    :param smiles: list - smiles representation of the molecules to make fingerprints of
    """
    
    log = logging.getLogger(__name__)
    
    mols = [mai.smiles_to_molecule(smile) for smile in smiles]
    fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
    
    return fps

def dask_substructure_checker(representation: str, substructures: list = None, smiles=True) -> list:
    """ 
    Function to find a substructure using SMARTS - Does not use dask but is used in functions that dask is used in
    :param smi: str - smiles
    :param substructures: iterable - SMARTS defining the substructure to search for
    :return: tuple - smiles substructure and True/False for looking for the substructure
    >>> dask_substructure_checker("CC", ["*CC*"])
    [1]
    """

    log = logging.getLogger(__name__)

    if smiles is True:
        mol = mai.smiles_to_molecule(representation)
    else:
        mol = mai.inchi_to_molecule(representation)

    substructs = [Chem.MolFromSmarts(substructure) for substructure in substructures]

    fp_vec = [int(mol.HasSubstructMatch(substruct)) for substruct in substructs]
    
    return fp_vec

def ccs_fp(representation: list, substructures: list = None, substructure_names: list = None,
           return_smarts_only: bool = False, version: int = 1,
           thresh: int = 1000, return_only_fingerprint: bool = False, return_fingerprints_as_str: bool = False,
           inchi_regex="InChI="):
    """
    Function to make a fingerprint out of the presence or not of a sub-structure using SMARTS. Note this uses
    Lazy dask parallel execution to make the porcess run in parallel if number of representations is >= thresh.
    :param representation: tuple/list - smiles representations of molecules to check for substructure presence or absence
    :param substructures: tuple or list - SMARTS tuple/list to look for to form the fingerprint
    :param substructure_names: tuple or list - names of the substructure if given a dataframe is returned as well
    :param return_smarts_only: bool - return only the keys for teh smarts in the substructure search
    :param version: int - version of the predefined ccs fingerprint to use
    :param thresh: int - number of smiles under which run in serial over or equal run in parallel with dask
    :param return_only_fingerprint: bool - return only the fingerprints
    :param inchi_regex: str - string to make sure a representation is inchi
    :Returns: list, dask dataframe, list
    >>> ccs_fp(["CCN"], return_fingerprints_as_str=True)
    ['010000100000000000000000000000000000000000000000010000000000000000000000']
    >>> ccs_fp(["InChI=1S/C2H7N/c1-2-3/h2-3H2,1H3"], return_fingerprints_as_str=True)
    ['010000100000000000000000000000000000000000000000010000000000000000000000']
    >>> ccs_fp(["CCN"], return_fingerprints_as_str=True, version=2)
    ['0100000000000000000000000000000000000000000000010000000000000000']
    >>> ccs_fp(["InChI=1S/C2H7N/c1-2-3/h2-3H2,1H3"], return_fingerprints_as_str=True, version=2)
    ['0100000000000000000000000000000000000000000000010000000000000000']
    """
    log = logging.getLogger(__name__)

    # Essentially these substructures look for the amine environment and groups which can interact with it. The
    # first elements identify specific groups. The later elements look for the motifs of closeness of certain
    # functional groups to the amine groups.
    ccus_substructs = ccus_fps(names=substructure_names, substructures=substructures, fingerprint_version=version,
                               log=log)
    substructure_names = ccus_substructs.names
    substructures = ccus_substructs.substructures


    if return_smarts_only is True:
        return substructures
    
    log.info("Number of substructures: {} Number of substructure names: {}".format(len(substructures),
                                                                                   len(substructure_names)))
    if len(substructures) != len(substructure_names):
        log.error("Number of substructures and names differ cannot produce dataframe: {} "
                  "{}".format(len(substructures), len(substructure_names)))
        for s, n in zip(substructures, substructure_names):
            log.info("{} {}".format(n, s))

    # ASSUMPTION: no one puts a mix of smiles and inchi in
    if inchi_regex in representation[0]:
        log.info("'inchi=' found in the first molecule, assume all molecules will be inchi not smiles!")
        inchi = representation

        log.info("Making fingerprint from {} InChI".format(len(inchi)))

        # chunk larger datasets manually
        if len(inchi) >= thresh:
            fps = []
            iters = int(np.floor(len(inchi) / thresh)) + 1
            limit = int(iters)
            bases = [0 + i * thresh for i in range(iters)]
            uppers = [thresh + i * thresh for i in range(iters)]
            uppers[-1] = None
            for b, u in zip(bases, uppers):
                log.info("fingerprints computed for {} InChI".format(b))
                log.info("InChI[{}:{}]".format(b, u))
                inchs = inchi[b:u]

                fp_tmp = [dask.delayed(dask_substructure_checker)(inc, substructures=substructures, smiles=False)
                          for inc in inchs]
                fps = fps + fp_tmp

            # compute fingerprints
            log.info("Running DASK computation .....")
            fps = dask.compute(*fps)
            log.info("DASK complete fingerprints generated.")
        else:
            log.info("Length of the InChI list is less than the threshold ({} change through function call) running "
                     "without DASK".format(thresh))
            fps = [dask_substructure_checker(inch, substructures=substructures, smiles=False) for inch in inchi]
            fps = dask.compute(*fps)

        log.info("Preparing fingerprints")
        if substructure_names is not None:
            log.info("Building dataframe .....")
            df = pd.DataFrame(data=fps, columns=substructure_names)
            log.info("Building RDKit bits .....")
            ffps = [DataStructs.cDataStructs.CreateFromBitString("".join([str(ent) for ent in f])) for f in fps]
            log.info("Fingerprint generation finished.")

            if return_fingerprints_as_str is True:
                return [bits_to_text(f) for f in ffps]
            elif return_only_fingerprint is True:
                return ffps
            else:
                return ffps, df, substructures
        else:
            ffps = [DataStructs.cDataStructs.CreateFromBitString("".join([str(ent) for ent in f])) for f in fps]

            if return_fingerprints_as_str is True:
                return [bits_to_text(f) for f in ffps]
            elif return_only_fingerprint is True:
                return ffps
            else:
                return ffps, substructures

    else:
        log.info("'inchi=' not found in the first molecule, assume all molecules will be SMILES not InChI!")
        smiles = representation

        log.info("Making fingerprint from {} SMILES".format(len(smiles)))

        # chunk larger datasets manually
        if len(smiles) >= thresh:
            fps = []
            iters = int(np.floor(len(smiles)/thresh)) + 1
            limit = int(iters)
            bases = [0 + i * thresh for i in range(iters)]
            uppers = [thresh + i * thresh for i in range(iters)]
            uppers[-1] = None
            for b, u in zip(bases, uppers):
                log.info("fingerprints computed for {} smiles".format(b))
                log.info("smiles[{}:{}]".format(b, u))
                smis = smiles[b:u]

                fp_tmp = [dask.delayed(dask_substructure_checker)(smi, substructures=substructures, smiles=True)
                          for smi in smis]
                fps = fps + fp_tmp

            # compute fingerprints
            log.info("Running DASK computation .....")
            fps = dask.compute(*fps)
            log.info("DASK complete fingerprints generated.")
        else:
            log.info("Length of the smiles list is less than the threshold ({} change through function call) running "
                     "without DASK".format(thresh))
            fps = [dask_substructure_checker(smi, substructures=substructures, smiles=True) for smi in smiles]
            fps = dask.compute(*fps)

        log.info("Preparing fingerprints")
        if substructure_names is not None:
            log.info("Building dataframe .....")
            df = pd.DataFrame(data=fps, columns=substructure_names)
            log.info("Building RDKit bits .....")
            ffps = [DataStructs.cDataStructs.CreateFromBitString("".join([str(ent) for ent in f])) for f in fps]
            log.info("Fingerprint generation finished.")

            if return_fingerprints_as_str is True:
                return [bits_to_text(f) for f in ffps]
            elif return_only_fingerprint is True:
                return ffps
            else:
                return ffps, df, substructures
        else:
            ffps = [DataStructs.cDataStructs.CreateFromBitString("".join([str(ent) for ent in f])) for f in fps]

            if return_fingerprints_as_str is True:
                return [bits_to_text(f) for f in ffps]
            elif return_only_fingerprint is True:
                return ffps
            else:
                return ffps, substructures

def substructure_checker(smiles: str, substructure: str = None) -> int:
    """ 
    Function to find a substructure using SMARTS
    :param smi: str - smiles
    :param substructure: str - SMARTS defining the substructure to search for
    :return: tuple - smiles substructure and True/False for looking for the substructure
    >>> substructure_checker("CC", "*CC*")
    1
    """
        
    mol = mai.smiles_to_molecule(smiles)

    substruct = Chem.MolFromSmarts(substructure)

    has_substructure = 0
    if mol.HasSubstructMatch(substruct):
        has_substructure = 1 

    return has_substructure

def fingerprint_similarity(fps1, fps2, dice: bool = False, return_distance: bool = False) -> float:
    """
    Function to calculate fingerprint similarity
    :param fps1: RDKit fingerprint - fingerprint of molecule 1
    :param fps2: RDKit fingerprint - fingerprint of molecule 2
    :param dice: true/false - Use dice similarity
    :param return_distance: bool - return distance (1 - similarity)
    """
    
    if dice is True:
        similarity = dice_similarity(fps1, fps2)
    else:
        similarity = DataStructs.TanimotoSimilarity(fps1, fps2)

    if return_distance is True:
        similarity = 1.0 - similarity

    return similarity

def bits_to_text(fp) -> str:
    """
    Function to convert bit vec to text 0s and 1s
    :param fp: RDKit bit fingerprint - RDKit bit fingerprint to be set to 1s and 0s
    >>> bits_to_text(maccskeys_fingerprints(["CC"])[0])
    '00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000100000000001000000'
    """
    
    text = DataStructs.cDataStructs.BitVectToText(fp)
    
    return text

def bulk_similarity(fp, fp_targets: list, test: bool = False, thresh: float = 0.5) -> pd.DataFrame:
    """
    Function to compare one fp with a list of others and get all the scores
    :param fp: RDKit fingerprint - fingerprint to compare to a list of fingerprint targets
    :param fp_targets: list - fingerprint targets to compare fp to
    :param test: bool - return only molecules with similarity greater than or equal to the thresh
    :param thresh: float - threshold for similarity to be returned
    :return:
    """
    
    tani_similarity = DataStructs.BulkTanimotoSimilarity(fp, fp_targets)
    data = np.array([[i for i in range(0, len(fp_targets))],[fp]*len(fp_targets), fp_targets, tani_similarity]).T
    df = pd.DataFrame(data=data, columns=["number", "fp_reference", "fp_target", "tanimoto_similarity"])
    
    if test is True:
        df = df[df["tanimoto_similarity"] >= thresh]
    return df

def dice_similarity(v1, v2):
    """
    Function to return dice similarity between two bitvectors
    :param v1: RDKit bit vector - chemcical represention as bit vector eg a bit vector fingerprint
    :param v2: RDKit bit vector - chemcical represention as bit vector eg a bit vector fingerprint
    """

    return DataStructs.DiceSimilarity(v1, v2)

def diverse_set_picking(fps: list, n_diverse_batch: int = 10):
    """
    A function using the commonly applied maxmin picking methods https://onlinelibrary.wiley.com/doi/epdf/10.1002/qsar.200290002
    essentially the algorithm selects a seed molecule calculates dissimilarlity from a fingerprint distance metric
    and adds the most dissimilar molecule to the set. This is stopped when either n molecules are picked or m threshold
    in the distance metric is surpassed by all molecules.
    :param fps: list of RDKit fingerprint - molecule fingerprints to use to pick a diverse set from
    :param n_diverse_batch : int - the number of set members to include in the diverse set
    """

    log = logging.getLogger(__name__)

    diversity_picker = SimDivFilters.rdSimDivPickers.MaxMinPicker()
    
    number_fps = len(fps)
    diverse_indices = diversity_picker.LazyBitVectorPick(fps, poolSize=number_fps, pickSize=n_diverse_batch,
                                                         seed=random_seed)

    log.debug("Diverse indices: {}".format(list(diverse_indices)))

    return diverse_indices


def contains_substructures(smiles: list , substructures: tuple =("[NH3]",
                                    "[NX3;H2][C;!$(C=[#7,#8])]",
                                    "[NX3;H1]([C;!$(C=[#7,#8])])[C;!$(C=[#7,#8])]",
                                    "[NX3]([C;!$(C=[#7,#8])])([C;!$(C=[#7,#8])])[C;!$(C=[#7,#8])]",
                                    "[$([nX3,X2](:[c,n,o,b,s]):[c,n,o,b,s])]"),

             substructure_names: tuple = ("ammonia",
                                   "primiary_amine", 
                                   "secondary_amine", 
                                   "tertiary_amine", 
                                   "aromatic_sp2_n"),
            version_name: int = 1,
            thresh: int =1000,
            remove_no_match_rows : bool = False,
            test: bool = False
            ):
    """
    Function to check if the smiles are amines or contain an aromatic N sp2
    :param smiles: str - smiles string to look for substructure
    :param substructures: iterable of str - smarts patterns to look for
    :param substructure_names: iterable of str - description of the SMARTS patterns
    :param version_name: int - ccs fp version number
    :param thresh: int - batch threshold for fingerprint code
    :param remove_no_match_rows: bool - remove rows with no matches
    :param test: bool - for testing the function
    :return: dataframe

    """
    
    log = logging.getLogger(__name__)
    
    log.info("Passing smiles and substructures to dask fp")
    log.info("Substructures:\n{}\n-----\n".format("\n".join(["{} ; {}".format(n, s) for n, s in zip(substructure_names,
                                                                                                    substructures)])))

    if isinstance(smiles, str):
        log.info("Smiles is expected to be a list assume it is one smiles and put in a list")
        smiles = [smiles]
    
    fingps, fingps_df, smarts = ccs_fp(smiles,
                                       substructures=substructures,
                                       substructure_names=substructure_names,
                                       return_smarts_only=False,
                                       version=version_name,
                                       thresh=thresh
                                      )
    
    any_true = fingps_df.any(axis=1)
    log.info("The following rows have at least one of the substructures found: {}".format(any_true))
    fingps_df["any_true"] = any_true

    # dataframe index values which have ata least one matching substructure 
    idx = fingps_df.index[fingps_df["any_true"] == 0]

    if remove_no_match_rows is True:
        log.info("{}".format(fingps_df))
        log.info("Dropping rows: {}".format(idx))
        fingps_df.drop(idx, axis=0, inplace=True)

    return fingps_df


def ccus_fp_bitstr(mol: rdkit.Chem.rdchem.Mol, substructures: list = None, substructure_names: list = None,
                   version: int = 1):
    """
    Function to find a substructure using SMARTS - Does not use dask but is used in functions that dask is used in
    return the ccus fingerprint as a cDatastructs array.
    :param mol: str - RDkit molecule
    :param substructures: iterable - SMARTS defining the substructure to search for
    :param substructure_names: iterable - names to describe the SMARTS substructure strings meaning
    :param version: int - version of the fingerprints to use
    :return: bitstr
    """

    log = logging.getLogger(__name__)

    if substructures is None:
        ccus_substructs = ccus_fps(names=substructure_names, substructures=substructures, fingerprint_version=version,
                                   verbose=False)
        if substructure_names is None:
            substructure_names = ccus_substructs.names
        substructures = ccus_substructs.substructures
    if substructure_names is None:
        ccus_substructs = ccus_fps(names=substructure_names, substructures=substructures, fingerprint_version=version,
                                   verbose=False)
        if substructures is None:
            substructures = ccus_substructs.substructures
        substructure_names = ccus_substructs.names

    substructs = [Chem.MolFromSmarts(substructure) for substructure in substructures]
    fps = [int(mol.HasSubstructMatch(substruct)) for substruct in substructs]
    ffps = DataStructs.cDataStructs.CreateFromBitString("".join([str(ent) for ent in fps]))

    return ffps

if __name__ == "__main__":
    import doctest
    doctest.testmod()
