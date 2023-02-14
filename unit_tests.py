#!/usr/bin/env python
# Copyright IBM Corporation 2022.
# SPDX-License-Identifier: EPL-2.0


import os
import unittest
import doctest
import re
from pathlib import Path
import pandas as pd
import shutil
import glob
import logging
import json
import networkx as nx




# Module to import to test

import ccsfp.informatics.carbon_capture as cc
import ccsfp.informatics.molecules_and_images as mai
import ccsfp.informatics.finger_prints as fp
import ccsfp.informatics.inchi as inc
import ccsfp.informatics.smiles as sm
import ccsfp.informatics.chemical_space_map as csm
import ccsfp.informatics.generation as ge
import ccsfp.informatics.pubchem_lookup as pcl

import ccsfp.utilities.utility_functions as util



class ccsfpUnitTests(unittest.TestCase):
    """
    The unit tests class for the bayesian optimization of DPD force field wrapper
    """

    def setUp(self):
        """
        Sets up the tests with variables, files and directories
        """

        self.path = os.getcwd()
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataFrameEqual)
        self.addTypeEqualityFunc(pd.Series, self.assertSeriesEqual)


    def assertDataFrameEqual(self, x, y, message:tuple = ("FAIL")):
        """
        Function to test if two pandas dataframes are equal
        :param x: Dataframe to check is equal to y
        :param y: Dataframe to check is equal to x
        :return:
        """
        try:
            pd.testing.assert_frame_equal(x, y)
        except AssertionError as aerr:
            aerr.args += message
            raise aerr

    def assertSeriesEqual(self, x, y, message: tuple = ("FAIL")):
        """
        Function to test if two pandas series are equal
        :param x: Pandas Series to check is equal to y
        :param y: Pandas Series to check is equal to x
        :return:
        """
        try:
            pd.testing.assert_series_equal(x, y)
        except AssertionError as aerr:
            aerr.args += message
            raise aerr

    def test_chemical_space_graph_maker(self):
        """
        Test Graph maker
        """

        df = pd.read_csv(os.path.join(self.path, "Test_data/test_data.csv"), header=0)
        g = csm.build_nx_graph(df, prop_key="made_up_capacity", smiles_key="smiles",label_key="label")
        self.assertIsInstance(g, nx.Graph)

    def test_chemical_space_graph_plotter(self):
            """
            Test graph plotting and node simulation
            """

            df = pd.read_csv(os.path.join(self.path, "Test_data/test_data.csv"), header=0)
            g = csm.build_nx_graph(df, prop_key="made_up_capacity", smiles_key="smiles",label_key="label")

            with open(os.path.join(self.path, "Test_data/graph_positions.json"), "r") as ji:
                target_positions = json.load(ji)
            # We just test x values as array testing through a dictionary is difficult
            target_positions_x = {int(k): float(v[0]) for k, v in target_positions.items()}

            pos, _ = csm.plot_graph(g, weight="weight")
            pos_x = {int(k): float(v[0]) for k, v in pos.items()}

            self.assertDictEqual(target_positions_x, pos_x)

    def test_chemical_space_graph_wrapper(self):
            """
            Test graph plotting and node simulation wrapper
            """

            df = pd.read_csv(os.path.join(self.path, "Test_data/test_data.csv"), header=0)

            g, pos, a = csm.plot_annotated_chemical_space(df, prop_key="made_up_capacity", smiles_key="smiles",
                                                          label_key="label", centroid=(0.7, 0.4), closest_n=4)
            pos_x = {int(k): float(v[0]) for k, v in pos.items()}

            with open(os.path.join(self.path, "Test_data/graph_positions.json"), "r") as ji:
                target_positions = json.load(ji)
            # We just test x values as array testing through a dictionary is difficult and does not really provide a
            # better test overall here
            target_positions_x = {int(k): float(v[0]) for k, v in target_positions.items()}

            self.assertDictEqual(target_positions_x, pos_x)

    def test_check_for_duplicates(self):
        """
        Function to test a utility to find duplicate entries on a molecule dataset
        :return:
        """

        duplicates = util.check_for_duplicates(datafile=os.path.join(self.path, "Test_data/test_set.csv"),
                                            compare_which="all")

        #believed_result = pd.Series(data=[False, False, False, False, True])
        believed_result = [5]

        self.assertListEqual(believed_result, duplicates, ("Failed", "got", "{}".format(duplicates),
                                                             "expected" "{}".format(believed_result)))


def ccsfp_test_finder():
    """
    """
    finder = doctest.DocTestFinder()
    parsedt = doctest.DocTestParser

    if os.path.isdir("ccsfp"):
        os.chdir("ccsfp")
    else:
        print("ERROR - Cannot find directory ccsfp")
        raise RuntimeError()

    for root, dirs, files in os.walk(os.getcwd()):
        fs = [f for f in files if re.search(r".*[a-z].py$", f)]
        print("Matches {} will look for doc tests in these files".format(fs))
        for f in fs:
            source_txt = Path(os.path.join(root, f)).read_text()
            #dt = finder.find(os.path.join(root, f))
            dt = parsedt.parse(source_txt)
            print(dt)


def clean_tests():
    """
    removes all of the test data after the unit tests
    """

    remove_files = glob.glob("*ccus_cp_similarity_image.png") + glob.glob("graph_plot*.png") + glob.glob("graph_positions*.json")

    files_to_delete = ["molecule.xyz", "amine_types.csv", "class.csv", 
            "fingerprint_strings.txt", "fingerprints.csv", "fps_df.csv", "smarts.csv",
            "tanimoto_similarities.csv", "run_example.log", "ccus_fingerprints.log",
            "tmp.csv", "tmp.png", "fps_data.csv", "formulation_features_fp_plus_conc.csv",
            "formulation_features_fp_scaled_by_conc.csv", "scaffolds.csv", "sidechains.csv",
            "all_generated_smiles.csv", "generated_smiles.csv"]

    remove_files = remove_files + files_to_delete 

    # remove testing files
    for remove_file in remove_files:
        try:
            os.remove(remove_file)
        except OSError:
            pass

    # Remove testing directories 
    try:
        shutil.rmtree("dask-worker-space")
    except OSError as err:
        print ("No {} directory to remove after running the unit tests - {}.".format(err.filename, err.strerror))


def run():
    """
    Run doc and unit tests
    """
    log = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(message)s"
    )
    log.setLevel(logging.INFO)

    testSuite = unittest.TestSuite()

    # unit tests in the class ccsfpUnitTests will run all of the tests
    testSuite.addTests(unittest.makeSuite(ccsfpUnitTests))

    # modules to run doctest for
    dt_modules = [cc, mai, fp, inc, sm, csm, ge, util, pcl]
    for dtm in dt_modules:
        log.info("Locating docstring tests in {}".format(dtm))
        testSuite.addTest(doctest.DocTestSuite(dtm))

    log.info("Running docstring and unittests now .....")
    r = unittest.TextTestRunner(verbosity=2).run(testSuite)

    log.info("Cleaning test outputs from directories .....")
    clean_tests()

    # if len(r.failures) > 0:
    #     log.error("FAILED")
    #     raise Exception("FAILED - some unit tests did not pass please fix the mistakes")
    #
    # if len(r.errors) > 0:
    #     log.error("FAILED - errors encountered")
    #     raise Exception("FAILED - some unit tests did not pass and errors were encountered. Please fix the mistakes")

if __name__ == "__main__":
    run()
