#!/usr/bin/env python
# Copyright IBM Corporation 2022.
# SPDX-License-Identifier: MIT

# https://www.rdkit.org/docs/GettingStartedInPython.html
# creative commons sa 4.0 tutorial used to learn rdkit methods
# https://creativecommons.org/licenses/by-sa/4.0/
# (C) 2007-2021 by Greg Landrum

"""
This module is intended to allow one to easily build a force directed graph of chemical space using the networkx python modules.

The codes require a pandas dataframe containing columns of names, smiles and a property of interest. The column names and the dataframe are
provided as input. The nodes are shaded based on the proptery and connected based on Tanimoto similarity from the smiles strings. A node 2D
coordinate set can be passed and the nearest molecules nodes will be annoted with moleucles images and names.
"""


# Python packages and utilities
import time
import pandas as pd
import numpy as np
import json
import os

#RDKit
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import DataStructs

# Logging
import logging

# networkx
import networkx as nx

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# own modules
from ccsfp.informatics import molecules_and_images as mai
from ccsfp.informatics import finger_prints as fp

random_seed = 15791

def build_nx_graph(df : pd.DataFrame,
                   prop_key : str = "property",
                   smiles_key : str = "smiles",
                   label_key : str = "names",
                   distance : bool = False,
                   fingerprints : list = None,
                   connection_threshold : float = 0.7,
                   connect_all : bool = False,
                   graph_file_name : str = None,
                   similarity_distance_list : list = None,
                   node_attributes : list = None) -> nx.Graph :
    """
    Function to build a defualt graph for chemical space map plots
    :param df: pandas dataframe - dataframe with at least a label column, smiles column and property column
    :param prop_key: str - column key for property of interest values
    :param smiles_key: str - column key for smiles of molecule srings
    :param label_key: str - column key for each entries labels
    :param distance: bool - use distance rather than similarity (distance = 1.0 - similarity)
    :param fingerprints: list - if None will use Morgan fingerprint radius 2 and 2048 bits users can input a list
                                otherwise of precomputed fingerprints from RDkit or ccsfp finger_print module should
                                be an RDKit ExplicitBitVect
    :param connection_threshold: float - the value that must be >= the similarity of distance to connect (form an edge)
                                         between two nodes
    :param connect_all: bool - connect all nodes with and apply to edge colour the value of grey those edges that don't
                               meet the threshold
    :param graph_file_name: str - if None don't save a pickle (pkl) of the graph other wise save a pkl file of the graph
                                  with this filename
    :param similarity_distance_list: list - pre-computed list of the similarity or distance list if none these are
                                            computed as Tanimoto similarities or distances
    :param node_attributes: : list of dicts - Should be in the same order as the rows of the df the list index which
                                              maps to the row index of df will have its dictionary added as node
                                              attributes
    :return: networkx.Graph() object
    """

    log = logging.Logger(__name__)

    log.info("Starting to build the chemical space graph, this can take quite some time .....")
    if fingerprints is None:
        log.info("No fingerprints given by user, will use Morgan circular fingerprints radius 2 bit length 2048.")
        fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), radius=2, nBits=2048) for s in
           df[smiles_key].values]
    else:
        fps = fingerprints

    log.info("Fingerprints: {}".format(fps))

    if similarity_distance_list is None:
        log.info("No smilarity/distance list given by user, will use Tanimoto metric")
        tc = [DataStructs.cDataStructs.BulkTanimotoSimilarity(fps[inx], fps) for inx in range(len(fps))]
    else:
        tc = similarity_distance_list

    log.info("Tanimoto similarity: {}".format(tc))

    # similarity or distance metrics - Not if you want to use a custom distance leave distance as False and pass in
    df_weights = pd.DataFrame(data=np.array(tc))
    if distance is True:
        df_weights = 1.0 - df_weights


    # initialize graph
    g = nx.Graph()

    # Start to build the graph with the nodes from the raw data
    if node_attributes is not None:

        for index, row in df.iterrows():
            log.debug("Index: {}".format(index))

            na = node_attributes[index]
            if not isinstance(dict, na):
                log.warning("WARNING - node_attribute index {} is not a dictionary {}\n{}".format(index, type(na), na))
                raise RuntimeError("ERROR - node attribute {} is not a dictionary".format(index))

            m = mai.smiles_to_molecule(row[smiles_key])

            g.add_node(index,
                       smiles=row[smiles_key],
                       name=row[label_key],
                       fingerprint=fps[index],
                       fingerprint_string=fp.bits_to_text(fps[index]),
                       inchi=Chem.inchi.MolToInchi(m),
                       inchikey=Chem.inchi.MolToInchiKey(m),
                       mr=Chem.rdMolDescriptors.CalcExactMolWt(m),
                       prop=row[prop_key],
                       **na
                      )

    else:
        for index, row in df.iterrows():
            log.debug("Index: {}".format(index))

            m = mai.smiles_to_molecule(row[smiles_key])

            g.add_node(index,
                       smiles=row[smiles_key],
                       name=row[label_key],
                       fingerprint=fps[index],
                       fingerprint_string=fp.bits_to_text(fps[index]),
                       inchi=Chem.inchi.MolToInchi(m),
                       inchikey=Chem.inchi.MolToInchiKey(m),
                       mr=Chem.rdMolDescriptors.CalcExactMolWt(m),
                       prop=row[prop_key]
                       )

    log.info("Number of nodes {}".format(len(list(g.nodes))))

    # Add connections
    edges = []

    if connect_all is True:
        log.info("Connecting all")

    for i in df_weights.index:
        for j in df_weights.index[i + 1:]:
            log.debug("Considering connection between node {} {}".format(i, j))
            if df_weights.loc[i, j] >= connection_threshold:
                edges.append((i, j, {"weight": df_weights.loc[i, j], "color": "blue"}))
            else:
                if connect_all is True:
                    edges.append((i, j, {"weight": df_weights.loc[i, j], "color": "grey"}))

    g.add_edges_from(edges)

    if graph_file_name is not None:
        if not os.path.isfile(graph_file_name):
            nx.readwrite.gpickle.write_gpickle(g, graph_file_name)
        else:
            current_time = time.strftime("%-Y%m-%d-%H-%M-%S")
            graph_file_name = "{}_{}.pkl".format(graph_file_name, current_time)
            log.info("File already existed will save to {}".format(graph_file_name))
            nx.readwrite.gpickle.write_gpickle(g, graph_file_name)

    return g


def plot_graph(g : nx.Graph,
               opt_dist : float = None,
               weight : str = None,
               iterations : int = 50,
               random_seed : int = 7,
               figure_size : tuple = (20, 20),
               node_colours : str ="b",
               node_size : int = 20,
               cmap : plt.cm = plt.cm.rainbow,
               file_name_no_extension : str = "graph",
               return_positions_only : bool = False,
               return_image_only : bool = False
               ):
    """
    A function to set node positions using the spring layout. This is a deterministic layout (if you set the seed) using
    a form of annealing. It uses the Fruchterman-Reingold force-directed algorithm. In simple terms the algorithm treats
    nodes as repulsive, connected nodes (i.e. they share an edge) are attracted to one another using the weight factor,
    if specified, where weight is the edge attribute holding a float as attractive strength. If weight is None all
    connected nodes are equally attractive, larger weight means more attractive. The opt_dist sets the optimal distance
    between nodes
    :param g: networkx Graph - Graph instance
    :param opt_dist: float - Optimal distance between nodes
    :param weight: str - edge attribute to use to determine the attractive force between nodes
    :param iterations: int - Number of annealing iterations to use
    :param random_seed: int - seed for node positions
    :param figure_size: tuple - size of the graph image
    :param node_colours: str or list - single colour str in matplotlib format or list of specific colours for each node
                                       in order of the node numbers
    :param node_size: int - Size of the nodes in the image
    :param cmap: matplotlib colormap - colormap
    :param file_name_no_extension: str - string to use to save positions and image to without an extension
    :param return_positions_only: bool - return only the positions not an image of the graph
    :param return_image_only: - return only and image not the positions
    :return: return_positions_only=False and return_image_only=False then return = dict, matplotlib figure
             return_positions_only=True and return_image_only=False then return = dict
             return_positions_only=False and return_image_only=True then return = matplotlib figure
             return_positions_only=True and return_image_only=True then return = dict
    """

    log = logging.getLogger(__name__)

    log.info("Setting nodes .....")
    position = nx.spring_layout(g,
                           k=opt_dist,
                           weight=weight,
                           iterations=iterations,
                           seed=random_seed
                           )

    pos_out = {k: ent.tolist() for k, ent in position.items()}

    positions_filename = "{}_positions".format(file_name_no_extension)
    if not os.path.isfile("{}.json".format(positions_filename)):
        with open("{}.json".format(positions_filename), "w") as jout:
            json.dump(pos_out, jout, indent=4)
    else:
        current_time = time.strftime("%-Y%m-%d-%H-%M-%S")
        positions_filename = "{}_{}".format(positions_filename, current_time)
        log.info("File already existed will save to {}.json".format(positions_filename))
        with open("{}.json".format(positions_filename), "w") as jout:
            json.dump(pos_out, jout, indent=4)

    log.info("Layout set\n-----\n")

    if return_positions_only is True:
        return position

    log.info("Got node positions .....")
    fig = plt.figure(figsize=figure_size)
    ax = plt.gca()

    log.info("Plotting .....")
    nx.draw(g,
            position,
            with_labels=False,
            node_color=node_colours,
            node_size=node_size,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            ax=ax)

    plot_filename = "{}_plot".format(file_name_no_extension)
    if not os.path.isfile("{}.png".format(plot_filename)):
        fig.savefig("{}_plot.png".format(file_name_no_extension))
    else:
        current_time = time.strftime("%-Y%m-%d-%H-%M-%S")
        plot_filename = "{}_{}".format(plot_filename, current_time)
        log.info("File already existed will save to {}.png".format(plot_filename))
        fig.savefig("{}_plot.png".format(plot_filename))

    if return_image_only is True:
        return fig
    else:
        return position, fig


def close_n_nodes(pos : dict,
                  centre : list = (0.0, 0.0),
                  close : float = None,
                  topn : int = 10) -> list:
    """
    From a dictionary of graph node positions (pos in the notebook) return a ranked list of n
    :param centre: np.ndarray - central point around which to find the nearest n points with in the threshold
    :param close: float - threshold for close if None just find the closest n
    :param topn: int - the number of points to find if None return all within close
    :return:
    """
    log =logging.getLogger(__name__)

    close_nodes = []

    if not isinstance(centre, np.ndarray):
        centre = np.array(centre)

    if close is not None:
        for k, node in pos.items():
            d = np.linalg.norm(centre - node, 2)

            if d < close:
                log.debug("Close node: {}".format(k))
                close_nodes.append(np.array([d, k]))
    else:
        for k, node in pos.items():
            d = np.linalg.norm(centre - node, 2)
            log.debug("Node distance: {}".format(k))
            close_nodes.append(np.array([d, k]))

    close_nodes = np.array(close_nodes)

    if topn is not None:
        sorted_close_nodes = close_nodes[np.argsort(close_nodes[:, 0])]
        close_node_keys = [int(ent) for ent in sorted_close_nodes[:topn, 1]]
    else:
        sorted_close_nodes = close_nodes[np.argsort(close_nodes[:, 0])]
        close_node_keys = [int(ent) for ent in sorted_close_nodes[:, 1]]

    return close_node_keys


def plot_image_annotated_chemical_space(g : nx.Graph,
                                      position : dict,
                                      ax : plt.axes,
                                      close_node_keys : list,
                                      property_key : str = "prop",
                                      x_axes_fraction_fixed : list = [-0.25, -0.05],
                                      yaxes_fraction_increments : list = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2,
                                                                          -0.4, -0.6, -0.8, -1.0],
                                      size : tuple = (150, 150),
                                      arrow : dict = dict(arrowstyle="simple",facecolor="grey",edgecolor="grey")
                                     ):
    """
    Function to add molecule images to the chemical space images
    :param g: networkx graph - networkx graph
    :param position: dict - positions of the nodes
    :param ax: matplotlib.axes - plot axes
    :param close_node_keys: list - keys of nodes close to one another on the grapg projection
    :param property_key: str - property in each node to include in the legend
    :param x_axes_fraction_fixed: float - x axis fixed position
    :param yaxes_fraction_increments: list y axis values to place images on should be one fore each image
    :param size: iterable - sizes of the indivdual molecules images
    :return: matplotlib.axes
    """

    log = logging.getLogger(__name__)

    if isinstance(yaxes_fraction_increments, tuple):
        yaxes_fraction_increments = list(yaxes_fraction_increments)

    gridimages = [(x, y) for x in x_axes_fraction_fixed for y in yaxes_fraction_increments]

    if len(gridimages) < len(close_node_keys):
        log.warning("Too few grid places for molecules close to node, this will cause an index error. Increase the number of grid places")

    for ith, cnk in enumerate(close_node_keys):

        # Get a molecule image from RDKit grid image so we can add a legend
        log.info("Ith {}: close node k {}".format(ith, cnk))
        message = "{}\n{}".format(g.nodes()[cnk]["smiles"], g.nodes()[cnk][property_key])
        log.info("plotting and adding legend {}".format(message))
        grid = Chem.Draw.MolsToGridImage([mai.smiles_to_molecule(g.nodes()[cnk]["smiles"], threed=False, addH=False)],
                                         molsPerRow=1,
                                         subImgSize=size,
                                         legends=[message],
                                         useSVG=False,
                                         returnPNG=False
                                         # maxMols=1
                                        )

        # Get an offset box in matplotlib add the image and legend as an annotation to the chemical space graph
        im_box = OffsetImage(grid)
        im_box.image.axes = ax
        log.info("Node of smiles position {}".format(position[cnk]))

        ab = AnnotationBbox(im_box,
                            position[cnk],
                            xybox=gridimages[ith],
                            xycoords="data",
                            boxcoords="axes fraction",
                            pad=0.55,
                            frameon=False,
                            arrowprops=arrow
                            )

        # Add the artist to the image
        ax.add_artist(ab)

    return ax


def plot_annotated_chemical_space(df : pd.DataFrame,
                                  prop_key : str = "property",
                                  smiles_key : str = "smiles",
                                  label_key : str = "names",
                                  closest_n : int = 4,
                                  centroid : list = (0.0, 0.0),
                                  connection_threshold : float = 0.7
                                  ) -> (nx.Graph, dict, plt.axes) :
    """
    Wrapper function to allow graoh building, plotting and annotation in a single call.
    :param df: pandas dataframe - dataframe with at least a label column, smiles column and property column
    :param prop_key: str - column key for property of interest values
    :param smiles_key: str - column key for smiles of molecule srings
    :param label_key: str - column key for each entries labels
    :param centroid: tuple - point around which to look for closest_n and annotate them
    :param closest_n: int - Number of points to look for around the centroid for annotation
    :param connection_threshold: float - connection threshold points with a tanimoto similarity >=
                                         the threshold are connected
    :return:
    """

    log = logging.getLogger(__name__)

    log.info("Starting chemical space plotting")
    log.info(f"{prop_key} {smiles_key} {label_key}")
    log.info(f"{df}")
    g = build_nx_graph(df, prop_key=prop_key, smiles_key=smiles_key, label_key=label_key,
                       connection_threshold=connection_threshold)
    pos, fig = plot_graph(g, weight="weight")
    close_by_nodes = close_n_nodes(pos, centre=centroid, topn=closest_n)
    ax = fig.gca()
    ax = plot_image_annotated_chemical_space(g, pos, ax, close_by_nodes)

    return g, pos, ax

if __name__ == "__main__":
    import doctest
    doctest.testmod()
