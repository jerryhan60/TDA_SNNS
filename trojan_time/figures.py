import os
from os.path import join
import random
import numpy as np
import pickle
from rich import print, inspect
from rich.progress import track
from tqdm import tqdm
import torch
import random
import logging
# logging.basicConfig(level=logging.INFO)
# logging.getLogger('hyperopt').setLevel(logging.WARNING)
from colorlog import ColoredFormatter
import time
from typing import List, Dict, Any
from pdb import set_trace as bp
import sys
sys.path.append("./TopoTrojDetection/")
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from matplotlib import use
import multiprocessing
import ripser
import persim
from sklearn.metrics.pairwise import pairwise_distances
from scipy import sparse

from scipy.sparse.linalg import eigsh
import scipy.sparse as sp

import rustworkx as rx
import igraph as ig
import networkx as nx
import cairo
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix

from mogutda import SimplicialComplex
import itertools

# import gtda.diagrams

from topological_feature_extractor import getGreedyPerm, getApproxSparseDM


from competition_model_data import ModelBasePaths, ModelData
from classifier_bin import xgb_classifier
from competition_classifier import load_all_models, featurize

root = "/Users/huxley/dataset_storage/snn_tda_mats/LENET_MODELS/competition_dataset"
# root = "/home/jerryhan/Documents/data"
# root = "/home/dataset_storage/TopoTrojDetect/competition_dataset"
models_dir = join(root, "all_models")
cache_dir = join(root, "calculated_features_cache")

def NCG():
    # load a model
    models = load_all_models(models_dir, cache_dir, percentage=0.05, load_fast=True)
    models = [x for x in models if x.architecture == "resnet50"]

    triggered = [x for x in models if x.label == 1][0]
    mat = triggered.fv["correlation_matrix"]
    mat[np.isnan(mat)] = 0
    # if the value is < 0.5, set it to 0
    np.fill_diagonal(mat, 0)
    mat[mat < 0.005] = 0

    # randomly zero 50% of the values
    mat = mat * (np.random.rand(*mat.shape) > 0.995)

    # zero out the bottom half
    mat = np.triu(mat, 0)
    mat = mat + mat.T


    # plt.matshow(mat)
    # plt.show()
    # exit()

    if not os.path.exists("graph.pkl") or True:
        g = ig.Graph.Weighted_Adjacency(mat.tolist(), mode="undirected")
        # pickle the g
        with open("graph.pkl", "wb") as f:
            pickle.dump(g, f)
    else:
        with open("graph.pkl", "rb") as f:
            g = pickle.load(f)

    out_fig_name = "graph.png"

    visual_style = {}

    colours = ['#fecc5c', '#a31a1c']
    visual_style["bbox"] = (3000,3000)
    visual_style["margin"] = 17
    visual_style["vertex_color"] = 'grey'
    visual_style["vertex_size"] = 20
    visual_style["vertex_label_size"] = 8
    visual_style["edge_curved"] = False
    my_layout = g.layout_fruchterman_reingold()
    visual_style["layout"] = my_layout
    # ig.plot(g, out_fig_name, **visual_style)
    # show the cairo plot
    ig.plot(g, out_fig_name, **visual_style)

def SC():
    # g = ig.Graph.Erdos_Renyi(n=300, p=0.01)
    g = ig.Graph.Watts_Strogatz(1, 300, 3, 0.1)
    # g = ig.Graph.SBM(300,
    #                  [
    #                      [0.1, 0.003, 0.001],
    #                      [0.003, 0.1, 0.003],
    #                      [0.001, 0.003, 0.1]
    #                      ], [120, 100, 80], loops=False)
    COMMUNITY_BINS = [130, 100, 70]

    # g = ig.Graph.SBM(300, [
    #     # [0.1, 0.01, 0.001],
    #     # [0.01, 0.1, 0.01],
    #     # [0.001, 0.01, 0.1]
    #     [0.01, 0.05, 0.03],
    #     [0.05, 0.01, 0.05],
    #     [0.03, 0.05, 0.01]
    # ], COMMUNITY_BINS, loops=False)
    # g = ig.Graph.Barabasi(300, 2)

    visual_style = {}
    # generate a random graph using IG
    # colours = ['#fecc5c', '#a31a1c']
    # colours = [ "#2c302eff", "#474a48ff", "#909590ff", "#9ae19dff", "#537a5aff" ]

    colours = [ "#BAEAFE", "#9ae19dff", "#537a5aff" ]
    visual_style["bbox"] = (3000,3000)
    visual_style["margin"] = 100
    # visual_style["vertex_color"] = 'grey'
    visual_style["vertex_size"] = 40
    visual_style["vertex_label_size"] = 8
    visual_style["edge_curved"] = False
    # my_layout = g.layout_fruchterman_reingold()
    # visual_style["layout"] = my_layout


    # assign one of 5 random colors to every single node
    # node_colours = [random.choice(colours) for x in range(len(g.vs))]
    # assign color based on the community, which
    # is determined by the bin
    node_colours = []
    for i, v in enumerate(g.vs):
        if i < COMMUNITY_BINS[0]:
            node_colours.append(colours[0])
        elif i < COMMUNITY_BINS[0] + COMMUNITY_BINS[1]:
            node_colours.append(colours[1])
        else:
            node_colours.append(colours[2])

    g.vs["color"] = node_colours


    # show the graph
    ig.plot(g, "graph.png", **visual_style)


def plot_vietoris_rips(num_points=200, epsilon=0.3):
    """
    Sample a bunch of random points on a sphere, create a Vietoris-Rips complex,
    and plot it using matplotlib with filled triangles and colored points.

    Args:
        num_points (int): The number of points to sample.
        epsilon (float): The maximum distance between points to form an edge.
    """
    # Generate random points on a unit sphere
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.arccos(2 * np.random.random(num_points) - 1)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    points = np.stack((x, y, z), axis=-1)

    # Calculate the distance matrix
    distances = distance_matrix(points, points)

    # Define colors for each third of the points
    colours = ["#40B6E7", "#3DD343", "#24512C"]
    color_indices = np.concatenate([
        np.full(num_points // 3, colours[0]),
        np.full(num_points // 3, colours[1]),
        np.full(num_points - 2 * (num_points // 3), colours[2])  # Handle any rounding issues
    ])

    # Plot points and edges
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=color_indices, s=50)

    # Collect triangles
    triangles = []
    for i, j, k in itertools.combinations(range(num_points), 3):
        if all([distances[i, j] <= epsilon, distances[j, k] <= epsilon, distances[k, i] <= epsilon]):
            triangles.append([i, j, k])

    # If there are triangles, plot them
    if triangles:
        ax.plot_trisurf(x, y, z, triangles=triangles, color='red', alpha=0.2)

    # Adjust the aspect ratio to avoid squishing
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1.28, 1]))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # ax.set_xlabel("X-axis")
    # ax.set_ylabel("Y-axis")
    # ax.set_zlabel("Z-axis")
    plt.show()





if __name__ == "__main__":
    # NCG()
    # SC()
    plot_vietoris_rips()
    pass





