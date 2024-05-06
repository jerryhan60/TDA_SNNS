import os
from os.path import join
import random
import json
import jsonpickle
from collections import defaultdict
from typing import List

import torch
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.covariance import EmpiricalCovariance
import xgboost as xgb
import argparse
import pickle as pkl
from datetime import date
from tqdm import tqdm
import glob
import time
import persim
from tqdm import tqdm
from scipy.spatial import distance

from rich import print, inspect
from typing import List, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import logging
import copy
import warnings

from classifier_bin import xgb_classifier
import sys
sys.path.append("./TopoTrojDetection/")

# ignore warnings from site-packages/torch/serialization.py # FIXME
warnings.filterwarnings("ignore", module="torch.serialization")

@dataclass
class ModelBasePaths:
    model_folder_path: str
    cache_dir_path: str


class ModelData:
    def __init__(self,
                 model_name: str,
                 base_paths: ModelBasePaths,
                 initialize_from_cache: bool = True,
                 skip_model_base: bool = False,
                 load_fast: bool = False,
                 ):

        ##############################
        # attributes to be initialized:
        ##############################
        self.label = None
        self.model = None
        self.architecture = None

        self.config = None
        self.fv = None
        self.PH_list = None
        self.PD_list = None


        # passed in
        self.model_name = model_name
        self.base_paths = base_paths
        self.load_fast = load_fast

        ##############################
        # begin initialization!
        ##############################

        if not skip_model_base: self.load_model_base()

        if initialize_from_cache:
            self.load_model_from_cache()
        else:
            self.calculate_features_from_weights() # TODO

    def load_model_base(self):
        """
            loads the data from the model_folder_path
            expects:
                - config.json
                - ground_truth.csv
                - model.pt
        """

        if not self.load_fast:
            self.model = torch.load(join(self.base_paths.model_folder_path, 'model.pt'), map_location='mps')
            self.model.eval()

        self.config = json.load(open(join(self.base_paths.model_folder_path, 'config.json')))
        self.architecture = self.config['MODEL_ARCHITECTURE']

        with open(join(self.base_paths.model_folder_path, 'ground_truth.csv'), 'r') as f:
            self.label = int(f.read().strip())

    def load_model_from_cache(self):
        """
            loads the precalced data from the cache_dir_path
            expects:
                - fv.pkl
                - PH_list.pkl
                - possibly, PD_list.pkl
        """

        self.fv = pkl.load(open(join(self.base_paths.cache_dir_path, 'fv.pkl'), 'rb'))

        self.PH_list = pkl.load(open(join(self.base_paths.cache_dir_path, 'PH_list.pkl'), 'rb'))
        # if not self.load_fast:
        #     self.PH_list = pkl.load(open(join(self.base_paths.cache_dir_path, 'PH_list.pkl'), 'rb'))

        self.PD_list = None
        if not self.load_fast and os.path.exists(join(self.base_paths.cache_dir_path, 'PD_list.pkl')):
            self.PD_list = pkl.load(open(join(self.base_paths.cache_dir_path, 'PD_list.pkl'), 'rb'))

    def calc_topo_feature(self, PH: List, dim: int)-> Dict:
        """
        Compute topological feature from the persistent diagram.
        Input args:
            PH (List) : Persistent diagram
            dim (int) : dimension to be focused on
        Return:
            Dictionary contains topological feature
        """
        pd_dim = PH[dim]
        if dim == 0:
            pd_dim = pd_dim[:-1]
        pd_dim = np.array(pd_dim)
        betti = len(pd_dim)
        ave_persis = sum(pd_dim[:, 1] - pd_dim[:, 0]) / betti if betti > 0 else 0
        ave_midlife = (sum((pd_dim[:, 0] + pd_dim[:, 1]) / 2) / betti) if betti > 0 else 0
        med_midlife = np.median((pd_dim[:, 0] + pd_dim[:, 1]) / 2) if betti > 0 else 0
        max_persis = (pd_dim[:, 1] - pd_dim[:, 0]).max() if betti > 0 else 0
        top_5_persis = np.mean(np.sort(pd_dim[:, 1] - pd_dim[:, 0])[-5:]) if betti > 0 else 0

        ave_death_time = sum(pd_dim[:, 1]) / betti if betti > 0 else 0

        topo_feature_dict = {"betti_" + str(dim): betti,
                            "avepersis_" + str(dim): ave_persis,
                            "avemidlife_" + str(dim): ave_midlife,
                            "maxmidlife_" + str(dim): med_midlife,
                            "maxpersis_" + str(dim): max_persis,
                            "toppersis_" + str(dim): top_5_persis,
                            "avedeath_" + str(dim): ave_death_time}
        return topo_feature_dict

    def load_PH(self):
        print("starting load_PH 1")
        self.PH_list = pkl.load(open(join(self.base_paths.cache_dir_path, 'PH_list.pkl'), 'rb'))
        PH_sample = random.choice(self.PH_list)
        print(len(self.PH_list))
        distances_0 = []
        distances_1 = []
        for i in tqdm(range(len(self.PH_list))):
            # print(i)
            PH = self.PH_list[i]
            distances_0.append((i, persim.sliced_wasserstein(PH_sample[0], PH[0])))
            distances_1.append((i, persim.sliced_wasserstein(PH_sample[1], PH[1])))

        return (distances_0, distances_1)

    def recalc_fv(self):
        # self.load_PH()
        topo_feature_pos=torch.zeros(
            len(self.PH_list),
            14
        )
        for i in range(len(self.PH_list)):
            PH = self.PH_list[i]
            clean_feature_0=self.calc_topo_feature(PH, 0)
            clean_feature_1=self.calc_topo_feature(PH, 1)
            topo_feature=[]
            for k in sorted(list(clean_feature_0)):
                topo_feature.append(clean_feature_0[k])
            for k in sorted(list(clean_feature_1)):
                topo_feature.append(clean_feature_1[k])
            topo_feature=torch.tensor(topo_feature)
            topo_feature_pos[i, :]=topo_feature
        topo_feature_pos = topo_feature_pos.reshape(5, 196, 14)
        self.fv['topo_feature_pos'] = topo_feature_pos

    def calculate_features_from_weights(self):
        raise NotImplementedError

    def __str__(self):
        """ pretty print the model data """
        return f"""
        Name: {self.model_name}
        Architecture: {self.architecture}
        Label: {self.label}
        loaded: {[x for x in [
            "model", "fv", "PH_list", "PD_list"
        ] if getattr(self, x) is not None]}
        """

    def trim_features(self):
       topo_feature_pos = self.fv['topo_feature_pos']
       # print(topo_feature_pos.shape)
       samples = 4
       trimmed_topo_ft = torch.zeros(5, samples, 12)

       def sort_vectors_by_outlierness(vectors):
           # Compute the mean vector
           vectors = vectors.numpy()

           mean_vector = np.mean(vectors, axis=0)
           std_vector = np.std(vectors, axis = 0)
           # Compute the covariance matrix
           cov_matrix = EmpiricalCovariance().fit(vectors)


           # Compute the Mahalanobis distance of each vector
           mahalanobis_dist = distance.cdist(vectors, mean_vector.reshape(1, -1), 'mahalanobis', VI=cov_matrix.covariance_)

           # Sort the vectors by their Mahalanobis distance
           sorted_indices = np.argsort(mahalanobis_dist, axis=0)
           sorted_indices = sorted_indices.flatten()[::-1]
           vv = vectors[sorted_indices]
           mean_vector = mean_vector.reshape(1,12)
           # std_vector = std_vector.reshape(1,12)
           # vv = np.concatenate((mean_vector, std_vector, vv), axis=0)
           vv = np.concatenate((mean_vector, vv), axis=0)
           return vv


       def protocol(eg_idx, class_idx):
           ft_eg = topo_feature_pos[eg_idx,:,:]

           def map_idx(idx):
               h = idx % 14
               w = idx//14
               return w, h

           v = sort_vectors_by_outlierness(ft_eg)
           return v[:samples,]


       for i in range(5):
           trimmed_topo_ft[i,:,:]= torch.from_numpy(protocol(0, 1))

       self.fv['topo_feature_pos'] = trimmed_topo_ft

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps')

    root = "/Users/huxley/dataset_storage/snn_tda_mats/LENET_MODELS/competition_dataset"
    models_dir = join(root, "all_models")
    cache_dir = join(root, "calculated_features_cache")

    models_list = sorted([x for x in os.listdir(models_dir) if x.startswith('id')])
    model_name = models_list[0]

    # make a new dataclass
    base_paths = ModelBasePaths(
        model_folder_path = join(models_dir, model_name),
        cache_dir_path    = join(cache_dir, model_name)
    )

    model = ModelData(model_name,
                      base_paths,
                      initialize_from_cache=True,
                      skip_model_base=False
                      )

    print(model)

