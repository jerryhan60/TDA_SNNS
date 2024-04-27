import os
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
import xgboost as xgb
import argparse
import pickle as pkl
from datetime import date
from tqdm import tqdm
import glob
import time

from rich import print, inspect
from typing import List, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import logging
import copy
from topo_utils import mat_bc_adjacency, parse_arch, feature_collect, sample_act, mat_discorr_adjacency, mat_cos_adjacency, mat_jsdiv_adjacency, mat_pearson_adjacency


# from topological_feature_extractor import topo_psf_feature_extract
# from run_crossval import run_crossval_xgb, run_crossval_mlp

# Algorithm Configuration
STEP_SIZE:  int = 7 # Stimulation stepsize used in PSF
PATCH_SIZE: int = 2 # Stimulation patch size used in PSF
STIM_LEVEL: int = 32 # Number of stimulation level used in PSF
N_SAMPLE_NEURONS: int = 1.5e3  # Number of neurons for sampling
USE_EXAMPLE: bool =  True     # Whether clean inputs will be given or not
CORR_METRIC: str = 'distcorr'   # Correlation metric to be used
CLASSIFIER: str  = 'xgboost'    # Classifier for the detection , choice = {xgboost, mlp}.
# Experiment Configuration
INPUT_SIZE: List = [1, 28, 28] # Input images' shape (default to be MNIST)
INPUT_RANGE: List = [0, 255]   # Input image range
TRAIN_TEST_SPLIT: float = 0.8  # Ratio of train to test



psf_config = {}
psf_config['step_size'] = STEP_SIZE
psf_config['stim_level'] = STIM_LEVEL
psf_config['patch_size'] = PATCH_SIZE
psf_config['input_shape'] = INPUT_SIZE
psf_config['input_range'] = INPUT_RANGE
psf_config['n_neuron'] = N_SAMPLE_NEURONS
psf_config['corr_method'] = CORR_METRIC

root = 'data/huxleydata/data'
model_name = 'CLEAN_id-00000010'
model_file_path = os.path.join(root, model_name, 'mnist_lambdatrigger_0.0', 'leenet5_0.0_poison.pt.1')
model_config_path = os.path.join(root, model_name, 'mnist_lambdatrigger_0.0', 'leenet5_0.0_poison.pt.1.stats.json')

model = torch.load(model_file_path)
model.eval()
model_config = jsonpickle.decode(open(model_config_path, "r").read())
gt = ('final_triggered_data_n_total' in model_config.keys())


# model, example_imgs




class ModelData:
    def __init__(self, model_folder_path: str, data_folder_path: str, model_name : str, troj_config: dict):
        # folder containing model weights
        self.model_folder_path = model_folder_path
        # folder containing example data 
        self.data_folder_path = data_folder_path
        self.model_name = model_name
        # configs for troj detection 
        self.troj_config = troj_config

        ##################################################
        # these are uninitialized attributes, but will be initialized by the end of the init call
        # self.label = None
        # self.model = None
        # self.stats = None

        self.load_model_data_base()
        self.load_example_data()
        # important: set model to eval mode
        self.model.eval()
        self.gen_activation_graph()
        # print(self.model)

    def load_model_data_base(self): 
        """
        - this function assumes the pt file is located right in the model_folder_path directory
        - load the .pt file into a torch model
        - load the endswith stats.json file into a dictionary
            - extract relevant features from the dict
        """

        # LOAD MODEL
        model_path = [x for x in os.listdir(model_folder_path) if x.endswith('.pt.1')][0]
        model = torch.load(os.path.join(model_folder_path, model_path))
        self.model = model

        # LOAD STATS
        stats_path = [x for x in os.listdir(model_folder_path) if x.endswith('stats.json')][0]
        stats = json.load(open(os.path.join(model_folder_path, stats_path)))
        self.stats = stats

        self.label = False # clean
        if "final_triggered_data_n_total" in stats:
            self.label = True # triggered

            if "CLEAN" in self.model_folder_path:
                logging.error("labelling a CLEAN model as DIRTY! you should *delete* this model folder")

    def load_model_data(self):
        """
        - check for the startswith mnist_lambdatrigger directory inside the model folder path
        - load the .pt file into a torch model
        - load the endswith stats.json file into a dictionary
            - extract relevant features from the dict
        """

        # check for the startswith mnist_lambdatrigger directory inside the model folder path
        dirs = os.listdir(self.model_folder_path)

        dirs = [x for x in dirs if os.path.isdir(os.path.join(self.model_folder_path, x))]
        model_dir = [x for x in dirs if x.startswith('mnist_lambdatrigger')][0]
        model_dir_path = os.path.join(self.model_folder_path, model_dir)

        # LOAD MODEL
        model_path = [x for x in os.listdir(model_dir_path) if x.endswith('.pt.1')][0]
        model = torch.load(os.path.join(model_dir_path, model_path))
        self.model = model

        # LOAD STATS
        stats_path = [x for x in os.listdir(model_dir_path) if x.endswith('stats.json')][0]
        stats = json.load(open(os.path.join(model_dir_path, stats_path)))
        self.stats = stats

        self.label = False # clean
        if "final_triggered_data_n_total" in stats:
            self.label = True # triggered

            if "CLEAN" in self.model_folder_path:
                logging.error("labelling a CLEAN model as DIRTY! you should *delete* this model folder")

    def load_example_data(self):
        examples = []
        for i in range(10):
            # search for the ith class
            file_list = glob.glob(self.data_folder_path + "/*" + str(i) + ".png")
            examples.append(file_list[0])
        # do some post-proc
        example_imgs = []
        for img_file in examples:
            img = torch.from_numpy(cv2.imread(img_file, cv2.IMREAD_UNCHANGED)).float()
            # example_imgs.append(img)
            img = img.unsqueeze(2) if len(img.shape) == 2 else img
            example_imgs.append(img.permute(2,0,1).unsqueeze(0))
        # set it to self.examples
        self.examples = example_imgs
        self.num_classes = len(examples)
        self.data_dim = example_imgs[0].shape[2:4]

    def gen_activation_graph(self):
        """
        """
        step_size=self.troj_config['step_size']
        stim_level=self.troj_config['stim_level']
        patch_size=self.troj_config['patch_size']
        input_shape=self.troj_config['input_shape']
        input_valuerange=self.troj_config['input_range']
        n_neuron_sample=self.troj_config['n_neuron']
        method=self.troj_config['corr_method']

        stim_seq=np.linspace(input_valuerange[0], input_valuerange[1], stim_level)
        # 2 represent score and conf
        feature_map_h=len(range(0, self.data_dim[0]-patch_size+1, step_size))
        feature_map_w=len(range(0, self.data_dim[1]-patch_size+1, step_size))
        num_per_class = len(stim_seq) * feature_map_h * feature_map_w

        perturbed_imgs = []
        for c in range(self.num_classes):
            input_eg = copy.deepcopy(self.examples[c])
            perturbed = input_eg.repeat(num_per_class, 1, 1, 1) 
            cnt = 0
            for pos_w in range(0, self.data_dim[0]-patch_size+1, step_size):
                for pos_h in range(0, self.data_dim[1]-patch_size+1, step_size):
                    for i in range(len(stim_seq)):
                        stim = stim_seq[i]
                        perturbed[cnt,0,pos_w:pos_w+patch_size, pos_h:pos_h+patch_size] = stim
                        cnt+=1
            perturbed_imgs.append(perturbed)
        prob_input = torch.cat(perturbed_imgs, dim = 0)
        pred = []
        feature_dict_c, output = feature_collect(self.model, prob_input)
        pred.append(output.detach())
        pred = torch.cat(pred)

        psf_score=pred
        psf_conf=torch.nn.functional.softmax(psf_score, 1)
        print(psf_score.shape)
        print(psf_conf.shape)

        # add this later
        # psf_feature_pos[0, c, feature_w_pos, feature_h_pos]=psf_score
        # psf_feature_pos[1, c, feature_w_pos, feature_h_pos]=psf_conf

        # Extract intermediate activating vectors
        neural_act = []
        # print(neural_act)
        # print(feature_dict_c)
        for k in feature_dict_c:
            if len(feature_dict_c[k][0].shape)==3:
                layer_act = [feature_dict_c[k][i].max(1)[0].max(1)[0].unsqueeze(1) for i in range(len(feature_dict_c[k]))]
            else:
                layer_act = [feature_dict_c[k][i].unsqueeze(1) for i in range(len(feature_dict_c[k]))]
            layer_act=torch.cat(layer_act, dim=1)
            # Standardize the activation layer-wisely
            layer_act=(layer_act-layer_act.mean(1, keepdim=True))/(layer_act.std(1, keepdim=True)+1e-30)
            neural_act.append(layer_act)
        neural_act=torch.cat(neural_act)
        layer_list=parse_arch(model)
        sample_n_neurons_list=None
        if len(neural_act)>1.5e3:
            neural_act, sample_n_neurons_list=sample_act(neural_act, layer_list, sample_size=n_neuron_sample)
        print("Neural act", neural_act.shape, neural_act)
        
        # save neural act
        self.neural_act = neural_act 


        # trimming the activation tensor otherwise memory limit exceeded
        neural_pd=mat_discorr_adjacency(neural_act[:,:1000])
        print(neural_pd.shape)
        # print(layer_act.shape, layer_act)
        # exit(0)
        pass

    def gen_TDA_features(self):
        """
        """
        pass


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_folder_path = "data/mnist_leenet5/id-00000000"
    data_folder_path = "data/mnist_clean/train"

    troj_config = {
        'step_size': 2,
        'stim_level': 32,
        'patch_size': 2,
        'input_shape': [1, 28, 28],
        'input_range': [0, 255],
        'n_neuron': 1.5e3,
        'corr_method': 'distcorr'
    }

    model_data = ModelData(model_folder_path, data_folder_path, None, troj_config)

    pass
