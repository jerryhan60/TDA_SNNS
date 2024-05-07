import numpy as np
from rich import print, inspect
import os
from os.path import join
import random
import numpy as np
from rich import print, inspect
from rich.progress import track
import torch
import random
import logging
# logging.basicConfig(level=logging.INFO)
# logging.getLogger('hyperopt').setLevel(logging.WARNING)
from colorlog import ColoredFormatter
import time
from typing import List, Dict, Any
from pdb import set_trace as bp
from tqdm import tqdm

import sys
sys.path.append("./TopoTrojDetection/")
from run_crossval import run_crossval_xgb
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

from competition_model_data import ModelBasePaths, ModelData
from classifier_bin import xgb_classifier, lgb_classifier, cnn_classifier, dnn_classifier
from competition_classifier import load_all_models, featurize

import os
import random
import json
import jsonpickle
from collections import defaultdict
from typing import List
import threading

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
from matplotlib import pyplot as plt

import multiprocessing

import warnings


def main():
    ######################
    # load the models up #
    ######################
    device = torch.device('cpu')

    root = "/Users/huxley/dataset_storage/snn_tda_mats/LENET_MODELS/competition_dataset"
    # root = "/home/jerryhan/Documents/data"
    cache_dir = join(root, "calculated_features_cache")
    models_dir = join(root, "all_models")

    # models = load_all_models(models_dir, cache_dir, percentage=0.03, load_fast=False)
    # models = models[:220]

    model_name = "id-00000031"

    base_paths = ModelBasePaths(
            model_folder_path = join(models_dir, model_name),
            cache_dir_path    = join(cache_dir, model_name)
        )

    m = ModelData(model_name,
                      base_paths,
                      initialize_from_cache=True,
                      skip_model_base=False,
                      load_fast=False
                      ).model

    start = time.time()
    # random.seed(start//1)
    # random.shuffle(models)

    # models = [x for x in models if x.architecture == "resnet50"]
    # models = [x for x in models if x.architecture != "resnet50"]


    # triggered = [x for x in models if x.label == 1]
    # clean = [x for x in models if x.label == 0] # NOT BALANCED

    # print(clean[1])

    ######################
    # anew!             #
    ######################

    # m = clean[1].model
    m = m.to(device)
    m.eval()

    # print(triggered[0].model)
    root_imgs_dir = "/Users/huxley/dataset_storage/snn_tda_mats/LENET_MODELS/competition_dataset/all_models/id-00000007/clean-example-data/class_"
    # images = os.listdir(root_imgs_dir)

    imgs_paths = [root_imgs_dir + str(i) + '_example_0.png' for i in range(5)]

    for i in imgs_paths:
        print(i)
        img_file = i
        img_file=glob.glob(
                img_file,
                recursive=True)[0]
        img = torch.from_numpy(cv2.imread(img_file, cv2.IMREAD_UNCHANGED)).float()
        # print("img", img)
        # plt.imshow(img)
        # plt.show()

        img = img.permute(2,0,1).unsqueeze(0).to(device)
        img = img.repeat(4,1,1,1)

        out = m(img)
        # print("out", out)

        psf_conf=torch.nn.functional.softmax(out, 1)
        print("psf_conf", psf_conf)




if __name__ == "__main__":
    main()


