from rich import print, inspect
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import json
import logging

class ModelData:
    def __init__(self, model_folder_path: str):
        self.model_folder_path = model_folder_path

        ##################################################
        # these are uninitialized attributes, but will be initialized by the end of the init call
        # self.label = None
        # self.model = None
        # self.stats = None

        self.load_model_data()

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

    def gen_activation_graph(self, examples: List):
        """
        """
        pass

    def gen_TDA_features(self):
        """
        """
        pass


if __name__ == '__main__':
    DATA_PATH = "/Users/huxley/dataset_storage/snn_tda_mats/LENET_MODELS/data/"
    test_model = DATA_PATH + "id-00000088/"

    model_data = ModelData(test_model)

    pass

