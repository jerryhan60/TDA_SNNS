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


import sys
sys.path.append("./TopoTrojDetection/")
from run_crossval import run_crossval_xgb
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

from competition_model_data import ModelBasePaths, ModelData
from classifier_bin import xgb_classifier, lgb_classifier

import multiprocessing

import warnings 

def seed_everything(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

log = logging.getLogger('pythonConfig')
def setup_logger():
    global log
    LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('hyperopt').setLevel(logging.WARNING)

    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(formatter)
    log = logging.getLogger('pythonConfig')
    log.setLevel(logging.INFO)
    log.addHandler(stream)

setup_logger()


def load_all_models(models_dir, cache_dir, percentage=1.0):
    start = time.time()
    seed_everything()

    # model_paths = sorted([x for x in os.listdir(cache_dir) if x.startswith('id')])
    model_paths = [x for x in os.listdir(cache_dir) if x.startswith('id')]
    random.shuffle(model_paths)
    model_paths = sorted(model_paths[:int(len(model_paths) * percentage)])

    # filter out all the models with empty cache dirs
    model_paths = [x for x in model_paths if len(os.listdir(join(cache_dir, x))) != 0]

    models = []
    for model_name in track(model_paths):
        base_paths = ModelBasePaths(
            model_folder_path = join(models_dir, model_name),
            cache_dir_path    = join(cache_dir, model_name)
        )

        model = ModelData(model_name,
                          base_paths,
                          initialize_from_cache=True,
                          skip_model_base=False,
                          load_fast=True
                          )

        models.append(model)

    log.info(f"loaded {len(models)} models in {time.time() - start:.2f} seconds")
    return models

def run_model_tests(feature, labels, model_list, thresholds = None, calc_thresholds=False):


    feature = feature
    labels = np.array(labels)
    dtest = xgb.DMatrix(np.array(feature), label=labels)

    y_pred = 0
    for i in range(len(model_list['models'])):
        best_bst=model_list['models'][i]
        weight=model_list['weight'][i]/sum(model_list['weight'])
        y_pred += best_bst.predict(dtest)*weight

    # T, b=model_list['threshold']
    # T, b = 0.1, 0.01

    auc = roc_auc_score(labels, y_pred)
    y_pred = y_pred / len(model_list)

    if not calc_thresholds and thresholds is not None:
        T, b = thresholds

        y_pred=torch.sigmoid(b*(torch.tensor(y_pred)-T)).numpy()
        acc = np.sum((y_pred >= 0.5)==labels)/len(y_pred)
        # ce_test = np.sum(-(labels * np.log(y_pred) + (1 - labels) * np.log(1 - y_pred))) / len(y_pred)
        ce = (np.mean(-(labels * np.log(y_pred + 1e-10) + (1 - labels) * np.log(1 - y_pred + 1e-10))))

        return {
                "acc": acc,
                "auc": auc,
                "ce": ce,
                "thresholds": (T, b)
                }

    log.info("calculating thresholds throuh grid search")
    accs = []
    ces = []
    thresholds = []

    # frick it let's just do this manually
    # do a grid search for the best T and b
    for T in np.linspace(0.0, 0.2, 100): # todo switch to a *correct* optimization method?
        for b in np.linspace(0.0001, 0.3, 100):

            test_y_pred=torch.sigmoid(b*(torch.tensor(y_pred)-T)).numpy()
            acc = np.sum((test_y_pred >= 0.5)==labels)/len(test_y_pred)
            # ce_test = np.sum(-(labels * np.log(test_y_pred) + (1 - labels) * np.log(1 - test_y_pred))) / len(test_y_pred)
            ce = (np.mean(-(labels * np.log(test_y_pred + 1e-10) + (1 - labels) * np.log(1 - test_y_pred + 1e-10))))

            accs.append(acc)
            ces.append(ce)
            thresholds.append((T, b))

    best_ind = np.argmax(accs)

    return {
            "acc": accs[best_ind],
            "auc": auc,
            "ce": ces[best_ind],
            "thresholds": thresholds[best_ind]
            }


def featurize(models: List[ModelData]):

    CLASSES = 5 # FIXME don't hardcode this
    n_classes = CLASSES
    fv_list = [x.fv for x in models]
    gt_list = [x.label for x in models]

    psf_feature=torch.cat([fv_list[i]['psf_feature_pos'].unsqueeze(0) for i in range(len(fv_list))])
    topo_feature = torch.cat([fv_list[i]['topo_feature_pos'].unsqueeze(0) for i in range(len(fv_list))])

    topo_feature[np.where(topo_feature==np.Inf)]=1
    n, _, nEx, fnW, fnH, nStim, C = psf_feature.shape
    psf_feature_dat=psf_feature.reshape(n, 2, -1, nStim, C)
    psf_diff_max=(psf_feature_dat.max(dim=3)[0]-psf_feature_dat.min(dim=3)[0]).max(2)[0].view(len(gt_list), -1)
    psf_med_max=psf_feature_dat.median(dim=3)[0].max(2)[0].view(len(gt_list), -1)
    psf_std_max=psf_feature_dat.std(dim=3).max(2)[0].view(len(gt_list), -1)
    psf_topk_max=psf_feature_dat.topk(k=min(3, n_classes), dim=3)[0].mean(2).max(2)[0].view(len(gt_list), -1)
    psf_feature_dat=torch.cat([psf_diff_max, psf_med_max, psf_std_max, psf_topk_max], dim=1)
    # dat = topo_feature.view(topo_feature.shape[0], -1)
    dat=torch.cat([psf_feature_dat, topo_feature.view(topo_feature.shape[0], -1)], dim=1)
    dat=preprocessing.scale(dat)
    gt_list=torch.tensor(gt_list)

    return {
        "features": np.array(dat),
        "labels": np.array(gt_list)
    }


if __name__ == "__main__":

    device = torch.device('mps')

    # TODO update this to ur device
    # root = "/Users/huxley/dataset_storage/snn_tda_mats/LENET_MODELS/competition_dataset"
    root = "/home/jerryhan/Documents/data"
    models_dir = join(root, "all_models")
    cache_dir = join(root, "calculated_features_cache")

    models = load_all_models(models_dir, cache_dir
                             # , percentage=0.2
                             )

    # filter for only resnets
    models = [x for x in models if x.architecture == "resnet50"]
    # models = [x for x in models if x.architecture != "resnet50"]
    print(models[0])
    # models = models[:50]

    triggered = [x for x in models if x.label == 1]
    clean = [x for x in models if x.label == 0]

    print(len(triggered), len(clean))

    min_len = min(len(triggered), len(clean))

    # balance the dataset # FIXME weighting
    triggered = triggered[:min_len]
    clean = clean[:min_len]

    models = triggered + clean
    np.random.shuffle(models)

    print(len(models))

    log.info("Featurizing...")

    TRAIN_TEST_SPLIT = 0.8
    _x = featurize(models)
    dat = _x['features']
    gt_list = _x['labels']

    N = len(gt_list)
    n_train = int(TRAIN_TEST_SPLIT * N)
    ind_reshuffle = np.random.choice(list(range(N)), N, replace=False)
    train_ind = ind_reshuffle[:n_train]
    test_ind = ind_reshuffle[n_train:]

    feature_train, feature_test = dat[train_ind], dat[test_ind]
    gt_train, gt_test = gt_list[train_ind], gt_list[test_ind]

    res = []

    for gamma in [0.07368421052631578]:

        general_params = {
            "num_epochs": 50*10,
            "test_percentage": 0.1
        }

        
        classifier_params = {
            'objective': 'binary',
            'num_threads': multiprocessing.cpu_count(),
            'metric': 'binary_error',
            'device': "cpu",

            'max_depth': 5,
            'learning_rate': 0.05,
            'lambda_l1': gamma,
            'lambda_l2': 0,
            'verbose': -1,

            'bagging_fraction': 0.5,
        }
        warnings.filterwarnings('ignore')
        model = lgb_classifier(features = {'train': feature_train, 'test': feature_test}, \
                            labels = {'train': gt_train, 'test': gt_test},
                            classifier_params=classifier_params,
                            general_params=general_params)
        

        #model = lgb_classifier(features = {'train': feature_train, 'test': feature_test}, \
        #                    labels = {'train': gt_train, 'test': gt_test})
        # print("Training...")
        model.train()
        # print("Evaluating")
        train_results, test_results = model.test()
        print("Gamma", gamma)
        print("Train", train_results)
        print("Test", test_results)
        res.append([train_results, test_results])
        
        # print("Gamma", gamma)
        #print("Train results", train_results)
        #print("Test results", test_results)
    #for r in res:
        #print("Train", r[0])
        #print("Test", r[1])
    # train_xgboost(models)
