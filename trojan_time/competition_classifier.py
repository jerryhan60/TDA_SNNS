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


def load_all_models(models_dir, cache_dir, percentage=1.0, seed=42):
    start = time.time()
    seed_everything(seed)

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

    # for model in models:
    #     model.trim_features()

    CLASSES = 5 # FIXME don't hardcode this
    n_classes = CLASSES
    fv_list = [x.fv for x in models]
    # ph_list = [x.PH_list for x in models]
    # exit(0)
    gt_list = [x.label for x in models]

    psf_feature=torch.cat([fv_list[i]['psf_feature_pos'].unsqueeze(0) for i in range(len(fv_list))])
    topo_feature = torch.cat([fv_list[i]['topo_feature_pos'].unsqueeze(0) for i in range(len(fv_list))])

    """
    corr_feature = []
    for i in range(len(fv_list)):
        corr_mat = fv_list[i]['correlation_matrix']
        top_k_values, top_k_indices = torch.from_numpy(fv_list[i]['correlation_matrix']).topk(100)
    """

    topo_feature[np.where(topo_feature==np.Inf)]=1
    n, _, nEx, fnW, fnH, nStim, C = psf_feature.shape
    psf_feature_dat=psf_feature.reshape(n, 2, -1, nStim, C)
    psf_diff_max=(psf_feature_dat.max(dim=3)[0]-psf_feature_dat.min(dim=3)[0]).max(2)[0].view(len(gt_list), -1)
    psf_med_max=psf_feature_dat.median(dim=3)[0].max(2)[0].view(len(gt_list), -1)
    psf_std_max=psf_feature_dat.std(dim=3).max(2)[0].view(len(gt_list), -1)
    psf_topk_max=psf_feature_dat.topk(k=min(3, n_classes), dim=3)[0].mean(2).max(2)[0].view(len(gt_list), -1)
    psf_feature_dat=torch.cat([psf_diff_max, psf_med_max, psf_std_max, psf_topk_max], dim=1)

    # print(topo_feature.shape)
    # topo_feature_mean = torch.mean(topo_feature, dim=[1,2])
    # topo_feature_std = torch.std(topo_feature, dim=[1,2])
    # topo_stat_features = torch.cat([topo_feature_mean, topo_feature_std], dim=1)
    # print(topo_stat_features)
    #print(topo_stat_features.shape)
    #print(psf_feature_dat.shape)

    # dat = topo_feature.view(topo_feature.shape[0], -1)
    # dat=torch.cat([topo_stat_features, psf_feature_dat, topo_feature.view(topo_feature.shape[0], -1)], dim=1)
    # dat=torch.cat([psf_feature_dat, topo_feature.view(topo_feature.shape[0], -1)], dim=1)
    # print(dat.shape)

    dat = topo_feature.view(topo_feature.shape[0], -1)
    # dat = dat[:, :]
    # dat = psf_feature_dat
    print(dat.shape)
    # exit()
    # dat = torch.cat([psf_feature_dat])

    dat=preprocessing.scale(dat)
    gt_list=torch.tensor(gt_list)

    return {
        "features": np.array(dat),
        "labels": np.array(gt_list)
    }


if __name__ == "__main__":

    device = torch.device('mps')

    # TODO update this to ur device
    root = "/Users/huxley/dataset_storage/snn_tda_mats/LENET_MODELS/competition_dataset"
    # root = "/home/jerryhan/Documents/data"
    models_dir = join(root, "all_models")
    cache_dir = join(root, "calculated_features_cache")

    models = load_all_models(models_dir, cache_dir, percentage=1)
    # models = models[:300]
    # np.random.shuffle(models)
    # np.random.shuffle(models)
    start = time.time()
    random.seed(start//1)
    random.shuffle(models)

    # filter for only resnets
    models = [x for x in models if x.architecture == "resnet50"]
    # models = [x for x in models if x.architecture != "resnet50"]
    # ph_outlier = models[0].load_PH()

    #for model in tqdm(models):
    #    model.recalc_fv()
    # models = [x for x in models if x.architecture != "resnet50"]
    print(models[0])
    # print(models[0].fv['topo_feature_pos'].shape)
    # print(models[0].fv['topo_feature_pos'].shape)
    # models = models[:2]

    triggered = [x for x in models if x.label == 1]
    clean = [x for x in models if x.label == 0]

    print(len(triggered), len(clean))

    min_len = min(len(triggered), len(clean))

    # balance the dataset # FIXME weighting
    triggered = triggered[:min_len]
    clean = clean[:min_len]

    models = triggered + clean
    np.random.shuffle(models)

    print("number of models: ", len(models))

    log.info("Featurizing...")

    TRAIN_TEST_SPLIT = 0.8
    _x = featurize(models)
    dat = _x['features']
    print(dat.shape)

    # dat = dat.reshape(len(models), 60, 14, 14)

    dat = torch.from_numpy(dat).float()

    gt_list = _x['labels']
    # gt_list = torch.tensor(gt_list, dtype=torch.float32).reshape(-1, 1)

    N = len(gt_list)
    n_train = int(TRAIN_TEST_SPLIT * N)
    ind_reshuffle = np.random.choice(list(range(N)), N, replace=False)
    train_ind = ind_reshuffle[:n_train]
    test_ind = ind_reshuffle[n_train:]

    feature_train = dat[train_ind]
    feature_test = dat[test_ind]

    gt_train, gt_test = gt_list[train_ind], gt_list[test_ind]

    # loop through our feature_test, and get which model it correponds to
    # then, check the model architecture
    # if it's a resnet, then we can use it for testing
    """
    new_feature_test = []
    new_gt_test = []
    for i in range(len(feature_test)):
        model = models[test_ind[i]]
        if model.architecture != "resnet50":
            new_feature_test.append(feature_test[i])
            new_gt_test.append(gt_test[i])

    feature_test = np.array(new_feature_test)
    gt_test = np.array(new_gt_test)
    """



    """

    model = cnn_classifier(
        features = {
            'train': feature_train, 'test': feature_test
        },
        labels = {
            'train': gt_train, 'test':gt_test
        }
    )


    model.train()


    model.eval_on_train()
    model.test()


    """





    print(gt_test)

    res = []
    # optimal gamma: 0.07368421052631578
    """
    for i in tqdm(range(100)):
        general_params = {
            "num_epochs": random.randint(100, 500),
            "test_percentage": 0.1
        }
        classifier_params = {
            'objective': 'binary',
            'num_threads': multiprocessing.cpu_count(),
            'metric': 'binary_error',
            'device': "cpu",

            'max_depth': random.randint(2, 5),
            'learning_rate': random.uniform(0.01, 0.1),
            'lambda_l1': random.uniform(0, 0.2),
            'lambda_l2': random.uniform(0, 0.2),
            'min_gain_to_split': random.uniform(0, 0.2),
            'verbose': -1,

            'feature_fraction': 0.9,
            'bagging_fraction': 0.5,
            'bagging_freq': 1
        }
        warnings.filterwarnings('ignore')
        model = lgb_classifier(features = {'train': feature_train, 'test': feature_test}, \
                            labels = {'train': gt_train, 'test': gt_test},
                            classifier_params=classifier_params,
                            general_params=general_params)


        #model = lgb_classifier(features = {'train': feature_train, 'test': feature_test}, \
        #                    labels = {'train': gt_train, 'test': gt_test})
        # print("Training...")
        x = model.train()
        print(general_params, classifier_params)
        print("Cross AUC", x)
        # print("Evaluating")
        train_results, test_results = model.test()
        print("Train", train_results)
        print("Test", test_results)
        res.append({
            "general_params" : general_params,
            "classifier_params" : classifier_params,
            "crossval_auc": x
        })
    """

    general_params = {
        "num_epochs": 30,
        "test_percentage": 0.2
    }

    classifier_params = {

        'boosting_type': 'gbdt',

        'objective': 'binary',
        'num_threads': multiprocessing.cpu_count(),
        'metric': 'binary_error',
        'device': "cpu",

        'n_estimators': 30,

        # 'num_leaves': 31,
        # 'min_data_in_leaf': 20,


        'max_depth': 2,
        # 'learning_rate': 0.2,
        'lambda_l1': 1.3,
        'lambda_l2': 5.0,
        # 'min_gain_to_split': 0.1,
        # 'verbose': -1,

        # 'feature_fraction': 0.5,
        # 'bagging_fraction': 0.5,
        # 'bagging_freq': 2
    }

    # warnings.filterwarnings('ignore')
    model = lgb_classifier(features = {'train': feature_train, 'test': feature_test}, \
                        labels = {'train': gt_train, 'test': gt_test},
                        classifier_params=classifier_params,
                        general_params=general_params)


    #model = lgb_classifier(features = {'train': feature_train, 'test': feature_test}, \
    #                    labels = {'train': gt_train, 'test': gt_test})
    # print("Training...")
    x = model.train()
    print(general_params, classifier_params)
    print("Cross AUC", x)
    # print("Evaluating")
    train_results, test_results = model.test()
    print("Train", train_results)
    print("Test", test_results)
    res.append({
        "general_params" : general_params,
        "classifier_params" : classifier_params,
        "crossval_auc": x
    })


        # print("Gamma", gamma)
        #print("Train results", train_results)
        #print("Test results", test_results)
    #for r in res:
        #print("Train", r[0])
        #print("Test", r[1])
    # train_xgboost(models)
