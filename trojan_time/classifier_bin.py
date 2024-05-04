import numpy as np
from rich import print, inspect
from sklearn.metrics import roc_auc_score
import torch
from sklearn.model_selection import train_test_split
import os
import multiprocessing
import sys
import xgboost as xgb
from typing import List, Dict, Any

import logging
from colorlog import ColoredFormatter

import lightgbm as lgb

from sklearn.metrics import accuracy_score, log_loss

# log = logging.getLogger('pythonConfig')
# def setup_logger():
#     global log
#     LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
#     logging.basicConfig(level=logging.INFO)
#     logging.getLogger('hyperopt').setLevel(logging.WARNING)

#     formatter = ColoredFormatter(LOGFORMAT)
#     stream = logging.StreamHandler()
#     stream.setLevel(logging.INFO)
#     stream.setFormatter(formatter)
#     log = logging.getLogger('pythonConfig')
#     log.setLevel(logging.INFO)
#     log.addHandler(stream)

# setup_logger()


class lgb_classifier:

    """
        - get_default_params
        - train
        - test
    """

    def __init__(self, features: Dict,
                 labels: Dict,
                 classifier_params = None,
                 general_params = None
                 ):
        """ this needs to be able to load the data, run the classifier
            and have a method for running tests
        """

        self.features = features
        self.labels = labels

        self.train_set = lgb.Dataset(features['train'], label=labels['train'])
        self.test_set = lgb.Dataset(features['test'], label=labels['test'])

        self.classifier_params = classifier_params
        self.general_params = general_params

        if self.classifier_params is None or self.general_params is None:
            self.get_default_params()

        self.model = None

    def get_default_params(self):

        self.general_params = {
            "num_epochs": 30*4,
            "test_percentage": 0.1
        }

        self.classifier_params = {
            'objective': 'binary',
            'num_threads': multiprocessing.cpu_count(),
            'metric': 'binary_error',
            'device': "cpu",

            'max_depth': 5,
            'learning_rate': 0.05,
            'lambda_l1': 0,
            'lambda_l2': 0,

            'bagging_fraction': 0.7,
            # 'feature_fraction': 0.8, # TODO consider turning this on when using PSF features,
                                       # but not when using topo features!
        }


    def train(self):

        # if self.classifier_params['device'] != "gpu":
            # logging.warning("you are not currently using GPU accel!")

        if self.general_params is None:
            logging.error("general_params not set!")
            return None

        bst = lgb.train(
            self.classifier_params,
            self.train_set,
            num_boost_round=self.general_params["num_epochs"],
            valid_sets=[self.test_set],
            verbose_eval=False
        )
        self.model = bst
        self.cv_results = lgb.cv(self.classifier_params, self.train_set, 
                                 num_boost_round=self.general_params["num_epochs"],
                                 nfold=5, metrics='auc', verbose_eval=False)
        # print("Cross-val", max(self.cv_results['auc-mean']))
        
        return max(self.cv_results['auc-mean'])

    def test(self):
        if self.model is None:
            logging.error("model not trained! run model.train first")
            return None

        def eval_metrics(labels, data, thresholds = None):

            y_pred = self.model.predict(data)

            auc = roc_auc_score(labels, y_pred)
            acc = accuracy_score(labels, y_pred >= 0.5)
            ce = log_loss(labels, y_pred) 

            return {
                    "acc": acc,
                    "auc": auc,
                    "ce": ce
                }
        train_results = eval_metrics(self.labels['train'], self.features['train'])
        test_results = eval_metrics(self.labels['test'], self.features['test'])                                   

        return train_results, test_results


class xgb_classifier:

    """
        - get_default_params
        - train
        - test
    """

    def __init__(self, features: Dict,
                 labels: Dict,
                 classifier_params = None,
                 general_params = None
                 ):
        """ this needs to be able to load the data, run the classifier
            and have a method for running tests
        """

        self.features = features
        self.labels = labels

        self.train_set = features['train']
        self.train_labels = labels['train']

        self.test_set = features['test']
        self.test_labels = labels['test']

        self.classifier_params = classifier_params
        self.general_params = general_params

        if self.classifier_params is None or self.general_params is None:
            self.get_default_params()

        self.model = None

    def get_default_params(self):

        self.general_params = {
            "num_epochs": 30*4,
            "test_percentage": 0.1
        }

        self.classifier_params = {
            'objective': 'binary:logistic',
            'nthread': multiprocessing.cpu_count(),
            'eval_metric':'error',
            'device': "cuda",

            'max_depth': 6,
            'eta': 0.05,
            'gamma': 0,
            'lambda': 0,
            'alpha': 0,

            'subsample': 0.7,
            # 'colsample_bytree': 0.8, # TODO consider turning this on when using PSF features,
                                       # but not when using topo features!
        }


    def train(self):
        import xgboost as xgb # Need to import here for HPO

        if self.classifier_params['device'] != "cuda":
            logging.warning("you are not currently using GPU accel!")

        if self.general_params is None:
            logging.error("general_params not set!")
            return None


        # Create sparse matrix for xgboost pipeline
        dtrain = xgb.DMatrix(self.train_set, label=self.train_labels)
        dtest = xgb.DMatrix(self.test_set, label=self.test_labels)

        evallist = [(dtest, 'eval'), (dtrain, 'train')]

        bst = xgb.train(
                self.classifier_params,
                dtrain,
                self.general_params["num_epochs"],
                evals=evallist,
                verbose_eval=False
                )

        self.model = bst

        return bst

    def test(self):
        if self.model is None:
            logging.error("model not trained! run model.train first")
            return None


        dtrain = xgb.DMatrix(self.train_set, label=self.train_labels)
        dtest = xgb.DMatrix(self.test_set, label=self.test_labels)

        def eval_metrics(labels, d, thresholds = None, calc_thresholds=False):

            y_pred = self.model.predict(d)

            auc = roc_auc_score(labels, y_pred)

            if not calc_thresholds and thresholds is not None:
                T, b = thresholds

                y_pred=torch.sigmoid(b*(torch.tensor(y_pred)-T)).numpy()
                acc = np.sum((y_pred >= 0.5)==labels)/len(y_pred)
                # ce_test = np.sum(-(labels * np.log(y_pred) + (1 - labels) * np.log(1 - y_pred))) / len(y_pred)
                ce = (np.mean(-(labels * np.log(y_pred + 1e-10) + (1 - labels) * np.log(1 - y_pred + 1e-10))))
                # FIXME wack

                return {
                        "acc": acc,
                        "auc": auc,
                        "ce": ce,
                        "thresholds": (T, b)
                        }

            logging.info("calculating thresholds throuh grid search")
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
        train_results = eval_metrics(self.train_labels, dtrain, calc_thresholds=True)
        test_results = eval_metrics(self.test_labels, dtest,
                                     thresholds=train_results['thresholds'],
                                     calc_thresholds=False
                                     )

        # print("train results: ", train_results)
        # print("test results: ", test_results)

        return train_results, test_results