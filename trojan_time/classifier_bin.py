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

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

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

        self.train_set = lgb.Dataset(features['train'], label=labels['train'], params={'verbose': -1})
        self.test_set = lgb.Dataset(features['test'], label=labels['test'], params={'verbose': -1})

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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(60, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)  # Increased dropout rate
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=0.2)  # Increased dropout rate
        self.conv3 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)  # Additional convolutional layer
        self.dropout3 = nn.Dropout(p=0.2)  # Increased dropout rate
        self.fc1 = nn.Linear(128 * 14 * 14, 256)  # Increased size and adjusted for the additional convolutional layer
        self.dropout4 = nn.Dropout(p=0.2)  # Increased dropout rate
        self.fc2 = nn.Linear(256, 128)  # Additional fully connected layer
        self.dropout5 = nn.Dropout(p=0.2)  # Increased dropout rate
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        x = x.view(-1, 128 * 14 * 14)
        x = F.relu(self.fc1(self.dropout4(x)))
        x = F.relu(self.fc2(self.dropout5(x)))
        x = torch.sigmoid(self.fc3(x))
        return x


class cnn_classifier:

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

        print(self.features['train'].shape)

        self.classifier_params = classifier_params
        self.general_params = general_params

        if self.classifier_params is None or self.general_params is None:
            self.get_default_params()

        self.model = CNN()
        self.model.to('mps')
        self.features['train'] = self.features['train'].to('mps')
        self.labels['train'] = self.labels['train'].to('mps')
        self.features['test'] = self.features['test'].to('mps')
        self.labels['test'] = self.labels['test'].to('mps')


    def get_default_params(self):
        pass

    def train(self):
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(12):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.features['train'], 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data
                labels = (self.labels['train'][i]).reshape(1,1)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                # print(labels.shape)
                outputs = self.model(inputs)
                # print("OK")
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def eval_on_train(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(self.features['train'], 0):
                images, labels = data, self.labels['train'][i]
                outputs = self.model(images)
                # print(outputs)
                predicted = (outputs.data > 0.5)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the train images: %d %%' % (
            100 * correct / total))

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(self.features['test'], 0):
                images, labels = data, self.labels['test'][i]
                outputs = self.model(images)
                # print(outputs)
                predicted = (outputs.data > 0.5)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))

class dnn_classifier:
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

        self.classifier_params = classifier_params
        self.general_params = general_params

        if self.classifier_params is None or self.general_params is None:
            self.get_default_params()

        self.model = self.create_model()

        self.model.to('mps')
        self.features['train'] = self.features['train'].to('mps')
        self.labels['train'] = self.labels['train'].to('mps')
        self.features['test'] = self.features['test'].to('mps')
        self.labels['test'] = self.labels['test'].to('mps')

    def create_model(self):

        class DNN(nn.Module):
            def __init__(self, superself):
                super(DNN, self).__init__()

                input_shape = superself.features['train'].shape[1]

                self.fc1 = nn.Linear(input_shape, 256)
                self.act1 = nn.ReLU()
                self.dropout1 = nn.Dropout(p=0.6)

                self.fc2 = nn.Linear(256, 128)
                self.act2 = nn.ReLU()
                self.dropout2 = nn.Dropout(p=0.6)

                self.fc3 = nn.Linear(128, 1)
                self.act3 = nn.Sigmoid()


            def forward(self, x):
                x = self.fc1(x)
                x = self.act1(x)
                x = self.dropout1(x)

                x = self.fc2(x)
                x = self.act2(x)
                x = self.dropout2(x)

                x = self.fc3(x)
                x = self.act3(x)

                return x
        return DNN(self)

    def get_default_params(self):
        pass

    def train(self):
        criterion = nn.BCELoss()
        # optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        for epoch in range(300):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.features['train'], 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data

                # labels = (self.labels['train'][i]).reshape(1,1)
                labels = self.labels['train'][i]


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                # print(labels.shape)
                outputs = self.model(inputs)
                # print("OK")
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 2000 mini-batches
                    # print('[%d, %5d] loss: %.3f' %
                    #       (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

                    self.eval_on_train()
                    self.test()

        print('Finished Training')

    def eval_on_train(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(self.features['train'], 0):
                images, labels = data, self.labels['train'][i]
                outputs = self.model(images)
                # print(outputs)
                predicted = (outputs.data > 0.5)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # calculate the AUC as well
        y_pred = self.model(self.features['train']).detach().cpu().numpy()
        y_true = self.labels['train'].detach().cpu().numpy()
        auc = roc_auc_score(y_true, y_pred)


        print(f'{round(100 * correct / total, 2)}, {round(auc, 2)}, train')
        # print('Accuracy of the network on the train images: %d %%' % (
        #     100 * correct / total))

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(self.features['test'], 0):
                images, labels = data, self.labels['test'][i]
                outputs = self.model(images)
                # print(outputs)
                predicted = (outputs.data > 0.5)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        y_pred = self.model(self.features['test']).detach().cpu().numpy()
        y_true = self.labels['test'].detach().cpu().numpy()
        auc = roc_auc_score(y_true, y_pred)

        print(f'{round(100 * correct / total, 2)}, {round(auc, 2)}, test \n\n')
