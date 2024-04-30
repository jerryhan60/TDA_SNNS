#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2021-12-17 12:00:00
# @Author  : Songzhu Zheng (imzszhahahaha@gmail.com)
# @Link    : https://songzhu-academic-site.netlify.app/

import math

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
import logging
# logging.basicConfig(level=logging.INFO)
from colorlog import ColoredFormatter

from networks import ResNet18, DenseNet121, MLP

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

def xgb_crossval(p):
    """
    Run cross validation to perform HPO on training set.
    Input args:
        p: A list contain hyper-parameter configuration
    Return:
        return_dict: A dictionary contain HPO results
    """

    import xgboost as xgb # Need to import here for HPO
    feature, gt_labels, max_depth, eta, gamma, lamb, alpha=p
    param = {
            'objective': 'binary:logistic',
            'nthread': 20,
            'eval_metric':'error'
            }
    param['max_depth']=int(max_depth)
    param['eta']=eta
    param['gamma']=gamma
    param['lambda']=lamb
    param['alpha']=alpha

    ################ added ################
    param['subsample']=0.7
    #param['colsample_bytree']=0.8

    #param['tree_method']='gpu_hist' :(
    log.warning("you are not currently using GPU accel!")

    acc = []
    auc = []
    crossEntropy = []

    auc_weights = []

    xgb_list=[]
    fold = 4
    # num_epochs=30*10
    num_epochs=30*5
    fold_ind=0

    y_pred_aggregated = []
    labels_aggregated = []

    kf = KFold(n_splits=fold, random_state=123, shuffle=True)
    kf.get_n_splits(feature)
    for train_index, test_index in kf.split(feature):
        train_set, test_set = feature[train_index], feature[test_index]
        train_y, test_y = gt_labels[train_index], gt_labels[test_index]

        # Create sparse matrix for xgboost pipeline
        dtrain = xgb.DMatrix(train_set, label=train_y)
        dtest = xgb.DMatrix(test_set, label=test_y)
        evallist = [(dtest, 'eval'), (dtrain, 'train')]

        bst = xgb.train(param, dtrain, num_epochs, evals=evallist, verbose_eval=False)
        y_pred = bst.predict(dtest)
        soft_pred = y_pred

        # Record accuracy and cross entropy
        acc_temp = np.sum((y_pred >= 0.5) == test_y) / len(y_pred)
        acc.append(acc_temp)


        # crossEntropy_temp = np.sum(-(test_y * np.log(soft_pred) + (1 - test_y) * np.log(1 - soft_pred))) / len(y_pred)
        crossEntropy_temp = (np.mean(-(test_y * np.log(y_pred + 1e-10) + (1 - test_y) * np.log(1 - y_pred + 1e-10))))

        crossEntropy.append(crossEntropy_temp)

        auc_temp = roc_auc_score(test_y, y_pred)
        auc.append(auc_temp)
        auc_weights.append(auc_temp) # idk # FIXME

        y_pred_aggregated.extend(y_pred)
        labels_aggregated.extend(test_y)

        xgb_list.append(bst)

        fold_ind += 1 # only for debug

        print('Fold [{}|{}] - Acc {:.3f}% - AUC {:.3f} - CE {:.3f}'
              .format(fold, fold_ind, acc_temp*100, auc_temp, crossEntropy_temp))

    """
    # Optimize the threshold
    # For each fold record corresponding auc and auc-weighted accuracy
    y_pred_ensemble=0
    for i in range(len(xgb_list)):
        xgb=xgb_list[i]
        y_pred_ensemble+=xgb.predict(dtest)*auc[i]/sum(auc)

    y_pred_ensemble=torch.tensor(y_pred_ensemble/len(xgb_list))

    # Search for a threshold to decide trojaned and clean model using SGD
    T=torch.ones(1, requires_grad=True)
    multi=torch.ones(1, requires_grad=True)
    optimizer=torch.optim.Adam([T]+[multi], lr=1e-1, weight_decay=5e-6)
    labels=torch.tensor(test_y).float()
    for t in range(5*1000):
        optimizer.zero_grad()
        score=torch.sigmoid(multi*(y_pred_ensemble-T))
        loss=-torch.mean(labels*torch.log(score)+(1-labels)*torch.log(1-score))
        loss.backward()
        optimizer.step()
    """

    y_pred_tensor = torch.tensor(y_pred_aggregated, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_aggregated, dtype=torch.float32)

    T = torch.tensor([0.5], requires_grad=True)
    b = torch.tensor([1.0], requires_grad=True)
    optimizer = torch.optim.Adam([T, b], lr=0.1)

    # for t in range(1000):
    for t in range(1): # TODO FIXME this is just broken,
                       # and is replaced by the grid search in
                       # competition_classifier>run_model_tests>grid search
        optimizer.zero_grad()
        logits = b * (y_pred_tensor - T)
        loss = F.binary_cross_entropy_with_logits(logits, labels_tensor)
        loss.backward()
        optimizer.step()

    # Optimize min acc
    goal=np.mean(auc)

    return_dict={'loss':1-goal,
                 'status':STATUS_OK,
                'models':xgb_list,
                'weight':auc,
                'threshold':[float(T.item()), float(b.item())]}

    return return_dict


def run_crossval_xgb(feature, gt_train):
    """
    Configure and perform xgboost HPO with input feature and labels.
    Input args:
        feature: np array that contains input features
        gt_train: np array that contains input targets
    Return:
        best_model_list: best xgboost model that achieve optimal object during cross validation
    """
    trials=Trials()
    hp_config=[]
    hp_config.append(feature)
    hp_config.append(gt_train)

    # hp_config.append(hp.qloguniform('max_depth', low=math.log(2), high=math.log(15), q=1))
    hp_config.append(hp.qloguniform('max_depth', low=0.9, high=8, q=1))

    # hp_config.append(hp.uniform('eta', low=0.5, high=1))
    # hp_config.append(hp.uniform('eta', low=0.1, high=0.9))
    # hp_config.append(hp.uniform('eta', low=0.005, high=0.05))
    hp_config.append(hp.uniform('eta', low=0.002, high=0.03))

    # hp_config.append(hp.loguniform('gamma', low=math.log(0.1), high=math.log(2)))
    hp_config.append(hp.loguniform('gamma', low=0.2, high=3))

    # hp_config.append(hp.loguniform('lamb', low=math.log(0.1), high=math.log(1)))
    hp_config.append(hp.loguniform('lamb', low=0, high=0.3))

    # hp_config.append(hp.loguniform('alpha', low=math.log(0.1), high=math.log(1)))
    hp_config.append(hp.loguniform('alpha', low=0, high=0.1))

    ####### added #######
    # hp_config.append(hp.loguniform('alpha', low=0, high=3))

    # Optimize object with hypoeropt
    # _=fmin(xgb_crossval, hp_config, algo=tpe.suggest, ax_evals=100, trials=trials)

    # _=fmin(xgb_crossval, hp_config, algo=tpe.suggest, max_evals=100, trials=trials)
    max_evals = 20

    if max_evals < 100:
        log.warning(f"max evals is {max_evals}, not running at full fidelity!")
    if max_evals >= 100:
        log.warning(f"max evals is {max_evals}, running at full fidelity! you can reduce this when prototyping for much faster speeds")

    _=fmin(xgb_crossval, hp_config, algo=tpe.suggest, max_evals=max_evals, trials=trials)


    best_model=getBestModelfromTrials(
            trials,
            optimize_over="loss",
            optimize_direction='min'
            )

    return best_model


def mlp_crossval(p):
    """
    Running MLP HPO with hyperopt pipeline.
    Input args:
        p: A list contains HPO configuration
    Return args:
        return_dict: A dictionary contains HPO results
    """

    feature, gt_train, nlayers_encoder_psf, nh_encoder_psf, nlayers_encoder_topo, nh_encoder_topo, nlayers_cls, nh_cls, lr, weight_decay=p
    nlayers_encoder_psf, nlayers_encoder_topo, nlayers_cls = int(nlayers_encoder_psf), int(nlayers_encoder_topo),  int(nlayers_cls)
    nh_encoder_psf, nh_encoder_topo , nh_cls = int(nh_encoder_psf), int(nh_encoder_topo), int(nh_cls)
    gt_train=np.array(gt_train)

    acc=[]
    auc=[]
    mlp_list=[]
    fold=4
    batch_size=32

    kf=KFold(n_splits=fold, random_state=123, shuffle=True)
    fold_ind=0
    for train_index, test_index in kf.split(gt_train):
        train_set=[feature[i] for i in train_index]
        test_set=[feature[i] for i in test_index]
        train_y, test_y = gt_train[train_index], gt_train[test_index]

        # Use branched MLP here. One MLP takes PSF feature and another takes Topological feature.
        encoder_psf = MLP(nlayers_encoder_psf, train_set[0]['psf_fv_pos_i'].shape[1], nh_encoder_psf, nh_encoder_psf).cuda()
        encoder_topo = MLP(nlayers_encoder_topo, train_set[0]['topo_fv_pos_i'].shape[1], nh_encoder_topo, nh_encoder_topo).cuda()
        # A further MLP is used to accept concatenate representation and give Trojan detection
        cls = MLP(nlayers_cls, nh_encoder_psf+nh_encoder_topo, nh_cls, 2).cuda()
        encoder_psf.train()
        encoder_topo.train()
        cls.train()

        optimizer = torch.optim.Adam([x for x in encoder_psf.parameters()]+[x for x in encoder_topo.parameters()]+[x for x in cls.parameters()],
                                     lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(50):
            correct = 0
            total = 0
            ind=torch.randperm(len(train_set))
            for i in range(0, max(int((len(train_set)-1)/batch_size), 1)):
                optimizer.zero_grad()
                batch_ind=ind[(batch_size*i):min(batch_size*(i+1), len(train_set))]
                batch=[train_set[j] for j in batch_ind]
                labels=[train_y[j] for j in batch_ind]
                embedding_list = []
                for single_input in batch:
                    psf_fv_pos_i=single_input['psf_fv_pos_i'].cuda()
                    topo_fv_pos_i=single_input['topo_fv_pos_i'].cuda()
                    embedding_psf=encoder_psf(psf_fv_pos_i)
                    # Handle single datapoint batch trick. Repeat one data 5 times each time add small Gaussian perturbation
                    if len(topo_fv_pos_i)==1:
                        topo_fv_pos_i=topo_fv_pos_i.repeat(5, 1)+torch.randn(5, topo_fv_pos_i.shape[1]).cuda()
                    embedding_topo=encoder_topo(topo_fv_pos_i)
                    embedding=torch.cat([embedding_psf.mean(0).flatten(), embedding_topo.mean(0).flatten()])
                    embedding_list.append(embedding)

                embeddings=torch.cat([x.unsqueeze(0) for x in embedding_list])
                labels=torch.tensor(labels).cuda().long()
                output = cls(embeddings)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                pred = output.argmax(1)
                correct += pred.eq(labels).sum().item()
                total += len(labels)

        train_acc=correct/total

        correct = 0
        total = 0
        encoder_psf.eval()
        encoder_topo.eval()
        cls.eval()
        labels_list=[]
        output_list=[]

        for i in range(0, max(int((len(test_set) - 1) / batch_size), 1)):

            batch = test_set[(batch_size*i):min(batch_size*(i+1), len(test_set))]
            labels = test_y[(batch_size*i):min(batch_size*(i+1), len(test_set))]
            embedding_list=[]
            for single_input in batch:
                psf_fv_pos_i = single_input['psf_fv_pos_i'].cuda()
                topo_fv_pos_i = single_input['topo_fv_pos_i'].cuda()
                if len(topo_fv_pos_i) == 1:
                    topo_fv_pos_i = topo_fv_pos_i.repeat(5, 1) + torch.randn(5, topo_fv_pos_i.shape[1]).cuda()
                embedding_psf = encoder_psf(psf_fv_pos_i)
                embedding_topo = encoder_topo(topo_fv_pos_i)
                embedding = torch.cat([embedding_psf.mean(0).flatten(), embedding_topo.mean(0).flatten()])
                embedding_list.append(embedding)

            embeddings = torch.cat([x.unsqueeze(0) for x in embedding_list])
            labels = torch.tensor(labels).cuda().long()
            output = cls(embeddings)
            pred = output.argmax(1)
            correct += pred.eq(labels).sum().item()
            total += len(labels)
            labels_list.append(labels.detach().cpu())
            output_list.append(output.detach().cpu())

        test_acc = correct / total
        labels=torch.cat(labels_list)
        output=torch.cat(output_list)
        test_auc=roc_auc_score(labels.numpy(), torch.nn.functional.softmax(output,1)[:, 1].numpy())

        acc.append(test_acc)
        auc.append(test_auc)
        mlp_list.append((encoder_psf, encoder_topo, cls))
        fold_ind+=1

        print('Fold [{}|{}] - Train Acc {:.3f}% - Test Acc {:.3f}% - Test AUC {:.3f}'.
              format(fold, fold_ind, train_acc*100, test_acc*100, test_auc))
    print('--------------------------------------------------------------------------------')

    goal = np.min(acc)
    return_dict = {'loss': 1 - goal,
                   'status': STATUS_OK,
                   'models': mlp_list,
                   'weight':acc}

    return return_dict


def run_crossval_mlp(feature, gt_train):
    """
    Configure and perform MLP HPO with input feature and labels.
    Input args:
        feature: np array that contains input features
        gt_train: np array that contains input targets
    Return:
        best_model_list: best MLP model that achieve optimal object during cross validation
    """
    trials=Trials()
    hp_config=[]
    hp_config.append(feature)
    hp_config.append(gt_train)
# TODO FIXME
    raise NotImplementedError
    # i uh messed this up, need to revert this to the original code from the repo srry :grimacing:

"""
    hp_config.append(hp.quniform('nlayers_encoder_psf', low=2, high=3, q=1)) # max depth

    hp_config.append(hp.qloguniform('nh_encoder_psf', low=math.log(128), high=math.log(256), q=1)) # eta
    # hp_config.append(hp.qloguniform('nh_encoder_psf', low=math.log(0.1), high=math.log(0.9), q=1)) # eta

    hp_config.append(hp.quniform('nlayers_encoder_topo', low=2, high=5, q=1)) # gamma
    # hp_config.append(hp.quniform('nlayers_encoder_topo', low=0, high=5, q=1)) # gamma

    hp_config.append(hp.qloguniform('nh_encoder_topo', low=math.log(128), high=math.log(256), q=1)) # lambda
    # hp_config.append(hp.qloguniform('nh_encoder_topo', low=0, high=5, q=1)) # lambda

    hp_config.append(hp.quniform('nlayers_cls', low=0, high=5, q=1)) # alpha

    hp_config.append(hp.qloguniform('nh_cls', low=math.log(45), high=math.log(128), q=1))
    hp_config.append(hp.loguniform('lr', low=math.log(1e-3), high=math.log(0.2)))
    hp_config.append(hp.loguniform('weight_decay', low=math.log(5e-5), high=math.log(5e-3)))

    _ = fmin(mlp_crossval, hp_config, algo=tpe.suggest, max_evals=50, trials=trials)

    best_model_list=getBestModelfromTrials(trials)

    return best_model_list
"""

def getBestModelfromTrials(trials, optimize_over='loss', optimize_direction='min'):
    """
    Helper function to get best model from HPO search.
    Input args:
        trials: HPO trials result
    """
    valid_trial_list = [trial for trial in trials if STATUS_OK == trial['result']['status']]
    # losses = [float(trial['result']['loss']) for trial in valid_trial_list]
    vals = [trial['result'][optimize_over] for trial in valid_trial_list]

    if optimize_direction == 'min':
        best_trial_obj = valid_trial_list[np.argmin(vals)]
    else:
        best_trial_obj = valid_trial_list[np.argmax(vals)]

    return best_trial_obj['result']
