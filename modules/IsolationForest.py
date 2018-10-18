#!/usr/bin/env python3
import copy
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from modules.modules import utils, dimension_reduction as dim_red, evaluation as eval, clustering as cluster
import sys
from sklearn.metrics import roc_auc_score, roc_curve

SEPARATOR = "==============================\n"


def isolation_forest(path='../data_in/PS.csv', dimension=5, is_product=True):
    # 0. Data loading
    if is_product:
        train, ytrain = utils.load_train_data(path, is_product)
    else:
        train, ytrain = utils.load_train_data(path, is_product)

    # 1. Dimension Reduction
    T = dimension
    projected = dim_red.pca(train, T, is_product)

    # 2. Clustering
    train["rate"] = cluster.isolation_forest_score(projected)
    train["label"] = ytrain

    # 3. Evaluation
    if is_product:
        return train
        # for i in train["rate"]:
        #     print(i)
    else:
        fpr, tpr, threshold = roc_curve(ytrain, train["rate"])
        t = np.arange(0., 5., 0.001)
        utils.plot(1, 1, fpr, tpr, 'b', t, t, 'r')
        print("AUC score : ", roc_auc_score(ytrain, train["rate"]))
        print("finish")
