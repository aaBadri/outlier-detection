#!/usr/bin/env python3 -W ignore::DeprecationWarning
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from modules.modules import utils, dimension_reduction as dim_red, clustering as cluster

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Using a non-tuple sequence")


# DIMENSION = 20 credit
# DIMENSION = 3  ps

SEPARATOR = "==============================\n"


def lof(path='../data_in/PS.csv', dimension=20, is_product=True):
    # 0. Data loading
    if is_product:
        train, ytrain = utils.load_train_data(path, is_product)
    else:
        train, ytrain = utils.load_train_data(path, is_product)

    # 1. Dimension Reduction
    T = dimension
    n = train.shape[0]
    projected = dim_red.pca(train, T, is_product)

    # 3. Clustering
    predict = cluster.LOF_score(projected)
    train["rate"] = predict
    train["label"] = ytrain

    # 4. Evaluation
    if is_product:
        for i in train["rate"]:
            print(i)
    else:
        fpr, tpr, threshold = roc_curve(ytrain, train["rate"])
        t = np.arange(0., 5., 0.001)
        utils.plot(1, 1, fpr, tpr, 'b', t, t, 'r')
        print("AUC score : ", roc_auc_score(ytrain, train["rate"]))
        print("finish")
