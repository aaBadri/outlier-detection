#!/usr/bin/env python3 -W ignore::DeprecationWarning
import warnings
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from modules.modules import utils, dimension_reduction as dim_red, clustering as cluster

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Using a non-tuple sequence")


# DIMENSION = 6 creditcard
# DIMENSION = 3 ps


def fast_voa(path='../data_in/PS.csv', dimension=5, is_product=True):
    # 0. Data loading
    if is_product:
        train, ytrain = utils.load_train_data(path, is_product)
    else:
        train, ytrain = utils.load_train_data(path, is_product)

    # 1. Dimension Reduction
    T = dimension
    n = train.shape[0]
    # projected = train
    projected = dim_red.random_projection(train, T)

    # 2. Clustering
    train["rate"] = cluster.fast_voa(projected, n, T, 1, 1)
    train["label"] = ytrain

    # 3. Evaluation
    if is_product:
        scores = train["rate"]
        scores = list(map(lambda x: float(x), scores))
        max = max(scores)
        for i in range(len(scores)):
            scores[i] = scores[i] / max
            print(scores[i])
    else:
        scores = train["rate"]
        scores = list(map(lambda x: float(x), scores))
        max = max(scores)
        for i in range(len(scores)):
            scores[i] = scores[i] / max
            # scores[i] = scores[i] * 1000
            # scores[i] = int(scores[i])
            # scores[i] = scores[i] / 1000

        fpr, tpr, threshold = roc_curve(ytrain, scores)
        t = np.arange(0., 5., 0.001)
        utils.plot(1, 1, fpr, tpr, 'b', t, t, 'r')
        print("AUC score : ", roc_auc_score(ytrain, scores))
        print("finish")
