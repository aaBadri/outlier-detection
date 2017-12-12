import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pprint
from operator import add

train_url = "./Book1.csv"
train = pd.read_csv(train_url, delimiter=',', header=None)
# train = train.sample(frac=1)
ytrain = train.iloc[:, -1]
train = train[:-1]
print("data is loaded")


def sample(record_number, train):
    origin_train = train
    origin_train["label"] = ytrain
    outliers = origin_train[origin_train["label"] == 1]
    normal = origin_train[origin_train["label"] == 0]
    outliers = outliers.sample(frac=1)
    outliers = outliers[:10]
    normal = normal[:record_number - 10]
    data = pd.concat([outliers, normal])
    return pd.DataFrame(data)


def plot(axisX, axisY, list1, list2, color, list12=[], list22=[], color2=None):
    if list12 is not []:
        plt.plot(list1, list2, color + 'o', list12, list22, color2 + 's')
        plt.axis([0, axisX, 0, axisY])
        plt.show()
    else:
        plt.plot(list1, list2, color + 'o')
        plt.axis([0, axisX, 0, axisY])
        plt.show()


def get_ROC(train):
    tp = fn = fp = tn = tpr = fpr = 0
    result = train["rate"]
    label = train["label"]
    tpr_list = []
    fpr_list = []

    for tr in range(int(np.min(result)), int(np.max(result)) + 1):
        for index, i in train.iterrows():
            if result[index] < tr:  # outlier
                if label[index] == 1:
                    tp += 1
                else:
                    fp += 1
            else:  # normal
                if label[index] == 1:
                    fn += 1
                else:
                    tn += 1
        # print(tp, fn, fp, tn)
        if tp == 0 and fn == 0:
            tpr = 0
        else:
            tpr = tp / (tp + fn)

        if fp == 0 and tn == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list


def random_projection(S, t):
    l = []
    for i in range(0, t):
        ri = []
        for j in range(0, S.shape[1]):
            ri.append(random.randint(0, 1))
        l.append([])
        for index, record in S.iterrows():
            dotted = np.dot(record, ri)
            l[i].append((index, dotted))
        l[i] = sorted(l[i], key=lambda x: x[1])
    return l


def first_moment_estimator(projected, t, n):
    f1 = [0] * n
    for i in range(0, t):
        cl = [0] * n
        cr = [0] * n
        li = projected[i]
        for j in range(0, n):
            idx = li[j][0]
            cl[idx] = j - 1
            cr[idx] = n - 1 - cl[idx]
        for j in range(0, n):
            f1[j] += cl[j] * cr[j]
    return list(map(lambda x: x * ((2 * math.pi) / (t * (n - 1) * (n - 2))), f1))


def frobenius_norm(train, t, n):
    f2 = [0] * n
    sl = np.random.choice([-1, 1], size=(n,), p=[1. / 2, 1. / 2])
    sr = np.random.choice([-1, 1], size=(n,), p=[1. / 2, 1. / 2])
    for i in range(0, t):
        amsl = [0] * n
        amsr = [0] * n
        li = train[i]
        for j in range(1, n):
            idx1 = li[j][0]
            idx2 = li[j - 1][0]
            amsl[idx1] = amsl[idx2] + sl[idx2]
        for j in range(n - 2, -1, -1):
            idx1 = li[j][0]
            idx2 = li[j + 1][0]
            amsr[idx1] = amsr[idx2] + sr[idx2]
        for j in range(0, n):
            f2[j] += amsl[j] * amsr[j]
    return f2


def fast_voa(s, t, s1, s2):
    n = s.shape[0]
    projected = random_projection(s, t)
    f1 = first_moment_estimator(projected, t, n)
    y = []
    for i in range(0, s2):
        s = [0] * n
        for j in range(0, s1):
            result = list(map(lambda x: x ** 2, frobenius_norm(projected, t, n)))
            s = list(map(add, s, result))
        s = list(map(lambda x: x / s1, s))
        y.append(s)
    y = list(map(list, zip(*y)))
    f2 = []
    for i in range(0, n):
        f2.append(np.average(y[i]))
    var = [0] * n
    for i in range(0, n):
        f2[i] = (4 * (math.pi ** 2) / (t * (t - 1) * (n - 1) * (n - 2))) * f2[i] - \
                (2 * math.pi * f1[i]) / (t - 1)
        var[i] = f2[i] - (f1[i] ** 2)
    return var


train["rate"] = fast_voa(train, 10, 10, 5)
train["label"] = ytrain
roc = get_ROC(train)
print("roc pair: ", roc[0], roc[1])
t = np.arange(0., 5., 0.01)

plot(1, 1, roc[1], roc[0], 'b', t, t, 'r')

print("finish")
train.to_csv("mammadAgha.csv", index=False)
