# TODO we do not check if center and a and b are in a line
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk

train_url = "./data_in/Book1.csv"

train = pd.read_csv(train_url, delimiter=',', header=None)
train = train.sample(frac=1)
ytrain = train.iloc[:, -1]
print("data is loaded")

MOD_RATE = 15
CLUSTER_NUMBER = 5


def sample(record_number, out_number, train):
    origin_train = train
    origin_train["label"] = ytrain
    outliers = origin_train[origin_train["label"] == 1]
    normal = origin_train[origin_train["label"] == 0]
    print(normal)
    outliers = outliers.sample(frac=1)
    final_outliers = outliers[:out_number]
    remain_outliers = outliers[out_number:]
    final_normal = normal[:record_number - out_number]
    remain_normal = normal[record_number - out_number:]
    remain_data = pd.concat([remain_normal, remain_outliers])
    remain_data = pd.DataFrame(remain_data)
    remain_data = remain_data.sample(frac=1)
    data = pd.concat([final_normal, final_outliers])
    data = pd.DataFrame(data)
    data = data.sample(frac=1)
    data.drop(['label'], axis=1, inplace=True)
    remain_data.drop(['label'], axis=1, inplace=True)
    return data, remain_data


def random_projection(S, t):
    l = []
    for i in range(0, t):
        ri = []
        for j in range(0, S.shape[1]):
            ri.append(random.randint(0, 1))
        l.append([])
        for index, record in S.iterrows():
            dotted = np.dot(record, ri)
            l[i].append(dotted)
        # l[i] = sorted(l[i], key=lambda x: x[1])
    return l


def hash_train(train, hash_rate):
    hashed_train = []
    print(train.shape[0], train.shape[1])
    for i in range(0, hash_rate):
        random_vector = []
        for j in range(0, train.shape[1]):
            random_vector.append(random.randint(-1, 1))
        hashed_train.append([])
        for index, record in train.iterrows():
            dotted = np.dot(record, random_vector)
            if dotted >= 0:
                hashed_train[i].append(1)
            else:
                hashed_train[i].append(-1)
    print("width1 ", len(hashed_train))
    print("height1 ", len(hashed_train[0]))
    hashed_train = list(map(list, zip(*hashed_train)))
    print("width ", len(hashed_train))
    print("height", len(hashed_train[0]))
    return pd.DataFrame(hashed_train)


def getABOF(vertex, a, b):
    va = a - vertex
    vb = b - vertex
    cosine_angle = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
    angle = np.arccos(cosine_angle)
    angle_degree = np.degrees(angle)
    dista = np.linalg.norm(va)
    distb = np.linalg.norm(vb)
    return angle_degree


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
    result = train["ABOF"]
    label = train["label"]
    print(result, label)
    tpr_list = []
    fpr_list = []

    for tr in range(int(np.min(result)), int(np.max(result))):
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
    return list(map(lambda x: x * ((2 * np.math.pi) / (t * (n - 1) * (n - 2))), f1))


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


def fast_voa(s1, s2, projected):
    n = projected.shape[0]
    t = projected.shape[1]
    f1 = first_moment_estimator(projected, t, n)
    y = []
    for i in range(0, s2):
        s = [0] * n
        for j in range(0, s1):
            result = list(map(lambda x: x ** 2, frobenius_norm(projected, t, n)))
            s = list(map(np.add, s, result))
        s = list(map(lambda x: x / s1, s))
        y.append(s)
    y = list(map(list, zip(*y)))
    f2 = []
    for i in range(0, n):
        f2.append(np.average(y[i]))
    var = [0] * n
    for i in range(0, n):
        f2[i] = (4 * (np.math.pi ** 2) / (t * (t - 1) * (n - 1) * (n - 2))) * f2[i] - \
                (2 * np.math.pi * f1[i]) / (t - 1)
        var[i] = f2[i] - (f1[i] ** 2)
    return var


def kmeans(train, test, train_record_number, outlier_number, n_clusters, n_init, max_iter):
    train_temp, remain_data = sample(train_record_number, outlier_number, train)
    print(train)
    kmeans = KMeans(n_clusters=n_clusters, random_state=None, n_init=n_init, max_iter=max_iter).fit(train_temp)
    result = kmeans.predict(test)
    return list(result)


test, remain_data = sample(150, 10, train)

test = random_projection(test, 20)
test = list(map(list, zip(*test)))
test = pd.DataFrame(test)
# print(test.shape)
remain_data = random_projection(remain_data, 20)
remain_data = list(map(list, zip(*remain_data)))
remain_data = pd.DataFrame(remain_data)
# print(remain_data.shape)
train.to_csv("./data_out/hash_train.csv", index=False)

cluster_labels = kmeans(remain_data, test, 600, 20, CLUSTER_NUMBER, 10, 300)
test['cluster'] = cluster_labels
# print(test)

clusters = list()
for i in range(0, CLUSTER_NUMBER):
    clusters.append(test[test['cluster'] == i])

calculated_clusters = []

for cluster in clusters:
    cluster.drop(['cluster'], axis=1, inplace=True)
    feature_number = cluster.shape[1]
    record_number = cluster.shape[0]
    original_cluster = cluster.copy()
    # cluster_list = cluster.tolist()
    for i in range(0, record_number):
        for j in range(0, feature_number):
            dotted = cluster.iloc[i][j]
            cluster.set_value(i, j, (i, dotted))
    l = []

    for i in range(0, record_number):
        record = cluster[i]
        l.append(record.tolist())
    cluster_list = list(map(list, zip(*l)))

    for i in range(0, feature_number):
        cluster_list[i] = sorted(cluster_list[i], key=lambda x: x[1])
    original_cluster["rate"] = fast_voa(10, 5, cluster_list)
    calculated_clusters.append(original_cluster)

concatted_clusters = pd.DataFrame([])
for i in range(0, CLUSTER_NUMBER):
    concatted_clusters = pd.concat([concatted_clusters, calculated_clusters[i]])

concatted_clusters["label"] = ytrain
roc = get_ROC(concatted_clusters)
print(roc[0], roc[1])
t = np.arange(0., 5., 0.01)

plot(1, 1, roc[1], roc[0], 'b', t, t, 'r')

print("finish")
concatted_clusters.to_csv("./data_out/mammadAgha.csv", index=False)
