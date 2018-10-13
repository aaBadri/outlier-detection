import pandas as pd
import matplotlib.pyplot as plt
from BasicModules.modules import clustering as cluster, dimension_reduction as dim_red, evaluation as eval
from mlxtend.evaluate import confusion_matrix


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


def prepare_projected_data(projected, t):
    result = list()
    for i in range(0, t):
        l = sorted(projected[i], key=lambda x: x[0])
        l = list(map(lambda x: x[1], l))
        result.append(l)
    result = list(map(list, zip(*result)))
    return result


# 0. Data loading
train_url = "./data_in/global.csv"
train = pd.read_csv(train_url, delimiter=',', header=None)
ytrain = train.iloc[:, -1]
train = train[:-1]
print("data is loaded")

T = 5
# 1. Dimension Reduction
n = train.shape[0]
projected = dim_red.random_projection(train, T)

# 2. Clustering
new_projected = prepare_projected_data(projected, T)
train["predict"] = cluster.k_means(new_projected)
train["label"] = ytrain

# 3. Evaluation
classes = [0, 1]
confusion_matrix_all = confusion_matrix(train["label"], train["predict"], binary=True)
plt.figure()
eval.plot_confusion_matrix(confusion_matrix_all, classes, normalize=False)
plt.show()

print("finish")