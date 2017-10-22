import math

import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

data_url = "./mammadAgha.csv"
data = pd.read_csv(data_url).dropna()

trainSize = math.ceil(data.shape[0] * 0.8)
data = data.sample(frac=1)
train = data[:trainSize]
test = data[trainSize:]

y_train = train['label']
# x_train = train.drop("label", axis=1)
x_train = train["ABOF"]

y_test = test['label']
# x_test = test.drop("label", axis=1)
x_test = test["ABOF"]

DT = DecisionTreeClassifier(random_state=0, max_depth=15, min_samples_leaf=2)

leanerSVML1 = LinearSVC(penalty='l1', loss='squared_hinge', dual=False,
                        random_state=0)
leanerSVML2 = LinearSVC(penalty='l2', loss='hinge', dual=True, random_state=0)

clf = svm.SVC(probability=True, verbose=True)

kf = KFold(n_splits=10, random_state=None, shuffle=False)

X = x_train.values
y = y_train.values


def classifing(classifier):
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        classifier.fit(X_train, Y_train)
        print("clf fitted")
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    print("Fscore", f1_score(y_test, prediction))
    print("accuracy", accuracy_score(y_test, prediction))
    print(recall_score(y_test, prediction, average=None))
    if classifier != leanerSVML1 and classifier != leanerSVML2:
        probPrediction = classifier.predict_proba(x_test)
        print("log loss", log_loss(y_test, probPrediction))
    return prediction


outlierList = classifing(DT)
print(outlierList)
#