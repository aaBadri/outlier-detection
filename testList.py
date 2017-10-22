# TODO we do not check if center and a and b are in a line
import math as math

import numpy as np
import pandas as pd

myList = [[13874873.3885, 25066.5069, 0.0552, 0.0708, 0.0047, 1.0457, 0.9973, 0.0013],
          [3015939.5849, 8627.855, 0.1001, 0, 0, 0.9971, 1.0038, 0.0019],
          [11702254.5972, 21149.1729, 0.001, 0, 0, 1.0277, 0.99, 0.0007],
          [20328628.9716, 23205.2734, 0.001, 0, 0.0278, 1.0023, 0.9954, 0.0019]]

# myList = [[0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [10, 10, 10, 0, 0, 0, 0, 0]]

train_url = "./dump_data_10k_rep_0/ipsweep_normal.csv"
train = pd.read_csv(train_url, delimiter=';', header=None)


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def getAngle(center, a, b):
    v1 = np.array(a) - np.array(center)
    v2 = np.array(b) - np.array(center)
    if length(v1) and length(v2):
        angle = math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
        print(angle)
        f = angle / (length(v1) ** 2) * (length(v2) ** 2)
        return f
    else:
        return -99999999


def getABOF(vertex, a, b):
    va = np.subtract(vertex, a)
    vb = np.subtract(vertex, b)
    # mammad = sum((a * b) for a, b in zip(va, vb))
    # cosine_angle = mammad / (np.linalg.norm(va) * np.linalg.norm(vb))
    # cosine_angle = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
    # print(cosine_angle)
    # angle = np.arccos(cosine_angle)
    # angle = math.acos(cosine_angle)
    # angle_degree = np.degrees(angle)
    # dista = np.linalg.norm(va)
    dista = length(va)
    # distb = np.linalg.norm(vb)
    distb = length(vb)

    angle = getAngle(list(vertex), list(a), list(b))
    f = angle / (dista ** 2) * (distb ** 2)
    # if (not f):
    #     # print("dista , distb", distb, distb)
    #     # print("Angle", cosine_angle)
    # # print("va : ", va)
    # #     print("vb : ", vb)
    # #     print("###########")
    return f


a = np.array([1, 0, 0])
b = np.array([1, 0, 0])
c = np.array([1, 0, 0])

varABOF = []
for center in myList:
    centerABOF = []
    for i in myList:
        if center != i:
            for j in range(myList.index(i), len(myList)):
                if center != myList[j] and i != myList[j]:
                    print(myList.index(center), myList.index(i), j)
                    centerABOF.append(getAngle(center, i, myList[j]))
    # print(centerABOF)
    varABOF.append(np.var(centerABOF))
print(varABOF)
# print(getABOF(a, b, c))
