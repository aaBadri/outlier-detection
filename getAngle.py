import numpy as np


def getAngle(vertex, a, b):
    va = a - vertex
    vb = b - vertex
    cosine_angle = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
    angle = np.arccos(cosine_angle)
    print(np.degrees(angle))


a = np.array([0, 0, 0])
b = np.array([0, 0, 1])
c = np.array([1, 0, 1])
getAngle(a, b, c)
