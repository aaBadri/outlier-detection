import random

import numpy as np
info = [[1, -2, 3, -4],[0, 1, -2, 1],[1, 2, 3, 4],[-1, -2, -3, -4]]
counter =0
hashed_train = []
for i in range(0, 3):
    random_vector = []
    for j in range(0, 4):
        random_vector.append(random.randint(-1, 1))
    print("random",random_vector)
    hashed_train.append([])
    for record in info:
        dotted = np.dot(random_vector, record)
        print("dotted ",dotted)
        if dotted >= 0:
            print("salam man")
            hashed_train[i].append(1)
        else:
            print("salam an")
            hashed_train[i].append(-1)
print("counter",counter)
print("hashed ",hashed_train)