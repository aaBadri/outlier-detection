#!/usr/bin/env python3
import warnings
import sys
import threading
from modules.IsolationForest import isolation_forest
from modules.fastVOA import fast_voa
from modules.LOF import lof

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings('ignore', '.*DeprecationWarning.*', )
warnings.filterwarnings("ignore", message="Using a non-tuple sequence")

lof_result = ''
fast_voa_result = ''
isolation_forest_result = ''

path = sys.argv[2]


def lof_caller():
    global lof_result
    lof_result = lof(path, dimension=20, is_product=True)
    lof_result = lof_result.tolist()


def fast_voa_caller():
    global fast_voa_result
    fast_voa_result = fast_voa(path, dimension=6, is_product=True)


def isolation_forest_caller():
    global isolation_forest_result
    isolation_forest_result = isolation_forest(path, dimension=20, is_product=True)
    isolation_forest_result = isolation_forest_result.tolist()


def fraud_detector():
    thread1 = threading.Thread(target=lof_caller(), )
    thread1.start()
    thread1.join()

    thread2 = threading.Thread(target=fast_voa_caller(), )
    thread2.start()
    thread2.join()

    thread3 = threading.Thread(target=isolation_forest_caller(), )
    thread3.start()
    thread3.join()

    # print("iso : " + str(len(isolation_forest_result)))
    # print("fast voa : " + str(len(fast_voa_result)))
    # print("lof : " + str(len(lof_result)))
    final_result = []
    for i in range(len(fast_voa_result)):
        score = 0
        if float(isolation_forest_result[i]) > 0:
            score += 1.5
        if float(lof_result[i]) > 2:
            score += 1
        if float(fast_voa_result[i]) > 0.5:
            score += 1
        if score >= 2.5:
            final_result.append(
                {
                    "index": str(i),
                    "detection": "fraud",
                    "if": isolation_forest_result[i],
                    "fastvoa": fast_voa_result[i],
                    "lof": lof_result[i]
                }
            )
        elif score > 1.5:
            final_result.append(
                {
                    "index": str(i),
                    "detection": "suspected",
                    "if": isolation_forest_result[i],
                    "fastvoa": fast_voa_result[i],
                    "lof": lof_result[i]
                }
            )
    return final_result
