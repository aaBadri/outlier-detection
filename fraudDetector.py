#!/usr/bin/env python3
import warnings

from sklearn.metrics import roc_curve, roc_auc_score

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Using a non-tuple sequence")
import sys
import subprocess
import threading
from modules.modules import clustering as cls
import pandas as pd
from modules.modules import utils
import numpy as np
import traceback

LOFR = ''
FAST_VOA = ''
ISOLATION_FOREST = ''

ROOT = "/home/aab/work/outlier-detection/modules/"


def LOF():
    global LOFR
    LOFR = subprocess.check_output(["python3", ROOT + "LOF.py", sys.argv[1]], shell=False)
    LOFR = LOFR.decode()
    LOFR = LOFR.replace(']', '').replace('[', '')
    LOFR = LOFR.split("\n")[:-1]


def fast_voa():
    global FAST_VOA
    FAST_VOA = subprocess.check_output(["python3", ROOT + "fastVOA.py", sys.argv[1]], shell=False)
    FAST_VOA = FAST_VOA.decode()
    FAST_VOA = FAST_VOA.replace(']', '').replace('[', '')
    FAST_VOA = FAST_VOA.split("\n")[:-1]


def isolation_forest():
    global ISOLATION_FOREST
    ISOLATION_FOREST = subprocess.check_output(
        ["python3", ROOT + "IsolationForest.py", sys.argv[1]], shell=False)
    ISOLATION_FOREST = ISOLATION_FOREST.decode()
    ISOLATION_FOREST = ISOLATION_FOREST.replace(']', '').replace('[', '')
    ISOLATION_FOREST = ISOLATION_FOREST.split("\n")[:-1]



thread1 = threading.Thread(target=LOF, )
thread1.start()
thread1.join()
thread2 = threading.Thread(target=fast_voa, )
thread2.start()
thread2.join()
thread3 = threading.Thread(target=isolation_forest, )
thread3.start()
thread3.join()

