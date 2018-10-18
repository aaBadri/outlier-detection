#!/usr/bin/env python3
import warnings
import sys
import threading
from modules.IsolationForest import isolation_forest
from modules.fastVOA import fast_voa
from modules.LOF import lof

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Using a non-tuple sequence")

LOFR = ''
FAST_VOA = ''
ISOLATION_FOREST = ''


def lof_caller():
    global LOFR
    LOFR = lof(sys.argv[1], dimension=5, is_product=True)
    LOFR = LOFR.decode()
    LOFR = LOFR.replace(']', '').replace('[', '')
    LOFR = LOFR.split("\n")[:-1]


def fast_voa_caller():
    global FAST_VOA
    FAST_VOA = fast_voa(sys.argv[1], dimension=5, is_product=True)
    FAST_VOA = FAST_VOA.decode()
    FAST_VOA = FAST_VOA.replace(']', '').replace('[', '')
    FAST_VOA = FAST_VOA.split("\n")[:-1]


def isolation_forest_caller():
    global ISOLATION_FOREST
    ISOLATION_FOREST = isolation_forest(sys.argv[1], dimension=5, is_product=True)
    ISOLATION_FOREST = ISOLATION_FOREST.decode()
    ISOLATION_FOREST = ISOLATION_FOREST.replace(']', '').replace('[', '')
    ISOLATION_FOREST = ISOLATION_FOREST.split("\n")[:-1]


thread1 = threading.Thread(target=lof_caller(), )
thread1.start()
thread1.join()

thread2 = threading.Thread(target=fast_voa_caller(), )
thread2.start()
thread2.join()

thread3 = threading.Thread(target=isolation_forest_caller(), )
thread3.start()
thread3.join()
