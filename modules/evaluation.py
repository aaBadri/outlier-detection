import numpy as np


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
