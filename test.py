# TODO we do not check if center and a and b are in a line
import pandas as pd

myList = [[0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [10, 10, 10, 0, 0, 0, 0, 0]]

train_url = "./dump_data_10k_rep_0/ipsweep_normal.csv"
train = pd.read_csv(train_url, delimiter=';', header=None)
print(list(train.iloc[5]))
# for i, row in train.iterrows():
#     print(list(row[:]))
