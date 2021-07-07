import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
np.set_printoptions(threshold=sys.maxsize)

tf.disable_v2_behavior()

print(" wine_Model !!!")
wine = pd.read_csv("./csv/winequality-red.csv", delimiter=';', dtype=float)

wine_copy = wine.copy()
_X = wine_copy.drop(labels=['quality'], axis=1).values
_Y = wine_copy.quality.values
# print(wine.shape)
# print(wine.head())
# print(wine.info())
# print(wine.describe())

train_index = np.random.choice(len(_X), round(len(_X) * 0.8), replace=False)
test_index = np.array(list(set(range(len(_X))) - set(train_index)))
train_X = _X[train_index]
train_Y = _Y[train_index]
test_X = _X[test_index]
test_Y = _Y[test_index]
train_Y = train_Y[:, np.newaxis]
test_Y = test_Y[:, np.newaxis]
print("============")
print(str(train_X[:1, :11]))
print(train_Y[1])
print("============")

def min_max_normalized(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)


train_X = min_max_normalized(train_X)
test_X = min_max_normalized(test_X)


print("==== min_max_normalized ")
print(str(train_X[:1, :11]))
print(train_Y[1])
# print(train_Y)
