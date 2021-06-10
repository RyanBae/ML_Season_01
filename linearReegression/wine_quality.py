
###################################
# Wine Quality Model
# fixed acidity : 고정 산도
# volatile acidity : 휘발성 산도 *
# citric acid : 시트르산 *
# residual sugar : 잔류 설탕
# chlorides : 염화물 *
# free sulfur dioxide : 자유 이산화황
# total sulfur dioxide : 총 이산화황
# density : 밀도
# pH : pH
# sulphates : 황산염 *
# alcohol : 알코올 *
# quality : 품질   # 종속변수
###################################

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

wine = pd.read_csv("./csv/winequality-white.csv", delimiter=';', dtype=float)
# wn = np.loadtxt("./csv/winequality-red.csv", delimiter=';', skiprows=1)

print(wine.info())
print(wine.describe())
# print(wine['quality'].describe())

# wine.hist(bins=25, figsize=(10, 10))

# #######
print(wine.shape)
# plt.hist(wine['quality'], bins=7, rwidth=0.7)
# plt.show()
print(wine.head())
# 정규화
wine_norm = (wine - wine.min()) / (wine.max() - wine.min())
# wine_norm = wine
print("정규화 ===")
print(wine_norm.head())
print(wine_norm.describe())


# 데이터 섞은 후 numpy array 로 변환
print("wine_shuffle ===")
wine_shuffle = wine_norm.sample(frac=1)
print(wine_shuffle.head())
print("===")
print(wine_shuffle.describe())
print("===")

wine_np = wine_shuffle.to_numpy()
print(wine_np[:5])

# 트레인, 테스트 셋 분할 하기
print(wine_np.shape)
print(wine_np[:5, :11])
print(wine_np[:5, 11:12])

train_idx = int(len(wine_np) * 0.7)
train_X, train_Y = wine_np[:train_idx, :-1], wine_np[:train_idx, -1]
test_X, test_Y = wine_np[train_idx:, :-1], wine_np[train_idx:, -1]
print("===")
print(len(wine_np))
print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)
print(len(wine_np))
print(train_X[:5])
print(train_Y[:5])
print(test_X[:5])
print(test_Y[:5])


X = tf.placeholder(tf.float32, shape=[None, 11])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([11, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

learning_rate = 0.001

# Hypothesis
hypothesis = tf.matmul(X, W)+b
# simplified cost / loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

train = optimizer.minimize(cost)

# Correct prediction Test model
prediction = tf.arg_max(hypothesis, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        cost_val, W_wal, _ = sess.run([cost, W, train], feed_dict={
                                      X: train_X, Y: train_Y})
        print(step, cost_val, W_wal)

    print("Prediction : ", sess.run(
        prediction, feed_dict={X: test_X, Y: test_Y}))
    print("Accuracy : ", sess.run(accuracy, feed_dict={X: test_X}))


# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# for step in range(1000):
#     # print(train_X[step])
#     cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={
#                                    X: train_X, Y: train_Y})
#     if step % 10 == 0:
#         print("Step : ", step, " | Cost : ", cost_val,
#               "\n    | Prediction : ", hy_val)
# #         # print("")

# print(wn.shape)
# print(wn[:100, -1])
# print(wine[:100, -1])


# plt.figure(figsize=[10, 6])

# ####### plot bar graph
# plt.bar(wine['quality'], wine['alcohol'], color='red')
# label x-axis
# plt.xlabel('quality')
# label y-axis
# plt.ylabel('alcohol')

# ####### ploting heatmap
# plt.figure(figsize=[19, 10])
# sns.heatmap(wine.corr(), annot=True)

# ####### line, point graph
# plt.show()

# ##### feature(x) 중 label(y)에 연관이 있는걸로 데이터 셋팅.


# ##### label(y)
# ##### bins : 구간을 나눠줄 숫자값
# https://rfriend.tistory.com/404
# 이진 분류기 만들기
# quality 에 한계를 줘서 good, bad 로 분류
# group_names = ['bad', 'good']
# bins = (2, 6.5, 8)
# wine_quality = wine['quality'].copy()
# print(wine_quality)

# #  으로 분류하여 변경
# wine_quality = pd.cut(wine_quality, bins=bins, labels=group_names)
# print(wine_quality)


# print(label_quality)
