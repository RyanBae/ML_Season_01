
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

wine_quality = wine['quality']
print(wine_quality)
wine_norm = (wine - wine.min()) / (wine.max() - wine.min())
# wine_norm = wine
print(wine_norm)


# 데이터 섞은 후 numpy array 로 변환
# wine_shuffle = wine_norm.sample(frac=1)
wine_np = wine_norm.to_numpy()
wine_y = wine_quality.to_numpy()
print(wine_y)
print("??")
wine_y = wine_y[:, np.newaxis]
print(wine_y)

# 트레인, 테스트 셋 분할 하기
train_idx = int(len(wine_np) * 0.7)
train_X, train_Y = wine_np[:train_idx, :-1], wine_y[:train_idx, -1]
test_X, test_Y = wine_np[train_idx:, :-1], wine_y[train_idx:, -1]
# print(train_Y.shape)
print("======")
# train_Y = np.expand_dims(train_Y, axis=0)
# test_Y = np.expand_dims(test_Y, axis=0)
train_Y = train_Y[:, np.newaxis]
test_Y = train_Y[:, np.newaxis]
# print(train_Y.shape)
print(train_X[:5, :])

X = tf.placeholder(tf.float32, shape=[None, 11])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([11, 1]), name='weight')
b = tf.Variable(tf.random_normal([1, 1]), name='bias')

learning_rate = 0.001

# Hypothesis
hypothesis = tf.matmul(X, W)+b
# simplified cost / loss function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

train = optimizer.minimize(cost)

# Correct prediction Test model
prediction = tf.round(tf.sigmoid(hypothesis))
is_correct = tf.cast(tf.equal(prediction, Y), dtype=tf.float32)
accuracy = tf.reduce_mean(is_correct)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     # for epoch in range(training_epochs):
#     #     avg_cost = 0
#     #     total_batch = int(wine.train.num_examples / batch_size)
#     #     for i in range(total_batch):
#     #         batch_xs, batch_ys =

#     for step in range(2000):
#         cost_val, W_wal, _ = sess.run([cost, W, train], feed_dict={
#                                       X: train_X, Y: train_Y})

#         print("| Step : ", step, "\n| Cost : ", cost_val,
#               "\n| Prediction : ", W_wal, "  | ", _)
#         # print("Acc : "+sess.run(accuracy, feed_dict={X: train_X}))

#     # print("Prediction : ", sess.run(
#         # prediction, feed_dict={X: test_X, Y: test_Y}))
#     print("Accuracy : ", sess.run(accuracy, feed_dict={X: test_X}))
#     # print("Accuracy : ", accuracy.eval(session=sess, feed_dict={X: test_X}))


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
