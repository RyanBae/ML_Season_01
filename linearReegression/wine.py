import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

wine = pd.read_csv("./csv/winequality-red.csv", delimiter=';', dtype=float)

print(wine.shape)
print(wine.head())
print(wine.info())
print(wine.describe())

# X, Y 분리
wine_copy = wine.copy()
# wine_copy = (wine_copy - wine_copy.min()) / (wine_copy.max() - wine_copy.min())
_X = wine_copy.drop(labels=['quality'], axis=1).values
_Y = wine_copy.quality.values


# 트레인, 테스트 셋 나누기
# train_idx = int(len(_X) * 0.7)
train_index = np.random.choice(len(_X), round(len(_X) * 0.8), replace=False)
test_index = np.array(list(set(range(len(_X))) - set(train_index)))
train_X = _X[train_index]
train_Y = _Y[train_index]
test_X = _X[test_index]
test_Y = _Y[test_index]
# train_X, train_Y = X[:train_idx, :-1], Y[:train_idx, -1]
# train_X, train_Y = _X[:train_idx], _Y[:train_idx]
# test_X, test_Y = X[train_idx:, :-1], Y[train_idx:, -1]
# test_X, test_Y = _X[train_idx:], _Y[train_idx:]

train_Y = train_Y[:, np.newaxis]
test_Y = test_Y[:, np.newaxis]

print("==== ")


# 노멀라이즈드
def min_max_normalized(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)


train_X = min_max_normalized(train_X)
test_X = min_max_normalized(test_X)
# print(train_X[:5, :])
print("==== train X")
print(train_X)
print("==== train Y")
print(train_Y)
print("==== test X")
print(test_X[:1])
print("==== test Y")
print(test_Y)

X = tf.placeholder(tf.float32, shape=[None, 11])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([11, 1]), name='weight')
b = tf.Variable(tf.random_normal([1, 1]), name='bias')


hypothesis = tf.matmul(X, W)+b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cost)

# prediction 정할때 0, 1 에서 찾는게 아니라면 tf.sigmoid 는 사용하지 않도록 한다!
prediction = tf.round(hypothesis)
is_correct = tf.cast(tf.equal(prediction, Y), dtype=tf.float32)
accuracy = tf.reduce_mean(is_correct)


# Start training model

loss_trace = []
train_acc = []
test_acc = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(100000):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, optimizer], feed_dict={X: train_X, Y: train_Y})
    acc, acc_val, _ = sess.run([accuracy, hypothesis, optimizer], feed_dict={X: test_X, Y: test_Y})
    train_acc_val = sess.run([accuracy], feed_dict={X: train_X, Y: train_Y})
    text_acc_val = sess.run([accuracy], feed_dict={X: test_X, Y: test_Y})

    if step % 10000 == 0:
        print("\n Step : ", step, "\n Cost : ", cost_val, "\n Prediction : ", hy_val)
        print(" Acc ", acc, " Acc Val : ", acc_val[:5])
        print(test_Y[:5])

    loss_trace.append(cost_val)
    train_acc.append(train_acc_val)
    test_acc.append(text_acc_val)


# Visualization of the results
# loss function
plt.plot(loss_trace)
plt.title('Cross Entropy Loss')
plt.xlabel('Step')
plt.ylabel('loss')
plt.show()

plt.plot(train_acc, 'b-', label='train accuracy')
plt.plot(test_acc, 'k-', label='test accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Train and Test Accuracy')
plt.legend(loc='best')

plt.show()


save = tf.train.Saver()
save.save(sess, 'model/ryan_ml_model_01')
