import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print(" wine_Model !!!")
wine = pd.read_csv("./csv/winequality-red.csv", delimiter=';', dtype=float)

wine_copy = wine.copy()
_X = wine_copy.drop(labels=['quality'], axis=1).values
_Y = wine_copy.quality.values

train_index = np.random.choice(len(_X), round(len(_X) * 0.8), replace=False)
test_index = np.array(list(set(range(len(_X))) - set(train_index)))
train_X = _X[train_index]
train_Y = _Y[train_index]
test_X = _X[test_index]
test_Y = _Y[test_index]
train_Y = train_Y[:, np.newaxis]
test_Y = test_Y[:, np.newaxis]


def min_max_normalized(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)


train_X = min_max_normalized(train_X)
test_X = min_max_normalized(test_X)

X = tf.placeholder(tf.float32, shape=[None, 11])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([11, 1]), name='weight')
b = tf.Variable(tf.random_normal([1, 1]), name='bias')


hypothesis = tf.matmul(X, W)+b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cost)

prediction = tf.round(hypothesis)
is_correct = tf.cast(tf.equal(prediction, Y), dtype=tf.float32)
accuracy = tf.reduce_mean(is_correct)


sess = tf.Session()
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
saver.restore(sess, './model/ryan_ml_model_01')

dict_input = np.array([[0.60550459, 0.13186813, 0.77777778, 0.09090909, 0.09534884, 0.25925926,
                        0.37588652, 0.55411255, 0.25438596, 0.15432099, 0.25]])

print("모델의 Accuracy : ", sess.run(accuracy, feed_dict={X: test_X, Y: test_Y}))
print("저장한 값으로 예측.")
print(sess.run(hypothesis, feed_dict={X: dict_input}))


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.import_meta_graph('./model/ryan_ml_model_01.meta')
#     saver.restore(sess, tf.train.latest_checkpoint('./model'))

#     test_accuracy, acc_val = sess.run(
#         [accuracy, hypothesis], feed_dict={X: test_X, Y: test_Y})

#     print("Accuracy :: ", test_accuracy)
#     print(sess.run('weight:0'))
