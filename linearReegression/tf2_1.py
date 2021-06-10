import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# 데이터 불러오기
data = pd.read_csv("./csv/winequality-white.csv", delimiter=';', dtype=float)
# print(data.shape)

# 정답(와인의 품질) 확인
g1 = data.groupby('quality').size()
print(g1)
plt.plot(g1)

# plt.show()


# 와인의 품질 수정
# data = np.loadtxt('data\\wine.csv', skiprows=1, delimiter=';')
# # print(data[data[:, -1] <= 4])
# data[data[:, -1] <= 4, -1] = 0
# data[(data[:, -1] > 4) & (data[:, -1] <= 7), -1] = 1    # & 연결시 조건별 ()
# data[data[:, -1] > 7, -1] = 2
# print(data[:, [-1]])

# # 수정한 데이터로 저장
# np.savetxt('data\\wine2.csv', data)

# # 수정한 데이터 불러오기
# data = np.loadtxt('data\\wine2.csv')
print(data)


# 데이터 나누기
xtrain = data[:3429, :11]
ytrain = data[:3429, [-1]]
print(xtrain.shape)
print(ytrain.shape)
xtest = data[3429:, :11]
ytest = data[3429:, [-1]]
print(xtest.shape)
print(ytest.shape)


# 변수 선언

x = tf.placeholder(tf.float32, [None, 11])
y = tf.placeholder(tf.int32, [None, 1])  # int32 <- 분류를 원핫배열로 바꾸기 위해
onehot = tf.reshape(tf.one_hot(y, 3), [-1, 3])
w = tf.Variable(tf.random_normal([11, 3]))
b = tf.Variable(tf.random_normal([3]))

logit = tf.matmul(x, w) + b
h = tf.nn.softmax(logit)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logit, labels=onehot))

train = tf.train.GradientDescentOptimizer(0.0005).minimize(cost)

dict = {x: xtrain, y: ytrain}
argmax_h = tf.argmax(h, 1)
argmax_y = tf.argmax(onehot, 1)
corr = tf.equal(argmax_h, argmax_y)
acc = tf.reduce_mean(tf.cast(corr, tf.float32))

# 학습

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 변수 초기화
    for i in range(3001):
        sess.run(train, feed_dict=dict)
        if i % 1000 == 0:
            print(sess.run(cost, feed_dict=dict))

    print('w=', sess.run(w))
    print('b=', sess.run(b))

    # 예측
    print('h=', sess.run(h, feed_dict={x: xtest}))

    # 검증
    # print(sess.run(h, {x:xtest}))   # feed_dict만 사용시 생략가능
    # print(sess.run(argmax_h, {x:xtest}))
    print('정확도 =', sess.run(acc, {x: xtest, y: ytest}))
