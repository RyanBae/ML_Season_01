import tensorflow as tf
import numpy as np
x_train = [1, 2, 3]
y_train = [1, 2, 3]
W = tf.Variable(tf.random.normal([1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias")
hypothesis = x_train*W+b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(1, input_dim=1))

model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(x_train, y_train, epochs=1000)

print(model.predict(np.array([5])))
