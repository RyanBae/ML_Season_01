import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import sys
from tensorflow.python.keras import activations
from tensorflow.python.keras.api._v2.keras import optimizers

from tensorflow.python.keras.layers.core import Dropout, Flatten, Dense 
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.training.tracking import base 


cifar10 = datasets.cifar10 
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
 
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
 
print("Train samples:", train_images.shape, train_labels.shape)
print("Test samples:", test_images.shape, test_labels.shape)
 

image_shape = train_images[0].shape;
print(image_shape)

vgg16_model = tf.keras.applications.VGG16(input_shape=image_shape, include_top=False, weights='imagenet', classes=10)

vgg16_model.summary()

# vgg16 모델에서 최상단 input 제외하고 Conv layer 를  base_model 추가
# 레이어만 가져오기!
base_model = tf.keras.Sequential()

# for layer in vgg16_model.layers[:]:
for layer in vgg16_model.layers:
    base_model.add(layer)

# base_model.summary()

# 학습하고자 하는 상단에 input 추가


# VGG 하단의 25개 레이어는 동결
for layer in vgg16_model.layers[:25]:
    if layer.name == 'block5_conv1':
        break

    layer.trainable = False
    print('Layer ' + layer.name + ' frozen.')

for layer in vgg16_model.layers:
    print(layer, layer.trainable)


# base_model.add()
# base_model.add(Dropout(0.25))
base_model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
base_model.add(BatchNormalization())
base_model.add(Dropout(0.25))
# base_model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
# base_model.add(BatchNormalization())
# base_model.add(Dropout(0.25))
# base_model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
# base_model.add(BatchNormalization())
# base_model.add(Dropout(0.25))


# base_model.add(vgg16_model)
base_model.add(Flatten())
base_model.add(Dense(256, activation='relu'))
base_model.add(Dropout(0.5))
# base_model.add(Dense(256, activation='relu'))
# base_model.add(Dropout(0.2))

base_model.add(Dense(128, activation='relu'))
base_model.add(Dropout(0.5))
base_model.add(Dense(64, activation='relu'))
# base_model.add(Dense(32, activation='relu'))
base_model.add(Dense(10, activation='softmax'))

base_model.summary()


# model_top = tf.keras.Sequential()
# model_top.add(Flatten(input_shpae=(32,32,3)))
# model_top.add(Dense(256, activation='relu'))
# model_top.add(Dropout(0.5))
# model_top.add(Dense(10, activation='softmax'))

# base_model.add(model_top)

# feature_batch = base_model(test_images)
# print(feature_batch.shape)

# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# feature_batch_average = global_average_layer(feature_batch)


# prediction_layer = keras.layers.Dense(10)
# prediction_batch = prediction_layer(feature_batch_average)
# print(prediction_batch.shape)
# print("=========")


# base_model = tf.keras.Sequential([
#     vgg16_model,
#     prediction_batch,
#     # Flatten(),
#     Dense(256, activation='relu'),
#     Dense(256, activation='relu'),
#     # Dropout(0.3),
#     # Dense(64, activation='relu'),
#     # Dense(32, activation='relu'),
#     Dense(10, activation='softmax')

# ])

base_model.summary()

opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
base_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


epochs = 20
batch_size = 142

trans_history = base_model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback], validation_split=0.3)


test_loss, test_acc = base_model.evaluate(test_images, test_labels)
 
print('Test accuracy:', test_acc)
 
predictions = base_model.predict(test_images)
 
# base_model.save("Ryan_CNN_10")





# 그래프
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
 
  plt.imshow(img, cmap=plt.cm.binary)
 
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
 
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label[0]]),
                                color=color)
 
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label[0]].set_color('blue')
 


i = 7
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()