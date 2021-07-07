import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import sys

from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.normalization import BatchNormalization 


# 트랜스퍼러닝 해보기. 
# vgg16 해보기.

cifar10 = datasets.cifar10 
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
 
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
 
print("Train samples:", train_images.shape, train_labels.shape)
print("Test samples:", test_images.shape, test_labels.shape)
 
print(train_images.shape)
print(train_images[1].shape)


# print("==========")
# print(test_images[0].shape)
# train_images = train_images.reshape((50000, 32, 32, 3))
# test_images = test_images.reshape((10000, 32, 32, 3))

# print(train_images.shape)
# sys.exit()
 
 
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
 
 
train_images = train_images/255.0
test_images = test_images/255.0
 

print("==> ")
print(train_images.shape)
datagen = ImageDataGenerator(
      featurewise_center=False,
      samplewise_center=False,
      featurewise_std_normalization=False,
      samplewise_std_normalization=False,
      zca_whitening=False,
      rotation_range=15,
      width_shift_range=0.1,
      height_shift_range=0.1,
      horizontal_flip=True,
      vertical_flip=False
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # horizontal_flip=True,
    # validation_split=0.3
)

datagen.fit(train_images)
print("==> GEN")
print(train_images.shape)

x_train_subset = train_images[:12]

data_test = datagen.flow(train_images)
# print("Flow ")
# print(data_test.shape)

# fig = plt.figure(figsize=(20,2))
# for i in range(0, len(x_train_subset)):
#     ax = fig.add_subplot(1, 12, i+1)
#     ax.imshow(x_train_subset[i])
# fig.suptitle('Subset of Original Training Images', fontsize=20)

# fig = plt.figure(figsize=(20,2))
# for x_batch in datagen.flow(x_train_subset, batch_size=12):
#     for i in range(0,12):
#         ax = fig.add_subplot(1,12,i+1)
#         ax.imshow(x_batch[i])
#     fig.suptitle('Augmented Images', fontsize=20)
#     plt.show()
#     break;



model = models.Sequential()

# 1
# ###############################################

# 특징을 32개 뽑을건데 그 특징이 3*3의 특징으로 할거다.
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# ###############################################
 
opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

checkpointer = ModelCheckpoint(filepath='aug_model.weights.best.hdf5', verbose=1, save_best_only=True)

# tensorBoard callback 
logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# model.fit(train_images, train_labels, epochs=50, callbacks=[tensorboard_callback], validation_split=0.3)

batch_size = 64
epochs = 50
# split validation
model.fit(train_images, train_labels, epochs=epochs, callbacks=[tensorboard_callback], validation_split=0.2)



# model.fit_generator(datagen.flow(train_images, train_labels, batch_size=batch_size), 
#                       steps_per_epoch=train_images.shape[0], 
#                       epochs=epochs, verbose=1, callbacks=[tensorboard_callback],validation_data=[test_images, test_labels])

# model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size), 
                      # steps_per_epoch=(train_images.shape[0]/batch_size), 
                      # epochs=epochs, verbose=1, callbacks=[tensorboard_callback], validation_data=[test_images, test_labels])

test_loss, test_acc = model.evaluate(test_images, test_labels)
 
print('Test accuracy:', test_acc)
 
predictions = model.predict(test_images)
 
# model.save("Ryan_CNN_10")

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