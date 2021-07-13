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

from tensorflow.python.keras.layers.core import Dropout, Flatten, Dense, Dropout 
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.training.tracking import base 

""" 
Transfer Learning 
-  머신러닝(Machine Learning)의 많은 모델은 적용하려는 데이터가 학습할 때의 데이터와 같은 분포를 가진다고 가정으로 했을 때 효율적이다. 
-  새로운 문제를 해결할 때 데이터의 분포가 바뀌면 기존의 통계적 모델을 새로운 데이터로 다시 만들어야 한다.
-  이는 비용이 많이 든다. 
    - layers의 갯수, activation, hyper parameters등 고려할 사항이 너무 많다. 
    - 복잡한 모델일 수록 학습에 시간 많이 소요됨
- 즉, 학습된 모델을 내가 원하는 데이터에 사용하기 위해서는 알맞게 작동 안 할 수 있다는 거. 
- 이미 잘 훈련된 모델이 있고, 특히 해당 모델과 유사한 문제를 해결시 Transfer Learning 을 사용한다. 

Transfer Learning 이란?
-   딥러닝을 feature extractor 로만 사용하고 그렇게 추출한 피처를 가지고 다른 모델을 학습 하는것.
-   기존에 만들어진 모델을 사용하여 새로운 모델을 만들시 학습을 빠르게 하며, 예측을 더 높이는 방법.
-   일반적으로 VGG, ResNet, gooGleNet 등 이미 이러한 사전에 학습이 완료된 모델(Pre-Training Model) 을 가지고 
    원하는 학습에 미세 조정. 즉, 작은 변화를 이용하여 학습시키는 방법이다.
-   이미 학습된 weight 들을 transfer(전송) 하여 자신의 model에 맞게 학습을 시키는 방법
-   신경망의 이러한 재학습 과정을 세부 조정(fine-tuning)이라 부른다.
-   실제로 CNN 을 구축하는 경우 대부분 처음부터 (random initialization) 학습하지는 않는다.
-   ImageNet과 같은 대형 데이터셋을 사용해서 pretrain 된 ConvNet 을 사용한다.

Fine-tuning 방법 *
-   Feature extraction (특징추출)
-   pre-trained model의 모델 구조를 이용
-   다른 레이어를 고정 시키고 일부분 layer 을 조정

Transfer Learning 종류
-  inception (googlenet), ms의 resNet, mobilenet, VGG 등
-   Pre trained VGG Model은 ImageNet 기반으로 학습이 된 Model
-   Inception v3는 ImageNet이라는 데이터를 분류하는데 학습이 되어 있다.

"""


cifar10 = datasets.cifar10 
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
 
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
 
print("Train samples:", train_images.shape, train_labels.shape)
print("Test samples:", test_images.shape, test_labels.shape)
 

image_shape = train_images[0].shape;
print(image_shape)

# vgg16 모델 가져오기.
# include_top(True) : 출력 레이러를 포함할것인지 여부로 개별 문제에 적합하다면 불필요
# weights (imagenet) : 로딩할 가중치. 처음부터 훈련시키는데 관심이 있다면 None을 통해 사전에 훈련된 가중치를 사용하지 않아도 된다.
# input_tensor (None) : 서로 다른 크기의 새로운 데이터에 모델을 맞추기 위한 새로운 입력 레이어
# input_shape (None) : 입력 레이어를 변경할 경우 모델이 가져올 것으로 기대하는 이미지의 크기
# pooling (None) : 출력 레이어의 새로운 세트를 훈련시킬 때 사용하는 풀링 타입
# classes (1000) : 출력 벡터와 같은 해당 모델의 클레스의 수
base_model = tf.keras.applications.VGG16(input_shape=image_shape, include_top=False, weights='imagenet', classes=10)


# 기존 모델에 아키택쳐 확인함.
base_model.summary()

# Trainable = False : 훈련 도중 모든 레이어의 가중치가 훈련 가능에서 불가능으로 바뀐다.
base_model.trainable = False

print(len(base_model.layers))
print(base_model.layers)

# for layer in base_model[:]:
#     layer.trainable = False

# for layer in base_model[10:]:
#     layer.trainable = True



feature_batch = base_model(test_images)
print(feature_batch.shape)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)


prediction_layer = keras.layers.Dense(10)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)
print("=========")

# 특징 추출기와 두 층을 쌓기
model = tf.keras.Sequential([
    base_model,

    # layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    # BatchNormalization(),
    # layers.MaxPooling2D((2, 2)),
    # Dropout(0.25),
    # layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    # BatchNormalization(),
    # layers.MaxPooling2D((2, 2)),
    # Dropout(0.25),

    # Flatten(),
    global_average_layer,
    # prediction_layer,
    Dropout(0.25),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dropout(0.25),
    Dense(64, activation='relu'),
    # Dense(32, activation='relu'),
    Dense(10, activation='softmax')

])

model.summary()
len(model.trainable_variables)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

initial_epochs = 10
validation_steps=20

logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

transfer_history = model.fit(train_images, train_labels, epochs=20, callbacks=[tensorboard_callback],validation_split=0.2)
# transfer_history = model.fit(train_images, 
#                             #  validation_data=(valid_data),
#                              steps_per_epoch=60,
#                              validation_steps=validation_steps,
#                              epochs=20, 
#                             )


test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("initial loss: {:.2f}".format(test_loss))
print("initial accuracy: {:.2f}".format(test_accuracy))

predictions = model.predict(test_images)