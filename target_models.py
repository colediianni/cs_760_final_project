# -*- coding: utf-8 -*-
"""
# Importing
"""

import os
import numpy
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import SGD

import tensorflow as tf

# Setting Seed
numpy.random.seed(seed=123)

# load fashion mnist images
def load_real_samples():
    # load dataset (total data is (60000, 28, 28))
    (trainX, trainY), (_, _) = load_data()
    trainX = trainX[int(len(trainX)/2):] # Taking second half of data for target model training
    trainY = trainY[int(len(trainY)/2):] # Taking second half of data for target model training
    # expand to 3d, e.g. add channels
    X = expand_dims(trainX, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    # convert labels to categorical
    Y = tf.keras.utils.to_categorical(trainY, num_classes=10)

    return X, Y

"""# Same Structure as Discriminator"""

def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# classifier
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

target_model_discriminator = define_discriminator()
x_train, y_train = load_real_samples()
target_model_discriminator.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, validation_split = 0.1)
model_save_path = os.path.join(os.getcwd(), 'target_models/GAN_discriminator_architecture_model.h5')
target_model_discriminator.save(model_save_path)

"""# Second Neural Network
From: https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification/
"""

def define_CNN_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

second_nn = define_CNN_model()
x_train, y_train = load_real_samples()
second_nn.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, validation_split = 0.1)
model_save_path = os.path.join(os.getcwd(), 'target_models/internet_model.h5')
second_nn.save(model_save_path)

"""# MLP Model"""

def define_MLP_model():
  model = Sequential()
  model.add(Flatten())
  model.add(Dense(28*28, activation='relu'))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(10, activation='softmax'))
  # compile model
  opt = SGD(lr=0.01, momentum=0.9)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

mlp = define_MLP_model()
x_train, y_train = load_real_samples()
mlp.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, validation_split = 0.1)
model_save_path = os.path.join(os.getcwd(), 'target_models/mlp_model.h5')
mlp.save(model_save_path)
