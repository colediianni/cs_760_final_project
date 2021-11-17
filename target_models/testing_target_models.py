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
from tensorflow.keras.models import load_model

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


x_train, y_train = load_real_samples()

"""Same Structure as Discriminator"""
GAN_discriminator_architecture_model = load_model('./GAN_discriminator_architecture_model.h5')
GAN_discriminator_architecture_model.evaluate(x_train, y_train)

"""Second Neural Network"""
internet_model = load_model('./internet_model.h5')
internet_model.evaluate(x_train, y_train)

"""MLP Model"""
mlp_model = load_model('./mlp_model.h5')
mlp_model.evaluate(x_train, y_train)
