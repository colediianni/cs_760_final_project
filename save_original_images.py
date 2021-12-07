import os
import pandas as pd
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
import tensorflow as tf
import argparse

# Setting Seed
numpy.random.seed(seed=123)

# example of loading the generator model and generating images
from tensorflow.keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot

def make_save_dirs(model_name):
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, 'saved_images', model_name)
    i = 1
    # don't overwrite previous saved models
    while os.path.isdir(save_dir):
        save_dir = os.path.join(cwd, 'saved_images', model_name + f'_{i}')
        i += 1

    os.makedirs(save_dir)
    log_file = os.path.join(save_dir, 'image_values.csv')
    return save_dir, log_file

# load fashion mnist images
def load_train_samples():
    # load dataset (total data is (60000, 28, 28))
    (trainX, _), (_, _) = load_data()
    trainX = trainX[:int(len(trainX)/2)] # Taking second half of data
    # expand to 3d, e.g. add channels
    X = expand_dims(trainX, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X

def load_test_samples():
    # load dataset (total data is (60000, 28, 28))
    (trainX, _), (_, _) = load_data()
    trainX = trainX[int(len(trainX)/2):] # Taking second half of data
    # expand to 3d, e.g. add channels
    X = expand_dims(trainX, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X

# create and save a plot of generated images (reversed grayscale)
def save_images_to_folder(examples, image_folder_path):
	# plot images
	pyplot.figure()
	for i in range(len(examples)):
		if i % 1000 == 0:
			print(i)
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
		save_path = os.path.join(os.getcwd(), image_folder_path, 'generated_image'+str(i))
		pyplot.savefig(save_path, format="png")
		pyplot.clf()

def main():
    image_folder_path, image_values_file_path = make_save_dirs("original_first_half_images")
    X = load_train_samples()
    print(X.shape)
    # save figure values to csv file
    X_to_save = pd.DataFrame(X.reshape((X.shape[0], X.shape[1] * X.shape[2])))
    X_to_save.to_csv(image_values_file_path, header=False, index=False)
    # plot the result
    save_images_to_folder(X, image_folder_path)

    image_folder_path, image_values_file_path = make_save_dirs("original_second_half_images")
    X = load_test_samples()
    print(X.shape)
    # save figure values to csv file
    X_to_save = pd.DataFrame(X.reshape((X.shape[0], X.shape[1] * X.shape[2])))
    X_to_save.to_csv(image_values_file_path, header=False, index=False)
    # plot the result
    save_images_to_folder(X, image_folder_path)




if __name__ == "__main__":
    main()
