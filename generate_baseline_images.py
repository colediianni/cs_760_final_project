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

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

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

	(trainX, _), (_, _) = load_data()
	parser = argparse.ArgumentParser()
	parser.add_argument("--folder-name", help="name to save model under")
	parser.add_argument("--generator-model-path", help="file to load generator model from")
	parser.add_argument("--num", help="number of samples to generate", type=int, default=len(trainX))
	args = parser.parse_args()

	image_folder_path, image_values_file_path = make_save_dirs(args.folder_name)

	# load model
	# 'GAN_models/generator.h5'
	model_path = os.path.join(os.getcwd(), args.generator_model_path)
	print(model_path)
	model = load_model(model_path)
	# generate images
	num_images_to_generate = args.num
	latent_points = generate_latent_points(100, num_images_to_generate)
	# generate images
	X = model.predict(latent_points)
	# save figure values to csv file
	X_to_save = pd.DataFrame(X.reshape((X.shape[0], X.shape[1] * X.shape[2])))
	X_to_save.to_csv(image_values_file_path, header=False, index=False)
	# plot the result
	save_images_to_folder(X, image_folder_path)


if __name__ == "__main__":
    main()
