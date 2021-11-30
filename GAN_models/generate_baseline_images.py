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
import tensorflow as tf

# Setting Seed
numpy.random.seed(seed=123)

# example of loading the generator model and generating images
from tensorflow.keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot

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
		save_path = os.path.join(os.getcwd(), 'GAN_models/baseline_images/generated_image'+str(i))
		pyplot.savefig(save_path, format="pdf")
		pyplot.clf()

def main():
	# load model
	model_path = os.path.join(os.getcwd(), 'GAN_models/generator.h5')
	print(model_path)
	model = load_model(model_path)
	(trainX, _), (_, _) = load_data()
	num_images_to_generate = 2 * int(len(trainX)/2)
	# generate images
	latent_points = generate_latent_points(100, num_images_to_generate)
	# generate images
	X = model.predict(latent_points)
	# plot the result
	image_folder_path = os.path.join(os.getcwd(), 'GAN_models/baseline_images')
	save_images_to_folder(X, 10)

if __name__ == "__main__":
    main()
