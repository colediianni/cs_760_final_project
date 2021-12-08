import os
import pandas as pd
import numpy
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from tensorflow.keras import backend as K
import argparse

# Setting Seed
numpy.random.seed(seed=123)

# example of loading the generator model and generating images
from tensorflow.keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot

def make_save_dirs(model_name, target_model_name):
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, 'saved_latent_points', model_name + "_on_" + target_model_name)
    i = 1
    # don't overwrite previous saved models
    while os.path.isdir(save_dir):
        save_dir = os.path.join(cwd, 'saved_latent_points', model_name + "_on_" + target_model_name + f'_{i}')
        i += 1

    os.makedirs(save_dir)
    log_file = os.path.join(save_dir, 'latent_point_values.csv')
    return save_dir, log_file

# generate points in latent space as input for the generator
def load_generated_images(model_name):
    # generate points in the latent space
    csv_path = os.path.join(os.getcwd(), "saved_images", model_name, "image_values.csv")
    generated_images = pd.read_csv(csv_path, header=None)
    generated_images = numpy.array(generated_images)
    generated_images = generated_images.reshape((len(generated_images), 28, 28, 1))
    return generated_images

def main():

    (trainX, _), (_, _) = load_data()
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-model-name", help="name to load/save images/latent points")
    parser.add_argument("--target-model-name", help="name of model to get latent space from")
    args = parser.parse_args()

    images_array = load_generated_images(args.generator_model_name)
    print(images_array.shape)

    latent_point_folder_path, latent_point_values_file_path = make_save_dirs(args.generator_model_name, args.target_model_name)

    # load model
    # 'GAN_models/generator.h5'
    target_model_path = os.path.join(os.getcwd(), "target_models", args.target_model_name)
    print(target_model_path)
    target_model = load_model(target_model_path)

    # remove last layer from model
    layer_name = target_model.layers[-2].name
    intermediate_layer_model = Model(inputs=target_model.input, outputs=target_model.get_layer(layer_name).output)

    # generate images
    intermediate_output = intermediate_layer_model.predict(images_array)
    # print(intermediate_output.shape)

    # save figure values to csv file
    X_to_save = pd.DataFrame(intermediate_output)

    # Saving latent space representation
    X_to_save.to_csv(latent_point_values_file_path, header=False, index=False)


if __name__ == "__main__":
    main()
