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

def make_save_dirs(model_name):
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, 'saved_latent_points', model_name)
    i = 1
    # don't overwrite previous saved models
    while os.path.isdir(save_dir):
        save_dir = os.path.join(cwd, 'saved_latent_points', model_name + f'_{i}')
        i += 1

    os.makedirs(save_dir)
    log_file = os.path.join(save_dir, 'latent_point_values.csv')
    return save_dir, log_file

# generate points in latent space as input for the generator
def load_latent_space(model_name):
    # generate points in the latent space
    csv_path = os.path.join(os.getcwd(), "saved_latent_points", model_name, "latent_point_values.csv")
    latent_space = pd.read_csv(csv_path, header=None)
    latent_space = numpy.array(latent_space)
    return latent_space

def get_closest_image(image_array, reference):
    error = reference - image_array
    sum_square_error = numpy.sum(numpy.square(error), axis=1)
    closest_image_index = numpy.argmin(sum_square_error)
    distance = sum_square_error[closest_image_index]
    return distance, closest_image_index

def prop_close_to_setA_vs_setB(gen_images, setA, setB, model_name):

    for image_index in range(len(gen_images)):
        if image_index % 1000 == 0:
            print(image_index)
        image = gen_images[image_index]
        distanceA, _ = get_closest_image(image, setA)
        distanceB, _ = get_closest_image(image, setB)
        if distanceA >= distanceB:
            count_close_to_setA += 1
        else:
            count_close_to_setB += 1
    return (count_close_to_setA/len(gen_images)), (count_close_to_setB/len(gen_images))


def main():
    (trainX, _), (_, _) = load_data()
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-model-name", help="name to load latent points from")
    args = parser.parse_args()

    target_model_training_set = load_latent_space("original_first_half_images")
    attacker_training_set = load_latent_space("original_second_half_images")

    latent_space = load_latent_space(args.generator_model_name)

    # Calc the proportion of generated images closer to target_model_training_set than to attacker_training_set
    prop_close_first_half, prop_close_first_half = prop_close_to_setA_vs_setB(latent_space,
                                                                                target_model_training_set,
                                                                                attacker_training_set,
                                                                                args.generator_model_name)
    print(prop_close_first_half, prop_close_first_half)



if __name__ == "__main__":
    main()
