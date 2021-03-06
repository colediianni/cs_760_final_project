# -*- coding: utf-8 -*-
"""
# Importing
"""

import os
import numpy
import matplotlib.pyplot as plt
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets.fashion_mnist import load_data
# Setting Seed
numpy.random.seed(seed=123)

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

import random
from tensorflow.keras.models import load_model

# from keras.utils.vis_utils import plot_model

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-discriminator", action='store_true',
            help="Do not include discriminator in attack model")
    parser.add_argument("--model-name", help="name to save model under")
    parser.add_argument("--target-model-name", help="file to load attack model from",
            default='GAN_discriminator_architecture_model.h5')
    parser.add_argument("--lr", help="learning rate for adam", type=float, default=0.0002)
    parser.add_argument("--load-generator", action="store_true",
            help="Use pre-trained generator instead of starting from scratch")
    parser.add_argument("--merged-loss", action="store_true",
            help="update based on two outputs simultaneously")
    parser.add_argument("-w", type=float, default=0.5,
            help="weight to give to attack model's loss, relative to discriminator")
    args = parser.parse_args()

    # Setting Seed
    numpy.random.seed(seed=123)
    random.seed(123)

    # size of the latent space
    latent_dim = 100
    # create the discriminator
    discriminator = define_discriminator()
    discriminator._name = "discriminator"
    # create the generator
    if args.load_generator:
        generator_path = os.path.join(os.getcwd(), 'GAN_models', 'generator.h5')
        generator = load_model(generator_path)
    else:
        generator = define_generator(latent_dim)
    generator._name = "generator"
    # create the gan
    gan_branch_model = define_gan(generator, discriminator)

    # load target and attack models
    # 'GAN_discriminator_architecture_model.h5'
    target_model_path = os.path.join(os.getcwd(), 'target_models', args.target_model_name)
    target_model = load_model(target_model_path)
    target_model._name = "target_model"
    attack_model_path = os.path.join(os.getcwd(), 'saved_models', 'attack_model.h5')
    attack_model = load_model(attack_model_path)
    attack_model._name = "attack_model"

    # create the membership construction branch
    if args.merged_loss:
        membership_construction_branch_model = define_merged_constructor(
                generator, discriminator, target_model, attack_model, learning_rate=args.lr,
                w = args.w)
    else:
        membership_construction_branch_model = define_membership_constructor(
                generator, target_model, attack_model, learning_rate=args.lr)

    # plot_model(membership_construction_branch_model, show_shapes=True,
    #         show_layer_names=True, to_file='constructor.png')

    # load image data
    dataset = load_real_samples()
    # train model
    gen_model = train(generator, discriminator, gan_branch_model,
            membership_construction_branch_model, dataset, latent_dim,
            with_discriminator=not args.no_discriminator, model_name=args.model_name,
            merged=args.merged_loss)



def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

"""# Connecting Models (in order) Updating GAN Weights"""
#
# GAN_model = load_model('/content/drive/MyDrive/ML_760/Final_Proj/generator.h5')
# target_model = load_model('/content/drive/MyDrive/ML_760/Final_Proj/GAN_discriminator_architecture_model.h5')
# attack_model = load_model('PATH')
#
# model = Sequential()
# model.add(GAN_model)
# model.add(target_model)
# model.add(attack_model)
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#
# model.summary()
#
# # Freeze layers (other than GAN)
# model.layers[1].trainable = False
# model.layers[2].trainable = False
#
# # Training
# epochs = 10:
# for i in range(epochs):
#   # generate images
#   x = generate_latent_points(100, 100)
#   y = ones((n_samples, 1))
#   model.train_on_batch(X, Y)
# model.save('/content/drive/MyDrive/ML_760/Final_Proj/basic_membership_attack_model.h5')

"""# GD On Noise Vector"""
#
# GAN_model = load_model('/content/drive/MyDrive/ML_760/Final_Proj/generator.h5')
# target_model = load_model('/content/drive/MyDrive/ML_760/Final_Proj/GAN_discriminator_architecture_model.h5')
# attack_model = load_model('PATH')
#
# model = Sequential()
# model.add(GAN_model)
# model.add(target_model)
# model.add(attack_model)
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#
# model.summary()
#
# # Freeze layers (other than GAN)
# model.layers[1].trainable = False
# model.layers[2].trainable = False
#
# # Training
# epochs = 10:
# for i in range(epochs):
#   # generate images
#   x = generate_latent_points(100, 100)
#   y = ones((n_samples, 1))
#   model.train_on_batch(X, Y)
# model.save('/content/drive/MyDrive/ML_760/Final_Proj/basic_membership_attack_model.h5')

"""# Connecting Models (GAN discriminator branch)"""

# define the standalone discriminator model
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
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# generate
	model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model


def define_merged_constructor(generator, discriminator, target_model, attack_model,
        learning_rate=0.0002, beta_1=0.5, w=0.5):
    """
    param w controls the weight given to the membership model's loss function
    """
    assert 0 <= w and w <= 1, "weight param w must be in [0,1]"
    target_model.trainable = False
    attack_model.trainable = False

    inputs = keras.Input(shape=(100,))
    imgs = generator(inputs)
    # branch 1
    membership = target_model(imgs)
    membership = tf.nn.top_k(membership, k=3, sorted=True, name="Top_k_final").values
    membership = attack_model(membership)
    # branch 2
    discrim = discriminator(imgs)

    model = keras.Model(inputs=inputs, outputs=[membership, discrim])
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)
    model.compile(loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'],
            loss_weights={
                'attack_model': w,
                'discriminator': 1-w})
    return model


def define_membership_constructor(generator, target_model, attack_model,
        learning_rate=0.0002, beta_1=0.5):

    # Freeze layers (other than generator)
    target_model.trainable = False
    attack_model.trainable = False


    inputs = keras.Input(shape=(100,)) # noise vector
    x = generator(inputs)
    x = target_model(x)
    #model3 = Lambda(lambda x: tf.nn.top_k(x, k=int(int(x.shape[-1])/2), sorted=True, name="Top_k_final").values)(model2)
    x = tf.nn.top_k(x, k=3, sorted=True, name="Top_k_final").values
    outputs = attack_model(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# load fashion mnist images
def load_real_samples():
	# load dataset
	(trainX, _), (_, _) = load_data()
	trainX = trainX[:int(len(trainX)/2)] # Taking first half of data for GAN training
	# expand to 3d, e.g. add channels
	X = expand_dims(trainX, axis=-1)
	# convert from ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X

# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = zeros((n_samples, 1))
	return X, y



def make_save_dirs(model_name):
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, 'saved_models', model_name)
    i = 1
    # don't overwrite previous saved models
    while os.path.isdir(save_dir):
        save_dir = os.path.join(cwd, 'saved_models', model_name + f'_{i}')
        i += 1

    os.makedirs(save_dir)
    log_file = os.path.join(save_dir, 'saved_results.log')
    return save_dir, log_file


def one_batch_merged(g_model, d_model, gan_model, merged_construction_model,
        dataset, latent_dim, n_batch, with_discriminator=True):
    half_batch = n_batch // 2
    # get randomly selected 'real' samples
    X_real, y_real = generate_real_samples(dataset, half_batch)
    # update discriminator model weights
    d_loss1, _ = d_model.train_on_batch(X_real, y_real)
    # generate 'fake' examples
    X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
    # update discriminator model weights
    d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

    # prepare points in latent space as input for the generator
    X_gan = generate_latent_points(latent_dim, n_batch)
    # create inverted labels for the fake samples
    discrim_out = ones((n_batch, 1))
    member_out  = keras.utils.to_categorical(ones((n_batch, 1)), 2)

    # update the generator via the discriminator's error
    mi_loss = merged_construction_model.train_on_batch(
            X_gan,
            {'attack_model': member_out,
             'discriminator': discrim_out})

    return mi_loss


def one_batch_split(g_model, d_model, gan_model, membership_construction_branch_model,
        dataset, latent_dim, n_batch, with_discriminator=True):
    """
    Update discriminator and generator sequentially on a single batch
    """
    half_batch = n_batch // 2
    if with_discriminator:
        # get randomly selected 'real' samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        d_loss1, _ = d_model.train_on_batch(X_real, y_real)
        # generate 'fake' examples
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model weights
        d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

    # prepare points in latent space as input for the generator
    X_gan = generate_latent_points(latent_dim, n_batch)
    # create inverted labels for the fake samples
    y_gan = ones((n_batch, 1))
    # update the generator via the discriminator's error
    g_loss = gan_model.train_on_batch(X_gan, y_gan)

    X_gan = generate_latent_points(latent_dim, n_batch)
    # create inverted labels for the fake samples
    y_gan = ones((n_batch, 1))
    # update labels to match the attack model's output
    y_gan = keras.utils.to_categorical(y_gan, 2)
    # update the generator via the discriminator's error
    mi_loss = membership_construction_branch_model.train_on_batch(X_gan, y_gan)

    if with_discriminator:
        return [d_loss1, d_loss2, g_loss, mi_loss]
    else:
        return [g_loss, mi_loss]


def save_partial_results(g_model, save_dir, epoch_no, latent_dim=100):
    g_model.save(os.path.join(save_dir, f"epoch{epoch_no}.h5"))
    imgs, _ = generate_fake_samples(g_model, latent_dim, 25)

    for k in range(25):
        plt.subplot(5, 5, k+1)
        plt.axis('off')
        plt.imshow(imgs[k, :, :, 0], cmap='gray_r')
    plt.savefig(os.path.join(save_dir, f"epoch{epoch_no}.png"))




# train the generator and discriminator
def train(g_model, d_model, gan_model, membership_construction_branch_model, dataset, latent_dim, n_epochs=100, n_batch=128, with_discriminator=True, model_name = 'constructor', merged=False):
    save_dir, log_file = make_save_dirs(model_name)

    bat_per_epo = int(dataset.shape[0] / n_batch)
    print("Number of batches:", bat_per_epo)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    with open(log_file, 'w') as fobj:
        fobj.write(f'{n_epochs} epochs, batch size {n_batch}, with discriminator: {with_discriminator}\n')
        for i in range(n_epochs):
            # enumerate batches over the training set
            for j in range(bat_per_epo):
               # summarize loss on this batch
                #print('>%d, d1=%.3f, d2=%.3f g=%.3f, mi=%.3f' % (i+1, d_loss1, d_loss2, g_loss, mi_loss))
                if merged:
                    f = one_batch_merged
                else:
                    f = one_batch_split

                losses = f(g_model, d_model, gan_model,
                        membership_construction_branch_model,
                        dataset, latent_dim, n_batch, with_discriminator=with_discriminator)
                loss_str = f"{i+1},{j+1}: {losses}"
                print(loss_str)
                fobj.write(loss_str + '\n')
            if (i+1)%5 == 0 or i == 0:
                # save partial results
                save_partial_results(g_model, save_dir, i+1, latent_dim=latent_dim)

    # save the generator model
    g_model.save(os.path.join(save_dir, model_name))
    return g_model

if __name__ == "__main__":
    main()
