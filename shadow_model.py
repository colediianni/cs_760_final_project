"""
    File: shadow_model.py
    Author: Ben Jacobsen
    Purpose: Implement a CNN and train it on one half of CIFAR-10, for the
        purpose of training a membership inference classifier later.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold
import os


def main():
    """
    Overall structure:
        Divide CIFAR-10 into 4 pieces:
            {x,y}_train_in:     Training data for shadow model and attack model
            {x,y}_test_in:      Training data for shadow model, test data for attack model
            {x,y}_train_out:    Not used for shadow model, training data for attack model
            {x,y}_test_out:     Not used for shadow model, test data for attack model
    """
    num_classes = 10
    save_dir = os.path.join(os.getcwd(), 'saved_models')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    (x_train, y_train), (x_test, y_test) = cifar10()

    ((x_train_in,    y_train_in,     x_test_in,     y_test_in),
    (x_train_out,    y_train_out,    x_test_out,    y_test_out)) = shadow_split(
            x_train, y_train, x_test, y_test)

    y_train_in = keras.utils.to_categorical(y_train_in, num_classes)
    y_train_out = keras.utils.to_categorical(y_train_out, num_classes)
    y_test_in = keras.utils.to_categorical(y_test_in, num_classes)
    y_test_out = keras.utils.to_categorical(y_test_out, num_classes)

    # TRAIN SHADOW MODEL
    model_name = "shadow_model"
    model_path = os.path.join(save_dir, model_name)

    X = keras.layers.Concatenate(axis=0)([x_train_in, x_test_in])
    y = keras.layers.Concatenate(axis=0)([y_train_in, y_test_in])

    X_test = keras.layers.Concatenate(axis=0)([x_train_out, x_test_out])
    y_test = keras.layers.Concatenate(axis=0)([y_train_out, y_test_out])
    try:
        cnn = keras.models.load_model(model_path)
        print(f"Loaded saved model from {model_path}")
    except OSError:
        cnn = get_cnn()
        cnn.summary()

        train_cnn(cnn, X, y, epochs=100)

        cnn.save(model_path)
        print(f"Saved model as {model_path}")

    #score = cnn.evaluate(X_test, y_test, verbose=0)
    #print("Test loss:", score[0])
    #print("Test accuracy:", score[1])

    # GENERATE DATA FOR ATTACK MODEL
    d = lambda x: tf.math.top_k(cnn.predict(x), k=3)[0]
    ones  = keras.initializers.Ones()
    zeros = keras.initializers.Zeros()

    in_train  = d(x_train_in)
    in_train_y = ones(shape=in_train.shape[0])
    in_test   = d(x_test_in)
    in_test_y = ones(shape=in_test.shape[0])
    out_train = d(x_train_out)
    out_train_y = zeros(shape=out_train.shape[0])
    out_test  = d(x_test_out)
    out_test_y = zeros(shape=out_test.shape[0])


    attack_model = get_attack_model()
    X = keras.layers.Concatenate(axis=0)([in_train, out_train])
    y = keras.layers.Concatenate(axis=0)([in_train_y, out_train_y])
    y = keras.utils.to_categorical(y, 2)

    X_test = keras.layers.Concatenate(axis=0)([in_test, out_test])
    y_test = keras.layers.Concatenate(axis=0)([in_test_y, out_test_y])
    y_test = keras.utils.to_categorical(y_test, 2)

    attack_model.summary()

    train_attack_model(attack_model, X,y, val=0, epochs=1000)

    #score = attack_model.evaluate(X_test, y_test, verbose=0)
    #print("Test loss:", score[0])
    #print("Test accuracy:", score[1])




def cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)


def shadow_split(x_train, y_train, x_test, y_test, seed=31415):
    """
    split the given dataset into two halves, preserving class proportions.
    one half should be used to train the shadow model, while the other is
    preserved as the 'out of test' sample for training the membership
    inference model.
    """
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)

    train_shadow, train_out = next(skf.split(x_train, y_train))
    test_shadow, test_out = next(skf.split(x_test, y_test))

    x_train_shadow = x_train[train_shadow]
    x_train_out    = x_train[train_out]
    x_test_shadow  = x_test[test_shadow]
    x_test_out     = x_test[test_out]

    y_train_shadow = y_train[train_shadow]
    y_train_out    = y_train[train_out]
    y_test_shadow  = y_test[test_shadow]
    y_test_out     = y_test[test_out]

    return ((x_train_shadow, y_train_shadow, x_test_shadow, y_test_shadow),
            (x_train_out,    y_train_out,    x_test_out,    y_test_out))




def get_attack_model(input_shape=3, num_classes=2):
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Dense(128, activation="relu"),
        #    kernel_regularizer = keras.regularizers.l2(0.1)),
        #layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ]
    )

    return model

def train_attack_model(model, x_train, y_train, batch_size=128, epochs=100,
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"],
        val=0.1):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(x_train, y_train,
            batch_size=batch_size, epochs=epochs, validation_split=val, shuffle=True)

def get_cnn(input_shape=(32,32,3), num_classes=10, k=3, p=2, d=0.25):
    """
    Creates and returns a simple convolution neural network. Parameters:
        input_shape: dimensions of the input vector
        num_classes: number of classes
        k: dimension of the kernels
        p: dimension of the max-pool operation
    """
    model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(k,k), activation="relu", padding="same"),
        layers.Conv2D(32, kernel_size=(k,k), activation="relu"),
        layers.MaxPooling2D(pool_size=(p,p)),
        #layers.Dropout(d),


        layers.Conv2D(64, kernel_size=(k,k), activation="relu", padding="same"),
        layers.Conv2D(64, kernel_size=(k,k), activation="relu"),
        layers.MaxPooling2D(pool_size=(p,p)),
        #layers.Dropout(d),


        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        #layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
    )

    return model


def train_cnn(model, x_train, y_train,
              batch_size=32, epochs=100, loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"], val=0.1):
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(x_train, y_train, 
              batch_size=batch_size, epochs=epochs, validation_split=val)








if __name__ == "__main__":
    main()
