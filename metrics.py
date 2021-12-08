
"""
    File: metrics.py
    Author: Ben Jacobsen
    Purpose: Measure success of dataset reconstruction attack
"""

import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset1")
    parser.add_argument("dataset2")

    args = parser.parse_args()

    col_names = ['feature' + str(i) for i in range(1,101)]

    df1 = pd.read_csv(args.dataset1, names=col_names)
    df2 = pd.read_csv(args.dataset2, names=col_names)

    distances = all_pairs_closest(df1, df2)

    dataset1 = extract_dataset_name(args.dataset1)
    dataset2 = extract_dataset_name(args.dataset2)
    print(dataset1, dataset2)

    closest = distances.sort_values('min_distance').iloc[:3]

    fig, axs = plt.subplots(3, 2)
    for i, (yi, xi) in enumerate(zip(closest.index, closest['argmin'])):
        print(yi, xi)
        yimg_name = os.path.join("saved_images", dataset2, 
                f"generated_image{yi}")
        ximg_name = os.path.join("saved_images", dataset2,
                f"generated_image{xi}")

        yimg = img.imread(yimg_name)
        ximg = img.imread(ximg_name)

        axs[i][0].imshow(ximg)
        axs[i][1].imshow(yimg)

    plt.show()



def all_pairs_closest(x, y):
    """
    For each row in y, find the row in x which is closest and report the
    euclidean distance
    """
    mins = dict()
    norm_x = np.linalg.norm(x, axis=1)
    # to avoid running out of memory, cut y into slices
    for i in range(0, len(y), 1000):
        y_slice = y.iloc[i:i+1000]
        norm_y = np.linalg.norm(y_slice, axis=1)
        xy = np.matmul(x.to_numpy(), y_slice.to_numpy().T)
        distances = np.add.outer(norm_x**2, norm_y**2) - 2 * xy
        min_dist, argmin_dist = np.min(distances, axis=0), np.argmin(distances, axis=0)
        for img in range(len(y_slice)):
            mins[img + i] = [min_dist[img], argmin_dist[img]]


    distances = pd.DataFrame.from_dict(mins, columns=['min_distance', 'argmin'], orient='index')
    return distances


def extract_dataset_name(path):
    """
    Given an input like:
        saved_latent_points/baseline_gan_on_mlp_model.h5/latent_point_values.csv

    return an output like:
        baseline_gan

    to make it easier to look up images
    """
    fullpath = os.path.abspath(path)
    # extract directory where latent points are stored
    dirname = os.path.basename(os.path.dirname(fullpath))
    # extract actual model name
    name = dirname.split("_on_", 1)[0]
    return name




if __name__ == "__main__":
    main()
