
"""
    File: metrics.py
    Author: Ben Jacobsen
    Purpose: Measure success of dataset reconstruction attack
"""

import argparse
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", help="name of the dataset being reconstructed",
            default="original_first_half_images")
    parser.add_argument("--target", help="name of target model", 
            default="mlp_model")
    parser.add_argument("gan_model", nargs='+', help="set of generator models to compare")

    args = parser.parse_args()

    pair_plot(args.original, args.target, args.gan_model[0])



def pair_plot(original, target, gan_model, thresholds=[1,20,25]):
    original_df = get_latent_points(original, target) 
    generated   = get_latent_points(gan_model, target)
    distances   = all_pairs_closest(generated, original_df)

    image_samples(distances, gan_model, original)



def image_samples(distances, dataset1, dataset2, thresholds=[1,10,25]):
    """
    find pairs of images that are separated by the given threshold, for
    illustration purposes
    """
    distances = distances.sort_values('min_distance')
    fig, axs = plt.subplots(len(thresholds), 2)

    for i, t in enumerate(thresholds):
        instance = distances[distances['min_distance'] > t].index[0]
        yi, xi = instance, distances.loc[instance, 'argmin']
        print(xi, yi)

        yimg_name = os.path.join("saved_images", dataset2, 
                f"generated_image{yi}")
        ximg_name = os.path.join("saved_images", dataset1,
                f"generated_image{xi}")

        yimg = img.imread(yimg_name)
        ximg = img.imread(ximg_name)

        axs[i][0].imshow(ximg)
        axs[i][0].set_title(f"Generated, t={t}")
        axs[i][0].set_axis_off()
        axs[i][1].imshow(yimg)
        axs[i][1].set_title(f"Original, t={t}")
        axs[i][1].set_axis_off()

    plt.show()


def proportion_plot(original, target, gan_models):
    models = dict()
    original_df = get_latent_points(original, target)
    for gan_model in gan_models:
        models[gan_model] = get_latent_points(gan_model, target)

    pd.set_option('display.max_rows', 100)

    fig, axs = plt.subplots(1,2)
    x_max1 = plot_proportion_within(models, original_df, ax=axs[0])
    x_max2 = plot_proportion_within(models, original_df, ax=axs[1], reverse=True)
    x_max = max(x_max1, x_max2)
    for ax in axs:
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(0.0, 1.0)
    plt.show()



def plot_proportion_within(models, original_df, ax=None, reverse=False):
    """
    returns smallest threshold such that 99% of observations fall below that
    threshold (for setting axis limits later)
    """
    x_max = 0
    for gan_model, df in models.items():
        if reverse:
            distances = all_pairs_closest(original_df, df)
        else:
            distances = all_pairs_closest(df, original_df)

        print(distances['min_distance'].quantile(q=0.99))
        x_max = max(x_max, distances['min_distance'].quantile(q=0.99))
        print(gan_model)
        print(distances['min_distance'].describe())
        stats_df = proportion_within(distances)
        sns.lineplot(data=stats_df, x=stats_df.index, y='cdf', label=gan_model, ax=ax)
        if reverse:
            title = "Proportion of original images within\nthreshold of generated image"
        else:
            title = "Proportion of generated images within\nthreshold of original image"
        if ax is None:
            plt.xlabel("Threshold")
            plt.ylabel("Proportion")
            plt.legend()
            plt.grid()
            plt.title(title)
        else:
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Proportion")
            ax.set_title(title)
            ax.legend()
            ax.grid()

    return x_max






def proportion_within(distances):
    """
    Compute the empirical cumulative distribution function for the distance
    between the two closest points in two sets
    """
    stats_df = (distances
               .groupby('min_distance')['min_distance']
               .agg('count')
               .pipe(pd.DataFrame)
               .rename(columns = {'min_distance': 'frequency'}))

    stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])

    stats_df['cdf'] = stats_df['pdf'].cumsum()

    return stats_df



def all_pairs_closest(x, y, testing=False):
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

        if testing:
            break

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



def get_latent_points(gan_model, target_model):
    fname = os.path.join("saved_latent_points", 
                        f"{gan_model}_on_{target_model}.h5",
                         "latent_point_values.csv")

    df = pd.read_csv(fname, header = None, prefix="Feature")
    return(df)




if __name__ == "__main__":
    main()
