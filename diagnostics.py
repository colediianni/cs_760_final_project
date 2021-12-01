"""
    File: diagnostics.py
    Author: Ben Jacobsen
    Purpose: Implements (mostly graphical) utilities to aid in debugging
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from shadow_model import get_data


def main():
    ((x_train_in,    y_train_in,     x_test_in,     y_test_in),
    (x_train_out,    y_train_out,    x_test_out,    y_test_out)) = get_data()

    shadow_model = keras.models.load_model('saved_models/shadow_model')
    attack_model = keras.models.load_model('saved_models/attack_model')

    attack_model_decision_boundary(attack_model, show=False)

    top1cdf(shadow_model, x_train_in, x_train_out, show=False)

    plt.show()



def top1cdf(model, in_data, out_data, show=True):
    """
    For measuring the actual difference in behavior between in-dataset and
    out-of-dataset predictions
    """
    d = lambda x: tf.math.top_k(model.predict(x), k=2)[0]

    in_preds = d(in_data).numpy()
    out_preds= d(out_data).numpy()

    print(in_preds)

    in_top = [x[0] for x in in_preds]
    in_sec = [x[1] for x in in_preds]
    out_top= [x[0] for x in out_preds]
    out_sec= [x[1] for x in out_preds]

    df = pd.DataFrame.from_dict({'in_top': in_top,
                                 'in_sec': in_sec,
                                 'out_top': out_top,
                                 'out_sec': out_sec})

    print(df)

    fig, (ax0, ax1) = plt.subplots(2)
    sns.ecdfplot(data=df[['out_top', 'in_top']], ax=ax0)
    ax0.set_title("ECDF of top-1 confidence")
    sns.scatterplot(data=df, x='out_top', y='out_sec', ax=ax1, label="Out", alpha=0.3, s=10)
    sns.scatterplot(data=df, x='in_top', y='in_sec', ax=ax1, label="In", alpha=0.3, s=10)
    ax1.set_title("Scatterplot of top-2 confidences")

    if show:
        plt.show()
    


def attack_model_decision_boundary(model, num_classes=10, show=True):
    n = num_classes
    data_points = list()
    for x in np.arange(1/n, 1, step=0.01):
        for y in np.arange((1-x)/(n-1), min(x, 1-x), step=0.01):
            for z in np.arange((1-x-y)/(n-2), min(y, (1-x-y)), step=0.01):
                data_points.append((x,y,z))

    df = pd.DataFrame(data_points, columns=['1st_pred','2nd_pred','3rd_pred'])
    predictions = model.predict(data_points)
    df['prediction'] = predictions[:,1]

    # collapse to 2d

    df = df.groupby(['1st_pred', '2nd_pred']).mean()

    sns.scatterplot(data=df, x='1st_pred', y='2nd_pred', hue='prediction',
            hue_norm=(0,1))
    plt.title("Confidence that input was in the training dataset")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
