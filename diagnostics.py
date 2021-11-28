"""
    File: diagnostics.py
    Author: Ben Jacobsen
    Purpose: Implements (mostly graphical) utilities to aid in debugging
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf


def top1cdf(model, in_data, out_data):
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
    sns.ecdfplot(data=df[['in_top', 'out_top']], ax=ax0)
    ax0.set_title("ECDF of top-1 confidence")
    sns.scatterplot(data=df, x='in_top', y='in_sec', ax=ax1, label="In", alpha=0.5)
    sns.scatterplot(data=df, x='out_top', y='out_sec', ax=ax1, label="Out", alpha=0.5)
    ax1.set_title("Scatterplot of top-2 confidences")
    plt.show()
    
