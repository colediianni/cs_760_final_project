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
    d = lambda x: tf.math.top_k(model.predict(x), k=1)[0]

    in_preds = d(in_data).numpy().flatten()
    out_preds= d(out_data).numpy().flatten()

    df = pd.DataFrame.from_dict({'in': in_preds, 'out': out_preds})

    sns.ecdfplot(data=df)
    plt.show()
    
