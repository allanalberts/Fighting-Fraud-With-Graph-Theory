import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import networkx as nx
import nxpd
import helpers as h

def plot_timeline(bitcoin_df, title):
    """ Plots positive and negative user rating counts over time
    Inputs:
        df: dataframe containg fields rating and date
        title: plot title
    """
    timeline = bitcoin_df.copy()
    timeline.set_index(pd.DatetimeIndex(timeline.date), inplace=True)

    fig, ax = plt.subplots(figsize=(15,4))
    ax.plot(timeline[timeline['rating'] >= 0]["rating"].resample("w").count(), c="b",label="Positive Ratings Count")
    ax.plot(timeline[timeline['rating'] < 0]["rating"].resample("w").count(), c="r",label="Negative Ratings Count")
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()

def add_user_to_graph(existing_graph, new_user, bitcoin_df):
    """ Returns a new graph object that now also includes the user's
    given and received ratings.
    Input: 
        existing_graph: graph object
        new_user: int
        bitcoin_g: graph containing new user node
    """
    # return nx.compose(existing_graph, nx.ego_graph(bitcoin_g, new_user, radius=1))
    return nx.compose(existing_graph, user_graph(new_user, bitcoin_df))

def confusion_pct(cm):
    """ Returns a confusion matrix array containing 
    percentage values for the predictions of class values.
    Input:
        cm: np.array representing a confusion matrix
    """
    tn, fp, fn, tp = cm.ravel()
    neg = tn + fp
    pos = fn + tp
    tnpct = tn/neg
    fppct = fp/neg
    fnpct = fn/pos
    tppct = tp/pos
    cm_pct = np.round(np.array([tnpct,fppct,fnpct, tppct]), 2)
    cm_pct.reshape((2,2))
    return cm_pct

def plot_confusion_matrix(ax, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.grid(False)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
