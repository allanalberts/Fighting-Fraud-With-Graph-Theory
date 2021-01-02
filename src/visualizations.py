import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import networkx as nx
import nxpd

def plot_timeline(df, title):
    """ Plots positive and negative user rating counts over time
    Inputs:
        df: dataframe containg fields rating and date
        title: plot title
    """
    timeline = df.copy()
    timeline.set_index(pd.DatetimeIndex(timeline.date), inplace=True)

    fig, ax = plt.subplots(figsize=(15,4))
    ax.plot(timeline[timeline['rating'] >= 0]["rating"].resample("w").count(), c="b",label="Positive Ratings Count")
    ax.plot(timeline[timeline['rating'] < 0]["rating"].resample("w").count(), c="r",label="Negative Ratings Count")
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()

def user_graph(user, bitcoin_G):
    """ Returns a graph of user's given and received ratings
    Input:
        user: int
        bitcoin_G: graph containing user node
    """
    return nx.ego_graph(bitcoin_G, user, radius=1)

def add_user_to_graph(existing_graph, new_user, bitcoin_g):
    """ Returns a new graph object that now also includes the user's
    given and received ratings.
    Input: 
        existing_graph: graph object
        new_user: int
        bitcoin_g: graph containing new user node
    """
    return nx.compose(existing_graph, nx.ego_graph(bitcoin_g, new_user, radius=1))