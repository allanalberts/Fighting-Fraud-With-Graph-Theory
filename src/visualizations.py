import pandas as pd
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