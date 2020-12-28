import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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

