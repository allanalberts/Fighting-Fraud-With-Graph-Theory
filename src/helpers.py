import pandas as pd
import numpy as np
import datetime

def load_bitcoin_data(filename):
    """
    Returns dataframe containing bitcoin data with default index and datetime field format
    Input: 
        filename: csv file with gzip compression
    Output:
        df: dataframe
    """
    df = pd.read_csv(filename,  compression='gzip', names=['rater','ratee','rating','date'])
    df['date'] = df['date'].apply(lambda d: datetime.datetime.fromtimestamp(d).strftime('%Y-%m-%d %H:%M:%S'))
    df['date'] = pd.to_datetime(df['date'])
    return df

if __name__ == '__main__':
    pass
