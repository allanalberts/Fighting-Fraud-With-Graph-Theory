import pandas as pd
import numpy as np
import datetime
import networkx as nx

def load_bitcoin_edge_data(filename):
    """ Returns dataframe containing bitcoin data with default index and datetime field format 
    and fraud classification field based on negative rating.
    Input: 
        filename: csv file with gzip compression
    Output:
        df: dataframe
    """
    df = pd.read_csv(filename,  
                    compression='gzip', 
                    names=['rater','ratee','rating','date'],
                    dtype={'rater': int, 'ratee': int, 'rating': int})
    df['date'] = df['date'].apply(lambda d: datetime.datetime.fromtimestamp(d).strftime('%Y-%m-%d %H:%M:%S'))
    df['date'] = pd.to_datetime(df['date'])
    df['color'] = np.where(df['rating'] < 0, 'red', 'blue')
    conditions  = [np.absolute(df['rating']) >= 8, np.absolute(df['rating']) >= 4, np.absolute(df['rating']) >= 2]
    choices     = [4, 3, 2]
    df['penwidth'] = np.select(conditions, choices, default=1)
    df = df.sort_values('date')
    return df

def user_data(bitcoin_df, user, user_type='target', rating_type='pos',rating_date=''):
    """ Returns a dataframe of a single users transactions prior to max date.
        Data may be filtered on user type (rater/ratee/all) and rating type (pos/neg/all)
    Input:
        bitcoin_df: dataframe of bitcoin ratings activity
        user: int
        user_type: str
        rating_type: str
        rating_date: date
    """
    df = bitcoin_df.copy()
    if rating_date != '':
        df = df[df['date'] < rating_date].sort_values('date')

    # restrict ratings to pos or neg is specified as parameter
    if rating_type == 'pos':
        df = df[df['rating']>0]
    elif rating_type == 'neg':
        df = df[df['rating']<0]

    if user_type == 'target':
        df = df[df['ratee'] == user]
    elif user_type == 'source':
        df = df[df['rater'] == user]   
    elif user_type =='all':   
        df = df[(df['ratee'] == user) | (df['rater'] == user)]
    else:
        print("Invalid user type")
    
    return df


def build_graph(bitcoin_df, user_lst=[], rating_type='all', rating_date=''):
    """ Returns a graph object containing bitcoin data in range up to and including 
    date (pos ratings only). Edge attributes created for each column in 
    dataframe that is not a node (rater or ratee).
    Input: 
        bitcoin_df:  dataframe containing bitcoin ratings as edges
        user_lst:    list containing rater and ratee users to filter for 
                     inclusion. If list is empty, then all users included.
        rating_type: string. Include only positive ratings if 'pos', only
                     negative ratings if 'neg', otherwise include all rating values
        rating_date: date. Only include records occurring prior to this date
    Output:
        graph:      attributes defined in bitcoin_df
                    color of edges: 
                       red=negative rating, 
                       blue=positive rating
                    weight of edges: 
                       1=rating of +/- [1], 
                       2=rating of +/- [2,3],
                       3=rating of +/- [4,5,6,7],
                       4=rating of +/- [8,9,10],
    """
    df = bitcoin_df.copy()
    if rating_date != '':
        df = df[(df['date'] < rating_date)].sort_values('date')

    # restrict ratings to pos or neg is specified as parameter
    if rating_type == 'pos':
        df = df[df['rating']>0]
    elif rating_type == 'neg':
        df = df[df['rating']<0]

    # if arguments include user list, restrict graph to just these users
    if len(user_lst) > 0:
        df = df[(df['ratee'].isin(user_lst)) | (df['rater'].isin(user_lst))]
        
    g = nx.from_pandas_edgelist(df, source='rater',
                                    target='ratee',
                                    edge_attr=True,
                                    create_using=nx.DiGraph())
    return g

if __name__ == '__main__':
    pass
