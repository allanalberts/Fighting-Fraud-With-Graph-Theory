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
    df['class'] = np.where(df['rating'] < 0, 1, 0)
    df['binomial_rating'] = np.where(df['rating'] < 0, -1, 1)
    df['color'] = np.where(df['rating'] < 0, 'red', 'blue')
    conditions  = [np.absolute(df['rating']) >= 8, np.absolute(df['rating']) >= 4, np.absolute(df['rating']) >= 2]
    choices     = [4, 3, 2]
    df['penwidth'] = np.select(conditions, choices, default=1)
    return df

def build_graph(bitcoin_df, user_lst=[], rating_type='all', maxdate='2016-01-24'):
    """ Returns a dataframe and graph object containing bitcoin data in range
    up to and including date. Edge attributes created for each column in 
    dataframe that is not a node (rater or ratee).
    Input: 
        bitcoin_df:  dataframe containing bitcoin ratings as edges
        user_lst:    list containing rater and ratee users to filter for 
                     inclusion. If list is empty, then all users included.
        rating_type: string. Include only positive ratings if 'pos', only
                     negative ratings if 'neg', otherwise include all rating values
        max_date:    date. Only include records occurring prior to this date
    Output:
        dataframe
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
    df = bitcoin_df[bitcoin_df['date'] <= maxdate].sort_values(['date', 'rating'],ascending=[True,False])
    if len(user_lst) > 0:
        df = df[(df['ratee'].isin(user_lst)) | (df['rater'].isin(user_lst))]
    if rating_type == 'pos':
        df = df[df['rating']>0]
    elif rating_type == 'neg':
        df = df[df['rating']<0]
        
    g = nx.from_pandas_edgelist(df, 
                                source='rater',
                                target='ratee',
                                edge_attr=True,
                                create_using=nx.DiGraph())
    return df, g

def user_stats(bitcoin_df, usertype="ratee"):
    """ Returns Dataframe of user activity stats based on whether the user
    is the rater or ratee. This dataframe is use in function
    user_activity_dataframe() to generate overall user stats
    Input:
        Dataframe of bitcoin marketplace ratings activity
    Output:
        Dataframe 
    """
    df = bitcoin_df.copy()
    if usertype == 'ratee':
        ratingstype = "Received"
        neg_ratingstype = "Fraud"
    elif usertype == 'rater':
        ratingstype = "Given"
        neg_ratingstype = "Victim"
    
    # aggregate ratee stats
    users_agg = df.groupby(usertype)['rating'].agg(['count','mean', 'median', 'min','max'])
    users_agg.rename_axis(index={usertype: 'user'}, inplace=True)
    users_agg['count'].fillna(0, inplace=True)
    users_agg.columns = ['Ratings' + ratingstype, 'AvgRating' + ratingstype, 
                         'MedianRating' + ratingstype, 'MinRating' + ratingstype, 
                         'MaxRating' + ratingstype]

    # aggregate ratee dates
    users_dt = df.groupby('ratee')['date'].agg(['min','max'])
    users_dt.rename_axis(index={'ratee': 'user'}, inplace=True)
    users_dt.columns = ['DateFirstRating' + ratingstype, 'DateLastRating' + ratingstype]
    
    # negative ratings
    users_nr = df[df['rating'] < 0].groupby(usertype)['rating'].agg({'count'}).fillna(0)
    users_nr.rename_axis(index={usertype: 'user'}, inplace=True)
    users_nr['count'] = users_nr['count'].fillna(0)
    users_nr.columns = [neg_ratingstype + 'Cnt']
    
    return pd.concat([users_agg, users_dt, users_nr], axis=1, sort=False)

def sequential_ratings_delay(bitcoin_df):
    """ Returns dataframe with user and their minimum 
    delay (forward or backwards) between sequential
    ratings. This function used for EDA purposes only
    for identifying automated bot ratings. Will use a
    velocity function for production.
    Input: 
        Dataframe of bitcoin marketplace ratings activity
    Output: 
        Dataframe
    """
    df = bitcoin_df.sort_values('date').copy()
    datetime_index = pd.DatetimeIndex(df['date'])
    df.set_index(datetime_index, inplace=True)
    
    df['previousdate'] = df.groupby('rater')['date'].shift(1)
    df['nextdate'] = df.groupby('rater')['date'].shift(-1)
    df['delta_previous'] = (df['date'] - df['previousdate']).astype('timedelta64[s]')
    df['delta_next'] = (df['nextdate'] - df['date']).astype('timedelta64[s]')
    df['delta'] = df[['delta_previous','delta_next']].min(axis=1)

    df = df.groupby('rater')['delta'].agg(['min'])
    df.columns = ['min_ratings_delta']
    
    # PLACEHOLDER FOR COUNT OF AUTOMATED ACTIVITY (NEED A DELAY THRESHOLD VALUE)
    # do I want to build a velocity function (ratings within previous X minutes)
    # 2 or more in one minute, 3 or more in 5 minutes.
    # can color code the edge to signify bot activity
    
    return df

def user_activity_dataframe(bitcoin_df):
    """Returns dataframe of metrics related to activity of each marketplace user
    Input:
        bitcoind_df: dataframe of bitcoin marketplace data
    Output:
        users: dataframe
    """
    ratee = user_stats(bitcoin_df, usertype="ratee")
    rater = user_stats(bitcoin_df, usertype="rater")
    rater_bot = sequential_ratings_delay(bitcoin_df)
    users = pd.concat([ratee, rater, rater_bot], axis=1, sort=False)

    # Time active 
    users['FirstActivity'] = users[['DateFirstRatingReceived', 'DateLastRatingReceived',
                                    'DateFirstRatingGiven', 'DateLastRatingGiven']].min(axis=1)
    users['LastActivity'] = users[['DateFirstRatingReceived', 'DateLastRatingReceived',
                                   'DateFirstRatingGiven', 'DateLastRatingGiven']].max(axis=1)
    users['TimeActive'] = users['LastActivity'] - users['FirstActivity']

    users['Victim'] = users['MinRatingGiven'] < 0
    users['Fraudster'] = users['MinRatingReceived'] < 0
    users['RatingsGivenRatio'] = users['RatingsGiven'] / users['RatingsReceived']
    users['BotActivity'] = users['min_ratings_delta'] == 0

    # set datatypes:
    # first need to deal with any null values in int columns
    # users_agg = users_agg.astype({'count':int, 'median':int, 'min':int, 'max':int})

    return users

if __name__ == '__main__':
    pass