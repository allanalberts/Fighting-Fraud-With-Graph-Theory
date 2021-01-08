import pandas as pd
import numpy as np
import networkx as nx

import helpers as h

def feature_creation(bitcoin_df, user, rate_date):
    """ Returns array containing predictive features for 
    an individual bitcoin rating.
    Input: 
        bitcoin_df:  Dataframe containing bitcoin ratings as edges
        user: int
        rate_date: date used for feature generation
    Output:
        array
    """
    df = bitcoin_df.copy()
    user_data_in = df[(df['ratee']==user) & ((df['date'] < rate_date) | ((df['date']==rate_date) & (df['rating'] > 0)))]
    if len(user_data_in)==0:
        return np.zeros(8)
    num_ratings_received = len(user_data_in)
    num_neg_received = user_data_in['class'].sum()
    num_pos_received = num_ratings_received - num_neg_received
    neg_ratings_pct = num_neg_received / num_ratings_received
    rating_sum = user_data_in['rating'].sum()
    days_active = (rate_date - user_data_in['date'].min()).days
    _, g = h.build_graph(df, maxdate=rate_date)
    cluster_coef = nx.clustering(g, user)
    g = g.to_undirected()
    num_cliques = nx.number_of_cliques(g, user)

    A = np.array([num_ratings_received, num_neg_received, num_pos_received, 
                  neg_ratings_pct, rating_sum, days_active, cluster_coef, num_cliques])
    A[np.isnan(A)] = 0
    return A

def feature_iteration(bitcoin_df):
    """ Returns datafame containing predictive features for 
    every bitcoin rating.
    Input: 
        bitcoin_df:  dataframe containing bitcoin ratings as edges
    Output:
        Dataframe
    """
    df = bitcoin_df.copy()
    for i, row in df.iterrows():
        user = row['ratee']
        rate_date = row['date']
        num_ratings_received, num_neg_received, num_pos_received, \
        neg_ratings_pct, rating_sum, days_active, cluster_coef, \
        num_cliques = feature_creation(df, user, rate_date)
        df.at[(i,'num_ratings_received')] = num_ratings_received
        df.at[(i,'num_neg_received')] = num_neg_received
        df.at[(i,'num_pos_received')] = num_pos_received
        df.at[(i,'neg_ratings_pct')] = neg_ratings_pct
        df.at[(i,'rating_sum')] = rating_sum
        df.at[(i,'days_active')] = days_active
        df.at[(i,'cluster_coef')] = cluster_coef
        df.at[(i,'num_cliques')] = num_cliques
    return df

    # for bitcoin alpha - needs to be modified for bitcoin_otc
def feature_creation_ato(bitcoin_df, user, rate_date):
    df = bitcoin_df.copy()
    user_data = df[(df['ratee']==user) & ((df['date'] < rate_date) | ((df['date']==rate_date) & (df['rating'] > 0)))]
    if len(user_data) <= 1:
        return 0, 0
    else:
        days_since_last_active = (rate_date - user_data.iloc[-2]['date']).days
        if user_data.iloc[-2]['rating'] < 0:
            successive_neg_rating = 1
            if len(user_data) > 2:
                if user_data.iloc[-3]['rating'] < 0:
                    successive_neg_rating = 2
                    if len(user_data) > 3:
                        if user_data.iloc[-4]['rating'] < 0:
                            successive_neg_rating = 3
        else:
            successive_neg_rating = 0
    return days_since_last_active, successive_neg_rating

def feature_iteration_ato(bitcoin_df):
    df = bitcoin_df.copy()
    df = df[['rater', 'ratee','rating','date']]
    for i, row in df.iterrows():
        user = row['ratee']
        rate_date = row['date']
        days_since_last_active, successive_neg_rating = feature_creation_ato(df, user, rate_date)
        df.at[(i,'days_since_last_active')] = days_since_last_active
        df.at[(i,'successive_neg_rating')] = successive_neg_rating
    return df   
