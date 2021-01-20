import pandas as pd
import numpy as np
import networkx as nx

import helpers as h

def edge_features(marketplace, bitcoin_df, user, max_date):
    """ Returns array containing predictive features for 
    an individual bitcoin rating.
    Input: 
        bitcoin_df:  Dataframe containing bitcoin ratings as edges
        user: int
        rate_date: date used for feature generation
    Output:
        array
    """
    target_user = h.user_data(marketplace, 
                            bitcoin_df, 
                            user, 
                            user_type='target',
                            rating_type='all',
                            max_date=max_date)
    if len(target_user)==0:
        return np.zeros(20)

    days_since_first_rated = (max_date - target_user['date'].min()).days
    user_all_activity  = h.user_data(marketplace, 
                            bitcoin_df, 
                            user, 
                            user_type='all',
                            rating_type='all',
                            max_date=max_date)
    days_active = (max_date - user_all_activity['date'].min()).daysad

    # base features
    num_ratings_received = len(target_user)
    num_neg_received = target_user['class'].sum()
    num_pos_received = num_ratings_received - num_neg_received
    if num_ratings_received > 0:
        neg_ratings_pct = num_neg_received / num_ratings_received
    else:
        neg_ratings_pct = 0
    rating_received_sum = target_user['rating'].sum()
    rating_received_avg = target_user['rating'].mean()

    # Sequential velocity feature
    if len(target_user) <= 1:
        days_since_last_rated = 0
        successive_neg_rating = 0
    else:
        days_since_last_rated = (max_date - target_user.iloc[-2]['date']).days
        if (target_user.iloc[-2]['rating'] < 0) & (len(target_user) > 1):
            successive_neg_rating = 1
            if len(target_user) > 2:
                if target_user.iloc[-3]['rating'] < 0:
                    successive_neg_rating = 2
                    if len(target_user) > 3:
                        if target_user.iloc[-4]['rating'] < 0:
                            successive_neg_rating = 3
        else:
            successive_neg_rating = 0

    # Graph Features
    ego_triad_300 = 0
    ego_triad_210 = 0
    ego_triad_201 = 0
    ego_triad_120 = 0
    ego_triad_all = 0
    ego_cluster_coef = 0
    ego_degree = 0
    ego_betweeness = 0
    ego_closeness = 0
    ego_num_cliques = 0
    g = h.build_graph(marketplace, bitcoin_df, rating_type='pos', max_date=max_date)
    if user in g: 
        ego_g = nx.ego_graph(nx.reverse_view(g), user, radius=1)
        if len(ego_g) > 2:
            node_census = nx.triadic_census(ego_g)
            ego_triad_300 = node_census['300']
            ego_triad_210 = node_census['210']
            ego_triad_201 = node_census['201']
            ego_triad_120 = node_census['120U'] + node_census['120D'] + node_census['120C']
            ego_triad_all = ego_triad_300 + ego_triad_210 + ego_triad_201 + ego_triad_120
            ego_cluster_coef = nx.clustering(ego_g, user)
            ego_degree = nx.degree(ego_g)[user]
            ego_betweeness = nx.betweenness_centrality(ego_g)[user]
            ego_closeness = nx.closeness_centrality(ego_g)[user]
            ego_g = ego_g.to_undirected()
            ego_num_cliques = nx.number_of_cliques(ego_g, user)

    arr = np.array([num_ratings_received, 
                    num_neg_received,
                    num_pos_received,
                    neg_ratings_pct,
                    rating_received_sum,
                    rating_received_avg,
                    days_since_first_rated,
                    days_since_last_rated,
                    days_active,
                    successive_neg_rating,
                    ego_triad_300,
                    ego_triad_210,
                    ego_triad_201,
                    ego_triad_120,
                    ego_triad_all,
                    ego_cluster_coef,
                    ego_degree,
                    ego_betweeness,
                    ego_closeness,
                    ego_num_cliques,
                    ])
    arr[np.isnan(arr)] = 0
    return arr

def iteration_feature_creation(marketplace, bitcoin_df):
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
        arr = edge_features(marketplace, df, user, rate_date)
        df.at[(i,'num_ratings_received')] = arr[0]
        df.at[(i,'num_neg_received')] = arr[1]
        df.at[(i,'num_pos_received')] = arr[2]
        df.at[(i,'neg_ratings_pct')] = arr[3]
        df.at[(i,'rating_received_sum')] = arr[4]
        df.at[(i,'rating_received_avg')] = arr[5]
        df.at[(i,'days_since_first_rated')] = arr[6]
        df.at[(i,'days_since_last_rated')] = arr[7]
        df.at[(i,'days_active')] = arr[8]
        df.at[(i,'successive_neg_rating')] = arr[9]
        df.at[(i,'ego_triad_300')] = arr[10]
        df.at[(i,'ego_triad_210')] = arr[11]
        df.at[(i,'ego_triad_201')] = arr[12]
        df.at[(i,'ego_triad_120')] = arr[13]
        df.at[(i,'ego_triad_all')] = arr[14]
        df.at[(i,'ego_cluster_coef')] = arr[15]
        df.at[(i,'ego_degree')] = arr[16]
        df.at[(i,'ego_betweeness')] = arr[17]
        df.at[(i,'ego_closeness')] = arr[18]
        df.at[(i,'ego_num_cliques')] = arr[19]
    return df

# def date_velocity(bitcoin_df, user, rate_date, vel_parm, user_type):
#     df = bitcoin_df.copy()
#     from_date = str(pd.Timestamp(rate_date) - pd.offsets.Hour(vel_parm))
#     vel_neg, vel_all = \
#     df[(df[user_type]==user) & (df['date'] < rate_date) & (df['date'] >= from_date)]['class'].agg(['sum', 'count'])
#     vel_pos = vel_all - vel_neg
#     A = np.array([vel_neg, vel_pos, vel_all])
#     A[np.isnan(A)] = 0
#     return A

def current_velocity(bitcoin_df, user, rate_date, vel_parm):
    """ Returns the current velocity associated with the user
    Input:
        bitcoin_df: dataframe of bitcoin activity
        user: int - from current record user
        rate_date: date - from current record
        vel_parm: int - the velocity timeframe in minutes
    """
    df = bitcoin_df.copy()
    from_date = str(pd.Timestamp(rate_date) - pd.offsets.Minute(vel_parm))
    velocity_current = \
    df[((df['rater']==user) | (df['ratee']==user)) & (df['date'] < rate_date) & (df['date'] >= from_date)]['rating'].count()
    np.nan_to_num(velocity_current, copy=False)
    return velocity_current

def max_velocity(bitcoin_df, user, rate_date):
    """ Returns the maximum velocity associated with the user
    Input:
        bitcoin_df: dataframe of bitcoin activity
        user: int - from current record user
        rate_date: date - from current record
    """
    df = bitcoin_df.copy()
    velocity_max = df[((df['rater']==user) | (df['ratee']==user)) & (df['date'] < rate_date)]['velocity_current'].max()
    return velocity_max

def feature_iteration_velocity(bitcoin_df):
    """ Returns dataframe with updated velocity transaction count metrics (current and max)
    Input:
        bitcoin_df: dataframe of bitcoin transactions
    """
    df = bitcoin_df.copy()
    df = df[['ratee', 'rater', 'rating','date']]
    for i, row in df.iterrows():
        user = row['ratee']
        rate_date = row['date']
        velocity_current = current_velocity(df, user, rate_date, vel_parm=10)
        df.at[(i,'velocity_current')] = velocity_current
        velocity_max = max_velocity(df, user, rate_date)
        df.at[(i,'velocity_max')] = velocity_max
    return df

# def feature_iteration_date_velocity(bitcoin_df):
#     df = bitcoin_df.copy()
#     df = df[['ratee', 'rater', 'rating','date','class']]
#     for i, row in df.iterrows():
#         user = row['ratee']
#         rate_date = row['date']
#         vel_24_in_neg, vel_24_in_pos, vel_24_in_all = date_velocity(df, user, rate_date, vel_parm=24, user_type="ratee")
#         vel_24_out_neg, vel_24_out_pos, vel_24_out_all = date_velocity(df, user, rate_date, vel_parm=24, user_type="rater")
#         vel_48_in_neg, vel_48_in_pos, vel_48_in_all = date_velocity(df, user, rate_date, vel_parm=48, user_type="ratee")
#         vel_48_out_neg, vel_48_out_pos, vel_48_out_all = date_velocity(df, user, rate_date, vel_parm=48, user_type="rater")
#         df.at[(i,'vel_24_in_neg')] = vel_24_in_neg
#         df.at[(i,'vel_48_in_neg')] = vel_48_in_neg
#     df.drop(['class'], axis=1)
#     return df
if __name__ == '__main__':

    otc_df = h.load_bitcoin_edge_data('../data/soc-sign-bitcoinotc.csv.gz')
    # df_features_otc = iteration_feature_creation(marketplace='otc', bitcoin_df=otc_df)
    # df_features_otc.to_csv('../data/df_features_otc.csv', index=False)

    df_otc_vel = feature_iteration_velocity(otc_df)
    df_otc_vel.to_csv('../data/df_otc_velocity.csv', index=False)

    # df_features_alpha = iteration_feature_creation(marketplace='alpha', bitcoin_df=alpha_df)
    # df_features_alpha.to_csv('../data/df_features_alpha2.csv', index=False)
