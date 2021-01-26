import pandas as pd
import numpy as np
import networkx as nx
import helpers as h


def graph_user_features(bitcoin_df, user, rate_date):
    """ Returns array containing predictive features for 
    an individual bitcoin rating.
    Input: 
        bitcoin_df:  Dataframe containing bitcoin ratings as edges
        user: int
        rate_date: date used for feature generation
    Output:
        array
    """
    g = h.build_graph(bitcoin_df, rating_type='pos', rating_date=rate_date)
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
            arr = np.array([ego_triad_300,
                            ego_triad_210,
                            ego_triad_201,
                            ego_triad_120,
                            ego_triad_all,
                            ego_cluster_coef,
                            ego_degree,
                            ego_betweeness,
                            ego_closeness,
                            ego_num_cliques])
            arr[np.isnan(arr)] = 0
            return arr  
    return np.zeros(10)


def historical_user_features(bitcoin_df, user, rate_date):
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
    df_user = df[(df['ratee']==user) & (df['date'] < rate_date)]

    num_ratings_received = len(df_user)
    # check for no historical user data
    if num_ratings_received ==0:
        return np.zeros(9)

    num_neg_received = df_user[df_user['rating'] < 0]['rating'].count()
    num_pos_received = num_ratings_received - num_neg_received
    neg_ratings_pct = np.where(num_ratings_received > 0, num_neg_received / num_ratings_received, 0)
    rating_received_sum = df_user['rating'].sum()
    rating_received_avg = df_user['rating'].mean()
    days_since_first_rated = (rate_date - df_user['date'].min()).days
    days_since_last_rated = (rate_date - df_user.iloc[-1]['date']).days
    last_rating_neg = np.where(df_user.iloc[-1]['rating'] < 0, 1, 0)

    # check for user having received a previous rating
    # if len(target_user) <= 1:
    #     days_since_last_rated = 0
    #     successive_neg_rating = 0
    # else:
    #     days_since_last_rated = (rate_date - target_user.iloc[-2]['date']).days
    #     successive_neg_rating = np.where(target_user.iloc[-2]['rating'] < 0, 1, 0)
        # code for having multiple successive negative ratings:
        # if (target_user.iloc[-2]['rating'] < 0) & (len(target_user) > 1):
        #     successive_neg_rating = 1
        #     if len(target_user) > 2:
        #         if target_user.iloc[-3]['rating'] < 0:
        #             successive_neg_rating = 2
        #             if len(target_user) > 3:
        #                 if target_user.iloc[-4]['rating'] < 0:
        #                     successive_neg_rating = 3
        # else:
        #     successive_neg_rating = 0
    arr = np.array([num_ratings_received, 
                    num_neg_received,
                    num_pos_received,
                    neg_ratings_pct,
                    rating_received_sum,
                    rating_received_avg,
                    days_since_first_rated,
                    days_since_last_rated,
                    last_rating_neg,
                    ])
    arr[np.isnan(arr)] = 0
    return arr

def velocity_user_features(bitcoin_df, user, rate_date, minutes):
    """ Returns array containt the current and maximum velocity 
    of activity per (minutes) time frame that is associated with 
    the user
    Input:
        bitcoin_df: dataframe of bitcoin activity
        user: int - from current record user
        rate_date: date - fryom current record
        minutes: int - the velocity time frame in minutes
    """
    df = bitcoin_df.copy()
    from_date = str(pd.Timestamp(rate_date) - pd.offsets.Minute(minutes))
    velocity_current = df[((df['rater']==user) | (df['ratee']==user)) & \
        (df['date'] < rate_date) & (df['date'] >= from_date)]['rating'].count()
    velocity_max = df[((df['rater']==user) | (df['ratee']==user)) & \
        (df['date'] < rate_date)]['velocity_current'].max()
    arr = np.array([velocity_current, velocity_max])
    arr[np.isnan(arr)] = 0
    return arr

def feature_creation_iteration(bitcoin_df, feature_type, feature_lst):
    """ Returns datafame containing predictive features for 
    every bitcoin rating.
    Input: 
        bitcoin_df:  dataframe containing bitcoin ratings as edges
        feature_type: string (graph/velocity/historical)
        feature_lst: list of feature names to create    
    """
    df = bitcoin_df.copy()
    for idx_df, row in df.iterrows():
        user = row['ratee']
        rate_date = row['date']
        if feature_type == 'graph':
            arr = graph_user_features(df, user, rate_date)
        elif feature_type == 'velocity':
            arr = velocity_user_features(df, user, rate_date, minutes=5)
        elif feature_type == 'historical':
            arr = historical_user_features(df, user, rate_date)
        else:
            print("Invalid Feature Type. Use: graph/velocity/historical")
            return
        for idx_lst, feature in enumerate(feature_lst):
            df.at[(idx_df, feature)] = arr[idx_lst]
    return df


if __name__ == '__main__':

    otc_df = h.load_bitcoin_edge_data('../data/soc-sign-bitcoinotc.csv.gz')

    historical_features = [
        'num_ratings_received',
        'num_neg_received',
        'num_pos_received',
        'neg_ratings_pct',
        'rating_received_sum',
        'rating_received_avg',
        'days_since_first_rated',
        'days_since_last_rated',
        'last_rating_neg']
    df_historical_features = feature_creation_iteration(otc_df, 'historical', historical_features)
    df_historical_features.to_csv('../data/historical_features.csv', index=False)

    velocity_features = [
        'velocity_current',
        'velocity_max']
    otc_df['velocity_current'] = 0  # instantiate field for velocity_max calculation
    df_velocity_features = feature_creation_iteration(otc_df, 'velocity', velocity_features)
    df_velocity_features.to_csv('../data/velocity_features.csv', index=False)

    graph_features = [
        'ego_triad_300',
        'ego_triad_210',
        'ego_triad_201',
        'ego_triad_120',
        'ego_triad_all',
        'ego_cluster_coef',
        'ego_degree',
        'ego_betweeness',
        'ego_closeness',
        'ego_num_cliques']
       