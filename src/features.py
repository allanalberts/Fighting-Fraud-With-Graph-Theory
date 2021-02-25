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
        # reverse the graph so that ego graph picks up those who rated the node,
        # then reverse it again, so that metrics are based on orginal directed structure
        ego_g = nx.ego_graph(nx.reverse_view(g), user, radius=1)
        ego_g = nx.reverse_view(ego_g)
        if len(ego_g) > 2:
            node_census = nx.triadic_census(ego_g)
            # fully connected traids:
            triad_300 = node_census['300']
            triad_210 = node_census['210']
            triad_120 = node_census['120U'] + node_census['120D'] + node_census['120C']
            triad_030T = node_census['030T']
            triad_030C = node_census['030C']
            # reciprocal but not fully connected:
            triad_201 = node_census['201']
            triad_111 = node_census['111U'] +node_census['111D']
            triad_102 = node_census['102']
            # no reciprocal ratings:
            triad_021 = node_census['021U'] + node_census['021D'] + node_census['021C']
            triad_all = sum(node_census.values())
            cluster_coef = nx.clustering(ego_g, user)
            neighbors_in = len(list(nx.reverse(ego_g).neighbors(user)))
            betweeness = nx.betweenness_centrality(ego_g)[user]
            excess_ratings_in = nx.in_degree_centrality(ego_g)[user] - nx.out_degree_centrality(ego_g)[user]
            arr = np.array([triad_300,
                            triad_210,
                            triad_120,
                            triad_030T,
                            triad_030C,
                            triad_201,
                            triad_111,
                            triad_102,
                            triad_021,
                            triad_all,
                            cluster_coef,
                            neighbors_in,
                            betweeness,
                            excess_ratings_in])
            arr[np.isnan(arr)] = 0
            return arr  
    return np.zeros(14)

def historical_source_user_features(bitcoin_df, user, rate_date):
    """ Returns array containing predictive features for 
    an individual bitcoin rating based on source of rating.
    Input: 
        bitcoin_df:  Dataframe containing bitcoin ratings as edges
        user: int
        rate_date: date used for feature generation
    Output:
        array
    """
    df = bitcoin_df.copy()
    df_user = df[(df['rater']==user) & (df['date'] < rate_date)]

    num_ratings_given = len(df_user)
    # check for no historical user data
    if num_ratings_given ==0:
        return np.zeros(4)
    rating_given_avg = df_user['rating'].mean()
    days_since_first_rating_source = (rate_date - df_user['date'].min()).days
    days_since_last_rating_source = (rate_date - df_user.iloc[-1]['date']).days

    arr = np.array([num_ratings_given, 
                    rating_given_avg,
                    days_since_first_rating_source,
                    days_since_last_rating_source])
    arr[np.isnan(arr)] = 0
    return arr


def historical_target_user_features(bitcoin_df, user, rate_date):
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
    days_since_first_rating_target = (rate_date - df_user['date'].min()).days
    days_since_last_rating_target = (rate_date - df_user.iloc[-1]['date']).days
    last_rating_neg = np.where(df_user.iloc[-1]['rating'] < 0, 1, 0)

    arr = np.array([num_ratings_received, 
                    num_neg_received,
                    num_pos_received,
                    neg_ratings_pct,
                    rating_received_sum,
                    rating_received_avg,
                    days_since_first_rating_target,
                    days_since_last_rating_target,
                    last_rating_neg,
                    ])
    arr[np.isnan(arr)] = 0
    return arr

def feature_creation_iteration(bitcoin_df, feature_type, feature_lst):
    """ Returns datafame containing predictive features for 
    every bitcoin rating.
    Input: 
        bitcoin_df:  dataframe containing bitcoin ratings as edges
        feature_type: string (graph/velocity/historical_target/historical_source)
        feature_lst: list of feature names to create    
    """
    df = bitcoin_df.copy()
    for idx_df, row in df.iterrows():
        target = row['ratee']
        source = row['rater']
        rate_date = row['date']
        if feature_type == 'graph_target':
            arr = graph_user_features(df, target, rate_date)
        elif feature_type == 'graph_source':
            arr = graph_user_features(df, source, rate_date)
        elif feature_type == 'historical_target':
            arr = historical_target_user_features(df, target, rate_date)
        elif feature_type == 'historical_source':
            arr = historical_source_user_features(df, source, rate_date)
        else:
            print("Invalid Feature Type. Use: graph/velocity/historical")
            return
        for idx_lst, feature in enumerate(feature_lst):
            df.at[(idx_df, feature)] = arr[idx_lst]
    return df

def normalize_source_graph_metrics(df_g):
    df = df_g.copy()
    df['neighbors_in_source'] = np.where(df['neighbors_in_source']==0, 1,df['neighbors_in_source'])
    df['210_norm_source'] = df['triad_210_source']/df['neighbors_in_source']
    df['120_norm_source'] = df['triad_120_source']/df['neighbors_in_source']
    df['300_norm_source'] = df['triad_300_source']/df['neighbors_in_source']
    df['030T_norm_source'] = df['triad_030T_source']/df['neighbors_in_source']
    df['201_norm_source'] = df['triad_201_source']/df['neighbors_in_source']
    df['111_norm_source'] = df['triad_111_source']/df['neighbors_in_source']
    df['102_norm_source'] = df['triad_102_source']/df['neighbors_in_source']
    df['021_norm_source'] = df['triad_021_source']/df['neighbors_in_source']
    df['all_norm_source'] = df['triad_all_source']/df['neighbors_in_source']
    drop_colls = ['triad_210_source',
                  'triad_120_source',
                  'triad_300_source',
                  'triad_030T_source',
                  'triad_201_source',
                  'triad_111_source',
                  'triad_102_source',
                  'triad_021_source',
                  'triad_all_source']
    df.drop(drop_colls, inplace=True, axis=1)    
    return df
    

def normalize_target_graph_metrics(df_g):
    """ Called by notebook program to normalize the graph data
    """
    df = df_g.copy()
    df['neighbors_in_target'] = np.where(df['neighbors_in_target']==0, 1,df['neighbors_in_target'])
    df['210_norm_target'] = df['triad_210_target']/df['neighbors_in_target']
    df['120_norm_target'] = df['triad_120_target']/df['neighbors_in_target']
    df['300_norm_target'] = df['triad_300_target']/df['neighbors_in_target']
    df['030T_norm_target'] = df['triad_030T_target']/df['neighbors_in_target']
    df['201_norm_target'] = df['triad_201_target']/df['neighbors_in_target']
    df['111_norm_target'] = df['triad_111_target']/df['neighbors_in_target']
    df['102_norm_target'] = df['triad_102_target']/df['neighbors_in_target']
    df['021_norm_target'] = df['triad_021_target']/df['neighbors_in_target']
    df['all_norm_target'] = df['triad_all_target']/df['neighbors_in_target']
    drop_colls = ['triad_210_target',
                  'triad_120_target',
                  'triad_300_target',
                  'triad_030T_target',
                  'triad_201_target',
                  'triad_111_target',
                  'triad_102_target',
                  'triad_021_target',
                  'triad_all_target']
    df.drop(drop_colls, inplace=True, axis=1)
    return df

def graph_metrics_source_target_difference(df_gg):
    df = df_gg.copy()
    df['neighbors_in_diff'] = df['neighbors_in_target'] - df['neighbors_in_source']
    df['210_diff'] = df['210_norm_target'] - df['210_norm_source']
    df['120_diff'] = df['120_norm_target'] - df['120_norm_source']
    df['300_diff'] = df['300_norm_target'] - df['300_norm_source']
    df['030T_diff'] = df['030T_norm_target'] - df['030T_norm_source']
    df['201_diff'] = df['201_norm_target'] - df['201_norm_source']
    df['111_diff'] = df['111_norm_target'] - df['111_norm_source']
    df['102_diff'] = df['102_norm_target'] - df['102_norm_source']
    df['021_diff'] = df['021_norm_target'] - df['021_norm_source']
    df['triads_diff'] = df['all_norm_target'] - df['all_norm_source']
    df['betweeness_diff'] = df['betweeness_target'] - df['betweeness_source']
    df['excess_ratings_in_diff'] = df['excess_ratings_in_target'] - df['excess_ratings_in_source']
    df['cluster_coef_diff'] = df['cluster_coef_target'] - df['cluster_coef_source']
    drop_colls = ['neighbors_in_source',
                  '210_norm_source',
                  '120_norm_source',
                  '300_norm_source',
                  '030T_norm_source',
                  '201_norm_source',
                  '111_norm_source',
                  '102_norm_source',
                  '021_norm_source',
                  'all_norm_source',
                  'betweeness_source',
                  'excess_ratings_in_source',
                  'cluster_coef_source']
    df.drop(drop_colls, inplace=True, axis=1)
    return df

if __name__ == '__main__':

    otc_df = h.load_bitcoin_edge_data('../data/soc-sign-bitcoinotc.csv.gz')

    historical_target_features = [
        'num_ratings_received',
        'num_neg_received',
        'num_pos_received',
        'neg_ratings_pct',
        'rating_received_sum',
        'rating_received_avg',
        'days_since_first_rating_target',
        'days_since_last_rating_target',
        'last_rating_neg']
    # df_historical_target_features = feature_creation_iteration(otc_df, 'historical_target', historical_target_features)
    # df_historical_target_features.to_csv('../data/historical_target_features.csv', index=False)

    historical_source_features = [
        'num_ratings_given',
        'rating_given_avg',
        'days_since_first_rating_source',
        'days_since_last_rating_source']
    # df_historical_source_features = feature_creation_iteration(otc_df, 'historical_source', historical_source_features)
    # df_historical_source_features.to_csv('../data/historical_source_features.csv', index=False)

    # Process graph data and write output to csv file
    graph_target_features = ['triad_300_target',
                            'triad_210_target',
                            'triad_120_target',
                            'triad_030T_target',
                            'triad_030C_target',
                            'triad_201_target',
                            'triad_111_target',
                            'triad_102_target',
                            'triad_021_target',
                            'triad_all_target',
                            'cluster_coef_target',
                            'neighbors_in_target',
                            'betweeness_target',
                            'excess_ratings_in_target']
    df_graph_target_features = feature_creation_iteration(otc_df, 'graph_target', graph_target_features)
    df_graph_target_features.to_csv('../data/graph_target_features.csv', index=False)

    graph_source_features = ['triad_300_source',
                            'triad_210_source',
                            'triad_120_source',
                            'triad_030T_source',
                            'triad_030C_source',
                            'triad_201_source',
                            'triad_111_source',
                            'triad_102_source',
                            'triad_021_source',
                            'triad_all_source',
                            'cluster_coef_source',
                            'neighbors_source',
                            'betweeness_source',
                            'excess_ratings_in_source']
    df_graph_source_features = feature_creation_iteration(otc_df, 'graph_source', graph_source_features)
    df_graph_source_features.to_csv('../data/graph_source_features.csv', index=False)
       