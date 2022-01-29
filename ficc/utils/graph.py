from collections import deque
import multiprocessing as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def _recent_trade_data_subset(df, N, appended_features_names_and_functions, categories=None, header=None):
    sorted_df = df.sort_values(by='trade_datetime')

    appended_features_names = list(appended_features_names_and_functions.keys())
    num_of_appended_features = len(appended_features_names)

    if categories == None:
        len_augmented_data = len(sorted_df)
    else:
        len_augmented_data = sum([tuple(row[categories]) == header for _, row in sorted_df.iterrows()])

    augmented_data = np.zeros(shape=(len_augmented_data, 1 + N * num_of_appended_features), dtype=np.float32)

    recent_trades = deque([])
    idx_adjustment = 0
    for idx, (i, row) in enumerate(sorted_df.iterrows()):
        if categories == None or tuple(row[categories]) == header:
            augmented_data[idx - idx_adjustment, 0] = i
            for j, neighbor in enumerate(recent_trades):
                for k, appended_features_name in enumerate(appended_features_names):
                    appended_features_function = appended_features_names_and_functions[appended_features_name]
                    augmented_data[idx - idx_adjustment, 1 + j * num_of_appended_features + k] = appended_features_function(row, neighbor)
        else:
            idx_adjustment += 1

        recent_trades.append(row)
        if len(recent_trades) > N:
            recent_trades.popleft()

    return augmented_data


'''
This function takes in a dataframe (`df`), a number of recent trades (`N`), a list 
of categories (`categories`), and a similarity function (`is_similar`). The function
`is_similar` is a similarity function which takes in two tuples of categories and 
returns True iff the categories are considered similar by the function. The goal is 
to augment each trade with previous trades that are similar to this one, where the 
`is_similar` function determines whether two trades are similar. If `is_similar` is 
equal to None, then this is equivalent to the similarity function enforcing that 
all categories amongst two trades must be equal in order to be considered similar.
'''
def append_recent_trade_data(df, N, appended_features_names_and_functions, categories=None, is_similar=None):
    assert 'trade_datetime' in df.columns, 'trade_datetime column is required'

    if categories is not None:
        augmented_data = []

        if is_similar is not None:
            subcategory_headers = []
            subcategory_dict = dict()
            for subcategory_header, subcategory_df in df.groupby(categories):
                if type(subcategory_header) != tuple:
                    subcategory_header = (subcategory_header,)
                subcategory_headers.append(subcategory_header)
                subcategory_dict[subcategory_header] = subcategory_df

            for subcategory_header in tqdm(subcategory_headers):
                if type(subcategory_header) != tuple:
                    subcategory_header = (subcategory_header,)
                related_subcategories = []

                for other_subcategory_header in subcategory_headers:
                    if type(other_subcategory_header) != tuple:
                        other_subcategory_header = (other_subcategory_header,)
                    if is_similar(categories, subcategory_header, other_subcategory_header):
                        related_subcategories.append(subcategory_dict[other_subcategory_header])
                related_subcategories_df = pd.concat(related_subcategories)

                augmented_data.append(_recent_trade_data_subset(related_subcategories_df, N, appended_features_names_and_functions, categories, subcategory_header))
        else:   
            for subcategory_header, subcategory_df in tqdm(df.groupby(categories)):
                augmented_data.append(_recent_trade_data_subset(subcategory_df, N, appended_features_names_and_functions, categories, subcategory_header))

        augmented_data = np.concatenate(augmented_data, axis=0)
    else:
        augmented_data = _recent_trade_data_subset(df, N, appended_features_names_and_functions)

    # Remove sorting
    augmented_data = augmented_data[augmented_data[:, 0].argsort()]

    appended_features_names = list(appended_features_names_and_functions.keys())
    num_of_appended_features = len(appended_features_names)
    for i in range(N):
        for j, appended_features_name in enumerate(appended_features_names):
            df.loc[:, f'{appended_features_name}_{i}'] = augmented_data[:, 1 + i * num_of_appended_features + j]


def _temporal_adjacency_subset(df, N=None):
    sorted_df = df.sort_values(by='trade_datetime')

    indices = []
    recent_trades = []
    for i, row in tqdm(sorted_df.iterrows(), total=len(sorted_df.index)):
        neighbors = []
        for neighbor in recent_trades:
            neighbors.append(neighbor)
        indices.append(neighbors)

        recent_trades.append(i)
        if len(recent_trades) > N:
            recent_trades = recent_trades[1:]

    return indices


def get_temporal_adjacency(df, categories=None, N=None):
    assert 'trade_datetime' in df.columns, "trade_datetime column is required"

    results = []

    if categories is not None:
        for _, subcategory_df in df.groupby(categories):
            latest = _temporal_adjacency_subset(subcategory_df, N)

            if len(results) == 0:
                results = latest
            else:
                results = results + latest
    else:
        results = _temporal_adjacency_subset(df, N)

    return tf.ragged.constant(results, dtype=tf.int64)
