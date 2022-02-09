from collections import deque
import multiprocessing as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

ONE_MINUTE_TO_SECONDS = 60

def _recent_trade_data_subset(df, N, appended_features_names_and_functions, categories=None, header=None):
    sorted_df = df.sort_values(by='trade_datetime')

    appended_features_names = list(appended_features_names_and_functions.keys())
    num_of_appended_features = len(appended_features_names)

    if categories == None:
        len_augmented_data = len(sorted_df)
    else:
        len_augmented_data = sum([tuple(row[categories]) == header for _, row in sorted_df.iterrows()])

    # initialization of augmented_data matrix
    augmented_data = np.zeros(shape=(len_augmented_data, 1 + N * num_of_appended_features), dtype=np.float32)
    for k, appended_features_name in enumerate(appended_features_names):
        _, init = appended_features_names_and_functions[appended_features_name]
        if init != 0:
            for j in range(N):
                augmented_data[:, 1 + k + j * num_of_appended_features] = init * np.ones(len_augmented_data)

    recent_trades = deque([])
    idx_adjustment = 0    # this index adjustment is for when the sorted_df has moved forward to another row that is not one of the rows for the current header (but is a similar one)
    for idx, (i, row) in enumerate(sorted_df.iterrows()):
        if categories == None or tuple(row[categories]) == header:
            augmented_data[idx - idx_adjustment, 0] = i    # put the row position in the first column of augmented_data
            for j, neighbor in enumerate(recent_trades):
                num_recent_trades_augmented = 0
                # the below condition ensures that recent trades don't come from the same CUSIP, since that 
                # is handled by the LSTM and that there is 15 minute window between two 'nearby' trades, since 
                # it only makes sense to have access to information from a prior trade (the 15 minute threshold 
                # is because that is the amount of time a trade must be reported to MSRB since being completed) 
                if (row['trade_datetime'] - neighbor['trade_datetime']).total_seconds() >= ONE_MINUTE_TO_SECONDS and row['cusip'] != neighbor['cusip']:
                    for k, appended_features_name in enumerate(appended_features_names):
                        appended_features_function, _ = appended_features_names_and_functions[appended_features_name]
                        augmented_data[idx - idx_adjustment, 1 + num_recent_trades_augmented * num_of_appended_features + k] = appended_features_function(row, neighbor)
                    num_recent_trades_augmented += 1
                
                    if num_recent_trades_augmented == N:   # going in here means that we have already filled the N recent trades for the current trade (`row`)
                        break
        else:
            idx_adjustment += 1

        recent_trades.appendleft(row)    # appendleft allows us to iterate from most recent to least recent when iterating through `recent_trades`

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
def append_recent_trade_data(df, N, appended_features_names_and_functions, categories=[], is_similar=None):
    assert 'trade_datetime' in df.columns, 'trade_datetime column is required'    # we use the datetime to determine the time of the trade

    if categories:    # entering here means that there are certain categories that need to be similar between two trades to be considered amongst the recent trades
        augmented_data = []

        if is_similar is not None:    # entering here means that there is a similarity function
            subcategory_headers = []
            subcategory_dict = dict()
            for subcategory_header, subcategory_df in df.groupby(categories):
                if type(subcategory_header) != tuple:    # this if statement converts a single item category value to a tuple to be consistent with the case when there are multiple categories
                    subcategory_header = (subcategory_header,)
                subcategory_headers.append(subcategory_header)
                subcategory_dict[subcategory_header] = subcategory_df

            for subcategory_header in tqdm(subcategory_headers):
                related_subcategories = []

                for other_subcategory_header in subcategory_headers:
                    if is_similar(categories, subcategory_header, other_subcategory_header):    # check if each subcategory header is similar to any of the other subcategory headers (trivally similar to itself)
                        related_subcategories.append(subcategory_dict[other_subcategory_header])
                related_subcategories_df = pd.concat(related_subcategories)

                augmented_data.append(_recent_trade_data_subset(related_subcategories_df, N, appended_features_names_and_functions, categories, subcategory_header))
        else:    # entering here means that there is no similarity function and all `categories` must be the same for two trades to be considered similar
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
