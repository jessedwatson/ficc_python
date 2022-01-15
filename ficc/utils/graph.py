import multiprocessing as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def _recent_trade_data_subset(df, orig_df, N):
    sorted_df = df.sort_values(by='trade_datetime')

    augmented_data = np.zeros(
        shape=(len(sorted_df), 1 + N * 3), dtype=np.float32)

    recent_trades = []
    for idx, (i, row) in tqdm(enumerate(sorted_df.iterrows()), total=len(sorted_df.index)):
        augmented_data[idx, 0] = i
        for j, neighbor in enumerate(recent_trades):
            augmented_data[idx, 1 + j * 3 + 0] = neighbor['yield_spread']
            augmented_data[idx, 1 + j * 3 + 1] = (
                row['trade_datetime'] - neighbor['trade_datetime']).total_seconds()
            augmented_data[idx, 1 + j * 3 + 2] = neighbor['par_traded']

        recent_trades.append(row)
        if len(recent_trades) > N:
            recent_trades = recent_trades[1:]

    return augmented_data


def append_recent_trade_data(df, N, categories=None,):
    assert 'trade_datetime' in df.columns, "trade_datetime column is required"

    if categories is not None:
        augmented_data = []
        for _, subcategory_df in df.groupby(categories):
            augmented_data.append(
                _recent_trade_data_subset(subcategory_df, df, N))

        augmented_data = np.concatenate(augmented_data, axis=0)
    else:
        augmented_data = _recent_trade_data_subset(df, N)

    # Remove sorting
    augmented_data = augmented_data[augmented_data[:, 0].argsort()]

    for i in range(N):
        df.loc[:,
               f'yield_spread_recent_{i}'] = augmented_data[:, 1 + i * 3 + 0]
        df.loc[:, f'seconds_ago_recent_{i}'] = augmented_data[:, 1 + i * 3 + 0]
        df.loc[:, f'par_traded_recent_{i}'] = augmented_data[:, 1 + i * 3 + 0]


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
