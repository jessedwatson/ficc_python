import multiprocessing as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


def _recent_trade_data_subset(df, N):
    sorted_df = df.sort_values(by='trade_datetime')

    recent_trades = []
    for i, row in tqdm(sorted_df.iterrows(), total=len(sorted_df.index)):
        for j, neighbor in enumerate(recent_trades):
            row[f'yield_spread_recent_{j}'] = neighbor['yield_spread']
            row[f'seconds_ago_recent_{j}'] = (row['trade_datetime'] - neighbor['trade_datetime']).total_seconds()
            row[f'par_traded_recent_{j}'] = neighbor['par_traded']

        recent_trades.append(row)
        if len(recent_trades) > N:
            recent_trades = recent_trades[1:]

def append_recent_trade_data(df, N, categories=None,):
    assert 'trade_datetime' in df.columns, "trade_datetime column is required"

    for i in range(N):
        df[f'yield_spread_recent_{i}'] = 0.0
        df[f'seconds_ago_recent_{i}'] = 0
        df[f'par_traded_recent_{i}'] = 0

    if categories is not None:
        for _, subcategory_df in df.groupby(categories):
            _recent_trade_data_subset(subcategory_df, N)
    else:
        _recent_trade_data_subset(df, N)


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
