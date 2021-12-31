import multiprocessing as mp
import pandas as pd
from tqdm import tqdm


def get_temporal_adjacency(df, N=None):
    sorted_df = df.sort_values(by='trade_datetime')

    indices = []
    weights = []
    recent_trades = []
    for i, row in tqdm(sorted_df[::-1].iterrows(), total=len(sorted_df.index)):
        for neighbor in recent_trades:
            indices.append((i, neighbor))
            weights.append(1.0)

        recent_trades.append(i)
        if len(recent_trades) > N:
            recent_trades = recent_trades[1:]

    return indices, weights


def get_restricted_temporal_adjacency(df, categories=None, N=None):
    indices = []
    weights = []

    if categories is not None:
        for _, subcategory_df in df.groupby(categories):
            new_indices, new_weights = get_temporal_adjacency(
                subcategory_df, N)

            indices = indices + new_indices
            weights = weights + new_weights
    else:
        indices, weights = get_temporal_adjacency(df, N)

    return indices, weights
