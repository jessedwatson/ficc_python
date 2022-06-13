
import numpy as np
import pandas as pd


def _unflatten(cell):
    num_dims = int(cell[0])
    shape = cell[1:1+num_dims].astype(int)
    return cell[1+num_dims:].reshape(shape)

def write_parquet(df, path, **kwargs):
    """
    Write a dataframe to a parquet file.
    """

    # for col in df.columns:
    #     print(f"{col}: {df[col].dtype} ({type(df[col].iloc[0])})")
    out_df = df.copy()
    for col in out_df.columns:
        # Flatten the arrays and replace it with a 1D array of (dim, *dimensions, flattened values...)
        if isinstance(df[col].iloc[0], np.ndarray):
            header = np.array([len(df[col].iloc[0].shape)] + list(df[col].iloc[0].shape), dtype = df[col].iloc[0].dtype)
            out_df.loc[:, col] = out_df.loc[:, col].apply(lambda x: np.concatenate((header, x.flatten())))
            out_df = out_df.copy()  # This is to defrag the dataframe
        if df.dtypes[col] == 'float64':
            out_df.loc[:, col] = out_df.loc[:, col].astype('float32')
        if df.dtypes[col] == 'datetime64[ns]':
            out_df.loc[:, col] = out_df.loc[:, col].astype('datetime64[us]')
    out_df.reset_index(inplace=True)

    with open(path, 'wb') as f:
        out_df.to_parquet(f, **kwargs)

def read_parquet(path, **kwargs):
    """
    Read a dataframe from a parquet file.
    """
    with open(path, 'rb') as f:
        df = pd.read_parquet(f, **kwargs)
    for col in df.columns:
        # Flatten the arrays and replace it with a 1D array of (dim, *dimensions, flattened values...)
        if isinstance(df[col].iloc[0], np.ndarray):
            df.loc[:, col] = df.loc[:, col].apply(_unflatten)
        if df.dtypes[col] == 'datetime64[us]':
            df.loc[:, col] = df.loc[:, col].astype('datetime64[ns]')
    return df

def write_feather(df, path, **kwargs):
    """
    Write a dataframe to a feather file.
    """

    out_df = df.copy()
    for col in out_df.columns:
        # Flatten the arrays and replace it with a 1D array of (dim, *dimensions, flattened values...)
        if isinstance(df[col].iloc[0], np.ndarray):
            header = np.array([len(df[col].iloc[0].shape)] + list(df[col].iloc[0].shape), dtype = df[col].iloc[0].dtype)
            out_df.loc[:, col] = out_df.loc[:, col].apply(lambda x: np.concatenate((header, x.flatten())))
            out_df = out_df.copy()  # This is to defrag the dataframe
        if df.dtypes[col] == 'float64':
            out_df.loc[:, col] = out_df.loc[:, col].astype('float32')
    out_df.reset_index(inplace=True)

    with open(path, 'wb') as f:
        out_df.to_feather(f, **kwargs)

def read_feather(path, **kwargs):
    """
    Read a dataframe from a feather file.
    """
    with open(path, 'rb') as f:
        df = pd.read_feather(f, **kwargs)
    for col in df.columns:
        # Flatten the arrays and replace it with a 1D array of (dim, *dimensions, flattened values...)
        if isinstance(df[col].iloc[0], np.ndarray):
            df.loc[:, col] = df.loc[:, col].apply(_unflatten)
    return df
