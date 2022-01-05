'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 10:04:41
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-01-04 15:17:48
 # @ Description: Source code to process trade history from BigQuery
 '''
import pandas as pd
import os
import numpy as np

# Pandaralled is a python package that is 
# used to multi-thread df apply
from pandarallel import pandarallel

from tqdm import tqdm
tqdm.pandas()

from ficc.utils.process_features import process_features
pandarallel.initialize(progress_bar=False)

import ficc.utils.globals as globals
from ficc.data.process_trade_history import process_trade_history
from ficc.utils.auxiliary_functions import sqltodf, drop_extra_columns, convert_dates, process_ratings
from ficc.utils.yield_curve import get_ficc_ycl




def process_data(query,client,SEQUENCE_LENGTH,NUM_FEATURES,PATH,YIELD_CURVE="FICC", **kwargs):
    # This global variable is used to be able to process data in parallel
    globals.YIELD_CURVE_TO_USE = YIELD_CURVE
    trades_df = process_trade_history(query,client,SEQUENCE_LENGTH, NUM_FEATURES, PATH)

    if YIELD_CURVE.upper() == "FICC":
        # Calculating yield spreads using ficc_ycl
        trades_df['ficc_ycl'] = trades_df.parallel_apply(get_ficc_ycl,axis=1)
        trades_df['yield_spread'] = trades_df['yield'] * 100 - trades_df['ficc_ycl']
    elif YIELD_CURVE.upper() == "S&P":
        # Converting the yield spread to basis points
        trades_df['yield_spread'] = trades_df['yield_spread'] * 100

    # Dropping columns which are not used for training
    trades_df = drop_extra_columns(trades_df)

    # Converting BigQuery Date data type to pandas datatime data type
    trades_df = convert_dates(trades_df)
    
    trades_df = process_ratings(trades_df)

    trades_df = process_features(trades_df)

    if 'training_features' in kwargs:
        trades_df = trades_df[kwargs['training_features']]
        trades_df.dropna(inplace=True)
    print(f"Numbers of samples {len(trades_df)}")
    return trades_df


    
    
    
    



