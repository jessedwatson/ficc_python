'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 10:04:41
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-03-01 10:50:29
 # @ Description: Source code to process trade history from BigQuery
 '''
import pandas as pd
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
from ficc.utils.yield_curve import get_ficc_ycl
from ficc.utils.get_mmd_ycl import get_mmd_ycl
from ficc.utils.auxiliary_functions import convert_dates


def process_data(query,client,SEQUENCE_LENGTH,NUM_FEATURES,PATH,YIELD_CURVE="FICC", estimate_calc_date = True, remove_short_maturity = False, remove_non_transaction_based = False, remove_trade_type = [], trade_history_delay=1, min_trades_in_history=2, **kwargs):
    # This global variable is used to be able to process data in parallel
    globals.YIELD_CURVE_TO_USE = YIELD_CURVE
    print(f'Running with\n estimate_calc_date:{estimate_calc_date}\n remove_short_maturity:{remove_short_maturity}\n remove_non_transaction_based:{remove_non_transaction_based}\n remove_trad_type:{remove_trade_type}\n trade_history_delay:{trade_history_delay}')

    trades_df = process_trade_history(query,
                                      client, 
                                      SEQUENCE_LENGTH,
                                      NUM_FEATURES,
                                      PATH,
                                      estimate_calc_date,
                                      remove_short_maturity,
                                      remove_non_transaction_based,
                                      remove_trade_type,
                                      trade_history_delay,
                                      min_trades_in_history)

    if YIELD_CURVE.upper() == "FICC":
        # Calculating yield spreads using ficc_ycl
        print("Calculating yield spread using ficc yield curve")
        trades_df['ficc_ycl'] = trades_df.parallel_apply(get_ficc_ycl,axis=1)
        # As ficc ycl is already in basis points
        trades_df['yield_spread'] = trades_df['yield'] * 100 - trades_df['ficc_ycl']
    
    elif YIELD_CURVE.upper() == "MMD":
        print("Calculating yield spreads using MMD yield curve")
        trades_df['mmd_ycl'] = trades_df.parallel_apply(get_mmd_ycl,axis=1)
        trades_df['yield_spread'] = (trades_df['yield'] - trades_df['mmd_ycl']) * 100
        
    elif YIELD_CURVE.upper() == "S&P":
        print("Using yield spreds created from the S&P muni index")
        # Converting the yield spread to basis points
        trades_df['yield_spread'] = trades_df['yield_spread'] * 100

    # Dropping columns which are not used for training
    # trades_df = drop_extra_columns(trades_df)
    trades_df = convert_dates(trades_df)

    print("Processing categorical features")
    trades_df = process_features(trades_df)

    if 'training_features' in kwargs:
        trades_df = trades_df[kwargs['training_features']]
        trades_df.dropna(inplace=True)
    
    print(f"Numbers of samples {len(trades_df)}")
    
    return trades_df


    
    
    
    


