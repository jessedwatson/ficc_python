'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 10:04:41
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2023-01-27 14:48:43
 # @ Description: Source code to process trade history from BigQuery
 '''
 
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np

# Pandaralled is a python package that is 
# used to multi-thread df apply
from pandarallel import pandarallel
from datetime import datetime, timedelta

from tqdm import tqdm
tqdm.pandas()

from ficc.utils.process_features import process_features
pandarallel.initialize(progress_bar=False)

import ficc.utils.globals as globals
from ficc.data.process_trade_history import process_trade_history
from ficc.utils.yield_curve import get_ficc_ycl
from ficc.utils.get_mmd_ycl import get_mmd_ycl
from ficc.utils.auxiliary_functions import convert_dates
from ficc.utils.auxiliary_variables import RELATED_TRADE_FEATURE_PREFIX, NUM_RELATED_TRADES, CATEGORICAL_REFERENCE_FEATURES_PER_RELATED_TRADE
from ficc.utils.get_treasury_rate import current_treasury_rate, get_all_treasury_rate, get_previous_treasury_difference
from ficc.utils.adding_flags import add_bookkeeping_flag, add_replica_count_flag, add_same_day_flag, add_ntbc_precursor_flag
from ficc.utils.related_trade import add_related_trades

def process_data(query, 
                 client, 
                 SEQUENCE_LENGTH, 
                 NUM_FEATURES, 
                 PATH, 
                 YIELD_CURVE="FICC_NEW", 
                 remove_short_maturity=False, 
                 trade_history_delay=1, 
                 min_trades_in_history=1, 
                 process_ratings=True, 
                 keep_nan=False, 
                 add_flags=False,
                 treasury_spread=False,
                 add_previous_treasury_rate=False,
                 add_previous_treasury_difference=False,
                 use_last_duration=False,
                 add_related_trades_bool=False,
                 production_set=False,
                 **kwargs):
    
    # This global variable is used to be able to process data in parallel
    globals.YIELD_CURVE_TO_USE = YIELD_CURVE
    print(f'Running with\n remove_short_maturity:{remove_short_maturity}\n trade_history_delay:{trade_history_delay}\n min_trades_in_hist:{min_trades_in_history}\n process_ratings:{process_ratings}\n add_flags:{add_flags}')
    
    trades_df = process_trade_history(query,
                                      client, 
                                      SEQUENCE_LENGTH,
                                      NUM_FEATURES,
                                      PATH,
                                      remove_short_maturity, 
                                      trade_history_delay,  
                                      min_trades_in_history,
                                      process_ratings,
                                      treasury_spread,
                                      production_set)

    if production_set == True:
        trades_df['trade_date'] = datetime.now().date() - BDay(1)
        trades_df['settlement_date'] = trades_df['trade_date'] + BDay(2)
    
    if YIELD_CURVE.upper() == "FICC" or YIELD_CURVE.upper() == "FICC_NEW":
        # Calculating yield spreads using ficc_ycl
        print("Calculating yield spread using ficc yield curve")
        temp = trades_df[['trade_date','calc_date']].parallel_apply(get_ficc_ycl,axis=1)
        trades_df[['ficc_ycl','ficc_ycl_3_month','ficc_ycl_1_month']] = pd.DataFrame(temp.tolist(), index=trades_df.index)
        
        # As ficc ycl is already in basis points
        if production_set == False:
            trades_df['yield_spread'] = trades_df['yield'] * 100 - trades_df['ficc_ycl']
            trades_df.dropna(subset=['yield_spread'],inplace=True)
    
    elif YIELD_CURVE.upper() == "MMD":
        print("Calculating yield spreads using MMD yield curve")
        trades_df['mmd_ycl'] = trades_df.parallel_apply(get_mmd_ycl,axis=1)  
        trades_df['yield_spread'] = (trades_df['yield'] - trades_df['mmd_ycl']) * 100
        
    elif YIELD_CURVE.upper() == "S&P":
        print("Using yield spreds created from the S&P muni index")
        # Converting the yield spread to basis points
        trades_df['yield_spread'] = trades_df['yield_spread'] * 100
    
    elif YIELD_CURVE.upper() == 'MSRB_YTW':
        trades_df['yield'] = trades_df['yield'] * 100 # converting it to basis points

    print('Yield spread calculated')

    if treasury_spread == True:
        if use_last_duration == True:
            trades_df['treasury_rate'] = trades_df[['trade_date','last_calc_date','last_settlement_date']].parallel_apply(current_treasury_rate, 
                                                                                                                         args = ([use_last_duration]), 
                                                                                                                         axis=1)
            trades_df['ficc_treasury_spread'] = trades_df['ficc_ycl'] - (trades_df['treasury_rate'] * 100)
        else:
            trades_df['treasury_rate'] = trades_df[['trade_date','calc_date','settlement_date']].parallel_apply(current_treasury_rate, 
                                                                                                                args=([use_last_duration]),
                                                                                                                axis=1)
            trades_df['ficc_treasury_spread'] = trades_df['ficc_ycl'] - (trades_df['treasury_rate'] * 100)
    
    if add_previous_treasury_rate == True: 
        # Adding the treasury rates
        temp = trades_df[['trade_date']].parallel_apply(get_all_treasury_rate, axis=1)
        trades_df[['t_rate_1',
                   't_rate_2', 
                   't_rate_3', 
                   't_rate_5', 
                   't_rate_7', 
                   't_rate_10', 
                   't_rate_20', 
                   't_rate_30']] = pd.DataFrame(temp.to_list(), index=trades_df.index)
        del temp
        print('Fetiching treasury rates')

    if add_previous_treasury_difference == True:
        temp = trades_df[['trade_date']].parallel_apply(get_previous_treasury_difference, axis=1)
        trades_df[['t_rate_diff_1', 
                   't_rate_diff_2', 
                   't_rate_diff_3', 
                   't_rate_diff_5', 
                   't_rate_diff_7', 
                   't_rate_diff_10', 
                   't_rate_diff_20', 
                   't_rate_diff_30']] = pd.DataFrame(temp.to_list(), index=trades_df.index)
        del temp
        print("Difference in treasury rates calculated")
        
    # Dropping columns which are not used for training
    # trades_df = drop_extra_columns(trades_df)
    trades_df = convert_dates(trades_df)

    print("Processing features")
    trades_df = process_features(trades_df, keep_nan, production_set)

    if remove_short_maturity == True:
        trades_df = trades_df[trades_df.days_to_maturity >= np.log10(1 + 400)]

    if 'training_features' in kwargs:
        trades_df = trades_df[kwargs['training_features']]
        trades_df.dropna(inplace=True)

    if add_flags:    # add additional flags to the data
        # trades_df = add_replica_flag(trades_df)    # the IS_REPLICA flag was originally designed to remove replica trades from the trade history
        trades_df = add_replica_count_flag(trades_df)
        trades_df = add_bookkeeping_flag(trades_df)
        trades_df = add_same_day_flag(trades_df)
        trades_df = add_ntbc_precursor_flag(trades_df)
    
    if add_related_trades_bool == True:
        print('Adding most recent related trade')
        trades_df = add_related_trades(trades_df,
                                       RELATED_TRADE_FEATURE_PREFIX, 
                                       NUM_RELATED_TRADES, 
                                       CATEGORICAL_REFERENCE_FEATURES_PER_RELATED_TRADE)
    

    print(f"Numbers of samples {len(trades_df)}")
    
    return trades_df
