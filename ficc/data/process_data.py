'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2021-12-16
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-01-08
 # @ Description: Source code to process trade history from BigQuery
 '''
import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
from pytz import timezone
pacific = timezone('US/Pacific')

# Pandaralled is a python package that is 
# used to multi-thread df apply
from pandarallel import pandarallel
from datetime import datetime, timedelta

import os

from ficc.utils.process_features import process_features
print(f'Initializing pandarallel with {os.cpu_count()/2} cores')
pandarallel.initialize(progress_bar=False, nb_workers=int(os.cpu_count()/2))

import ficc.utils.globals as globals
from ficc.data.process_trade_history import process_trade_history
from ficc.utils.yield_curve import get_ficc_ycl
from ficc.utils.auxiliary_functions import convert_dates
from ficc.utils.auxiliary_variables import RELATED_TRADE_FEATURE_PREFIX, NUM_RELATED_TRADES, CATEGORICAL_REFERENCE_FEATURES_PER_RELATED_TRADE
from ficc.utils.get_treasury_rate import current_treasury_rate
from ficc.utils.adding_flags import add_bookkeeping_flag, add_replica_count_flag, add_same_day_flag, add_ntbc_precursor_flag
from ficc.utils.related_trade import add_related_trades


def process_data(query, 
                 client, 
                 SEQUENCE_LENGTH, 
                 NUM_FEATURES, 
                 PATH, 
                 YIELD_CURVE='FICC_NEW', 
                 remove_short_maturity=False, 
                 trade_history_delay=0.2, 
                 min_trades_in_history=0, 
                 treasury_spread=False, 
                 add_flags=False, 
                 add_related_trades_bool=False, 
                 add_rtrs_in_history=False, 
                 only_dollar_price_history=False, 
                 **kwargs):
    
    # This global variable is used to be able to process data in parallel
    globals.YIELD_CURVE_TO_USE = YIELD_CURVE
    print(f'Running with\n remove_short_maturity:{remove_short_maturity}\n trade_history_delay:{trade_history_delay}\n min_trades_in_hist:{min_trades_in_history}\n add_flags:{add_flags}')
    
    trades_df = process_trade_history(query,
                                      client, 
                                      SEQUENCE_LENGTH,
                                      NUM_FEATURES,
                                      PATH,
                                      remove_short_maturity, 
                                      trade_history_delay,  
                                      min_trades_in_history,
                                      treasury_spread,
                                      add_rtrs_in_history,
                                      only_dollar_price_history)
    
    if trades_df is None: return None    # no new trades

    if only_dollar_price_history == False:
        if YIELD_CURVE.upper() == 'FICC' or YIELD_CURVE.upper() == 'FICC_NEW':
            # Calculating yield spreads using ficc_ycl
            print('Calculating yield spread using ficc yield curve')
            trades_df['ficc_ycl'] = trades_df[['trade_date', 'calc_date']].parallel_apply(get_ficc_ycl, axis=1)
            
             
        trades_df['yield_spread'] = trades_df['yield'] * 100 - trades_df['ficc_ycl']
        trades_df.dropna(subset=['yield_spread'],inplace=True)
        print('Yield spread calculated')

        if treasury_spread == True:
            trades_df['treasury_rate'] = trades_df[['trade_date','calc_date','settlement_date']].parallel_apply(current_treasury_rate, 
                                                                                                                            axis=1)
            trades_df['ficc_treasury_spread'] = trades_df['ficc_ycl'] - (trades_df['treasury_rate'] * 100)
    
        
    # Dropping columns which are not used for training
    # trades_df = drop_extra_columns(trades_df)
    trades_df = convert_dates(trades_df)

    print('Processing features')
    trades_df = process_features(trades_df)

    if remove_short_maturity == True:
        trades_df = trades_df[trades_df.days_to_maturity >= np.log10(1 + 400)]

    if 'training_features' in kwargs:
        trades_df = trades_df[kwargs['training_features']]
        trades_df.dropna(inplace=True)

    if add_flags == True:    # add additional flags to the data
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
    

    print(f'Numbers of samples {len(trades_df)}')
    
    return trades_df
