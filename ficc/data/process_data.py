'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 10:04:41
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-08-09 13:42:41
 # @ Description: Source code to process trade history from BigQuery
 '''
 
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
from ficc.utils.auxiliary_variables import IS_BOOKKEEPING, IS_SAME_DAY, IS_DUPLICATE
from ficc.utils.adding_flags import add_bookkeeping_flag, add_same_day_flag, add_duplicate_flag


def process_data(query, 
                 client, 
                 SEQUENCE_LENGTH, 
                 NUM_FEATURES, 
                 PATH, 
                 YIELD_CURVE="FICC", 
                 estimate_calc_date=False, 
                 remove_short_maturity=False, 
                 remove_non_transaction_based=False, 
                 remove_trade_type=[], 
                 remove_duplicates_from_trade_history=False, 
                 trade_history_delay=1, 
                 min_trades_in_history=2, 
                 process_ratings=True, 
                 keep_nan=False, 
                 add_flags=False, 
                 **kwargs):
    
    # This global variable is used to be able to process data in parallel
    globals.YIELD_CURVE_TO_USE = YIELD_CURVE
    print(f'Running with\n estimate_calc_date:{estimate_calc_date}\n remove_short_maturity:{remove_short_maturity}\n remove_non_transaction_based:{remove_non_transaction_based}\n remove_trade_type:{remove_trade_type}\n remove_duplicates_from_trade_history:{remove_duplicates_from_trade_history}\n trade_history_delay:{trade_history_delay}\n min_trades_in_hist:{min_trades_in_history}\n process_ratings:{process_ratings}\n add_flags:{add_flags}')
    
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
                                      remove_duplicates_from_trade_history, 
                                      min_trades_in_history,
                                      process_ratings)

    if YIELD_CURVE.upper() == "FICC" or YIELD_CURVE.upper() == "FICC_NEW":
        # Calculating yield spreads using ficc_ycl
        print("Calculating yield spread using ficc yield curve")
        trades_df['ficc_ycl'] = trades_df.parallel_apply(get_ficc_ycl,axis=1)
        # As ficc ycl is already in basis points
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
    
    print('Yield spread calculated')

    # Dropping columns which are not used for training
    # trades_df = drop_extra_columns(trades_df)
    trades_df = convert_dates(trades_df)

    print("Processing features")
    trades_df = process_features(trades_df, keep_nan)

    if remove_short_maturity == True:
        trades_df = trades_df[trades_df.days_to_maturity >= np.log10(400)]

    if 'training_features' in kwargs:
        trades_df = trades_df[kwargs['training_features']]
        trades_df.dropna(inplace=True)

    if add_flags:    # add additional flags to the data
        if IS_DUPLICATE not in trades_df.columns:
            trades_df = add_duplicate_flag(trades_df, IS_DUPLICATE)
        trades_df = add_bookkeeping_flag(trades_df, IS_BOOKKEEPING)
        trades_df = add_same_day_flag(trades_df, IS_SAME_DAY)
    
    print(f"Numbers of samples {len(trades_df)}")
    
    return trades_df
