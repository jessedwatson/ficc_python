'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2021-12-16
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-02-14
 # @ Description: Source code to process trade history from BigQuery
 '''
import warnings
import numpy as np
# Pandaralled is a python package that is 
# used to multi-thread df apply
from pandarallel import pandarallel

import os

from ficc.utils.process_features import process_features
print(f'Initializing pandarallel with {os.cpu_count()/2} cores')
pandarallel.initialize(progress_bar=False, nb_workers=int(os.cpu_count()/2))

from ficc.data.process_trade_history import process_trade_history
from ficc.utils.yield_curve import get_ficc_ycl
from ficc.utils.auxiliary_functions import convert_dates
from ficc.utils.auxiliary_variables import RELATED_TRADE_FEATURE_PREFIX, NUM_RELATED_TRADES, CATEGORICAL_REFERENCE_FEATURES_PER_RELATED_TRADE
from ficc.utils.get_treasury_rate import get_treasury_rate_dict, current_treasury_rate
from ficc.utils.adding_flags import add_bookkeeping_flag, add_replica_count_flag, add_same_day_flag, add_ntbc_precursor_flag
from ficc.utils.related_trade import add_related_trades
from ficc.utils.yield_curve_params import yield_curve_params


def process_data(query, 
                 client, 
                 num_trades_in_history, 
                 num_features_for_each_trade_in_history, 
                 path, 
                 yield_curve='FICC_NEW', 
                 remove_short_maturity=False, 
                 trade_history_delay=12, 
                 min_trades_in_history=0, 
                 use_treasury_spread=False, 
                 add_flags=False, 
                 add_related_trades_bool=False, 
                 add_rtrs_in_history=False, 
                 only_dollar_price_history=False, 
                 save_data=True, 
                 **kwargs):
    if len(kwargs) != 0: warnings.warn(f'**kwargs is not empty and has following arguments: {kwargs.keys()}', category=RuntimeWarning)
        
    yield_curve = yield_curve.upper()
    if yield_curve == 'FICC' or yield_curve == 'FICC_NEW':
        print('Grabbing yield curve params')
        try:
            nelson_params, scalar_params, shape_parameter = yield_curve_params(client, yield_curve)
        except Exception as e:
            print('Unable to grab yield curve parameters')
            raise e
    
    print(f'Running with\n remove_short_maturity: {remove_short_maturity}\n trade_history_delay: {trade_history_delay}\n use_treasury_spread: {use_treasury_spread}\n min_trades_in_hist: {min_trades_in_history}\n add_flags: {add_flags}\n add_related_trades_bool: {add_related_trades_bool}\n add_rtrs_in_history: {add_rtrs_in_history}\n only_dollar_price_history: {only_dollar_price_history}\n save_data: {save_data}')
    treasury_rate_dict = get_treasury_rate_dict(client) if use_treasury_spread is True else None
    trades_df = process_trade_history(query, 
                                      client, 
                                      num_trades_in_history, 
                                      num_features_for_each_trade_in_history, 
                                      path, 
                                      remove_short_maturity, 
                                      trade_history_delay,  
                                      min_trades_in_history, 
                                      use_treasury_spread, 
                                      add_rtrs_in_history, 
                                      only_dollar_price_history, 
                                      yield_curve, 
                                      treasury_rate_dict, 
                                      nelson_params, 
                                      scalar_params, 
                                      shape_parameter, 
                                      save_data)
    
    if trades_df is None: return None    # no new trades

    if only_dollar_price_history is False:
        if yield_curve == 'FICC' or yield_curve == 'FICC_NEW':
            print('Calculating yield spread using ficc yield curve')
            trades_df['ficc_ycl'] = trades_df[['trade_date', 'calc_date']].parallel_apply(lambda trade: get_ficc_ycl(trade, nelson_params, scalar_params, shape_parameter), axis=1)
             
        trades_df['yield_spread'] = trades_df['yield'] * 100 - trades_df['ficc_ycl']
        num_trades_before_dropping_null_yield_spreads = len(trades_df)
        trades_df.dropna(subset=['yield_spread'], inplace=True)
        print(f'Yield spread calculated; removed {num_trades_before_dropping_null_yield_spreads - len(trades_df)} trades since these had a null yield spread')

        if use_treasury_spread is True:
            trades_df['treasury_rate'] = trades_df[['trade_date', 'calc_date', 'settlement_date']].parallel_apply(lambda trade: current_treasury_rate(treasury_rate_dict, trade), axis=1)
            null_treasury_rate = trades_df['treasury_rate'].isnull()
            if null_treasury_rate.sum() > 0:
                trade_dates_corresponding_to_null_treasury_rate = trades_df.loc[null_treasury_rate, 'trade_date']
                print(f'The following `trade_date`s have no corresponding `treasury_rate`, so all {null_treasury_rate.sum()} trades with these `trade_date`s have been removed: {trade_dates_corresponding_to_null_treasury_rate.unique()}')
                trades_df = trades_df[~null_treasury_rate]
            trades_df['ficc_treasury_spread'] = trades_df['ficc_ycl'] - (trades_df['treasury_rate'] * 100)

    if len(trades_df) == 0:
        print(f'After dropping trades for not having a treasury rate, the dataframe is empty')
        return None
        
    # Dropping columns which are not used for training
    # trades_df = drop_extra_columns(trades_df)
    trades_df = convert_dates(trades_df)
    trades_df = process_features(trades_df)

    if remove_short_maturity is True:
        print('Removing short maturity')
        trades_df = trades_df[trades_df.days_to_maturity >= np.log10(1 + 400)]

    if 'training_features' in kwargs:
        trades_df = trades_df[kwargs['training_features']]
        trades_df.dropna(inplace=True)

    if add_flags is True:    # add additional flags to the data
        # trades_df = add_replica_flag(trades_df)    # the IS_REPLICA flag was originally designed to remove replica trades from the trade history
        trades_df = add_replica_count_flag(trades_df)
        trades_df = add_bookkeeping_flag(trades_df)
        trades_df = add_same_day_flag(trades_df)
        trades_df = add_ntbc_precursor_flag(trades_df)
    
    if add_related_trades_bool is True:
        print('Adding most recent related trade')
        trades_df = add_related_trades(trades_df,
                                       RELATED_TRADE_FEATURE_PREFIX, 
                                       NUM_RELATED_TRADES, 
                                       CATEGORICAL_REFERENCE_FEATURES_PER_RELATED_TRADE)
    
    print(f'{len(trades_df)} trades at the end of `process_data(...)` ranging from trade datetimes of {trades_df.trade_datetime.min()} to {trades_df.trade_datetime.max()}')
    return trades_df
