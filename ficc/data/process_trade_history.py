'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2021-12-17
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-02-13
 # @ Description:
 '''
import os
import pandas as pd
import pickle5 as pickle

from ficc.utils.auxiliary_functions import sqltodf, process_ratings
from ficc.utils.pad_trade_history import pad_trade_history
from ficc.utils.trade_list_to_array import trade_list_to_array


def fetch_trade_data(query, client, PATH='data.pkl', save_data=True):
    if os.path.isfile(PATH):
        print(f'Data file {PATH} found, reading data from it')
        with open(PATH, 'rb') as f: 
            (q, trades_df) = pickle.load(f)
        if q == query:
            return trades_df
        else:
            raise Exception (f'Saved query is incorrect:\n{q}')
    
    print(f'Grabbing data from BigQuery')
    trades_df = sqltodf(query, client)

    if save_data:
        print(f'Saving query and data to {PATH}')
        ds = (query, trades_df)
        with open(PATH, 'wb') as f: 
            pickle.dump(ds, f)
    return trades_df


def process_trade_history(query: str,
                          client, 
                          num_trades_in_history: int, 
                          num_features_for_each_trade_in_history: int, 
                          PATH: str,  
                          remove_short_maturity: bool,
                          trade_history_delay: int, 
                          min_trades_in_history: int, 
                          use_treasury_spread: bool,
                          add_rtrs_in_history: bool,
                          only_dollar_price_history: bool, 
                          yield_curve_to_use: str, 
                          treasury_rate_dict: dict = None, 
                          nelson_params: dict = None, 
                          scalar_params: dict = None, 
                          shape_parameter: dict = None, 
                          save_data: bool = True):
    trades_df = fetch_trade_data(query, client, PATH, save_data)
    if len(trades_df) == 0:
        print('Raw data contains 0 trades')
        return None
    print(f'Raw data contains {len(trades_df)} trades ranging from trade datetimes of {trades_df.trade_datetime.min()} to {trades_df.trade_datetime.max()}')
    
    trades_df = process_ratings(trades_df)
    # trades_df = convert_object_to_category(trades_df)

    print('Creating trade history')
    if remove_short_maturity is True: print('Removing trades with shorter maturity')
    print(f'Removing trades less than {trade_history_delay} seconds in the history')
    temp = pd.DataFrame(data=None, index=trades_df.index, columns=['trade_history', 'temp_last_features'])
    temp = trades_df.recent.parallel_apply(trade_list_to_array, args=([remove_short_maturity,
                                                                       trade_history_delay,
                                                                       use_treasury_spread,
                                                                       add_rtrs_in_history,
                                                                       only_dollar_price_history, 
                                                                       yield_curve_to_use, 
                                                                       treasury_rate_dict, 
                                                                       nelson_params, 
                                                                       scalar_params, 
                                                                       shape_parameter]))
                                                                        
    trades_df[['trade_history', 'temp_last_features']] = pd.DataFrame(temp.tolist(), index=trades_df.index)
    del temp
    print('Trade history created')
    print('Getting last trade features')
    trades_df[['last_yield_spread', 
               'last_ficc_ycl', 
               'last_rtrs_control_number', 
               'last_yield', 
               'last_dollar_price', 
               'last_seconds_ago', 
               'last_size', 
               'last_calc_date', 
               'last_maturity_date', 
               'last_next_call_date', 
               'last_par_call_date', 
               'last_refund_date', 
               'last_trade_datetime', 
               'last_calc_day_cat', 
               'last_settlement_date', 
               'last_trade_type']] = pd.DataFrame(trades_df['temp_last_features'].tolist(), index=trades_df.index)
    trades_df = trades_df.drop(columns=['temp_last_features', 'recent'])

    print(f'Restricting the trade history to the {num_trades_in_history} most recent trades')
    trades_df.trade_history = trades_df.trade_history.apply(lambda history: history[:num_trades_in_history])

    print('Padding history')
    print(f'Minimum number of trades required in the history: {min_trades_in_history}')
    trades_df.trade_history = trades_df.trade_history.parallel_apply(pad_trade_history, args=[num_trades_in_history, 
                                                                                              num_features_for_each_trade_in_history,
                                                                                              min_trades_in_history])
    print('Padding completed')
    
    num_trades_before_removing_null_history = len(trades_df)
    trades_df.dropna(subset=['trade_history'], inplace=True)
    print(f'Processed trade history contains {len(trades_df)} trades. Prior to removing null histories, it contained {num_trades_before_removing_null_history} trades.')
    return trades_df
