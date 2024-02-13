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
import ficc.utils.globals as globals
from ficc.utils.yield_curve_params import yield_curve_params
from ficc.utils.trade_list_to_array import trade_list_to_array
from ficc.utils.get_treasury_rate import get_treasury_rate


def fetch_trade_data(query, client, PATH='data.pkl', save_data=True):
    if os.path.isfile(PATH):
        print(f'Data file {PATH} found, reading data from it')
        with open(PATH, 'rb') as f: 
            (q, trade_dataframe) = pickle.load(f)
        if q == query:
            return trade_dataframe
        else:
            raise Exception (f'Saved query is incorrect:\n{q}')
    
    print(f'Grabbing data from BigQuery')
    trade_dataframe = sqltodf(query, client)

    if save_data:
        print(f'Saving query and data to {PATH}')
        ds = (query, trade_dataframe)
        with open(PATH, 'wb') as f: 
            pickle.dump(ds, f)
    return trade_dataframe


def process_trade_history(query,
                          client, 
                          SEQUENCE_LENGTH, 
                          num_features_for_each_trade_in_history, 
                          PATH,  
                          remove_short_maturity,
                          trade_history_delay, 
                          min_trades_in_history, 
                          treasury_spread,
                          add_rtrs_in_history,
                          only_dollar_price_history, 
                          save_data=True):
    if globals.YIELD_CURVE_TO_USE.upper() == 'FICC' or globals.YIELD_CURVE_TO_USE.upper() == 'FICC_NEW':
        print('Grabbing yield curve params')
        try:
            yield_curve_params(client, globals.YIELD_CURVE_TO_USE.upper())
        except Exception as e:
            raise e 
    
    if treasury_spread == True: get_treasury_rate(client)

    trade_dataframe = fetch_trade_data(query, client, PATH, save_data)
    print(f'Raw data contains {len(trade_dataframe)} samples')
    if len(trade_dataframe) == 0: return None
    
    trade_dataframe = process_ratings(trade_dataframe)
    # trade_dataframe = convert_object_to_category(trade_dataframe)

    print('Creating trade history')
    if remove_short_maturity == True: print('Removing trades with shorter maturity')
    print(f'Removing trades less than {trade_history_delay} seconds in the history')
    temp = pd.DataFrame(data=None, index=trade_dataframe.index, columns=['trade_history', 'temp_last_features'])
    temp = trade_dataframe.recent.parallel_apply(trade_list_to_array, args=([remove_short_maturity,
                                                                             trade_history_delay,
                                                                             treasury_spread,
                                                                             add_rtrs_in_history,
                                                                             only_dollar_price_history]))
                                                                        
    trade_dataframe[['trade_history', 'temp_last_features']] = pd.DataFrame(temp.tolist(), index=trade_dataframe.index)
    del temp
    print('Trade history created')
    print('Getting last trade features')
    trade_dataframe[['last_yield_spread',
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
                     'last_trade_type']] = pd.DataFrame(trade_dataframe['temp_last_features'].tolist(), index=trade_dataframe.index)
    trade_dataframe = trade_dataframe.drop(columns=['temp_last_features', 'recent'])

    print(f'Restricting the trade history to the {SEQUENCE_LENGTH} most recent trades')
    trade_dataframe.trade_history = trade_dataframe.trade_history.apply(lambda history: history[:SEQUENCE_LENGTH])

    print('Padding history')
    print(f'Minimum number of trades required in the history: {min_trades_in_history}')
    trade_dataframe.trade_history = trade_dataframe.trade_history.parallel_apply(pad_trade_history, args=[SEQUENCE_LENGTH, 
                                                                                                          num_features_for_each_trade_in_history,
                                                                                                          min_trades_in_history])
    print('Padding completed')
    
    num_trades_before_removing_null_history = len(trade_dataframe)
    trade_dataframe.dropna(subset=['trade_history'], inplace=True)
    print(f'Processed trade history contains {len(trade_dataframe)} samples. Prior to removing null histories, it contained {num_trades_before_removing_null_history} samples.')
    return trade_dataframe
