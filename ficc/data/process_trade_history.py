'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-17 14:44:20
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-07-19 13:19:45
 # @ Description:
 '''

import os
import pandas as pd
import pickle5 as pickle

from ficc.utils.auxiliary_functions import sqltodf, process_ratings, convert_object_to_category, convert_calc_date_to_category
from ficc.utils.auxiliary_variables import IS_DUPLICATE
from ficc.utils.adding_flags import add_duplicate_flag
from ficc.utils.pad_trade_history import pad_trade_history
import ficc.utils.globals as globals
from ficc.utils.ficc_calc_end_date import calc_end_date
from ficc.utils.yield_curve_params import yield_curve_params
from ficc.utils.trade_list_to_array import trade_list_to_array
from ficc.utils.create_mmd_data import create_mmd_data


def fetch_trade_data(query, client, PATH='data.pkl'):

    if os.path.isfile(PATH):
        print(f"Data file {PATH} found, reading data from it")
        with open(PATH, 'rb') as f: 
            (q, trade_dataframe) = pickle.load(f)
        if q == query:
            return trade_dataframe
        else:
            raise Exception (f"Saved query is incorrect:\n{q}")
    
    print(f'Grabbing data from BigQuery')
    trade_dataframe = sqltodf(query,client)
    print(f"Saving query and data to {PATH}")
    
    ds = (query, trade_dataframe)
    
    with open(PATH, 'wb') as f: 
        pickle.dump(ds, f)
    
    return trade_dataframe

def process_trade_history(query,
                          client, 
                          SEQUENCE_LENGTH, 
                          NUM_FEATURES, 
                          PATH, 
                          estimate_calc_date, 
                          remove_short_maturity, 
                          remove_non_transaction_based,
                          remove_trade_type, 
                          trade_history_delay, 
                          remove_duplicates, 
                          min_trades_in_history, 
                          drop_ratings):
    
    if globals.YIELD_CURVE_TO_USE.upper() == "FICC":
        print("Grabbing yield curve params")
        try:
            yield_curve_params(client)
        except Exception as e:
            print("Failed to grab yield curve parameters")
            raise e
    
    if globals.YIELD_CURVE_TO_USE.upper() == "MMD":
        print("Grabbing MMD yield curve level")
        try:
            create_mmd_data(client)
        except Exception as e:
            print("Failed to grab MMD ycl")
            raise e

    trade_dataframe = fetch_trade_data(query, client, PATH)
    trade_dataframe = process_ratings(trade_dataframe, drop_ratings)
    trade_dataframe = convert_object_to_category(trade_dataframe)

    print(f'Raw data contains {len(trade_dataframe)} samples')
    
    # Dropping empty trades
    print("Dropping empty trades")
    trade_dataframe['empty_trade'] = trade_dataframe.recent.apply(lambda x: x[0]['rtrs_control_number'] is None)
    trade_dataframe = trade_dataframe[trade_dataframe.empty_trade == False]

    # Taking only the most recent trades
    # trade_dataframe.recent = trade_dataframe.recent.apply(lambda x: x[:SEQUENCE_LENGTH])

    if estimate_calc_date == True:
        trade_dataframe['calc_date'] = trade_dataframe.apply(calc_end_date, axis=1)
        print('Estimating calculation date')
        print(trade_dataframe[['maturity_date','next_call_date','calc_date']])

    print('Creating trade history')
    
    if remove_short_maturity == True:
        print("Removing trades with shorter maturity")
    
    if len(remove_trade_type) > 0:
        print(f"Removing trade types {remove_trade_type}")

    if remove_duplicates:
        print(f'Removing trades that are marked with the {IS_DUPLICATE} flag.')
        trade_dataframe = add_duplicate_flag(trade_dataframe, IS_DUPLICATE)

    print('Getting last dollar price and calc date')
    temp_df = trade_dataframe.recent.apply(lambda x:(x[0]['dollar_price'],x[0]['calc_date']))
    trade_dataframe[['last_dollar_price','last_calc_date']] = pd.DataFrame(temp_df.tolist(), index=trade_dataframe.index)    
    trade_dataframe['last_calc_day_cat'] = trade_dataframe.apply(convert_calc_date_to_category, axis=1)
    print('Getting last dollar price and calc date')

    print(f'Removing trades less than {trade_history_delay} minutes in the history')
    trade_dataframe['trade_history'] = trade_dataframe.recent.parallel_apply(trade_list_to_array, args=([remove_short_maturity,
                                                                                                        remove_non_transaction_based,
                                                                                                        remove_trade_type,
                                                                                                        trade_history_delay, 
                                                                                                        remove_duplicates]))
    print('Trade history created')

    trade_dataframe.drop(columns=['recent', 'empty_trade'],inplace=True)
    
    print(f"Restricting the trade history to the {SEQUENCE_LENGTH} most recent trades")
    trade_dataframe.trade_history = trade_dataframe.trade_history.apply(lambda x: x[:SEQUENCE_LENGTH])

    print("Padding history")
    print(f"Minimum number of trades required in the history {min_trades_in_history}")
    trade_dataframe.trade_history = trade_dataframe.trade_history.apply(pad_trade_history, args=[SEQUENCE_LENGTH, NUM_FEATURES, min_trades_in_history])
    print("Padding completed")
     
    trade_dataframe.dropna(subset=['trade_history', 'yield_spread'], inplace=True)
    print(f'Processed trade history contain {len(trade_dataframe)} samples')
    return trade_dataframe