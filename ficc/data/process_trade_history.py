'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-17 14:44:20
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-01-24 13:06:45
 # @ Description:
 '''

import os
import pandas as pd
import numpy as np

from ficc.utils.auxiliary_functions import sqltodf
from ficc.utils.auxiliary_variables import PREDICTORS
from ficc.utils.pad_trade_history import pad_trade_history
import ficc.utils.globals as globals
from ficc.utils.ficc_calc_end_date import calc_end_date
from ficc.utils.yield_curve_params import yield_curve_params
from ficc.utils.trade_list_to_array import trade_list_to_array

def fetch_trade_data(query, client, PATH='data.pkl'):

    if os.path.isfile(PATH):
        print(f"Data file found {PATH}, reading data from file")
        trade_dataframe = pd.read_pickle(PATH)
        print("File read")
    else:
        print("Data file not found, Running query to fetch data")
        trade_dataframe = sqltodf(query,client)
        print("Saving data")
        print(PATH)
        trade_dataframe.to_pickle(PATH)

    return trade_dataframe

def process_trade_history(query, client, SEQUENCE_LENGTH, NUM_FEATURES, PATH, estimate_calc_date):
    if globals.YIELD_CURVE_TO_USE.upper() == "FICC":
        print("Grabbing yield curve params")
        try:
            yield_curve_params(client)
        except Exception as e:
            print("Failed to grab yield curve parameters")
            raise e
    
    trade_dataframe = fetch_trade_data(query, client, PATH)

    #Dropping empty trades
    print("Dropping empty trades")
    trade_dataframe['empty_trade'] = trade_dataframe.recent.apply(lambda x: x[0]['rtrs_control_number'] is None)
    trade_dataframe = trade_dataframe[trade_dataframe.empty_trade == False]

    # Moved to the query
    # print("Dropping trades less that $10,000")
    # trade_dataframe = trade_dataframe[trade_dataframe.par_traded > 10000]
    
    # Taking only the most recent trades
    trade_dataframe.recent = trade_dataframe.recent.apply(lambda x: x[:SEQUENCE_LENGTH])

    trade_dataframe['calc_date'] = trade_dataframe.parallel_apply(calc_end_date, axis=1)
    trade_dataframe.recent =  trade_dataframe.apply(lambda x: np.append(x['recent'],np.array(x['calc_date'])),axis=1)

    if estimate_calc_date == True:
        print('Estimating calculation date')
        print(trade_dataframe[['maturity_date','next_call_date','calc_date']])

    # the trade history correctly
    print('Creating trade history')
    trade_dataframe['trade_history'] = trade_dataframe.recent.parallel_apply(trade_list_to_array)
    print('Trade history created')

    if estimate_calc_date == True:
        trade_dataframe.drop(columns=['recent', 'empty_trade'],inplace=True)
    else:
        trade_dataframe.drop(columns=['recent', 'empty_trade','calc_date'],inplace=True)

    print("Padding history")
    trade_dataframe.trade_history = trade_dataframe.trade_history.apply(pad_trade_history, args=[SEQUENCE_LENGTH, NUM_FEATURES])
    print("Padding completed")

    trade_dataframe.dropna(subset=['trade_history', 'yield_spread'], inplace=True)

    return trade_dataframe