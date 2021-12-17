'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 10:04:41
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2021-12-16 15:10:31
 # @ Description: Code to process trade history from BigQuery
 '''
import pandas as pd
import os
import numpy as np

# Pandaralled is a python package that is 
# used to multi-thread df apply
from pandarallel import pandarallel
pandarallel.initialize()

from ficc.utils.ficc_calc_end_date import calc_end_date
from ficc.utils.yield_curve import yield_curve_params
from ficc.utils.trade_list_to_array import trade_list_to_array
from ficc.utils.core import sqltodf
import ficc.utils.globals as globals
from ficc.utils.pad_trade_history import pad_trade_history

def fetch_trade_data(query, client, SEQUENCE_LENGTH, PATH='data.pkl'):

    if os.path.isfile(PATH):
        print(f"Processed file found {PATH}, reading data from file")
        trade_dataframe = pd.read_pickle(PATH)
        print("File read")
    else:
        print("Processed data file not found, Running query to fetch data")
        trade_dataframe = sqltodf(query,client)
        print("Saving data")
        trade_dataframe.to_pickle(PATH)
    
    #Dropping empty trades
    print("Dropping empty trades")
    trade_dataframe['empty_trade'] = trade_dataframe.recent.apply(lambda x: x[0]['rtrs_control_number'] is None)
    trade_dataframe = trade_dataframe[trade_dataframe.empty_trade == False]

    print("Dropping trades less that 10000$")
    trade_dataframe = trade_dataframe[trade_dataframe.par_traded > 10000]
    #Taking only the most recent trades
    trade_dataframe.recent = trade_dataframe.recent.apply(lambda x: x[:SEQUENCE_LENGTH])

    print('Estimating calculation date')
    trade_dataframe['calc_date'] = trade_dataframe.parallel_apply(calc_end_date, axis=1)
    trade_dataframe.recent =  trade_dataframe.apply(lambda x: np.append(x['recent'],np.array(x['calc_date'])),axis=1)

    print(trade_dataframe[['maturity_date','next_call_date','calc_date']])

    # the trade history correctly
    print('Creating trade history')
    trade_dataframe['trade_history'] = trade_dataframe.recent.parallel_apply(trade_list_to_array)
    print('Trade history created')

    trade_dataframe.drop(columns=['recent', 'empty_trade'],inplace=True)

    return trade_dataframe

def process_trade_history(query,client,SEQUENCE_LENGTH, NUM_FEATURES,PATH):
    print("Grabbing vield curve params")
    try:
        yield_curve_params(client)
    except Exception as e:
        print("Failed to grab yield curve parameters")
        raise e
    
    trade_dataframe = fetch_trade_data(query, client, SEQUENCE_LENGTH, PATH)

    print("Padding history")
    trade_dataframe.trade_history = trade_dataframe.trade_history.apply(pad_trade_history, args=[SEQUENCE_LENGTH, NUM_FEATURES])
    print("Padding completed")

    trade_dataframe.dropna(subset=['trade_history', 'yield_spread'], inplace=True)
    
    print(f"Number of samples {len(trade_dataframe)}")

    return trade_dataframe

