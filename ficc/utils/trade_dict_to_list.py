'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 13:58:58
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-03-01 22:08:46
 # @ Description:The trade_dict_to_list converts the recent trade dictionary to a list.
 # The SQL arrays from BigQuery are converted to a dictionary when read as a pandas dataframe. 
 # 
 # A few blunt normalization are performed on the data. We will experiment with others as well. 
 # Multiplying the yield spreads by 100 to convert into basis points. 
 # Taking the log of the size of the trade to reduce the absolute scale 
 # Taking the log of the number of seconds between the historical trade and the latest trade
 '''

import numpy as np
from datetime import datetime, timedelta

from pandas import ExcelFile

from ficc.utils.yield_curve import yield_curve_level
from ficc.utils.mmd_ycl import mmd_ycl
import ficc.utils.globals as globals

def trade_dict_to_list(trade_dict: dict, calc_date, remove_short_maturity, remove_non_transaction_based, remove_trade_type, trade_history_delay) -> list:
    trade_type_mapping = {'D':[0,0],'S': [0,1],'P': [1,0]}
    trade_list = []

    if trade_dict['rtrs_control_number'] is None:
        print('rtrs control number is missing, skipping this trade')
        return None

    # Checking if the seconds a go feature is missing
    if trade_dict['seconds_ago'] is None:
        print('Seconds a go missing, skipping this trade')
        return None
    elif trade_dict['seconds_ago'] < (trade_history_delay * 60):
        return None
    

    # We do not have weighted average maturity before July 27 for ficc yc
    if globals.YIELD_CURVE_TO_USE.upper() == 'FICC' and trade_dict['trade_datetime'] is not None and trade_dict['trade_datetime'] < datetime(2021,7,27):
        target_date = datetime(2021,7,27).date()
    elif trade_dict['trade_datetime'] is not None:
        target_date = trade_dict['trade_datetime'].date()
    else:
        print("Trade date is missing, skipping this trade")
        return None
    
    if remove_non_transaction_based == True and trade_dict['is_non_transaction_based_compensation'] == True:
        return None

    if trade_dict['settlement_date'] is None and remove_short_maturity == True:
        if trade_dict['trade_date'] is not None:
            trade_dict['settlement_date'] = trade_dict['trade_date'] + timedelta(days=3)
        else:
            return None
    elif remove_short_maturity == True:
        try:
            days_to_maturity = (trade_dict['maturity_date'] - trade_dict['settlement_date']).days
        except Exception as e:
            print("Failed to remove this trade")
            for key in trade_dict.keys():
                print(f'{key} : {trade_dict[key]}')
        if days_to_maturity < 400:
            return None
    
    if len(remove_trade_type) > 0 and trade_dict['trade_type'] in remove_trade_type:
        return None

    #calculating the time to maturity in years from the trade_date
    if globals.YIELD_CURVE_TO_USE.upper() == "FICC":
        time_to_maturity = (calc_date - target_date).days/365.25
        global nelson_params
        global scalar_params
        yield_at_that_time = yield_curve_level(time_to_maturity,target_date.strftime('%Y-%m-%d'),
                                            globals.nelson_params, globals.scalar_params)

        if trade_dict['yield'] is not None:
            trade_list.append(trade_dict['yield'] * 100 - yield_at_that_time)
        else:
            print('Yield is missing, skipping trade')
            return None
    
    elif globals.YIELD_CURVE_TO_USE.upper() == "MMD":
        time_to_maturity = (calc_date - target_date).days/365.25
        yield_at_that_time = mmd_ycl(target_date.strftime('%Y-%m-%d'), time_to_maturity)
        if trade_dict['yield'] is not None:
            trade_list.append( (trade_dict['yield'] - yield_at_that_time) * 100 )
        else:
            print('Yield is missing, skipping trade')
            return None

    elif globals.YIELD_CURVE_TO_USE.upper() == "S&P":
        if trade_dict['yield'] is not None:
            trade_list.append(trade_dict['yield_spread'] * 100)
        else:
            print('Yield is missing, skipping this trade')
            return None
        
    for key in ['par_traded','trade_type','seconds_ago']:
        if trade_dict[key] is None:
            print(f'{key} is missing, skipping this trade')
            return None
    
    trade_list.append(np.float32(np.log10(trade_dict['par_traded'])))        
    trade_list += trade_type_mapping[trade_dict['trade_type']]

    # For some trades the seconds ago feature is negative.
    # This is because the publish time is after the trade datetime.
    # We have verified that this is an anomaly on MSRBs end.
    if trade_dict['seconds_ago'] < 0:
        trade_list.append(0)
    else:
        trade_list.append(np.log10(1+trade_dict['seconds_ago']))

    return np.stack(trade_list)