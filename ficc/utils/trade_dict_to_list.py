'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 13:58:58
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-01-26 15:56:43
 # @ Description:The trade_dict_to_list converts the recent trade dictionary to a list.
 # The SQL arrays from BigQuery are converted to a dictionary when read as a pandas dataframe. 
 # 
 # A few blunt normalization are performed on the data. We will experiment with others as well. 
 # Multiplying the yield spreads by 100 to convert into basis points. 
 # Taking the log of the size of the trade to reduce the absolute scale 
 # Taking the log of the number of seconds between the historical trade and the latest trade
 '''

import numpy as np
from datetime import datetime

from ficc.utils.yield_curve import yield_curve_level
import ficc.utils.globals as globals

def trade_dict_to_list(trade_dict: dict, calc_date, remove_short_maturity) -> list:
    trade_type_mapping = {'D':[0,0],'S': [0,1],'P': [1,0]}
    trade_list = []

    # This 
    if trade_dict['trade_datetime'] < datetime(2021,7,27):
        target_date = datetime(2021,7,27).date()
    else:
        target_date = trade_dict['trade_datetime'].date()
    
    if remove_short_maturity == True:
        days_to_calc = (calc_date - trade_dict['settlement_date']).days
        if days_to_calc < 360:
            return None

    #calculating the time to maturity in years from the trade_date
    if globals.YIELD_CURVE_TO_USE.upper() == "FICC":
        time_to_maturity = (calc_date - target_date).days/365.25
        global nelson_params
        global scalar_params
        yield_at_that_time = yield_curve_level(time_to_maturity,target_date.strftime('%Y-%m-%d'),
                                            globals.nelson_params, globals.scalar_params)

    
        trade_list.append(trade_dict['yield'] * 100 - yield_at_that_time)
    
    elif globals.YIELD_CURVE_TO_USE.upper() == "S&P":
        trade_list.append(trade_dict['yield_spread'] * 100)
        
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