'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 13:58:58
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2023-09-29 16:28:59
 # @ Description:The trade_dict_to_list converts the recent trade dictionary to a list.
 # The SQL arrays from BigQuery are converted to a dictionary when read as a pandas dataframe. 
 # 
 # A few blunt normalization are performed on the data. We will experiment with others as well. 
 # Multiplying the yield spreads by 100 to convert into basis points. 
 # Taking the log of the size of the trade to reduce the absolute scale 
 # Taking the log of the number of seconds between the historical trade and the latest trade
 '''

from time import time
import numpy as np
from datetime import datetime
import pandas as pd

from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from ficc.utils.yield_curve import yield_curve_level
import ficc.utils.globals as globals
from pandas.tseries.offsets import BDay


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def trade_dict_to_list(trade_dict: dict, 
                       remove_short_maturity, 
                       trade_history_delay,
                       treasury_spread,
                       add_rtrs_in_history,
                       only_dollar_price_history) -> list:

    trade_type_mapping = {'D':[0,0],'S': [0,1],'P': [1,0]}
    trade_list = []

    for feature in ['rtrs_control_number','seconds_ago','settlement_date','par_traded','trade_type','seconds_ago','trade_datetime']:
        if trade_dict[feature] is None:
            return None, None

    # Making sure that the most recent trade
    if trade_dict['seconds_ago'] < (trade_history_delay * 60):
        return None, None

    
    if only_dollar_price_history == True:
        trade_list.append(trade_dict['dollar_price'])
        trade_list.append(np.float32(np.log10(trade_dict['par_traded'])))        
        trade_list += trade_type_mapping[trade_dict['trade_type']]
        trade_list.append(np.log10(1+trade_dict['seconds_ago']))
        yield_at_that_time = None
        yield_spread = None
    
    else:    
        # The ficc yield curve coefficients are only present before 27th July for the old yield curve and 2nd August for the new yield curve
        if globals.YIELD_CURVE_TO_USE.upper() == 'FICC'and trade_dict['trade_datetime'] < datetime(2021,7,27):
            return None, None
        elif globals.YIELD_CURVE_TO_USE.upper() == 'FICC_NEW' and trade_dict['trade_datetime'] < datetime(2021,8,3):
            return None, None
        elif trade_dict['trade_datetime'] is not None:
            target_date = trade_dict['trade_datetime'].date()
        else:
            return None, None

        
        if remove_short_maturity == True:
            try:
                days_to_maturity = (trade_dict['maturity_date'] - trade_dict['settlement_date']).days
            except Exception as e:
                print("Failed to remove this trade")
                for key in trade_dict.keys():
                    print(f'{key} : {trade_dict[key]}')
            if days_to_maturity < 400:
                return None, None
        

        calc_date = trade_dict['calc_date']
        time_to_maturity = diff_in_days_two_dates(calc_date,target_date)/NUM_OF_DAYS_IN_YEAR
        #calculating the time to maturity in years from the trade_date
        if globals.YIELD_CURVE_TO_USE.upper() == "FICC" or globals.YIELD_CURVE_TO_USE == 'FICC_NEW':
            global nelson_params
            global scalar_params
            yield_at_that_time = yield_curve_level(time_to_maturity,
                                                target_date,
                                                globals.nelson_params, 
                                                globals.scalar_params,
                                                globals.shape_parameter)

            if trade_dict['yield'] is not None and yield_at_that_time is not None:
                yield_spread = trade_dict['yield'] * 100 - yield_at_that_time
                trade_list.append(yield_spread)
            else:
                # print(f"Yield is missing, skipping trade {trade_dict['rtrs_control_number']}")
                return None, None
            
        
        if treasury_spread == True:
            # add all the maturities and the difference in the levels among them and the ted spread
            treasury_maturities = np.array([1,2,3,5,7,10,20,30])
            maturity = min(treasury_maturities, key=lambda x:abs(x-time_to_maturity))
            maturity = 'year_'+str(maturity)
            
            try:
                t_rate = globals.treasury_rate[target_date][maturity]
            except Exception as e:
                return None, None
            
            t_spread = (trade_dict['yield'] - t_rate) * 100
            trade_list.append(np.round(t_spread,3))

        #trade_list.append(np.float32(trade_dict['dollar_price']))
        trade_list.append(np.float32(np.log10(trade_dict['par_traded'])))        
        trade_list += trade_type_mapping[trade_dict['trade_type']]
        trade_list.append(np.log10(1+trade_dict['seconds_ago']))

    if add_rtrs_in_history == True:
        trade_list.append(trade_dict['is_non_transaction_based_compensation'])
        trade_list.append(int(trade_dict['rtrs_control_number']))

    return np.stack(trade_list) , (yield_spread,
                                   yield_at_that_time,
                                   int(trade_dict['rtrs_control_number']),
                                   trade_dict['yield'] * 100 if trade_dict['yield'] is not None else None,
                                   trade_dict['dollar_price'], 
                                   trade_dict['seconds_ago'], 
                                   float(trade_dict['par_traded']),
                                   trade_dict['calc_date'], 
                                   trade_dict['maturity_date'], 
                                   trade_dict['next_call_date'], 
                                   trade_dict['par_call_date'], 
                                   trade_dict['refund_date'], 
                                   trade_dict['trade_datetime'], 
                                   trade_dict['calc_day_cat'], 
                                   trade_dict['settlement_date'], 
                                   trade_dict['trade_type'])