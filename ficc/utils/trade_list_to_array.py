'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 13:56:59
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2023-08-01 20:27:28
 # @ Description:The trade_list_to_array function uses the trade_dict_to_list 
 # function to unpack the list of dictionaries and creates a list of historical trades. 
 # With each element in the list containing all the information for that particular trade
 '''

import numpy as np
from ficc.utils.trade_dict_to_list import trade_dict_to_list


def trade_list_to_array(trade_history, 
                        remove_short_maturity: bool, 
                        trade_history_delay: int, 
                        use_treasury_spread: bool,
                        add_rtrs_in_history: bool,
                        only_dollar_price_history: bool, 
                        treasury_rate_dict: dict = None):
    empty_last_trade_features = [None] * 16
    if len(trade_history) == 0: return np.array([]), empty_last_trade_features

    trades_list = []
    last_trade_features = None
    for entry in trade_history:
        trades, temp_last_features = trade_dict_to_list(entry,
                                                        remove_short_maturity,
                                                        trade_history_delay,
                                                        use_treasury_spread,
                                                        add_rtrs_in_history,
                                                        only_dollar_price_history, 
                                                        treasury_rate_dict)
        if trades is not None: trades_list.append(trades)
        if last_trade_features is None: last_trade_features = temp_last_features
        
    if len(trades_list) == 0: return [], empty_last_trade_features
    try:
        return np.stack(trades_list), last_trade_features
    except Exception as e:
        print(f'Unable to call np.stack(trades_list) due to error: {e}')
        print('trades_list')
        for trade in trades_list: print(trade)
        print('last_trade_features')
        for last_trade_feature in last_trade_features: print(last_trade_feature)
        raise e
        