'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 13:56:59
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-03-01 10:42:16
 # @ Description:The trade_list_to_array function uses the trade_dict_to_list 
 # function to unpack the list of dictionaries and creates a list of historical trades. 
 # With each element in the list containing all the information for that particular trade
 '''


import numpy as np
from ficc.utils.trade_dict_to_list import trade_dict_to_list

def trade_list_to_array(trade_history, remove_short_maturity, remove_non_transaction_based, remove_trade_type, trade_history_delay):
    
    if len(trade_history) == 0:
        return np.array([])

    # The calc date for a trade is added as the last
    # feautre in the trade history
    calc_date = trade_history[-1]
    trade_history = trade_history[:-1] 
    trades_list = []

    for entry in trade_history:
        trades = trade_dict_to_list(entry,calc_date, remove_short_maturity, remove_non_transaction_based, remove_trade_type, trade_history_delay)
        if trades is not None:
            trades_list.append(trades)

    if len(trades_list) > 0:
        try:
            return np.stack(trades_list)
        except Exception as e:
            for i in trades_list:
                print(i)
            raise Exception("Failed to stack the arrays")
    else:
        return []