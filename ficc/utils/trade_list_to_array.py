'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 13:56:59
 # @ Modified by: Mitas Ray
 # @ Modified time: 2022-08-17 14:16:00
 # @ Description:The trade_list_to_array function uses the trade_dict_to_list 
 # function to unpack the list of dictionaries and creates a list of historical trades. 
 # With each element in the list containing all the information for that particular trade
 '''

import numpy as np
from ficc.utils.trade_dict_to_list import trade_dict_to_list


def trade_list_to_array(trade_history, 
                        remove_short_maturity, 
                        remove_non_transaction_based, 
                        remove_trade_type, 
                        trade_history_delay, 
                        remove_replicas_from_trade_history, 
                        rtrs_control_number_and_is_replica_flag):
    '''The `remove_replicas_from_trade_history` is a boolean variable that 
    determines whether replica trades should be excluded from the trade 
    history. If this variable is set to `True`, then we must have a dataframe 
    in `rtrs_control_number_and_is_replica_flag` which contains both the 
    `rtrs_control_number` and the corresponding `is_replica_flag`.'''
    
    if len(trade_history) == 0:
        return np.array([])

    # The calc date for a trade is added as the last
    # feature in the trade history
    # calc_date = trade_history[-1]
    # trade_history = trade_history[:-1] 
    trades_list = []

    for entry in trade_history:
        trades = trade_dict_to_list(entry, remove_short_maturity, remove_non_transaction_based, remove_trade_type, trade_history_delay, remove_replicas_from_trade_history, rtrs_control_number_and_is_replica_flag)
        if trades is not None:
            trades_list.append(trades)

    if len(trades_list) > 0:
        try:
            return np.stack(trades_list)
        except Exception:
            for i in trades_list:
                print(i)
            raise Exception("Failed to stack the arrays")
    else:
        return []