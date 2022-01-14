'''
 # @ Author: Mitas Ray
 # @ Create Time: 2022-01-13 17:58:00
 # @ Description: This file implements functions for bonds that have been called.
 '''

import pandas as pd

'''
This function provides the end date for a called bond.
'''
def end_date_for_called_bond(trade):
    if not pd.isnull(trade.called_redemption_date):
        return trade.called_redemption_date
    else:
        raise ValueError(f"Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) is called, but no call redemption date.")

'''
This function provides the par value for a called bond.
'''
def par_for_called_bond(trade, default_par):
    par = default_par
    if not pd.isnull(trade.refund_price):
        par = trade.refund_price
    elif not pd.isnull(trade.next_call_price):
        par = trade.next_call_price
    else:
        print(f"Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) is called, but no refund price or next call price.")    # printing instead of raising an error to not disrupt processing large quantities of trades
    return par