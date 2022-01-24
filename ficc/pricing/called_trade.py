'''
 # @ Author: Mitas Ray
 # @ Create Time: 2022-01-13 17:58:00
 # @ Description: This file implements functions for bonds that have been called.
 '''

import pandas as pd

'''
This function provides the end date for a called bond. 
The variable `called_redemption_date` is a field that 
we create in the notebook create_ICE_flat. The assumptions 
that went into the creation of this field are faulty, in 
particular, it doesn't correctly deal with bonds that have 
been escrowed to maturity or pre-refunded but for which the 
call options have not been voided or defeased. The correct 
ICE field is `refund_date`, which corresponds to `refund_price` 
below. At some future date, we should change this module to 
correctly reflect this.
'''
def end_date_for_called_bond(trade):
    if not pd.isnull(trade.called_redemption_date):
        return trade.called_redemption_date
    else:
        raise ValueError(f"Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) is called, but no call redemption date.")

'''
This function provides the par value for a called bond.
'''
def refund_price_for_called_bond(trade, default_par):
    par = default_par
    if not pd.isnull(trade.refund_price):
        par = trade.refund_price
    elif not pd.isnull(trade.next_call_price):
        par = trade.next_call_price
    else:
        raise ValueError(f"Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) is called, but no refund price or next call price.")
    return par