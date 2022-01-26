'''
 # @ Author: Jesse Watson
 # @ Create Time: 2021-10-04 09:38:00
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-01-25 15:36:08
 # @ Description: This function estimates the date used to
 # calculate the yield for the trade
 '''


import pandas as pd
import numpy as np 

def calc_end_date(trade, column_used='dollar_price'):

    '''
    This function calculates the end date (sometimes called redemption 
    or calculation date) for calculating the yield to worst given a particular
    trade. We check first whether the bond has been called, then we implicitly 
    assume the bond may be callable.  If a series conditions are not met, we price
    the bond to maturity. 
     '''

    # First we check whether the dollar price is null.  If this is the case,
    # we check for a proximate price. 

    if column_used in trade and trade[column_used] is not None: 
        approx_price = trade[column_used]
    elif 'last_price' in trade and trade.last_price is not None:
        approx_price = trade.last_price
    else:
        approx_price = trade.issue_price

    # Next we check if the bond has been called. 
    if trade.is_called:
        if not pd.isnull(trade.called_redemption_date):
    # If the bond has been called, we use ICE's called_redemption_date.
            end_date = trade.called_redemption_date
        elif pd.isnull(trade.refund_date):
            if trade.called_redemption_type in [1,5]: 
    # Called_redemption_types 1 and 5 are escrowed to maturity. 
                end_date = trade.maturity_date
            else:
                end_date = trade.next_call_date
        else:
            end_date = trade.refund_date

        if not(pd.isnull(trade.refund_date)) and trade.refund_date < trade.settlement_date:
            print("anomalous refund date:", trade.cusip, "settlement:", trade.settlement_date, "refunding:", trade.refund_date)

    # Below for bonds which may be callable. 
    
    elif approx_price < 100:
    # If price is below par, then the bond should be priced to maturity. 
        end_date = trade.maturity_date
    elif approx_price > trade.next_call_price and not pd.isnull(trade.next_call_date):
    # Next we check if price is greater than next_call_price. 
    # We check next_call_price before par_call_price in case the next call is at a premium.
        end_date = trade.next_call_date
    elif approx_price > trade.par_call_price and not pd.isnull(trade.par_call_date):
        end_date = trade.par_call_date
    else:
        end_date = trade.maturity_date
    
    return end_date



