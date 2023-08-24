'''
 # @ Author: Mitas Ray
 # @ Create Time: 2022-01-13 17:58:00
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2023-08-10 12:24:38
 # @ Description: This file implements functions for bonds that have been called.
 '''
import pandas as pd


def end_date_for_called_bond(trade):
    '''This function provides the end date for a called bond.'''
    if not pd.isnull(trade.refund_date):
        return trade.refund_date
    else:
        raise ValueError(f'Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) is called, but no refund date.')


def refund_price_for_called_bond(trade):
    '''This function provides the par value for a called bond.'''
    if not pd.isnull(trade.refund_price):
        return trade.refund_price
    else:
        raise ValueError(f'Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) is called, but no refund price.')