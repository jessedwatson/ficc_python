'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-20 10:00:17
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2021-12-20 10:42:56
 # @ Description: This file implements a function to calculate the difference in 
 # days between two days in accordance to the provision of MSRB rule 33G
 '''

import pandas as pd

def diff_in_days(trade,convention="360/30",**kwargs):
    #See MSRB Rule 33-G for details
    if 'calc_type' in kwargs:
        if kwargs['calc_type'] == 'accural' and not pd.isnull(trade.accrual_date):
            start_date = trade.accrual_date
            end_date = trade.settlement_date
        else:
            start_date = trade.dated_date
            end_date = trade.settlement_date

    Y2 = end_date.year
    Y1 = start_date.year
    M2 = end_date.month
    M1 = start_date.month
    D2 = end_date.day #(end_date - relativedelta(days=1)).day 
    D1 = start_date.day
    if convention == "360/30":
        D1 = min(D1, 30)
        if D1 == 30: D2 = min(D2,30)
        difference_in_days = (Y2 - Y1) * 360 + (M2 - M1) * 30 + (D2 - D1)
    else: 
        print("unknown convention", convention)
    return difference_in_days
