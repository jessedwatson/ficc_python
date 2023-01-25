'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-20 10:00:17
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2023-01-19 15:53:08
 # @ Description: This file implements a function to calculate the difference in 
 # days between two days in accordance to the provision of MSRB rule 33G
 '''

import pandas as pd


'''This function calculates the difference in days using the 360/30 
convention specified in MSRB Rule Book G-33, rule (e).'''
def _diff_in_days_two_dates_360_30(end_date, start_date):
    Y2 = end_date.year
    Y1 = start_date.year
    M2 = end_date.month
    M1 = start_date.month
    D2 = end_date.day
    D1 = start_date.day
    D1 = min(D1, 30)
    if D1 == 30: 
        D2 = min(D2, 30)
    return (Y2 - Y1) * 360 + (M2 - M1) * 30 + (D2 - D1)


def _diff_in_days_two_dates_exact(end_date, start_date):
    diff = end_date - start_date
    if isinstance(diff, pd.Series): return diff.dt.days    # https://stackoverflow.com/questions/60879982/attributeerror-timedelta-object-has-no-attribute-dt
    else: return diff.days


ACCEPTED_CONVENTIONS = {'360/30': _diff_in_days_two_dates_360_30, 
                        'exact': _diff_in_days_two_dates_exact}


def diff_in_days_two_dates(end_date, start_date, convention="360/30"):
    if convention not in ACCEPTED_CONVENTIONS:
        print("unknown convention", convention)
        return None
    return ACCEPTED_CONVENTIONS[convention](end_date, start_date)

    
def diff_in_days(trade, convention="360/30", **kwargs):
    #See MSRB Rule 33-G for details
    if 'calc_type' in kwargs:
        if kwargs['calc_type'] == 'accrual' and not pd.isnull(trade.accrual_date):
            start_date = trade.accrual_date
            end_date = trade.settlement_date
        else:
            raise ValueError('Invalid arguments')
    else:
        start_date = trade.dated_date
        end_date = trade.settlement_date

    return diff_in_days_two_dates(end_date, start_date, convention)