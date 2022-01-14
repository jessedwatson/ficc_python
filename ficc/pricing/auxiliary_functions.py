'''
 # @ Author: Mitas Ray
 # @ Create Time: 2022-01-13 17:44:00
 # @ Description: This file implements functions to help with pricing bonds
 and computing yields.
 '''

import pandas as pd

from ficc.utils.auxiliary_functions import compare_dates, dates_are_equal

'''
This function computes the next time a coupon is paid.
Note that this function could return a `next_coupon_date` that is after the end_date. 
This does not create a problem since we deal with the final coupon separately in 
`price_of_bond_with_multiple_periodic_interest_payments`.
'''
def get_next_coupon_date(first_coupon_date, start_date, time_delta):
    date = first_coupon_date
    while compare_dates(date, start_date) < 0:
        date = date + time_delta
    return date
#     cannot use the below code since division is not valid between datetime.timedelta and relativedelta, and converting between types introduces potential for errors
#     num_of_time_periods = int(np.ceil((start_date - first_coupon_date) / time_delta))    # `int` wraps the `ceil` function because the `ceil` function returns a float
#     return first_coupon_date + time_delta * num_of_time_periods

'''
This function computes the previous time a coupon was paid for this bond 
by relating it to the next coupon date.
'''
def get_previous_coupon_date(first_coupon_date, start_date, accrual_date, time_delta, next_coupon_date=None):
    if next_coupon_date == None:
        next_coupon_date = get_next_coupon_date(first_coupon_date, start_date, time_delta)
        
    if dates_are_equal(next_coupon_date, first_coupon_date):
        return accrual_date
    return next_coupon_date - time_delta

'''
This function is valid for bonds that don't pay coupons, whereas the previous 
two functions assume the bond pays coupons.
'''
def get_prev_coupon_date_and_next_coupon_date(trade, frequency, time_delta):
    if frequency == 0:
        my_next_coupon_date = trade.maturity_date
        my_prev_coupon_date = trade.accrual_date
    else:
        if pd.isnull(trade.next_coupon_payment_date):
            my_next_coupon_date = get_next_coupon_date(trade.first_coupon_date, trade.settlement_date, time_delta)
        else:
            my_next_coupon_date = pd.to_datetime(trade.next_coupon_payment_date)

        if pd.isnull(trade.previous_coupon_payment_date):
            my_prev_coupon_date = get_previous_coupon_date(trade.first_coupon_date, trade.settlement_date, trade.accrual_date, time_delta, my_next_coupon_date)
        else:
            my_prev_coupon_date = pd.to_datetime(trade.previous_coupon_payment_date)

    return my_prev_coupon_date, my_next_coupon_date

'''
This function returns the number of interest payments and the final coupon 
date based on the next coupon date, the end date, and the gap between coupon 
payments. This function returns both together because one is always a 
byproduct of computing the other.
'''
def get_num_of_interest_payments_and_final_coupon_date(next_coupon_date, end_date, time_delta): 
    if compare_dates(next_coupon_date, end_date) > 0:
        return 0, next_coupon_date    # return 1, end_date (would be valid in isolation)
    
    num_of_interest_payments = 1
    final_coupon_date = next_coupon_date
    while compare_dates(final_coupon_date + time_delta, end_date) <= 0:
        num_of_interest_payments += 1
        final_coupon_date += time_delta
    return num_of_interest_payments, final_coupon_date