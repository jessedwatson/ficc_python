'''
 # @ Author: Mitas Ray
 # @ Create Time: 2022-01-13 23:04:00
 # @ Modified by: Mitas Ray
 # @ Modified time: 2022-01-24 12:17:00
 # @ Description: This file implements functions to compute the price of a trade
 # given the yield.
 '''
import numpy as np
import pandas as pd

from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from ficc.utils.auxiliary_functions import compare_dates
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.frequency import get_time_delta_from_interest_frequency
from ficc.utils.truncation import trunc_and_round_price
from ficc.pricing.auxiliary_functions import get_num_of_interest_payments_and_final_coupon_date, \
                                             price_of_bond_with_multiple_periodic_interest_payments, \
                                             get_prev_coupon_date_and_next_coupon_date
from ficc.pricing.called_trade import end_date_for_called_bond, refund_price_for_called_bond

'''
This function is a helper function for `compute_price`. This function calculates the price of a trade, where `yield_rate` 
is a specific yield and `end_date` is a fixed repayment date. All dates must be valid relative to the settlement 
date, as opposed to the trade date. Note that "yield" is a reserved word in Python and should not be used as the name 
of a variable or column.

Formulas are from https://www.msrb.org/pdf.aspx?url=https%3A%2F%2Fwww.msrb.org%2FRules-and-Interpretations%2FMSRB-Rules%2FGeneral%2FRule-G-33.aspx.
For all bonds, `base` is the present value of future cashflows to the buyer. 
The clean price is this price minus the accumulated amount of simple interest that the buyer must pay to the seller, which is called `accrued`.
Zero-coupon bonds are handled first. For these, the yield is assumed to be compounded semi-annually, i.e., once every six months.
For bonds with non-zero coupon, the first and last interest payment periods may have a non-standard length,
so they must be handled separately.

When referring to the formulas in the MSRB handbook (link above), the below variables map to the code.
A: prev_coupon_date_to_settlement_date
B: NUM_OF_DAYS_IN_YEAR
Y: yield_rate
N: num_of_interest_payments
E: num_of_days_in_period
F: settlement_date_to_next_coupon_date
P: price
D: settlement_date_to_end_date
H: prev_coupon_date_to_end_date
R: coupon
'''
def get_price(cusip, 
              prev_coupon_date, 
              first_coupon_date, 
              next_coupon_date, 
              end_date, 
              settlement_date, 
              accrual_date, 
              frequency, 
              yield_rate, 
              coupon, 
              RV, 
              time_delta, 
              last_period_accrues_from_date):
    if pd.isnull(end_date):
        return np.inf
    
    yield_rate = yield_rate / 100
    
    # Right now we do not disambiguate zero coupon from interest at maturity. More specfically, 
    # we should add logic that separates the cases of MSRB Rule Book G-33, rule (b) and rule (c)
    if frequency == 0:
        # MSRB Rule Book G-33, rule (b)(i)(A)
        accrual_date_to_settlement_date = diff_in_days_two_dates(settlement_date, accrual_date)
        settlement_date_to_end_date = diff_in_days_two_dates(end_date, settlement_date)
        base = (RV + (settlement_date_to_end_date / NUM_OF_DAYS_IN_YEAR)) / \
               (1 + (settlement_date_to_end_date - accrual_date_to_settlement_date) / NUM_OF_DAYS_IN_YEAR * yield_rate)
        accrued = coupon * accrual_date_to_settlement_date / NUM_OF_DAYS_IN_YEAR
        price = base - accrued
    else:
        num_of_interest_payments, final_coupon_date = get_num_of_interest_payments_and_final_coupon_date(next_coupon_date, 
                                                                                                         end_date, 
                                                                                                         time_delta)
        prev_coupon_date_to_settlement_date = diff_in_days_two_dates(settlement_date, prev_coupon_date)
            
        num_of_days_in_period = NUM_OF_DAYS_IN_YEAR / frequency    # number of days in interest payment period 
        assert num_of_days_in_period == round(num_of_days_in_period)
         
        if compare_dates(end_date, next_coupon_date) <= 0:
            # MSRB Rule Book G-33, rule (b)(i)(B)(1)
            settlement_date_to_end_date = diff_in_days_two_dates(end_date, settlement_date)
            final_coupon_date_to_end_date = diff_in_days_two_dates(end_date, final_coupon_date)
            interest_due_at_end_date = coupon * final_coupon_date_to_end_date / NUM_OF_DAYS_IN_YEAR
            base = (RV + coupon / frequency + interest_due_at_end_date) / \
                   (1 + (yield_rate / frequency) * settlement_date_to_end_date / num_of_days_in_period)
            accrued = coupon * prev_coupon_date_to_settlement_date / NUM_OF_DAYS_IN_YEAR
            price = base - accrued
        else:
            # MSRB Rule Book G-33, rule (b)(i)(B)(2)
            price = price_of_bond_with_multiple_periodic_interest_payments(cusip, 
                                                                           settlement_date, 
                                                                           accrual_date, 
                                                                           first_coupon_date, 
                                                                           prev_coupon_date, 
                                                                           next_coupon_date, 
                                                                           final_coupon_date, 
                                                                           end_date,  
                                                                           frequency,
                                                                           num_of_interest_payments, 
                                                                           yield_rate,
                                                                           coupon, 
                                                                           RV, 
                                                                           time_delta, 
                                                                           last_period_accrues_from_date)              
    return trunc_and_round_price(price)

'''
This function computes the price of a trade. For bonds that have not been called, the price is the lowest of
three present values: to the next call date (which may be above par), to the next par call date, and to maturity.
'''
def compute_price(trade, yield_rate=None):
    if yield_rate == None:
        yield_rate = trade['yield']
    elif type(yield_rate) == str:
        raise ValueError('Yield rate argument cannot be a string. It must be a numerical value.')

    frequency = trade.interest_payment_frequency
    time_delta = get_time_delta_from_interest_frequency(frequency)
    my_prev_coupon_date, my_next_coupon_date = get_prev_coupon_date_and_next_coupon_date(trade, frequency, time_delta)

    par = 100    # can we rewrite this to not hard code the par value?
    if trade.is_called:
        end_date = end_date_for_called_bond(trade)
        if compare_dates(end_date, trade.settlement_date) < 0:
            print(f"Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) has an end date ({end_date}) which is after the settlement date ({trade.settlement_date}).")    # printing instead of raising an error to not disrupt processing large quantities of trades
            # raise ValueError(f"Bond (CUSIP: {trade.cusip}, RTRS: {trade.rtrs_control_number}) has an end date ({end_date}) which is after the settlement date ({trade.settlement_date}).")
            
        par = refund_price_for_called_bond(trade, par)
    else:
        end_date = trade.maturity_date    # not used later

    get_price_caller = lambda end_date, par: get_price(trade.cusip, 
                                                       my_prev_coupon_date, 
                                                       trade.first_coupon_date, 
                                                       my_next_coupon_date, 
                                                       end_date, 
                                                       trade.settlement_date, 
                                                       trade.accrual_date, 
                                                       frequency, 
                                                       yield_rate, 
                                                       trade.coupon, 
                                                       par, 
                                                       time_delta, 
                                                       trade.last_period_accrues_from_date)

    if trade.is_called:
        final = get_price_caller(end_date, par)
        calc = end_date
    else:
        next_price = get_price_caller(trade.next_call_date, trade.next_call_price)
        to_par_price = get_price_caller(trade.par_call_date, trade.par_call_price)
        maturity_price = get_price_caller(trade.maturity_date, 100)

        prices_and_dates = [(next_price, trade.next_call_date), 
                            (to_par_price, trade.par_call_date), 
                            (maturity_price, trade.maturity_date)]
        final, calc = min(prices_and_dates, key=lambda x:x[0])    # this function is stable and will choose the tuple which appears first in the case of ties with the sorting condition
    return final, calc
