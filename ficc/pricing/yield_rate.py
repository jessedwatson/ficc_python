'''
 # @ Author: Mitas Ray
 # @ Create Time: 2022-01-13 23:20:00
 # @ Description: This file implements functions to compute the price of a trade
 # given the yield.
 '''
import pandas as pd
import scipy.optimize as optimize

from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from ficc.utils.auxiliary_functions import compare_dates
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.frequency import get_time_delta_from_interest_frequency
from ficc.utils.truncation import trunc_and_round_yield
from ficc.pricing.auxiliary_functions import get_num_of_interest_payments_and_final_coupon_date, \
                                             price_of_bond_with_multiple_periodic_interest_payments, \
                                             get_prev_coupon_date_and_next_coupon_date
from ficc.pricing.called_trade import end_date_for_called_bond, par_for_called_bond

'''
This function is a helper function for `compute_yield`. This function calculates the yield of a trade given the price and other trade features, using
the MSRB Rule Book G-33 linked at: https://www.msrb.org/pdf.aspx?url=https%3A%2F%2Fwww.msrb.org%2FRules-and-Interpretations%2FMSRB-Rules%2FGeneral%2FRule-G-33.aspx.

When referring to the formulas in the MSRB handbook (link above), the below variables map to the code.
B: NUM_OF_DAYS_IN_YEAR
E: number of days in interest payment period
M: frequency
N: num_of_interest_payments
R: coupon
'''
def get_yield(cusip, 
              prev_coupon_date, 
              first_coupon_date, 
              next_coupon_date, 
              end_date, 
              settlement_date, 
              accrual_date,
              frequency, 
              price, 
              coupon, 
              RV, 
              time_delta, 
              last_period_accrues_from_date):
    settlement_date_to_end_date = diff_in_days_two_dates(end_date, settlement_date)    # hold period days

    if coupon != 0 and frequency != 0:    # coupon paid every M periods
        num_of_interest_payments, final_coupon_date = get_num_of_interest_payments_and_final_coupon_date(next_coupon_date, 
                                                                                                         end_date, 
                                                                                                         time_delta)
        prev_coupon_date_to_settlement_date = diff_in_days_two_dates(settlement_date, prev_coupon_date)    # accrued days from beginning of the interest payment period, used to be labelled `A`
        prev_coupon_date_to_end_date = diff_in_days_two_dates(end_date, prev_coupon_date)    # accrual days for final paid coupon

        if compare_dates(end_date, next_coupon_date) <= 0:
            # MSRB Rule Book G-33, rule (b)(ii)(B)(1)
            # Recall: number of interest payments per year * number of days in interest payment period = number of days in a year, i.e., E * M = NUM_OF_DAYS_IN_YEAR
            yield_estimate = (((RV + (coupon * prev_coupon_date_to_end_date / NUM_OF_DAYS_IN_YEAR)) / \
                               (price + (coupon * prev_coupon_date_to_settlement_date / NUM_OF_DAYS_IN_YEAR))) - 1) * \
                             (NUM_OF_DAYS_IN_YEAR / settlement_date_to_end_date)
        else:
            # MSRB Rule Book G-33, rule (b)(ii)(B)(2)
            ytm_func = lambda Y: -price + price_of_bond_with_multiple_periodic_interest_payments(cusip, 
                                                                                                 settlement_date, 
                                                                                                 accrual_date, 
                                                                                                 first_coupon_date, 
                                                                                                 prev_coupon_date, 
                                                                                                 next_coupon_date, 
                                                                                                 final_coupon_date, 
                                                                                                 end_date,  
                                                                                                 frequency,
                                                                                                 num_of_interest_payments, 
                                                                                                 Y,
                                                                                                 coupon, 
                                                                                                 RV, 
                                                                                                 time_delta, 
                                                                                                 last_period_accrues_from_date)
            try:
                guess = 0.01
                yield_estimate = optimize.newton(ytm_func, guess, maxiter=100)
            except Exception as e:
                print(e)
                return None
    elif coupon == 0:    # THIS LOGIC IS CURRENTLY UNTESTED
        # MSRB Rule Book G-33, rule (b)(ii)(A), since coupon == 0, the formula is simplified with R == 0
        yield_estimate = ((RV - price) / price) * (NUM_OF_DAYS_IN_YEAR / settlement_date_to_end_date)
    return trunc_and_round_yield(yield_estimate * 100)

'''
This function computes the yield of a trade.
'''
def compute_yield(trade):
    frequency = trade.interest_payment_frequency
    time_delta = get_time_delta_from_interest_frequency(frequency)
    my_prev_coupon_date, my_next_coupon_date = get_prev_coupon_date_and_next_coupon_date(trade, frequency, time_delta)

    get_yield_caller = lambda end_date, par: get_yield(trade.cusip, 
                                                       my_prev_coupon_date, 
                                                       trade.first_coupon_date, 
                                                       my_next_coupon_date,
                                                       end_date, 
                                                       trade.settlement_date, 
                                                       trade.accrual_date, 
                                                       trade.interest_payment_frequency,
                                                       trade.dollar_price, 
                                                       trade.coupon, 
                                                       par, 
                                                       time_delta, 
                                                       trade.last_period_accrues_from_date)

    par = 100
    if (not trade.is_called) and (not trade.is_callable):
        end_date = trade.maturity_date
        yield_to_maturity = get_yield_caller(end_date, par)
        return yield_to_maturity, trade.maturity_date
    
    if trade.is_called:
        end_date = end_date_for_called_bond(trade)
        par = par_for_called_bond(trade, par)
        yield_to_maturity = get_yield_caller(end_date, par)
        return yield_to_maturity, trade.called_redemption_date
    else:
        yield_to_next_call = float("inf")
        yield_to_maturity = float("inf")
        yta = float("inf")
        yield_to_par_call = float("inf")
        
        if not pd.isnull(trade.par_call_date):
            end_date = trade.par_call_date
            par = trade.par_call_price    
            yield_to_par_call = get_yield_caller(end_date, par)

        end_date = trade.next_call_date
        par = trade.next_call_price
        yield_to_next_call = get_yield_caller(end_date, par)
        
        end_date = trade.maturity_date
        par = 100
        yield_to_maturity = get_yield_caller(end_date, par)
        
        dict_yields = {"yield_to_next_call": yield_to_next_call, 
                       "yield_to_maturity": yield_to_maturity, 
                       "yta": yta, 
                       "yield_to_par_call": yield_to_par_call}
        our_choice = min(dict_yields, key=dict_yields.get)
        date_dict = {"yield_to_next_call": trade.next_call_date, 
                     "yield_to_maturity": trade.maturity_date, 
                     "yta": "error", 
                     "yield_to_par_call": trade.par_call_date}
        return (dict_yields[our_choice], date_dict[our_choice])