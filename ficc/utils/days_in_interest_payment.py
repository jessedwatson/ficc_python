'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-20 10:45:07
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2021-12-20 10:56:33
 # @ Description: This file implements the function that calculates
 #  the number of days in the interest payment period
 '''

from ficc.utils.auxiliary_variables import COUPON_FREQUENCY_TYPE, LARGE_NUMBER

def days_in_interest_payment(trade):
    frequency = COUPON_FREQUENCY_TYPE[trade['interest_payment_frequency']]
    if frequency == 0:
        frequency = LARGE_NUMBER
    return 360 / frequency