'''
Author: Ahmad Shayaan
Date: 2021-12-15
Last Editor: Mitas Ray
Last Edit Date: 2025-05-06
Description: This file contains the code to get the ficc yield curve level using the calc_date. Currently unused, but kept for reference.
'''
from datetime import datetime

from ficc.utils.nelson_siegel_model import yield_curve_level
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR


def get_ficc_ycl(trade, nelson_params, scalar_params, shape_parameter, **kwargs):
    if 'date' in kwargs:
        calc_date = kwargs['date']
    else:
        calc_date = trade['calc_date']
    
    target_datetime = trade['trade_datetime']
    if trade['trade_datetime'] < datetime(2021, 7, 27):    # we only have reliable data to estimate the yield curve after 2021-07-27
        target_datetime = datetime(2021, 7, 27)

    duration = diff_in_days_two_dates(calc_date.date(), target_datetime.date()) / NUM_OF_DAYS_IN_YEAR
    return yield_curve_level(duration, target_datetime, nelson_params, scalar_params, shape_parameter)
