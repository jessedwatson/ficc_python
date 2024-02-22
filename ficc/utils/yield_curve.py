'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2021-12-15
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-02-13
 # @ Description: This file contains the code to get the ficc yield curve level using the calc_date.
 '''
from datetime import datetime

from ficc.utils.nelson_siegel_model import yield_curve_level
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR


def get_ficc_ycl(trade, nelson_params, scalar_params, shape_parameter, **kwargs):
    target_date = None
    if 'date' in kwargs: trade.calc_date = kwargs['date']
    
    try:    # we only have reliable data to estimate the yield curve after 2021-07-27
        if trade.trade_date < datetime(2021, 7, 27).date()  :
            target_date = datetime(2021, 7, 27).date()
        else:
            target_date = trade.trade_date
    except Exception:
        print('Cannot compare timestamp to date, trying with datetime')
        if trade.trade_date < datetime(2021, 7, 27):
            target_date = datetime(2021, 7, 27)
        else:
            target_date = trade.trade_date

    try:
        duration = diff_in_days_two_dates(trade.calc_date, target_date) / NUM_OF_DAYS_IN_YEAR
    except Exception:
        duration = diff_in_days_two_dates(trade.calc_date.date(),target_date) / NUM_OF_DAYS_IN_YEAR
    
    ficc_ycl = yield_curve_level(duration,
                                 target_date,
                                 nelson_params,
                                 scalar_params,
                                 shape_parameter)
    return ficc_ycl
