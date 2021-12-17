'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-15 13:59:54
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2021-12-17 11:14:30
 # @ Description: This file contains the code to get 
 # the ficc yield curve level using the calc_date
 '''


import pandas as pd
from datetime import datetime

from ficc.utils.nelson_seigel_model import yield_curve_level
import ficc.utils.globals as globals

def get_ficc_ycl(trade):
    target_date = None

    # We only have reliable data to estimate the yield curve after
    # july 27th, 2021.
    if trade.trade_date < datetime(2021, 7, 27).date():
        target_date = datetime(2021, 7, 27).date()
    else:
        target_date = trade.trade_date
    duration = (trade.calc_date - target_date).days/365.25
    ficc_yl = yield_curve_level(duration,
                                target_date.strftime('%Y-%m-%d'),
                                globals.nelson_params,
                                globals.scalar_params)
    return ficc_yl
