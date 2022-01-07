'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-15 13:59:54
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-01-07 13:48:08
 # @ Description: This file contains the code to get 
 # the ficc yield curve level using the calc_date
 '''


import pandas as pd
from datetime import datetime
import sys
from ficc.utils.nelson_seigel_model import yield_curve_level
from ficc.utils.yield_curve_params import yield_curve_params
import ficc.utils.globals as globals

def get_ficc_ycl(trade, **kwargs):
    target_date = None

    # We only have reliable data to estimate the yield curve after
    # july 27th, 2021.
    if trade.trade_date < datetime(2021, 7, 27).date():
        target_date = datetime(2021, 7, 27).date()
    else:
        target_date = trade.trade_date
    duration = (trade.calc_date - target_date).days/365.25
    
    try:
        ficc_yl = yield_curve_level(duration,
                                target_date.strftime('%Y-%m-%d'),
                                globals.nelson_params,
                                globals.scalar_params)
    except Exception as e:
        if 'client' not in kwargs:
            raise Exception("Need to provied bigquery client if being used as a stand alone function")
            sys.exit(0)
        
        bq_client = kwargs['client']
        yield_curve_params(bq_client)
        ficc_yl = yield_curve_level(duration,
                                target_date.strftime('%Y-%m-%d'),
                                globals.nelson_params,
                                globals.scalar_params)
    return ficc_yl
