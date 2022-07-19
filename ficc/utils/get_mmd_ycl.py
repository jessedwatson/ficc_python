'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2022-02-09 13:28:17
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-07-19 16:00:00
 # @ Description:
 '''

import numpy as np
import ficc.utils.globals as globals
from ficc.utils.mmd_ycl import mmd_ycl
from ficc.utils.create_mmd_data import create_mmd_data
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from datetime import datetime

def get_mmd_ycl(trade, **kwargs):
    
    if 'date' in kwargs:
        trade.calc_date = kwargs['date']
    
    target_date = trade.trade_date
    duration = diff_in_days_two_dates(trade.calc_date, target_date)
    
    try:
        ficc_yl = mmd_ycl(target_date,duration)
    
    except Exception as e:
        print(trade.rtrs_control_number, trade.calc_date, target_date, duration)
        raise e
        if 'client' not in kwargs:
            raise Exception("Need to provide bigquery client if being used as a stand alone function")
            sys.exit(0)
        
        bq_client = kwargs['client']
        create_mmd_data(bq_client)
        ficc_yl = mmd_ycl(duration,target_date.strftime('%Y-%m-%d'))
    return ficc_yl
