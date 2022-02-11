'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2022-02-09 13:28:17
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-02-09 13:48:10
 # @ Description:
 '''

import numpy as np
import ficc.utils.globals as globals
from ficc.utils.mmd_ycl import mmd_ycl
from ficc.utils.create_mmd_data import create_mmd_data
from datetime import datetime

def get_mmd_ycl(trade, **kwargs):
    
    if 'date' in kwargs:
        trade.calc_date = kwargs['date']
    
    target_date = trade.trade_date
    duration = (trade.calc_date - target_date).days/365.25
    
    try:
        ficc_yl = mmd_ycl(target_date.strftime('%Y-%m-%d'),duration)
    
    except Exception as e:
        if 'client' not in kwargs:
            raise Exception("Need to provide bigquery client if being used as a stand alone function")
            sys.exit(0)
        
        bq_client = kwargs['client']
        create_mmd_data(bq_client)
        ficc_yl = mmd_ycl(duration,target_date.strftime('%Y-%m-%d'))
    return ficc_yl
