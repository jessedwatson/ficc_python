'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-15 13:59:54
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2023-01-27 14:34:54
 # @ Description: This file contains the code to get 
 # the ficc yield curve level using the calc_date
 '''


from datetime import datetime
from ficc.utils.nelson_siegel_model import yield_curve_level
from ficc.utils.yield_curve_params import yield_curve_params
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
import ficc.utils.globals as globals

def get_ficc_ycl(trade, **kwargs):
    target_date = None
    
    if 'date' in kwargs:
        trade.calc_date = kwargs['date']
    
    # We only have reliable data to estimate the yield curve after
    # july 27th, 2021.
    try:
        if trade.trade_date < datetime(2021, 7, 27).date()  :
            target_date = datetime(2021, 7, 27).date()
        else:
            target_date = trade.trade_date
    except Exception as e:
        print("Cannot compare timestamp to date, trying with datetime")
        if trade.trade_date < datetime(2021, 7, 27):
            target_date = datetime(2021, 7, 27)
        else:
            target_date = trade.trade_date

    try:
        duration = diff_in_days_two_dates(trade.calc_date,target_date)/NUM_OF_DAYS_IN_YEAR
    except Exception as e:
        duration = diff_in_days_two_dates(trade.calc_date.date(),target_date)/NUM_OF_DAYS_IN_YEAR
    
    try:
        ficc_ycl = yield_curve_level(duration,
                                    target_date,
                                    globals.nelson_params,
                                    globals.scalar_params,
                                    globals.shape_parameter)

        ficc_ycl_3_month = yield_curve_level(0.25,
                                    target_date,
                                    globals.nelson_params,
                                    globals.scalar_params,
                                    globals.shape_parameter)                                    

        ficc_ycl_1_month  = yield_curve_level(1/12,
                                    target_date,
                                    globals.nelson_params,
                                    globals.scalar_params,
                                    globals.shape_parameter)

        return ficc_ycl, ficc_ycl_3_month, ficc_ycl_1_month
        
    except Exception as e:
        raise e
        if 'client' not in kwargs:
            raise Exception("Need to provide bigquery client if being used as a stand alone function")
            
        bq_client = kwargs['client']
        yield_curve_params(bq_client,'FICC_NEW')
        ficc_yl = yield_curve_level(duration,
                                    target_date,
                                    globals.nelson_params,
                                    globals.scalar_params,
                                    globals.shape_parameter)
        return ficc_yl
