'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2022-09-29 14:41:45
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-10-07 11:37:02
 # @ Description:
 '''

import numpy as np

from ficc.utils.auxiliary_functions import sqltodf
from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from ficc.utils.diff_in_days import diff_in_days_two_dates
import ficc.utils.globals as globals


def get_treasury_rate(client):
    query = '''SELECT * FROM `eng-reactor-287421.treasury_yield.daily_yield_rate` order by Date desc;'''
    globals.treasury_rate = sqltodf(query, client)
    globals.treasury_rate.set_index("Date", drop=True, inplace=True)
    globals.treasury_rate = globals.treasury_rate.transpose().to_dict()

def current_treasury_rate(trade):
    treasury_maturities = np.array([1,2,3,5,7,10,20,30])
    time_to_maturity = diff_in_days_two_dates(trade['calc_date'],trade['trade_date'])/NUM_OF_DAYS_IN_YEAR
    maturity = min(treasury_maturities, key=lambda x:abs(x-time_to_maturity))
    maturity = 'year_'+str(maturity)
    t_rate = globals.treasury_rate[trade['trade_date']][maturity]
    return t_rate