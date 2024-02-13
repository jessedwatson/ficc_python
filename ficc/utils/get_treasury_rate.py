'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2022-09-29
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-02-13
 # @ Description:
 '''
import numpy as np
from pandas.tseries.offsets import BDay

from ficc.utils.auxiliary_functions import sqltodf
from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from ficc.utils.diff_in_days import diff_in_days_two_dates
import ficc.utils.globals as globals


def get_treasury_rate(client):
    query = '''SELECT * FROM `eng-reactor-287421.treasury_yield.daily_yield_rate` order by Date desc'''
    globals.treasury_rate = sqltodf(query, client)
    globals.treasury_rate = globals.treasury_rate.drop_duplicates(keep='first')    # from testing or manual corrections, sometimes there are duplicate entries in the table
    globals.treasury_rate.set_index('Date', drop=True, inplace=True)
    globals.treasury_rate = globals.treasury_rate.transpose().to_dict()


def get_all_treasury_rate(trade_date):
    t_rate = globals.treasury_rate[trade_date.values[0]]
    return list(t_rate.values())
    
def get_previous_treasury_difference(trade_date):
    # getting the previous business day
    day_before = (trade_date.values[0] - BDay(1)).date()
    while day_before not in globals.treasury_rate.keys():
        day_before = (day_before - BDay(1)).date()

    t_rate = np.array(list(globals.treasury_rate[trade_date.values[0]].values()))
    t_rate_before = np.array(list(globals.treasury_rate[day_before].values()))
    diff_rate = (t_rate - t_rate_before)*100
    return diff_rate.tolist()


def current_treasury_rate(trade):
    '''If the trade date corresponding to `trade` is not found in `globals.treasury_rate`, 
    then return np.nan. This is later filtered out in `process_data(...)`.'''
    trade_date = trade['trade_date']
    if trade_date not in globals.treasury_rate: return np.nan
    treasury_maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30])
    time_to_maturity = diff_in_days_two_dates(trade['calc_date'], trade['settlement_date']) / NUM_OF_DAYS_IN_YEAR
    maturity = min(treasury_maturities, key=lambda treasury_maturity: abs(treasury_maturity - time_to_maturity))
    maturity = 'year_' + str(maturity)
    t_rate = globals.treasury_rate[trade['trade_date']][maturity]
    return t_rate