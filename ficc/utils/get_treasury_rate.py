'''
Author: Ahmad Shayaan
Date: 2022-09-29
Last Editor: Mitas Ray
Last Edit Date: 2024-02-13
'''
import numpy as np

from ficc.utils.auxiliary_functions import sqltodf
from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from ficc.utils.diff_in_days import diff_in_days_two_dates


def get_treasury_rate_dict(client) -> dict:
    query = '''SELECT * FROM `eng-reactor-287421.treasury_yield.daily_yield_rate` order by Date desc'''
    treasury_rate_df = sqltodf(query, client)
    treasury_rate_df = treasury_rate_df.drop_duplicates(keep='first')    # from testing or manual corrections, sometimes there are duplicate entries in the table
    treasury_rate_df.set_index('Date', drop=True, inplace=True)
    return treasury_rate_df.transpose().to_dict()


def current_treasury_rate(treasury_rate_dict: dict, trade):
    '''If the trade date corresponding to `trade` is not found in `treasury_rate_dict`, 
    then return np.nan. This is later filtered out in `process_data(...)`.'''
    trade_date = trade['trade_date']
    if trade_date not in treasury_rate_dict: return np.nan
    treasury_maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30])
    time_to_maturity = diff_in_days_two_dates(trade['calc_date'], trade['settlement_date']) / NUM_OF_DAYS_IN_YEAR
    maturity = min(treasury_maturities, key=lambda treasury_maturity: abs(treasury_maturity - time_to_maturity))
    maturity = 'year_' + str(maturity)
    t_rate = treasury_rate_dict[trade['trade_date']][maturity]
    return t_rate
