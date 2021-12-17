'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-15 13:59:54
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2021-12-16 15:09:15
 # @ Description:
 '''


import pandas as pd
from datetime import datetime

from ficc.utils.core import sqltodf
from ficc.utils.yield_value import yield_curve_level
from ficc.utils.ficc_calc_end_date import calc_end_date
import ficc.utils.globals as globals

def yield_curve_params(client):
    # The following fetches Nelson-Siegel coefficient and standard scalar parameters from BigQuery and sends them to a dataframe.
    globals.nelson_params = sqltodf(
        "select * from `eng-reactor-287421.yield_curves.nelson_siegel_coef_daily` order by date desc", client)
    globals.scalar_params = sqltodf(
        "select * from`eng-reactor-287421.yield_curves.standardscaler_parameters_daily` order by date desc", client)

    # The below sets the index of both dataframes to date column and converts the data type to datetime.
    globals.nelson_params.set_index("date", drop=True, inplace=True)
    globals.scalar_params.set_index("date", drop=True, inplace=True)
    globals.scalar_params.index = pd.to_datetime(globals.scalar_params.index)
    globals.nelson_params.index = pd.to_datetime(globals.nelson_params.index)


def get_ficc_ycl(trade):
    target_date = None

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
