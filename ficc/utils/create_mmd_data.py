'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2022-02-03 15:43:05
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-02-08 19:25:20
 # @ Description:
 '''

import pandas as pd
import numpy as np
from ficc.utils.auxiliary_functions import sqltodf
import ficc.utils.globals as globals


MMD_HISTORICAL_QUERY = """ Select * from eng-reactor-287421.yield_curves.new_mmd_approximation"""

def create_mmd_data(client):
    mmd_hist = sqltodf(MMD_HISTORICAL_QUERY, client)
    mmd_hist.Date = pd.to_datetime(mmd_hist.Date)
    mmd_hist.set_index('Date', inplace=True, drop=True)
    globals.mmd_ycl = mmd_hist

