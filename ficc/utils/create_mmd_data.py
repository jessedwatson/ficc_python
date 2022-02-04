'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2022-02-03 15:43:05
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-02-03 20:18:59
 # @ Description:
 '''

import pandas as pd
import numpy as np
from ficc.utils.auxiliary_functions import sqltodf
import ficc.utils.globals as globals


MMD_HISTORICAL_QUERY = """ Select * from eng-reactor-287421.yield_curves.mmd_approximation_history """
MMD_APPROX_QUERY = """ SELECT *  FROM eng-reactor-287421.yield_curves.mmd_approximation  """

def create_mmd_data(client):

    mmd_hist = sqltodf(MMD_HISTORICAL_QUERY, client)[['close_date','MMD_5Y','MMD_10Y']]
    mmd_hist = mmd_hist.sort_values('close_date',ascending=False)
    mmd_hist.close_date = pd.to_datetime(mmd_hist.close_date).dt.date
    mmd_hist.MMD_5Y = mmd_hist.MMD_5Y.astype('float')
    mmd_hist.MMD_10Y = mmd_hist.MMD_10Y.astype('float')
    mmd_hist.set_index('close_date',inplace=True)

    mmd_approx = sqltodf(MMD_APPROX_QUERY, client)[['date','Maturity','AAA']]
    mmd_approx.date = pd.to_datetime(mmd_approx.date).dt.date
    mmd_approx.AAA = mmd_approx.AAA.astype('float')

    data = []
    for i in [5,10]:
        temp_df = mmd_approx[mmd_approx.Maturity == i].sort_values('date',ascending=False)
        if i == 30:
            data.append(list(temp_df['AAA'].values))
        else:
            data.append(list(temp_df['AAA'].values))

    index = list(mmd_approx[mmd_approx.Maturity == 5].sort_values('date',ascending=False)['date'].values)
    processed_mmd_data = pd.DataFrame(data).T
    processed_mmd_data.rename(columns={0:'MMD_5Y',1:"MMD_10Y"}, inplace=True)
    processed_mmd_data.index = index

    combined_mmd = pd.concat([processed_mmd_data, mmd_hist])

    missing_data = pd.DataFrame({'MMD_5Y': [0.63, 0.63, 0.63, 0.63, 0.63, 0.63 ],
                             'MMD_10Y': [1.13, 1.13, 1.13, 1.13, 1.13, 1.13]},
                           index = ['2021-11-02','2021-11-03','2021-11-04','2021-11-05','2021-11-06','2021-11-07'])
    missing_data.index = pd.to_datetime(missing_data.index).date
    combined_mmd = pd.concat([combined_mmd, missing_data])
    combined_mmd.sort_index(inplace=True)
    combined_mmd.index = pd.to_datetime(combined_mmd.index)
    globals.mmd_ycl = combined_mmd