
'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-17 10:40:14
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2023-10-11 17:23:51
 # @ Description:
 '''

import ficc.utils.globals as globals
from ficc.utils.auxiliary_functions import sqltodf
import pandas as pd

def yield_curve_params(client, yield_curve_to_use):
    # The following fetches Nelson-Siegel coefficient and standard scalar parameters from BigQuery and sends them to a dataframe.
    
    if yield_curve_to_use == "FICC":
        globals.nelson_params = sqltodf(
            "select * from `eng-reactor-287421.yield_curves.nelson_siegel_coef_daily` order by date desc", client)
        globals.scalar_params = sqltodf(
            "select * from`eng-reactor-287421.yield_curves.standardscaler_parameters_daily` order by date desc", client)
    
    elif yield_curve_to_use == "FICC_NEW":
        globals.nelson_params = sqltodf(
            "select * from `eng-reactor-287421.ahmad_test.nelson_siegel_coef_daily` order by date desc", client)
        globals.scalar_params = sqltodf(
            "select * from `eng-reactor-287421.ahmad_test.standardscaler_parameters_daily` order by date desc", client)

    globals.shape_parameter = sqltodf("SELECT *  FROM `eng-reactor-287421.ahmad_test.shape_parameters` order by Date desc", client)

    import gcsfs
    fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
    with fs.open('ahmad_data/historical_yield_curves.csv') as f:
        df = pd.read_csv(f)

    df.date = pd.to_datetime(df.date).dt.date
    df = df.sort_values('date',ascending=False)
    

    temp_nelson_params = df[['date','const','exponential','laguerre']].copy()
    temp_scalar_params = df[['date','exponential_mean','exponential_std','laguerre_mean','laguerre_std']].copy()
    temp_shape_parameter = df[['date','L']].copy()
    temp_shape_parameter = temp_shape_parameter.rename(columns={'date':'Date'})

    globals.nelson_params = pd.concat([globals.nelson_params, temp_nelson_params])
    globals.scalar_params = pd.concat([globals.scalar_params, temp_scalar_params])
    globals.shape_parameter = pd.concat([globals.shape_parameter, temp_shape_parameter])

    # The below sets the index of dataframes to date column and converts the data type to datetime.
    globals.nelson_params.set_index("date", drop=True, inplace=True)
    globals.scalar_params.set_index("date", drop=True, inplace=True)
    globals.shape_parameter.set_index("Date", drop=True, inplace=True)

    # Drop rows with duplicate indices, keeping the first such row. This approach was measured fastest per https://stackoverflow.com/a/34297689
    globals.nelson_params = globals.nelson_params[~globals.nelson_params.index.duplicated(keep='first')]
    globals.scalar_params = globals.scalar_params[~globals.scalar_params.index.duplicated(keep='first')]
    globals.shape_parameter = globals.shape_parameter[~globals.shape_parameter.index.duplicated(keep='first')]

    globals.nelson_params = globals.nelson_params.transpose().to_dict()
    globals.scalar_params = globals.scalar_params.transpose().to_dict()
    globals.shape_parameter = globals.shape_parameter.transpose().to_dict()
