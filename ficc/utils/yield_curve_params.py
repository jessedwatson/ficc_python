'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-17 10:40:14
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-01-28 12:30:43
 # @ Description:
 '''

import pandas as pd

import ficc.utils.globals as globals
from ficc.utils.auxiliary_functions import sqltodf

def yield_curve_params(client):
    # The following fetches Nelson-Siegel coefficient and standard scalar parameters from BigQuery and sends them to a dataframe.
    globals.nelson_params = sqltodf(
        "select * from `eng-reactor-287421.yield_curves.nelson_siegel_coef_daily` order by date desc", client)
    globals.scalar_params = sqltodf(
        "select * from`eng-reactor-287421.yield_curves.standardscaler_parameters_daily` order by date desc", client)

    # The below sets the index of both dataframes to date column and converts the data type to datetime.
    globals.nelson_params.set_index("date", drop=True, inplace=True)
    globals.scalar_params.set_index("date", drop=True, inplace=True)

    # Drop rows with duplicate indices, keeping the first such row. This approach was measured fastest per https://stackoverflow.com/a/34297689
    globals.nelson_params = globals.nelson_params[~globals.nelson_params.index.duplicated(keep='first')]
    globals.scalar_params = globals.scalar_params[~globals.scalar_params.index.duplicated(keep='first')]

    globals.nelson_params = globals.nelson_params.to_dict()
    globals.scalar_params = globals.scalar_params.to_dict()
