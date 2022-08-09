
'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-17 10:40:14
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-08-09 12:46:10
 # @ Description:
 '''

import ficc.utils.globals as globals
from ficc.utils.auxiliary_functions import sqltodf

def yield_curve_params(client, yield_crurve_to_use):
    # The following fetches Nelson-Siegel coefficient and standard scalar parameters from BigQuery and sends them to a dataframe.
    
    if yield_crurve_to_use == "FICC":
        globals.nelson_params = sqltodf(
            "select * from `eng-reactor-287421.yield_curves.nelson_siegel_coef_daily` order by date desc", client)
        globals.scalar_params = sqltodf(
            "select * from`eng-reactor-287421.yield_curves.standardscaler_parameters_daily` order by date desc", client)
    
    elif yield_crurve_to_use == "FICC_NEW":
        globals.nelson_params = sqltodf(
            "select * from `eng-reactor-287421.ahmad_test.nelson_siegel_coef_daily` order by date desc", client)
        globals.scalar_params = sqltodf(
            "select * from `eng-reactor-287421.ahmad_test.standardscaler_parameters_daily` order by date desc", client)

    # The below sets the index of both dataframes to date column and converts the data type to datetime.
    globals.nelson_params.set_index("date", drop=True, inplace=True)
    globals.scalar_params.set_index("date", drop=True, inplace=True)

    # Drop rows with duplicate indices, keeping the first such row. This approach was measured fastest per https://stackoverflow.com/a/34297689
    globals.nelson_params = globals.nelson_params[~globals.nelson_params.index.duplicated(keep='first')]
    globals.scalar_params = globals.scalar_params[~globals.scalar_params.index.duplicated(keep='first')]

    # Transpose here so we can index along the longer of the two dimensions once. Otherwise we would have to index
    # the target_date once for each sub-param
    globals.nelson_params = globals.nelson_params.transpose().to_dict()
    globals.scalar_params = globals.scalar_params.transpose().to_dict()
