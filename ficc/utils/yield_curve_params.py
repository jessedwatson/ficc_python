
'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-17 10:40:14
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2023-10-11 17:23:51
 # @ Description:
 '''
import pandas as pd
import gcsfs

from ficc.utils.auxiliary_functions import sqltodf


def yield_curve_params(client, yield_curve_to_use):
    supported_yield_curves = ('FICC', 'FICC_NEW')
    assert yield_curve_to_use in supported_yield_curves, f'Yield curve of {yield_curve_to_use} is not supported. Supported yield curves: {supported_yield_curves}'
    table_name = 'yield_curves' if yield_curve_to_use == 'FICC' else 'ahmad_test'
    nelson_params = sqltodf(f'SELECT * FROM `eng-reactor-287421.{table_name}.nelson_siegel_coef_daily` ORDER BY date DESC', client)
    scalar_params = sqltodf(f'SELECT * FROM `eng-reactor-287421.{table_name}.standardscaler_parameters_daily` ORDER BY date DESC', client)
    shape_parameter = sqltodf('SELECT *  FROM `eng-reactor-287421.ahmad_test.shape_parameters` order by Date desc', client)
    
    fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
    with fs.open('ahmad_data/historical_yield_curves.csv') as file:
        historical_yield_curves = pd.read_csv(file)

    historical_yield_curves.date = pd.to_datetime(historical_yield_curves.date).dt.date
    historical_yield_curves = historical_yield_curves.sort_values('date', ascending=False)
    
    temp_nelson_params = historical_yield_curves[['date', 'const', 'exponential', 'laguerre']].copy()
    temp_scalar_params = historical_yield_curves[['date', 'exponential_mean', 'exponential_std', 'laguerre_mean', 'laguerre_std']].copy()
    temp_shape_parameter = historical_yield_curves[['date', 'L']].copy()
    temp_shape_parameter = temp_shape_parameter.rename(columns={'date': 'Date'})

    nelson_params = pd.concat([nelson_params, temp_nelson_params])
    scalar_params = pd.concat([scalar_params, temp_scalar_params])
    shape_parameter = pd.concat([shape_parameter, temp_shape_parameter])

    # set the index of dataframes to date column and converts the data type to datetime
    nelson_params.set_index('date', drop=True, inplace=True)
    scalar_params.set_index('date', drop=True, inplace=True)
    shape_parameter.set_index('Date', drop=True, inplace=True)

    # drop rows with duplicate indices, keeping the first such row; this approach was measured fastest per https://stackoverflow.com/a/34297689
    nelson_params = nelson_params[~nelson_params.index.duplicated(keep='first')]
    scalar_params = scalar_params[~scalar_params.index.duplicated(keep='first')]
    shape_parameter = shape_parameter[~shape_parameter.index.duplicated(keep='first')]

    nelson_params = nelson_params.transpose().to_dict()
    scalar_params = scalar_params.transpose().to_dict()
    shape_parameter = shape_parameter.transpose().to_dict()
    
    return nelson_params, scalar_params, shape_parameter
