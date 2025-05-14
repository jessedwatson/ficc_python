'''
Author: Mitas Ray
Date: 2025-05-13
Last Editor: Mitas Ray
Last Edit Date: 2025-05-13
'''
import pandas as pd


from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR, PROJECT_ID
from ficc.utils.auxiliary_functions import function_timer, sqltodf
from ficc.utils.nelson_siegel_model import yield_curve_level
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.initialize_pandarallel import initialize_pandarallel


YIELD_CURVE_DATASET_NAME = 'yield_curves_v2'


def get_duration(row, use_last_calc_day_cat: bool) -> float:
    '''Get the duration in years from `row`. If `use_last_calc_day_cat` is `True`, then the function will 
    use the last duration for the yield curve level. Otherwise, it will use the current duration by 
    using the calculation date computed upstream.

    NOTE: there is an argument the logic when setting `use_last_calc_day_cat` to `False` causes leakage 
    because in production we do not have the `calc_date` column. However, the objective is to train a 
    model that performs as well as possible IF we had the correct calc date. In production, we have a 
    subprocedure to infer what the calc date is, so it makes sense to train a model that performs as well 
    as possible if the calc date were to be correct. Subpoint: in most cases, the calc date is known (e.g., 
    for bond without a call option, a called bond, etc.).'''
    start_date = row['settlement_date']    # this value is never null (verified from inspecting the materialized trade history on 2025-05-13)
    
    maturity_date = row['maturity_date']    # this value is never null (verified from inspecting the materialized trade history on 2025-05-13)
    if use_last_calc_day_cat:
        # the reason that we need to use the logic below instead of just using row['last_calc_date'] is because 
        # (1) if the bond has been called, then the last calc date is incorrect
        # (2) if the reference data is updated, then using the category and selecting the correct date is more accurate since last_calc_date may be a date in the past since the next_call_date has been updated
        is_called, is_callable, last_calc_day_cat = row['is_called'], row['is_callable'], row['last_calc_day_cat']
        if is_called is True:
            end_date = row['refund_date']    # this value is never null if `is_called` is True (verified from inspecting the materialized trade history on 2025-05-13)
        elif is_callable is False:
            end_date = maturity_date
        elif last_calc_day_cat == 0 and pd.notna(row['next_call_date']):    # sometimes the case that `next_call_date` is null, but `first_call_date` is not null, and in this case, the correct value is perhaps `first_call_date` but requires complicated upstream code to correct it and deeper investigation before making the change here (1.3% of materialized trade history on 2025-05-13, and only new issues)
            end_date = row['next_call_date']
        elif last_calc_day_cat == 1 and pd.notna(row['par_call_date']):
            end_date = row['par_call_date']
        else:
            end_date = maturity_date
    else:    # `use_last_calc_day_cat` is `False`
        calc_date = row['calc_date']
        if pd.isna(calc_date):
            end_date = maturity_date
        else:
            end_date = calc_date

    return diff_in_days_two_dates(end_date, start_date) / NUM_OF_DAYS_IN_YEAR


def get_yield_curve_level(row, nelson_params: dict, scalar_params: dict, shape_params: dict, end_of_day: bool, use_last_calc_day_cat: bool) -> float:
    '''`end_of_day` is a boolean that indicates whether the data is end-of-day yield curve or the real-time (minute) yield curve. 
    `use_last_calc_day_cat` is a boolean that indicates whether to use the last duration for the yield curve level. If `use_last_calc_day_cat` 
    is `True`, then the function will use the last duration for the yield curve level. Otherwise, it will use the current duration by 
    using the calculation date computed upstream.'''
    return yield_curve_level(get_duration(row, use_last_calc_day_cat), row['trade_datetime'], nelson_params, scalar_params, shape_params, end_of_day)


def get_parameters(table_name: str, bq_client, date_column_name: str = 'date') -> dict:
    '''Return the parameters from `table_name` as a dictionary.'''
    params = sqltodf(f'SELECT * FROM `{PROJECT_ID}.{YIELD_CURVE_DATASET_NAME}.{table_name}` ORDER BY {date_column_name} DESC', bq_client)
    params.set_index(date_column_name, drop=True, inplace=True)
    params = params[~params.index.duplicated(keep='first')]
    return params.transpose().to_dict()


@function_timer
def add_yield_curve(data, bq_client, end_of_day: bool = False, use_last_calc_day_cat: bool = False) -> pd.DataFrame:
    '''Add 'new_ficc_ycl' field to `data`. `end_of_day` is a boolean that indicates whether the data is end-of-day yield curve or the real-time (minute) yield curve. 
    `use_last_calc_day_cat` is a boolean that indicates whether to use the last duration for the yield curve level. If `use_last_calc_day_cat` is `True`, then the function 
    will use the last duration for the yield curve level. Otherwise, it will use the current duration by using the calculation date computed upstream.'''
    initialize_pandarallel()    # only initialize if needed

    # TODO: extract the minimum date from `data`, and pass it into `get_parameters(...)` to get the parameters for that date +/- a few days; this will speed up the querying and other downstream procedures
    if end_of_day:
        nelson_params = get_parameters('nelson_siegel_coef_daily', bq_client)
    else:
        nelson_params = get_parameters('nelson_siegel_coef_minute', bq_client)

    scalar_daily_params = get_parameters('standardscaler_parameters_daily', bq_client)
    shape_params = get_parameters('shape_parameters', bq_client, 'Date')    # 'Date' is capitalized for this table which is a typo when initially created

    columns_needed_to_compute_ycl = ['calc_date', 'settlement_date', 'trade_datetime', 'last_calc_day_cat', 'is_called', 'is_callable', 'refund_date', 'next_call_date', 'par_call_date', 'maturity_date']
    columns_received_from_computing_ycl = ['new_ficc_ycl', 'const', 'exponential', 'laguerre', 'target_datetime_for_nelson_params', 'exponential_mean', 'exponential_std', 'laguerre_mean', 'laguerre_std', 'target_date_for_scaler_params', 'shape_parameter', 'target_date_for_shape_parameter']
    get_yield_curve_level_caller = lambda row: get_yield_curve_level(row, nelson_params, scalar_daily_params, shape_params, end_of_day, use_last_calc_day_cat)
    data[columns_received_from_computing_ycl] = data[columns_needed_to_compute_ycl].parallel_apply(get_yield_curve_level_caller, axis=1, result_type='expand')
    return data
