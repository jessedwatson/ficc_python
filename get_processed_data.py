'''
 # @ Author: Mitas Ray
 # @ Create date: 2024-04-19
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-04-23
 # @ Description: Gather train / test data from materialized trade history. First, find all dates for which there are trades. Then, 
 use multiprocessing to read the data from BigQuery for each date, since the conversion of the query results to a dataframe is costly. 
 This file was created to test different ways of getting the raw data to determine which one was faster: getting it all at once, or 
 getting it day by day using multiprocessing and then concatenating it together.
 '''
from tqdm import tqdm
import multiprocess as mp
from datetime import datetime
import pandas as pd
from pandas.tseries.offsets import BusinessDay

from ficc.utils.auxiliary_functions import sqltodf, function_timer

from automated_training_auxiliary_variables import EASTERN, YEAR_MONTH_DAY, EARLIEST_TRADE_DATETIME, OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA_YIELD_SPREAD, OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA_DOLLAR_PRICE, MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME
from automated_training_auxiliary_functions import BQ_CLIENT, get_data_query, check_that_model_is_supported, get_new_data, get_optional_arguments_for_process_data, add_trade_history_derived_features, drop_features_with_null_value, save_data


MODEL = 'yield_spread_with_similar_trades'
check_that_model_is_supported(MODEL)

MULTIPROCESSING = True
SAVE_DATA = True

TESTING = False
if TESTING is True:
    SAVE_DATA = False
    EARLIEST_TRADE_DATETIME = (datetime.now(EASTERN) - BusinessDay(2)).strftime(YEAR_MONTH_DAY) + 'T00:00:00'    # 2 business days before the current datetime (start of the day) to have enough days for training and testing; same logic as `automated_training_auxiliary_functions::decrement_business_days(...)` but cannot import from there due to circular import issue


data_query = get_data_query(EARLIEST_TRADE_DATETIME, 'yield_spread_with_similar_trades')
print('Getting data from the following query:\n', data_query)

distinct_dates_query = 'SELECT DISTINCT trade_date ' + data_query[data_query.find('FROM') : data_query.find('ORDER BY')]    # remove all the original selected features and just get each unique `trade_date`; need to remove the `ORDER BY` clause since the `trade_datetime` feature is not selected in this query
distinct_dates = sqltodf(distinct_dates_query, BQ_CLIENT)
distinct_dates = sorted(distinct_dates['trade_date'].astype(str).values, reverse=True)    # convert the one column dataframe with column name `trade_date` fron `sqltodf(...)` into a numpy array sorted by `trade_date`; going in descending order since the the query gets the trades in descending order of `trade_datetime` and so concatenating all the trades from each of the days will be in descending order of `trade_datetime` if the trade dates are in descending order
print('Distinct dates:', distinct_dates)


@function_timer
def check_that_df_is_sorted_by_column(df, column_name, desc: bool = False):
    '''`desc` is a boolean flag that determines whether we are checking for ascending order 
    (`desc` is `False`) or descending order (`desc` is `True`).'''
    column_values = df[column_name].to_numpy()
    larger = column_values[:-1] if desc else column_values[1:]
    smaller = column_values[1:] if desc else column_values[:-1]
    assert all(larger >= smaller)


@function_timer
def get_processed_trades_for_particular_date(start_date_as_string: str, end_date_as_string: str = None) -> pd.DataFrame:
    '''If `end_date_as_string` is `None`, then we get trades only for `start_date_as_string`.'''
    print(start_date_as_string)
    if end_date_as_string is None: end_date_as_string = start_date_as_string
    data_query = get_data_query(start_date_as_string + 'T00:00:00', 'yield_spread_with_similar_trades')
    data_query_date = data_query[:data_query.find('ORDER BY')] + f'AND trade_datetime <= "{end_date_as_string}T23:59:59" ' + data_query[data_query.find('ORDER BY'):]    # add condition of restricting all trades to the specified `date_as_string`
    print(data_query_date)

    optional_arguments_for_process_data = get_optional_arguments_for_process_data(MODEL)
    use_treasury_spread = optional_arguments_for_process_data.get('use_treasury_spread', False)
    _, processed_data, _, _, _ = get_new_data(None, MODEL, use_treasury_spread, optional_arguments_for_process_data, data_query_date, False)
    return processed_data    # get just the raw data with `sqltodf(data_query_date, BQ_CLIENT)`


@function_timer
def get_trades_for_all_dates(dates: list, all_at_once: bool = False) -> pd.DataFrame:
    '''`dates` is a list of strings.'''
    if all_at_once is True:
        trades_for_all_dates = get_processed_trades_for_particular_date(min(dates), max(dates))
    else:
        if len(dates) > 1 and MULTIPROCESSING:
            print(f'Using multiprocessing for calling `get_trades_for_particular_date(...) on each of the {len(dates)} items in {dates}')
            with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html; consider trying with lesser processes if running out of RAM (put an argument of e.g. `os.cpu_count() // 2` as the argument to `mp.Pool()` so it reads `mp.Pool(os.cpu_count() // 2)`)
                trades_for_all_dates = pool_object.map(get_processed_trades_for_particular_date, dates)
        else:
            trades_for_all_dates = [get_processed_trades_for_particular_date(date) for date in tqdm(dates)]
        trades_for_all_dates = pd.concat(trades_for_all_dates).reset_index(drop=True)    # `pd.concat(...)` is a bottleneck (even though the calls to `get_trades_for_particular_date(...)` take less than a minute, the `pd.concat(...)` takes >10 mins
    check_that_df_is_sorted_by_column(trades_for_all_dates, 'trade_datetime')

    optional_arguments_for_process_data = get_optional_arguments_for_process_data(MODEL)
    trades_for_all_dates = add_trade_history_derived_features(trades_for_all_dates, MODEL, optional_arguments_for_process_data.get('use_treasury_spread', False))
    trades_for_all_dates = drop_features_with_null_value(trades_for_all_dates, MODEL)
    if SAVE_DATA: save_data(trades_for_all_dates, MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME[MODEL])
    return trades_for_all_dates


trades_for_all_dates = get_trades_for_all_dates(distinct_dates, False)
trades_for_all_dates.to_pickle('files/trades_for_all_dates_from_get_processed_data.pkl')
print(trades_for_all_dates)
