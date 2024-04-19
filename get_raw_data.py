'''
 # @ Author: Mitas Ray
 # @ Create date: 2024-04-19
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-04-19
 # @ Description: Gather train / test data from materialized trade history. First, find all dates for which there are trades. Then, 
 use multiprocessing to read the data from BigQuery for each date, since the conversion of the query results to a dataframe is costly. 
 This file was created to test different ways of getting the raw data to determine which one was faster: getting it all at once, or 
 getting it day by day using multiprocessing and then concatenating it together.
 '''
import multiprocess as mp
from datetime import datetime
import pandas as pd
from pandas.tseries.offsets import BusinessDay

from ficc.utils.auxiliary_functions import sqltodf, function_timer

from automated_training_auxiliary_variables import EASTERN, YEAR_MONTH_DAY, EARLIEST_TRADE_DATETIME
from automated_training_auxiliary_functions import BQ_CLIENT, get_data_query


MULTIPROCESSING = True

TESTING = False
if TESTING is True:
    EARLIEST_TRADE_DATETIME = (datetime.now(EASTERN) - BusinessDay(2)).strftime(YEAR_MONTH_DAY) + 'T00:00:00'    # 2 business days before the current datetime (start of the day) to have enough days for training and testing; same logic as `automated_training_auxiliary_functions::decrement_business_days(...)` but cannot import from there due to circular import issue


data_query = get_data_query(EARLIEST_TRADE_DATETIME, 'yield_spread_with_similar_trades')
print('Getting data from the following query:\n', data_query)

distinct_dates_query = 'SELECT DISTINCT trade_date ' + data_query[data_query.find('FROM') : data_query.find('ORDER BY')]    # remove all the original selected features and just get each unique `trade_date`; need to remove the `ORDER BY` clause since the `trade_datetime` feature is not selected in this query
distinct_dates = sqltodf(distinct_dates_query, BQ_CLIENT)
distinct_dates = sorted(distinct_dates['trade_date'].astype(str).values)    # convert the one column dataframe with column name `trade_date` fron `sqltodf(...)` into a numpy array sorted by `trade_date`
print('Distinct dates:', distinct_dates)


@function_timer
def get_trades_for_particular_date(start_date_as_string: str, end_date_as_string: str = None) -> pd.DataFrame:
    '''If `end_date_as_string` is `None`, then we get trades only for `start_date_as_string`.'''
    print(start_date_as_string)
    if end_date_as_string is None: end_date_as_string = start_date_as_string
    data_query = get_data_query(start_date_as_string + 'T00:00:00', 'yield_spread_with_similar_trades')
    data_query_date = data_query[:data_query.find('ORDER BY')] + f'AND trade_datetime <= "{end_date_as_string}T23:59:59" ' + data_query[data_query.find('ORDER BY'):]    # add condition of restricting all trades to the specified `date_as_string`
    print(data_query_date)
    return sqltodf(data_query_date, BQ_CLIENT)


@function_timer
def get_trades_for_all_dates(dates: list, all_at_once: bool = False) -> pd.DataFrame:
    '''`dates` is a list of strings.'''
    if all_at_once is True:
        trades_for_all_dates = get_trades_for_particular_date(min(dates), max(dates))
    else:
        if len(dates) > 1 and MULTIPROCESSING:
            print(f'Using multiprocessing for calling `get_trades_for_particular_date(...) on each item in {dates}')
            with mp.Pool() as pool_object:    # using template from https://docs.python.org/3/library/multiprocessing.html
                trades_for_all_dates = pool_object.map(get_trades_for_particular_date, dates)
        else:
            trades_for_all_dates = [get_trades_for_particular_date(date) for date in dates]
        trades_for_all_dates = pd.concat(trades_for_all_dates).reset_index(drop=True)    # `pd.concat(...)` is a bottleneck (even though the calls to `get_trades_for_particular_date(...)` take less than a minute, the `pd.concat(...)` takes >10 mins
    return trades_for_all_dates


trades_for_all_dates = get_trades_for_all_dates(distinct_dates, True)
trades_for_all_dates.to_pickle('files/trades_for_all_dates_from_get_raw_data.pkl')
print(trades_for_all_dates)
