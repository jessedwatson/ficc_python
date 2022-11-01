'''
 # @ Author: Anis Ahmad 
 # @ Create Time: 2021-12-15 13:59:54
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-07-19 13:10:59
 # @ Description: This file contains function to help the functions 
 # to process training data
 '''

import pandas as pd
import numpy as np

from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from ficc.utils.diff_in_days import diff_in_days_two_dates

def sqltodf(sql, bq_client):
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe()

def convert_dates(df):
    date_cols = [col for col in list(df.columns) if 'DATE' in col.upper()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    
    return df

'''
This function  
'''
def process_ratings(df, drop_ratings):
    # MR is for missing ratings
    df.sp_long.fillna('MR', inplace=True)
    if drop_ratings == True:
        df = df[df.sp_long.isin(['BBB+','A-','A','A+','AA-','AA','AA+','AAA','NR','MR'])] 
    df['rating'] = df['sp_long']
    return df
    
'''
This function extracts the features of the latest trade from 
the trade history array
'''
def get_latest_trade_feature(x):
    recent_trade = x[0]
    return recent_trade[-1], recent_trade[0] , recent_trade[1]

'''
This function compares two date objects whether they are in Timestamp or datetime.date. 
The different types are causing a future warning. If date1 occurs after date2, return 1. 
If date1 equals date2, return 0. Otherwise, return -1.
'''
def compare_dates(date1, date2):
    if type(date1) == pd.Timestamp:
        date1 = date1.to_pydatetime()
    if type(date2) == pd.Timestamp:
        date2 = date2.to_pydatetime()
    
    if date1 > date2:
        return 1
    elif date1 == date2:
        return 0
    elif date1 < date2:
        return -1

'''
This function directly calls `compare_dates` to check if two dates are equal.
'''
def dates_are_equal(date1, date2):
    return compare_dates(date1, date2) == 0

'''
This function converts the columns with object datatypes to category data types
'''
def convert_object_to_category(df):
    print("Converting object data type to categorical data type")
    for col_name in df.columns:
        if col_name.endswith("event") or col_name.endswith("redemption") or col_name.endswith("history") or col_name.endswith("date") or col_name.endswith("issue"):
            continue

        if df[col_name].dtype == "object" and col_name not in ['organization_primary_name','security_description','recent','issue_text','series_name','recent_trades_by_series','recent_trades_same_calc_day']:
            df[col_name] = df[col_name].astype("category")
    return df

def calculate_a_over_e(df):
    if not pd.isnull(df.previous_coupon_payment_date):
        A = (df.settlement_date - df.previous_coupon_payment_date).days
        return A/df.days_in_interest_payment
    else:
        return df['accrued_days']/NUM_OF_DAYS_IN_YEAR

'''
This function converts calc date to calc date category
these labels are used to train the calc date model
'''
def convert_calc_date_to_category(row):
    if row.last_calc_date == row.next_call_date:
        calc_date_selection = 0
    elif row.last_calc_date == row.par_call_date:
        calc_date_selection = 1
    elif row.last_calc_date == row.maturity_date:
        calc_date_selection = 2
    elif row.last_calc_date == row.refund_date:
        calc_date_selection = 3
    else:
        calc_date_selection = 4
    return calc_date_selection


'''Computes the dollar error from the predicted yield spreads and MSRB data. 
Assumes that the predicted yield spreads are in basis points.'''
def calculate_dollar_error(df, predicted_ys):
    assert len(predicted_ys) == len(df), 'There must be a predicted yield spread for each of the trades in the passed in dataframe'
    columns_set = set(df.columns)
    assert 'quantity' in columns_set    # assumes that the quantity is log10 transformed
    assert 'ficc_ycl' in columns_set    # represents the ficc yield curve level
    assert 'yield' in columns_set    # represents yield to worst from the MSRB data
    assert 'calc_date' in columns_set and 'settlement_date' in columns_set    # need these two features to compute the number of years from the settlement date to the calc date
    years_to_calc_date = diff_in_days_two_dates(df.calc_date,df.settlement_date) / NUM_OF_DAYS_IN_YEAR    # the division by `np.timedelta64(NUM_OF_DAYS_IN_YEAR, 'D')` converts the quantity to years according to the MSRB convention of NUM_OF_DAYS_IN_YEAR in a year
    ytw_error = ((predicted_ys + df['ficc_ycl']) / 100 - df['yield']) / 100    # the second divide by 100 is because the unit of the dividend is in percent
    return ytw_error * (10 ** df['quantity']) * years_to_calc_date    # dollar error = duration * quantity * ytw error; duration = calc_date - settlement_date [in years]
