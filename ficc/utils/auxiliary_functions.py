'''
 # @ Author: Anis Ahmad 
 # @ Create Time: 2021-12-15 13:59:54
 # @ Modified by: Mitas Ray
 # @ Modified time: 2022-07-13 11:52:00
 # @ Description: This file contains function to help the functions 
 # to process training data
 '''
import pandas as pd
import numpy as np

def sqltodf(sql, bq_client):
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe()


def drop_extra_columns(df):
    df.drop(columns=[
                 'sp_stand_alone',
                 'sp_icr_school',
                 'sp_icr_school',
                 'sp_icr_school',
                 'sp_watch_long',
                 'sp_outlook_long',
                 'sp_prelim_long',
                 'MSRB_maturity_date',
                 'MSRB_INST_ORDR_DESC',
                 'MSRB_valid_from_date',
                 'MSRB_valid_to_date',
                 'upload_date',
                 'sequence_number',
                 'ICE_valid_from_date',
                 'ICE_valid_TO_date',
                 'additional_next_sink_date',
                 'last_period_accrues_from_date',
                 'primary_market_settlement_date',
                 'assumed_settlement_date',
                 'sale_date','q','d'],
                  inplace=True)
    
    
    return df


def convert_dates(df):
    date_cols = [col for col in list(df.columns) if 'DATE' in col.upper()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
    
    return df

'''
This function  
'''
def process_ratings(df, process_ratings):
    # MR is for missing ratings
    df.sp_long.fillna('MR', inplace=True)
    if process_ratings == True:
        df = df[df.sp_long.isin(['BBB+','A-','A','A+','AA-','AA','AA+','AAA','NR','MR'])] 
    df['rating'] = df['sp_long']
    return df
    
'''
This function extracts the features of the latest trade from 
the trade history array
'''
def get_latest_trade_feature(x, feature):
    recent_trade = x[0]
    if feature == 'yield_spread':
        return recent_trade[0]
    elif feature == 'seconds_ago':
        return recent_trade[-1]
    elif feature == 'par_traded':
        return recent_trade[1]

'''
This function compares two date objects whether they are in Timestamp or datetime.date. 
The different types are causing a future warning. If date1 occurs after date2, return 1. 
If date1 equals date2, return 0. Otherwise, return -1.
'''
def compare_dates(date1, date2):
    if type(date1) == pd.Timestamp:
        date1 = date1.date()
    if type(date2) == pd.Timestamp:
        date2 = date2.date()
    
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
        return df['accrued_days']/360


'''Computes the dollar error from the predicted yield spreads and MSRB data. 
Assumes that the predicted yield spreads are in basis points.'''
def calculate_dollar_error(df, predicted_ys):
    assert len(predicted_ys) == len(df), 'There must be a predicted yield spread for each of the trades in the passed in dataframe'
    columns_set = set(df.columns)
    assert 'quantity' in columns_set
    assert 'ficc_ycl' in columns_set    # represents the ficc yield curve level
    assert 'yield' in columns_set    # represents yield to worst from the MSRB data
    assert 'calc_date' in columns_set and 'settlement_date' in columns_set    # need these two features to compute the number of years from the settlement date to the calc date
    years_to_calc_date = (df['calc_date'] - df['settlement_date']) / np.timedelta64(1, 'Y')    # the division by `np.timedelta64(1, 'Y')` converts the quantity to years
    return ((predicted_ys + df['ficc_ycl']) / 100 - df['yield']) * df['quantity'] * years_to_calc_date    # dollar error = duration * quantity * (ytw - ficc_ytw) [in percentage points]; duration = calc_date - settlement_date [in years]
