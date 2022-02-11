'''
 # @ Author: Anis Ahmad 
 # @ Create Time: 2021-12-15 13:59:54
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-02-10 10:49:09
 # @ Description: This file contains function to help the functions 
 # to process training data
 '''
import pandas as pd

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
def process_ratings(df):
    df = df[df.sp_long.isin(['BBB+','A-','A','A+','AA-','AA','AA+','AAA','NR'])] 
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
        if col_name.endswith("event") or col_name.endswith("redemption") or col_name.endswith("history") or col_name.endswith("date"):
            continue

        if df[col_name].dtype == "object" and col_name not in ['organization_primary_name','security_description','recent','issue_text','series_name']:
            df[col_name] = df[col_name].astype("category")
    return df