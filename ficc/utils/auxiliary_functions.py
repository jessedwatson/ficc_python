'''
 # @ Author: Anis Ahmad 
 # @ Create Time: 2021-12-15 13:59:54
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2023-10-26 13:01:18
 # @ Description: This file contains function to help the functions 
 # to process training data
 '''
import datetime
import pandas as pd

from ficc.utils.auxiliary_variables import NUM_OF_DAYS_IN_YEAR
from ficc.utils.diff_in_days import diff_in_days_two_dates


'''Quote a string twice: e.g., double_quote_a_string('hello') -> "'hello'". This 
function is used to put string arguments into formatted string expressions and 
maintain the quotation.'''
double_quote_a_string = lambda potential_string: f'"{str(potential_string)}"' if type(potential_string) == str else potential_string


def sqltodf(sql, bq_client):
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe()

def convert_dates(df):
    date_cols = [col for col in list(df.columns) if 'DATE' in col.upper()]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception as e:
            print('************ ERROR ************')
            print(f'Failed to convert date for {col}')
            print('*******************************')
            continue
    
    return df

'''
This function  
'''
def process_ratings(df):
    # MR is for missing ratings
    df.sp_long.fillna('MR', inplace=True)
    df['rating'] = df['sp_long']
    return df


'''
Converts an object, either of type pd.Timestamp or datetime.datetime to a 
datetime.date object.
'''
def convert_to_date(date):
    if isinstance(date, pd.Timestamp): date = date.to_pydatetime()
    if isinstance(date, datetime.datetime): date = date.date()
    return date    # assumes the type is datetime.date

'''
This function compares two date objects whether they are in Timestamp or datetime.date. 
The different types are causing a future warning. If date1 occurs after date2, return 1. 
If date1 equals date2, return 0. Otherwise, return -1.
'''
def compare_dates(date1, date2):
    return (convert_to_date(date1) - convert_to_date(date2)).total_seconds()

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


