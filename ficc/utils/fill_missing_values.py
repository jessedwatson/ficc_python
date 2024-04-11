'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2021-12-17
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-04-10
 # @ Description: fill in features with the corresponding default values.
 '''
import warnings

import pandas as pd


FEATURES_AND_DEFAULT_VALUES = {'purpose_class': 0,    # unknown
                               'call_timing': 0,    # unknown
                               'call_timing_in_part': 0,    # unknown
                               'sink_frequency': 0,    # under special circumstances
                               'sink_amount_type': 10, 
                               'issue_text': 'No issue text', 
                               'state_tax_status': 0, 
                               'series_name': 'No series name', 
                               'transaction_type': 'I', 
                               'next_call_price': 100, 
                               'par_call_price': 100, 
                               'min_amount_outstanding': 0, 
                               'max_amount_outstanding': 0, 
                               'days_to_par': 0, 
                               'maturity_amount': 0, 
                               'issue_price': lambda df: df.issue_price.mean(),    # leakage; computing the mean over the entire dataset uses the test data (low priority, since this barely affects the model)
                               'orig_principal_amount': lambda df: df.orig_principal_amount.mean(),    # leakage; computing the mean over the entire dataset uses the test data (low priority, since this barely affects the model)
                               'par_price': 100, 
                               'called_redemption_type': 0, 
                               'extraordinary_make_whole_call': False, 
                               'make_whole_call': False, 
                               'default_indicator': False, 
                               'called_redemption_type': 0, 
                               'days_to_settle': 0, 
                               'days_to_maturity': 0, 
                               'days_to_call': 0, 
                               'days_to_refund': 0, 
                               'days_to_par': 0, 
                               'call_to_maturity': 0,
                               'last_seconds_ago': 0, 
                               'last_yield_spread': 0.0, 
                               'last_dollar_price': 0.0, 
                               'days_in_interest_payment': 180}


def replace_nan_with_value(df, feature, default_value):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.SettingWithCopyWarning)    # inplace replacements raise `SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy return self._update_inplace(result)`
        if callable(default_value):    # checks whether the default_value is a function that needs to be called on the dataframe
            df[feature].fillna(default_value(df), inplace=True)
        else:
            df[feature].fillna(default_value, inplace=True)


def fill_missing_values(df):
    # df.dropna(subset=['instrument_primary_name'], inplace=True)
    for feature, default_value in FEATURES_AND_DEFAULT_VALUES.items():
        try:
            replace_nan_with_value(df, feature, default_value)
        except Exception:
            print(f'Feature {feature} not in dataframe')
    return df
