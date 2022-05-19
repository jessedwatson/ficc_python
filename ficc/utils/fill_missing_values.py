'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-17 12:32:03
 # @ Modified by: Mitas Ray
 # @ Modified time: 2022-03-17 13:19:00
 # @ Description: fill in features with the corresponding default values.
 '''


FEATURES_AND_DEFAULT_VALUES = (('purpose_class', 0),    # unknown
                               ('call_timing', 0),    # unknown
                               ('call_timing_in_part', 0),    # unknown
                               ('sink_frequency', 0),    # under special circumstances
                               ('sink_amount_type', 10), 
                               ('issue_text', 'No issue text'), 
                               ('state_tax_status', 0), 
                               ('series_name', 'No series name'), 
                               ('transaction_type', 'I'), 
                               ('next_call_price', 100), 
                               ('par_call_price', 100), 
                               ('min_amount_outstanding', 0), 
                               ('max_amount_outstanding', 0), 
                               ('days_to_par', 0), 
                               ('maturity_amount', 0), 
                               ('issue_price', lambda df: df.issue_price.mean()), 
                               ('orig_principal_amount', lambda df: df.orig_principal_amount.mean()), 
                               ('par_price', 100), 
                               ('called_redemption_type', 0), 
                               ('extraordinary_make_whole_call', False), 
                               ('make_whole_call', False), 
                               ('default_indicator', False), 
                               ('called_redemption_type', 0), 
                               ('days_to_settle', 0), 
                               ('days_to_maturity', 0), 
                               ('days_to_call', 0), 
                               ('days_to_refund', 0), 
                               ('days_to_par', 0), 
                               ('call_to_maturity', 0))

def fill_missing_values(df, feature_to_default_value_dict, add_missingness_flag):
    df.dropna(subset=['instrument_primary_name'], inplace=True)

    for feature, default_value in FEATURES_AND_DEFAULT_VALUES:
        if df[feature].isnull().values.any():    # checks if any of the values for `feature` are null
            if add_missingness_flag:
                df[feature + '_missing'] = df[feature].isnull()

            if feature in feature_to_default_value_dict:
                default_value = feature_to_default_value_dict

            if callable(default_value):    # checks whether the default_value is a function that needs to be called on the dataframe
                df[feature].fillna(default_value(df), inplace=True)
            else:
                df[feature].fillna(default_value, inplace=True)
    
    return df