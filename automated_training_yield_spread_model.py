'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2023-01-23
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-03-26
 '''
import sys
import numpy as np
import pandas as pd
from ficc.utils.auxiliary_functions import function_timer
from ficc.utils.auxiliary_variables import PREDICTORS

from automated_training_auxiliary_functions import SAVE_MODEL_AND_DATA, \
                                                   get_storage_client, \
                                                   get_bq_client, \
                                                   setup_gpus, \
                                                   get_new_data, \
                                                   combine_new_data_with_old_data, \
                                                   add_trade_history_derived_features, \
                                                   drop_features_with_null_value, \
                                                   save_data, \
                                                   train_save_evaluate_model


setup_gpus()


STORAGE_CLIENT = get_storage_client()
BQ_CLIENT = get_bq_client()

OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA = {'use_treasury_spread': True, 
                                       'only_dollar_price_history': False}


if 'ficc_treasury_spread' not in PREDICTORS: PREDICTORS.append('ficc_treasury_spread')
if 'target_attention_features' not in PREDICTORS: PREDICTORS.append('target_attention_features')


def update_data():
    '''Updates the master data file that is used to train and deploy the model. Returns a tuple of (pd.DataFrame, datetime, int).
    NOTE: if any of the variables in `process_data(...)` or `NUM_TRADES_IN_HISTORY_YIELD_SPREAD_MODEL` are changed, then we need 
    to rebuild the entire `processed_data_test.pkl` since that data is will have the old preferences; an easy way to do that is 
    to manually set `last_trade_date` to a date way in the past (the desired start date of the data).'''
    file_name = 'processed_data_test.pkl'
    use_treasury_spread = OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA.get('use_treasury_spread', False)
    data_before_last_trade_datetime, data_from_last_trade_datetime, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath = get_new_data(file_name, 
                                                                                                                                                              'yield_spread', 
                                                                                                                                                              STORAGE_CLIENT, 
                                                                                                                                                              BQ_CLIENT, 
                                                                                                                                                              use_treasury_spread, 
                                                                                                                                                              OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA)
    data = combine_new_data_with_old_data(data_before_last_trade_datetime, data_from_last_trade_datetime, 'yield_spread')
    data = add_trade_history_derived_features(data, 'yield_spread', use_treasury_spread)
    data = drop_features_with_null_value(data, PREDICTORS)
    if SAVE_MODEL_AND_DATA: save_data(data, file_name, STORAGE_CLIENT)
    return data, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath


def apply_exclusions(data: pd.DataFrame, dataset_name: str = None):
    from_dataset_name = f' from {dataset_name}' if dataset_name is not None else ''
    data_before_exclusions = data[:]
    
    previous_size = len(data)
    data = data[(data.days_to_call == 0) | (data.days_to_call > np.log10(400))]
    current_size = len(data)
    if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having 0 < days_to_call <= 400')
    
    previous_size = current_size
    data = data[(data.days_to_refund == 0) | (data.days_to_refund > np.log10(400))]
    current_size = len(data)
    if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having 0 < days_to_refund <= 400')
    
    previous_size = current_size
    data = data[(data.days_to_maturity == 0) | (data.days_to_maturity > np.log10(400))]
    current_size = len(data)
    if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having 0 < days_to_maturity <= 400')
    
    previous_size = current_size
    data = data[data.days_to_maturity < np.log10(30000)]
    current_size = len(data)
    if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having days_to_maturity >= 30000')
    
    ## null last_calc_date exclusion was removed on 2024-02-19
    # previous_size = current_size
    # data = data[~data.last_calc_date.isna()]
    # current_size = len(data)
    # if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having a null value for last_calc_date')

    return data, data_before_exclusions


@function_timer
def main():
    current_date_passed_in = sys.argv[1] if len(sys.argv) == 2 else None
    train_save_evaluate_model('yield_spread', update_data, apply_exclusions, current_date_passed_in)


if __name__ == '__main__':
    main()
