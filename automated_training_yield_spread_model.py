'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2023-01-23
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-03-21
 '''
import numpy as np
import pandas as pd
from ficc.utils.auxiliary_functions import function_timer
from ficc.utils.auxiliary_variables import PREDICTORS
from datetime import datetime

from automated_training_auxiliary_functions import EASTERN, \
                                                   YEAR_MONTH_DAY, \
                                                   TESTING, \
                                                   SAVE_MODEL_AND_DATA, \
                                                   EMAIL_RECIPIENTS, \
                                                   get_storage_client, \
                                                   get_bq_client, \
                                                   setup_gpus, \
                                                   decrement_business_days, \
                                                   get_new_data, \
                                                   combine_new_data_with_old_data, \
                                                   add_trade_history_derived_features, \
                                                   drop_features_with_null_value, \
                                                   save_data, \
                                                   save_update_data_results_to_pickle_files, \
                                                   train_model, \
                                                   get_model_results, \
                                                   save_model, \
                                                   remove_file, \
                                                   send_results_email_multiple_tables, \
                                                   send_no_new_model_email


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
    current_datetime = datetime.now(EASTERN)
    print(f'automated_training_yield_spread_model.py starting at {current_datetime} ET')
    data, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath = save_update_data_results_to_pickle_files('yield_spread', update_data)
    
    current_date = current_datetime.date().strftime(YEAR_MONTH_DAY)
    previous_business_date = decrement_business_days(current_date, 11)
    model, previous_business_date_model, previous_business_date_model_date, encoders, mae, result_df_list = train_model(data, last_trade_date, 'yield_spread', num_features_for_each_trade_in_history, previous_business_date, apply_exclusions)
    current_date_data_current_date_model_result_df, current_date_data_previous_business_date_model_result_df = result_df_list
    last_trade_date_data_previous_business_date_model_result_df = get_model_results(data, last_trade_date, 'yield_spread', previous_business_date_model, encoders, apply_exclusions)

    if raw_data_filepath is not None:
        print(f'Removing {raw_data_filepath} since training is complete')
        remove_file(raw_data_filepath)
    
    if not TESTING and model is None:
        send_no_new_model_email(last_trade_date, EMAIL_RECIPIENTS, 'yield_spread')
        raise RuntimeError('No new data was found, so the procedure is terminating gracefully and without issue. Raising an error only so that the shell script terminates.')
    else:
        if SAVE_MODEL_AND_DATA: save_model(model, encoders, STORAGE_CLIENT, dollar_price_model=False)
        try:
            result_df_list = [current_date_data_current_date_model_result_df, current_date_data_previous_business_date_model_result_df, last_trade_date_data_previous_business_date_model_result_df]
            description_list = [f'The below table shows the accuracy of the newly trained model for the trades that occurred after {last_trade_date}', 
                                f'The below table shows the accuracy of the model trained on {previous_business_date_model_date} which was the one deployed on {previous_business_date_model_date} for the trades that occurred after {last_trade_date} (same data as first table but different model)', 
                                f'The below table shows the accuracy of the model trained on {previous_business_date_model_date} which was the one deployed on {previous_business_date_model_date} for the trades that occurred on {last_trade_date} (same model as second table but different data)']
            send_results_email_multiple_tables(result_df_list, description_list, last_trade_date, EMAIL_RECIPIENTS, 'yield_spread')
            # send_results_email_table(current_date_data_current_date_model_result_df, last_trade_date, EMAIL_RECIPIENTS, 'yield_spread')
            # send_results_email(mae, last_trade_date, EMAIL_RECIPIENTS, 'yield_spread')
        except Exception as e:
            print('Error:', e)


if __name__ == '__main__':
    main()
