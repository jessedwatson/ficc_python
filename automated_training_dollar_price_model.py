'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2023-01-23
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-03-21
 '''
import pandas as pd
from ficc.utils.auxiliary_functions import function_timer
from ficc.utils.auxiliary_variables import PREDICTORS_DOLLAR_PRICE
from datetime import datetime

from automated_training_auxiliary_functions import EASTERN, \
                                                   TESTING, \
                                                   SAVE_MODEL_AND_DATA, \
                                                   EMAIL_RECIPIENTS, \
                                                   get_storage_client, \
                                                   get_bq_client, \
                                                   setup_gpus, \
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
                                                   send_results_email_table, \
                                                   send_no_new_model_email


setup_gpus()


STORAGE_CLIENT = get_storage_client()
BQ_CLIENT = get_bq_client()

OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA = {'use_treasury_spread': False, 
                                       'only_dollar_price_history': True}


def update_data():
    '''Updates the master data file that is used to train and deploy the model. Returns a tuple of (pd.DataFrame, datetime, int).
    NOTE: if any of the variables in `process_data(...)` or `NUM_TRADES_IN_HISTORY_DOLLAR_PRICE_MODEL` are changed, then we need 
    to rebuild the entire `processed_data_test.pkl` since that data is will have the old preferences; an easy way to do that is 
    to manually set `last_trade_date` to a date way in the past (the desired start date of the data).'''
    file_name = 'processed_data_dollar_price.pkl'
    data_before_last_trade_datetime, data_from_last_trade_datetime, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath = get_new_data(file_name, 
                                                                                                                                                              'dollar_price', 
                                                                                                                                                              STORAGE_CLIENT, 
                                                                                                                                                              BQ_CLIENT, 
                                                                                                                                                              optional_arguments_for_process_data=OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA)
    data = combine_new_data_with_old_data(data_before_last_trade_datetime, data_from_last_trade_datetime, 'dollar_price')
    data = add_trade_history_derived_features(data, 'dollar_price')
    data = drop_features_with_null_value(data, PREDICTORS_DOLLAR_PRICE)
    if SAVE_MODEL_AND_DATA: save_data(data, file_name, STORAGE_CLIENT)
    return data, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath


@function_timer
def main():
    print(f'automated_training_dollar_price_model.py starting {datetime.now(EASTERN)} ET')
    data, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath = save_update_data_results_to_pickle_files('dollar_price', update_data)
    model, last_trade_date_model, encoders, mae, result_df_list = train_model(data, last_trade_date, num_features_for_each_trade_in_history)
    current_date_data_current_date_model_result_df, last_trade_date_data_current_date_model_result_df = result_df_list
    last_trade_date_data_last_trade_date_model_result_df = get_model_results(data, last_trade_date, 'dollar_price', last_trade_date_model)

    if raw_data_filepath is not None:
        print(f'Removing {raw_data_filepath} since training is complete')
        remove_file(raw_data_filepath)
    
    if not TESTING and model is None:
        send_no_new_model_email(last_trade_date, EMAIL_RECIPIENTS, 'dollar_price')
        raise RuntimeError('No new data was found. Raising an error so that the shell script terminates.')
    else:
        if SAVE_MODEL_AND_DATA: save_model(model, encoders, STORAGE_CLIENT, dollar_price_model=True)
        try:
            send_results_email_table(current_date_data_current_date_model_result_df, last_trade_date, EMAIL_RECIPIENTS, 'dollar_price')
            # send_results_email(mae, last_trade_date, EMAIL_RECIPIENTS, 'dollar_price')
        except Exception as e:
            print('Error:', e)


if __name__ == '__main__':
    main()
