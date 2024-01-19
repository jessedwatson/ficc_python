'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2023-01-23
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-01-18
 '''
import pandas as pd
from ficc.utils.auxiliary_variables import PREDICTORS_DOLLAR_PRICE, NON_CAT_FEATURES_DOLLAR_PRICE, BINARY_DOLLAR_PRICE, CATEGORICAL_FEATURES_DOLLAR_PRICE
from ficc.utils.auxiliary_functions import function_timer
from datetime import datetime
from dollar_model import dollar_price_model

from automated_training_auxiliary_functions import SEQUENCE_LENGTH_DOLLAR_PRICE_MODEL, \
                                                   TESTING, \
                                                   SAVE_MODEL_AND_DATA, \
                                                   EMAIL_RECIPIENTS, \
                                                   get_storage_client, \
                                                   get_bq_client, \
                                                   setup_gpus, \
                                                   get_new_data, \
                                                   combine_new_data_with_old_data, \
                                                   add_trade_history_derived_features, \
                                                   save_data, \
                                                   save_update_data_results_to_pickle_files, \
                                                   create_input, \
                                                   get_trade_date_where_data_exists_after_this_date, \
                                                   fit_encoders, \
                                                   train_and_evaluate_model, \
                                                   save_model, \
                                                   send_results_email, \
                                                   send_no_new_model_email


setup_gpus()


STORAGE_CLIENT = get_storage_client()
BQ_CLIENT = get_bq_client()

OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA = {'treasury_spread': False, 
                                       'only_dollar_price_history': True}


def update_data() -> (pd.DataFrame, datetime, int):
    '''Updates the master data file that is used to train and deploy the model. NOTE: if any of the variables in 
    `process_data(...)` or `SEQUENCE_LENGTH_DOLLAR_PRICE_MODEL` are changed, then we need to rebuild the entire 
    `processed_data_dollar_price.pkl` since that data is will have the old preferences; an easy way to do that 
    is to manually set `last_trade_date` to a date way in the past (the desired start date of the data).'''
    file_name = 'processed_data_dollar_price.pkl'
    data_before_last_trade_date, data_from_last_trade_date, last_trade_date, num_features_for_each_trade_in_history = get_new_data(file_name, 
                                                                                                                                   'dollar_price', 
                                                                                                                                   BQ_CLIENT, 
                                                                                                                                   optional_arguments_for_process_data=OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA)
    data = combine_new_data_with_old_data(data_before_last_trade_date, data_from_last_trade_date, 'dollar_price')
    print(f'Number of data points after combining new and old data: {len(data)}')
    data = add_trade_history_derived_features(data, 'dollar_price')
    data = data.rename(columns={'trade_history': 'trade_history_dollar_price'})    # change the trade history column name to match with `PREDICTORS_DOLLAR_PRICE`
    data.dropna(inplace=True, subset=PREDICTORS_DOLLAR_PRICE)
    if SAVE_MODEL_AND_DATA: save_data(data, file_name, STORAGE_CLIENT)
    return data, last_trade_date, num_features_for_each_trade_in_history


@function_timer
def train_model(data, last_trade_date, num_features_for_each_trade_in_history):
    encoders, fmax = fit_encoders(data, CATEGORICAL_FEATURES_DOLLAR_PRICE, 'dollar_price')

    if TESTING: last_trade_date = get_trade_date_where_data_exists_after_this_date(last_trade_date, data)
    test_data = data[data.trade_date > last_trade_date]
    if len(test_data) == 0: return None, None, None

    train_data = data[data.trade_date <= last_trade_date]
    print(f'Training set contains {len(train_data)} data points (on or before {last_trade_date})')
    print(f'Test set contains {len(test_data)} data points (after {last_trade_date})')
    
    x_train = create_input(train_data, encoders, NON_CAT_FEATURES_DOLLAR_PRICE, BINARY_DOLLAR_PRICE, CATEGORICAL_FEATURES_DOLLAR_PRICE, 'dollar_price')
    y_train = train_data.dollar_price

    x_test = create_input(test_data, encoders, NON_CAT_FEATURES_DOLLAR_PRICE, BINARY_DOLLAR_PRICE, CATEGORICAL_FEATURES_DOLLAR_PRICE, 'dollar_price')
    y_test = test_data.dollar_price

    model = dollar_price_model(x_train, 
                               SEQUENCE_LENGTH_DOLLAR_PRICE_MODEL, 
                               num_features_for_each_trade_in_history, 
                               CATEGORICAL_FEATURES_DOLLAR_PRICE, 
                               NON_CAT_FEATURES_DOLLAR_PRICE, 
                               BINARY_DOLLAR_PRICE, 
                               fmax)
    
    model, mae, history = train_and_evaluate_model(model, x_train, y_train, x_test, y_test)
    return model, encoders, mae


@function_timer
def main():
    print(f'automated_training_dollar_price_model.py starting {datetime.now()}')
    data, last_trade_date, num_features_for_each_trade_in_history = save_update_data_results_to_pickle_files('dollar_price', update_data)
    model, encoders, mae = train_model(data, last_trade_date, num_features_for_each_trade_in_history)

    if not TESTING and model is None:
        send_no_new_model_email(EMAIL_RECIPIENTS)
    else:
        if SAVE_MODEL_AND_DATA: save_model(model, encoders, STORAGE_CLIENT, dollar_price_model=True)
        if not TESTING: send_results_email(mae, last_trade_date, EMAIL_RECIPIENTS)


if __name__ == '__main__':
    main()
