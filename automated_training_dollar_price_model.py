'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2023-01-23
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-01-08
 '''
import numpy as np
import pandas as pd
import tensorflow as tf
from ficc.data.process_data import process_data
from ficc.utils.auxiliary_variables import PREDICTORS_DOLLAR_PRICE, NON_CAT_FEATURES_DOLLAR_PRICE, BINARY_DOLLAR_PRICE, CATEGORICAL_FEATURES_DOLLAR_PRICE
from ficc.utils.auxiliary_functions import function_timer, get_dp_trade_history_features
from ficc.utils.gcp_storage_functions import upload_data
from datetime import datetime
from dollar_model import dollar_price_model

from automated_training_auxiliary_functions import SEQUENCE_LENGTH_DOLLAR_PRICE_MODEL, \
                                                   QUERY_FEATURES, \
                                                   QUERY_CONDITIONS, \
                                                   ADDITIONAL_QUERY_FEATURES_FOR_DOLLAR_PRICE_MODEL, \
                                                   BUCKET_NAME, \
                                                   SAVE_MODEL_AND_DATA, \
                                                   EMAIL_RECIPIENTS, \
                                                   get_storage_client, \
                                                   get_bq_client, \
                                                   get_trade_history_columns, \
                                                   target_trade_processing_for_attention, \
                                                   replace_ratings_by_standalone_rating, \
                                                   create_input, \
                                                   get_data_and_last_trade_date, \
                                                   return_data_query, \
                                                   fit_encoders, \
                                                   train_and_evaluate_model, \
                                                   trade_history_derived_features_dollar_price, \
                                                   save_model, \
                                                   send_results_email


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


STORAGE_CLIENT = get_storage_client()
BQ_CLIENT = get_bq_client()

OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA = {'treasury_spread': False, 
                                       'only_dollar_price_history': True}


@function_timer
def update_data() -> (pd.DataFrame, datetime.datetime, int):
    '''Updates the master data file that is used to train and deploy the model. NOTE: if any of the variables in 
    `process_data(...)` or `SEQUENCE_LENGTH_DOLLAR_PRICE_MODEL` are changed, then we need to rebuild the entire 
    `processed_data_dollar_price.pkl` since that data is will have the old preferences; an easy way to do that 
    is to manually set `last_trade_date` to a date way in the past (the desired start date of the data).'''
    file_name = 'processed_data_dollar_price.pkl'
    
    data, last_trade_date = get_data_and_last_trade_date(BUCKET_NAME, file_name)
    print(f'last trade date: {last_trade_date}')
    DATA_QUERY = return_data_query(last_trade_date, QUERY_FEATURES + ADDITIONAL_QUERY_FEATURES_FOR_DOLLAR_PRICE_MODEL, QUERY_CONDITIONS)
    file_timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')

    dp_trade_history_features = get_dp_trade_history_features()
    num_features_for_each_trade_in_history = len(dp_trade_history_features)
    data_from_last_trade_date = process_data(DATA_QUERY, 
                                             BQ_CLIENT, 
                                             SEQUENCE_LENGTH_DOLLAR_PRICE_MODEL, 
                                             num_features_for_each_trade_in_history, 
                                             f'raw_data_{file_timestamp}.pkl', 
                                             **OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA)
    
    if data_from_last_trade_date is not None:    # there is new data since `last_trade_date`
        print(f'Restricting history to {SEQUENCE_LENGTH_DOLLAR_PRICE_MODEL} trades')
        data_from_last_trade_date.trade_history = data_from_last_trade_date.trade_history.apply(lambda x: x[:SEQUENCE_LENGTH_DOLLAR_PRICE_MODEL])
        data.trade_history = data.trade_history.apply(lambda x: x[:SEQUENCE_LENGTH_DOLLAR_PRICE_MODEL])

        data_from_last_trade_date = replace_ratings_by_standalone_rating(data_from_last_trade_date)
        data_from_last_trade_date['yield'] = data_from_last_trade_date['yield'] * 100
        data_from_last_trade_date['target_attention_features'] = data_from_last_trade_date.parallel_apply(target_trade_processing_for_attention, axis=1)

        #### removing missing data
        data_from_last_trade_date['trade_history_sum'] = data_from_last_trade_date.trade_history.parallel_apply(lambda x: np.sum(x))
        data_from_last_trade_date.issue_amount = data_from_last_trade_date.issue_amount.replace([np.inf, -np.inf], np.nan)
        data_from_last_trade_date.dropna(inplace=True, subset=PREDICTORS_DOLLAR_PRICE + ['trade_history_sum'])

        print('Adding new data to master file')
        data = pd.concat([data_from_last_trade_date, data])    # concatenating `data_from_last_trade_date` to the original `data` dataframe

    ####### Adding trade history features to the data ###########
    print('Adding features from previous trade history')
    temp = data[['cusip', 'trade_history', 'quantity', 'trade_type']].parallel_apply(trade_history_derived_features_dollar_price, axis=1)
    DP_COLS = get_trade_history_columns('dollar_price')
    data[DP_COLS] = pd.DataFrame(temp.tolist(), index=data.index)
    del temp

    data.sort_values('trade_datetime', ascending=False, inplace=True)
    #############################################################
    data.dropna(inplace=True, subset=PREDICTORS_DOLLAR_PRICE)
    
    if SAVE_MODEL_AND_DATA:
        print(f'Saving data to pickle file with name {file_name}')
        data.to_pickle(file_name)  
        print(f'Uploading data to {BUCKET_NAME}/{file_name}')
        upload_data(STORAGE_CLIENT, BUCKET_NAME, file_name)
    return data, last_trade_date, num_features_for_each_trade_in_history


@function_timer
def train_model(data, last_trade_date, num_features_for_each_trade_in_history):
    encoders, fmax = fit_encoders(data, CATEGORICAL_FEATURES_DOLLAR_PRICE, 'dollar_price')

    train_data = data[data.trade_date <= last_trade_date]
    test_data = data[data.trade_date > last_trade_date]
    
    x_train = create_input(train_data, encoders, NON_CAT_FEATURES_DOLLAR_PRICE, BINARY_DOLLAR_PRICE, CATEGORICAL_FEATURES_DOLLAR_PRICE)
    y_train = train_data.dollar_price

    x_test = create_input(test_data, encoders, NON_CAT_FEATURES_DOLLAR_PRICE, BINARY_DOLLAR_PRICE, CATEGORICAL_FEATURES_DOLLAR_PRICE)
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


def main():
    print(f'\n\nFunction starting {datetime.now()}')

    print('Processing data')
    data, last_trade_date, num_features_for_each_trade_in_history = update_data()
    print('Data processed')
    
    print('Training model')
    model, encoders, mae = train_model(data, last_trade_date, num_features_for_each_trade_in_history)
    print('Training done')

    if SAVE_MODEL_AND_DATA:
        print('Saving model')
        save_model(model, encoders, STORAGE_CLIENT, dollar_price_model=True)
        print('Finished saving the model\n\n')

    print('sending email')
    send_results_email(mae, last_trade_date, EMAIL_RECIPIENTS)
    print(f'Function executed {datetime.now()}\n\n')


if __name__ == '__main__':
    main()
