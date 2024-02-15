'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2023-01-23
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-02-15
 '''
import numpy as np
import pandas as pd
from google.cloud import bigquery
from ficc.utils.auxiliary_functions import function_timer
from ficc.utils.auxiliary_variables import PREDICTORS, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES
from datetime import datetime
from yield_model import yield_spread_model

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from automated_training_auxiliary_functions import NUM_TRADES_IN_HISTORY_YIELD_SPREAD_MODEL, \
                                                   EASTERN, \
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
                                                   create_input, \
                                                   get_trade_date_where_data_exists_after_this_date, \
                                                   fit_encoders, \
                                                   train_and_evaluate_model, \
                                                   save_model, \
                                                   remove_file, \
                                                   send_email, \
                                                   send_no_new_model_email


setup_gpus()


STORAGE_CLIENT = get_storage_client()
BQ_CLIENT = get_bq_client()

HISTORICAL_PREDICTION_TABLE = 'eng-reactor-287421.historic_predictions.historical_predictions'

OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA = {'use_treasury_spread': True, 
                                       'only_dollar_price_history': False}


if 'ficc_treasury_spread' not in PREDICTORS: PREDICTORS.append('ficc_treasury_spread')
if 'ficc_treasury_spread' not in NON_CAT_FEATURES: NON_CAT_FEATURES.append('ficc_treasury_spread')
if 'target_attention_features' not in PREDICTORS: PREDICTORS.append('target_attention_features')


def get_table_schema_for_predictions():
    '''Returns the schema required for the bigquery table storing the predictions.'''
    schema = [
        bigquery.SchemaField('rtrs_control_number', 'INTEGER', 'REQUIRED'),
        bigquery.SchemaField('cusip', 'STRING', 'REQUIRED'),
        bigquery.SchemaField('trade_date', 'DATE', 'REQUIRED'),
        bigquery.SchemaField('dollar_price', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('yield','FLOAT', 'REQUIRED'),
        bigquery.SchemaField('new_ficc_ycl', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('new_ys', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('new_ys_prediction', 'FLOAT', 'REQUIRED'),
        bigquery.SchemaField('prediction_datetime', 'DATETIME', 'REQUIRED')
    ] 
    return schema


def upload_predictions(data: pd.DataFrame):
    '''Upload the coefficient and scalar dataframeto BigQuery.'''
    job_config = bigquery.LoadJobConfig(schema=get_table_schema_for_predictions(), write_disposition='WRITE_APPEND')
    job = BQ_CLIENT.load_table_from_dataframe(data, HISTORICAL_PREDICTION_TABLE, job_config=job_config)
    try:
        job.result()
        print('Upload Successful')
    except Exception as e:
        print('Failed to Upload')
        raise e


def update_data() -> (pd.DataFrame, datetime, int):
    '''Updates the master data file that is used to train and deploy the model. NOTE: if any of the variables in 
    `process_data(...)` or `NUM_TRADES_IN_HISTORY_YIELD_SPREAD_MODEL` are changed, then we need to rebuild the entire `processed_data_test.pkl` 
    since that data is will have the old preferences; an easy way to do that is to manually set `last_trade_date` to a 
    date way in the past (the desired start date of the data).'''
    file_name = 'processed_data_test.pkl'
    use_treasury_spread = OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA.get('use_treasury_spread', False)
    data_before_last_trade_datetime, data_from_last_trade_datetime, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath = get_new_data(file_name, 
                                                                                                                                                              'yield_spread', 
                                                                                                                                                              BQ_CLIENT, 
                                                                                                                                                              use_treasury_spread, 
                                                                                                                                                              OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA)
    data = combine_new_data_with_old_data(data_before_last_trade_datetime, data_from_last_trade_datetime, 'yield_spread')
    print(f'Number of data points after combining new and old data: {len(data)}')
    data = add_trade_history_derived_features(data, 'yield_spread', use_treasury_spread)
    data = drop_features_with_null_value(data, PREDICTORS)
    if SAVE_MODEL_AND_DATA: save_data(data, file_name, STORAGE_CLIENT)
    return data, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath


def segment_results(data: pd.DataFrame):
    data['delta'] = np.abs(data.predicted_ys - data.new_ys)
    delta = data['delta']

    def get_mae_and_count(condition):
        return np.round(np.mean(delta[condition]), 3), data[condition].shape[0]    # round mae to 3 digits after the decimal point to reduce noise

    investment_grade_ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']
    total_mae, total_count = np.mean(delta), data.shape[0] 
    
    inter_dealer = data.trade_type == 'D'
    dd_mae, dd_count = get_mae_and_count(inter_dealer)
    dealer_purchase = data.trade_type == 'P'
    dp_mae, dp_count = get_mae_and_count(dealer_purchase)
    dealer_sell = data.trade_type == 'S'
    ds_mae, ds_count = get_mae_and_count(dealer_sell)
    aaa = data.rating == 'AAA'
    aaa_mae, aaa_count = get_mae_and_count(aaa)
    investment_grade = data.rating.isin(investment_grade_ratings)
    investment_grade_mae, investment_grade_count = get_mae_and_count(investment_grade)
    par_traded_greater_than_or_equal_to_100k = data.par_traded >= 1e5
    hundred_k_mae, hundred_k_count = get_mae_and_count(par_traded_greater_than_or_equal_to_100k)

    result_df = pd.DataFrame(data=[[total_mae, total_count],
                                   [dd_mae, dd_count],
                                   [dp_mae, dp_count],
                                   [ds_mae, ds_count], 
                                   [aaa_mae, aaa_count], 
                                   [investment_grade_mae, investment_grade_count],
                                   [hundred_k_mae, hundred_k_count]],
                             columns=['Mean absolute Error', 'Trade count'],
                             index=['Entire set', 'Dealer-Dealer', 'Dealer-Purchase', 'Dealer-Sell', 'AAA', 'Investment Grade', 'Trade size >= 100k'])
    return result_df


def apply_exclusions(data: pd.DataFrame):
    data_before_exclusions = data[:]
    data = data[(data.days_to_call == 0) | (data.days_to_call > np.log10(400))]
    data = data[(data.days_to_refund == 0) | (data.days_to_refund > np.log10(400))]
    data = data[(data.days_to_maturity == 0) | (data.days_to_maturity > np.log10(400))]
    data = data[data.days_to_maturity < np.log10(30000)]
    data = data[~data.last_calc_date.isna()]
    return data, data_before_exclusions


@function_timer
def train_model(data: pd.DataFrame, last_trade_date, num_features_for_each_trade_in_history: int):
    encoders, fmax = fit_encoders(data, CATEGORICAL_FEATURES, 'yield_spread')

    if TESTING: last_trade_date = get_trade_date_where_data_exists_after_this_date(last_trade_date, data, exclusions_function=apply_exclusions)
    test_data = data[data.trade_date > last_trade_date]
    test_data, test_data_before_exclusions = apply_exclusions(test_data)
    if len(test_data) == 0:
        print(f'No model is trained since there are no trades in `test_data`; `train_model(...)` is terminated')
        return None, None, None

    train_data = data[data.trade_date <= last_trade_date]
    print(f'Training set contains {len(train_data)} trades ranging from trade datetimes of {train_data.trade_datetime.min()} to {train_data.trade_datetime.max()}')
    print(f'Test set contains {len(test_data)} trades ranging from trade datetimes of {test_data.trade_datetime.min()} to {test_data.trade_datetime.max()}')
    
    x_train = create_input(train_data, encoders, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES, 'yield_spread')
    y_train = train_data.new_ys

    x_test = create_input(test_data, encoders, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES, 'yield_spread')
    y_test = test_data.new_ys

    model = yield_spread_model(x_train, 
                               NUM_TRADES_IN_HISTORY_YIELD_SPREAD_MODEL, 
                               num_features_for_each_trade_in_history,
                               CATEGORICAL_FEATURES, 
                               NON_CAT_FEATURES, 
                               BINARY, 
                               fmax)

    model, mae, history = train_and_evaluate_model(model, x_train, y_train, x_test, y_test)

    # creating table to send over email
    try:
        test_data['predicted_ys'] = model.predict(x_test, batch_size=1000)
        result_df = segment_results(test_data)
    except Exception as e:
        print(e)
        result_df = pd.DataFrame()

    try:
        print(result_df.to_markdown())
    except Exception as e:
        print('Need to run `pip install tabulate` on this machine in orer to display the dataframe in an easy to read way')

    # uploading predictions to bigquery
    if SAVE_MODEL_AND_DATA:
        try:
            test_data_before_exclusions_x_test = create_input(test_data_before_exclusions, encoders, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES, 'yield_spread')
            test_data_before_exclusions['new_ys_prediction'] = model.predict(test_data_before_exclusions_x_test, batch_size=1000)
            test_data_before_exclusions = test_data_before_exclusions[['rtrs_control_number', 'cusip', 'trade_date', 'dollar_price', 'yield', 'new_ficc_ycl', 'new_ys', 'new_ys_prediction']]
            test_data_before_exclusions['prediction_datetime'] = pd.to_datetime(datetime.now().replace(microsecond=0))
            test_data_before_exclusions['trade_date'] = pd.to_datetime(test_data_before_exclusions['trade_date']).dt.date
            upload_predictions(test_data_before_exclusions)
        except Exception as e:
            print('Failed to upload predictions to BigQuery')
            print(e)
    return model, encoders, mae, result_df


def send_results_email_table(result_df, last_trade_date, recipients: list, model: str):
    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'
    print(f'Sending email to {recipients}')
    sender_email = 'notifications@ficc.ai'
    
    msg = MIMEMultipart()
    msg['Subject'] = f'Mae for {model} model trained till {last_trade_date}'
    msg['From'] = sender_email

    html_table = result_df.to_html(index=True)
    body = MIMEText(html_table, 'html')
    msg.attach(body)
    send_email(sender_email, msg, recipients)


@function_timer
def main():
    print(f'automated_training_yield_spread_model.py starting at {datetime.now(EASTERN)} ET')
    data, last_trade_date, num_features_for_each_trade_in_history, raw_data_filepath = save_update_data_results_to_pickle_files('yield_spread', update_data)
    model, encoders, mae, result_df = train_model(data, last_trade_date, num_features_for_each_trade_in_history)

    print(f'Removing {raw_data_filepath} since training is complete')
    remove_file(raw_data_filepath)
    
    if not TESTING and model is None:
        send_no_new_model_email(last_trade_date, EMAIL_RECIPIENTS, 'yield_spread')
        raise RuntimeError('No new data was found. Raising an error so that the shell script terminates.')
    else:
        if SAVE_MODEL_AND_DATA: save_model(model, encoders, STORAGE_CLIENT, dollar_price_model=False)
        try:
            if not TESTING: send_results_email_table(result_df, last_trade_date, EMAIL_RECIPIENTS, 'yield_spread')
            # send_results_email(mae, last_trade_date, EMAIL_RECIPIENTS, 'yield_spread')
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
