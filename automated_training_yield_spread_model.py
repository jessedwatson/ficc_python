'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2023-01-23
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-01-09
 '''
import numpy as np
import pandas as pd
from google.cloud import bigquery
from ficc.utils.auxiliary_functions import function_timer
from ficc.utils.auxiliary_variables import PREDICTORS, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES
from datetime import datetime
from yield_model import yield_spread_model

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from automated_training_auxiliary_functions import SEQUENCE_LENGTH_YIELD_SPREAD_MODEL, \
                                                   SAVE_MODEL_AND_DATA, \
                                                   EMAIL_RECIPIENTS, \
                                                   get_storage_client, \
                                                   get_bq_client, \
                                                   setup_gpus, \
                                                   get_new_data, \
                                                   combine_new_data_with_old_data, \
                                                   add_trade_history_derived_features, \
                                                   save_data, \
                                                   create_input, \
                                                   fit_encoders, \
                                                   train_and_evaluate_model, \
                                                   save_model


setup_gpus()


STORAGE_CLIENT = get_storage_client()
BQ_CLIENT = get_bq_client()

HISTORICAL_PREDICTION_TABLE = 'eng-reactor-287421.historic_predictions.historical_predictions'

OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA = {'treasury_spread': True, 
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


def upload_predictions(data:pd.DataFrame):
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
    `process_data(...)` or `SEQUENCE_LENGTH_YIELD_SPREAD_MODEL` are changed, then we need to rebuild the entire `processed_data_test.pkl` 
    since that data is will have the old preferences; an easy way to do that is to manually set `last_trade_date` to a 
    date way in the past (the desired start date of the data).'''
    file_name = 'processed_data_test.pkl'
    using_treasury_spread = OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA.get('treasury_spread', False)
    data_before_last_trade_date, data_from_last_trade_date, last_trade_date, num_features_for_each_trade_in_history = get_new_data(file_name, 
                                                                                                                                   'yield_spread', 
                                                                                                                                   BQ_CLIENT, 
                                                                                                                                   using_treasury_spread, 
                                                                                                                                   OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA)
    data = combine_new_data_with_old_data(data_before_last_trade_date, data_from_last_trade_date, PREDICTORS, 'yield_spread')
    data = add_trade_history_derived_features(data, 'yield_spread', using_treasury_spread)
    data.dropna(inplace=True, subset=PREDICTORS)
    if SAVE_MODEL_AND_DATA: save_data(data, file_name, STORAGE_CLIENT)
    return data, last_trade_date, num_features_for_each_trade_in_history


def segment_results(data):
    data['delta'] = np.abs(data.predicted_ys - data.new_ys)

    investment_grade = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']

    total_mae, total_count = np.mean(data.delta), data.shape[0] 

    dd_mae, dd_count = np.mean(data['delta'][data.trade_type == 'D']), data[data.trade_type == 'D'].shape[0]
    dp_mae, dp_count = np.mean(data['delta'][data.trade_type == 'P']), data[data.trade_type == 'P'].shape[0]
    ds_mae, ds_count = np.mean(data['delta'][data.trade_type == 'S']), data[data.trade_type == 'S'].shape[0]

    AAA_mae, AAA_count = np.mean(data['delta'][data.rating == 'AAA']), data[data.rating == 'AAA'].shape[0]
    investment_grade_mae, investment_grade_count = np.mean(data['delta'][data.rating.isin(investment_grade)]), data[data.rating.isin(investment_grade)].shape[0]
    hundred_k_mae, hundred_k_count = np.mean(data['delta'][data.par_traded >= 1e5]), data[data.par_traded >= 1e5].shape[0]

    result_df = pd.DataFrame(data=[[total_mae, total_count],
                                   [dd_mae, dd_count],
                                   [dp_mae, dp_count],
                                   [ds_mae, ds_count], 
                                   [AAA_mae, AAA_count], 
                                   [investment_grade_mae, investment_grade_count],
                                   [hundred_k_mae, hundred_k_count]],
                             columns=['Mean absolute Error', 'Trade count'],
                             index=['Entire set', 'Dealer-Dealer', 'Dealer-Purchase', 'Dealer-Sell', 'AAA', 'Investment Grade', 'Trade size > 100k'])
    return result_df


@function_timer
def train_model(data, last_trade_date, num_features_for_each_trade_in_history):
    encoders, fmax = fit_encoders(data, CATEGORICAL_FEATURES, 'yield_spread')

    train_data = data[data.trade_date <= last_trade_date]
    test_data = data[data.trade_date > last_trade_date]
    prediction_data = test_data[:]    # `test_data` before exclusions applied
    
    test_data = test_data[(test_data.days_to_call == 0) | (test_data.days_to_call > np.log10(400))]
    test_data = test_data[(test_data.days_to_refund == 0) | (test_data.days_to_refund > np.log10(400))]
    test_data = test_data[(test_data.days_to_maturity == 0) | (test_data.days_to_maturity > np.log10(400))]
    test_data = test_data[test_data.days_to_maturity < np.log10(30000)]
    test_data = test_data[~test_data.last_calc_date.isna()]
    
    x_train = create_input(train_data, encoders, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES)
    y_train = train_data.new_ys

    x_test = create_input(test_data, encoders, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES)
    y_test = test_data.new_ys

    model = yield_spread_model(x_train, 
                               SEQUENCE_LENGTH_YIELD_SPREAD_MODEL, 
                               num_features_for_each_trade_in_history,
                               CATEGORICAL_FEATURES, 
                               NON_CAT_FEATURES, 
                               BINARY, 
                               fmax)

    model, mae, history = train_and_evaluate_model(model, x_train, y_train, x_test, y_test)

    ## Creating table to send over email    
    try:
        test_data['predicted_ys'] = model.predict(x_test, batch_size=1000)
        result_df = segment_results(test_data)
    except Exception as e:
        print(e)
        result_df = pd.DataFrame()

    ## Uploading prediction to BQ
    if SAVE_MODEL_AND_DATA:
        try:
            prediction_data_x_test = create_input(prediction_data, encoders, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES)
            prediction_data['new_ys_prediction'] = model.predict(prediction_data_x_test, batch_size=1000)
            prediction_data = prediction_data[['rtrs_control_number', 'cusip', 'trade_date', 'dollar_price', 'yield', 'new_ficc_ycl', 'new_ys', 'new_ys_prediction']]
            prediction_data['prediction_datetime'] = pd.to_datetime(datetime.now().replace(microsecond=0))
            prediction_data['trade_date'] = pd.to_datetime(prediction_data['trade_date']).dt.date
            upload_predictions(prediction_data)
        except Exception as e:
            print('Failed to upload predictions to BigQuery')
            print(e)
    return model, encoders, mae, result_df           


def send_results_email_table(result_df, last_trade_date, recipients:list):
    print(f'Sending email to {recipients}')
    sender_email = 'notifications@ficc.ai'
    
    msg = MIMEMultipart()
    msg['Subject'] = f'Mae for model trained till {last_trade_date}'
    msg['From'] = sender_email

    html_table = result_df.to_html(index=True)
    body = MIMEText(html_table, 'html')
    msg.attach(body)

    smtp_server = 'smtp.gmail.com'
    port = 587

    with smtplib.SMTP(smtp_server, port) as server:
        try:
            server.starttls()
            server.login(sender_email, 'ztwbwrzdqsucetbg')
            for receiver in EMAIL_RECIPIENTS:
                server.sendmail(sender_email, receiver, msg.as_string())
        except Exception as e:
            print(e)
        finally:
            server.quit()


@function_timer
def main():
    print(f'\n\nautomated_training_yield_spread_model.py starting at {datetime.now()}')
    data, last_trade_date, num_features_for_each_trade_in_history = update_data()
    model, encoders, mae, result_df = train_model(data, last_trade_date, num_features_for_each_trade_in_history)

    if SAVE_MODEL_AND_DATA:
        print('Saving model')
        save_model(model, encoders, STORAGE_CLIENT, dollar_price_model=False)
        print('Finished saving the model\n\n')

    # send_results_email(mae, last_trade_date)
    try:
        send_results_email_table(result_df, last_trade_date, EMAIL_RECIPIENTS)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
