'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2023-01-23
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-01-08
 '''
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import bigquery
from ficc.data.process_data import process_data
from ficc.utils.auxiliary_functions import sqltodf
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.auxiliary_variables import PREDICTORS, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES, NUM_OF_DAYS_IN_YEAR
from ficc.utils.nelson_siegel_model import yield_curve_level
from ficc.utils.gcp_storage_functions import upload_data
from datetime import datetime
from yield_model import yield_spread_model

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from automated_training_auxiliary_functions import NUM_FEATURES, \
                                                   SEQUENCE_LENGTH_YIELD_SPREAD_MODEL, \
                                                   QUERY_FEATURES, \
                                                   QUERY_CONDITIONS, \
                                                   ADDITIONAL_QUERY_CONDITIONS_FOR_YIELD_SPREAD_MODEL, \
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
                                                   trade_history_derived_features_yield_spread, \
                                                   train_and_evaluate_model, \
                                                   save_model


print('***********************')
if tf.test.is_gpu_available():
    print('****** USING GPU ******')
else:
    print('** NO GPU AVAILABLE ***')
print('***********************')


STORAGE_CLIENT = get_storage_client()
BQ_CLIENT = get_bq_client()

OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA = {'treasury_spread': True, 
                                       'only_dollar_price_history': False}


if 'ficc_treasury_spread' not in PREDICTORS: PREDICTORS.append('ficc_treasury_spread')
if 'ficc_treasury_spread' not in NON_CAT_FEATURES: NON_CAT_FEATURES.append('ficc_treasury_spread')
if 'target_attention_features' not in PREDICTORS: PREDICTORS.append('target_attention_features')


HISTORICAL_PREDICTION_TABLE = 'eng-reactor-287421.historic_predictions.historical_predictions'


nelson_params = sqltodf('SELECT * FROM `eng-reactor-287421.ahmad_test.nelson_siegel_coef_daily` order by date desc', BQ_CLIENT)
nelson_params.set_index('date', drop=True, inplace=True)
nelson_params = nelson_params[~nelson_params.index.duplicated(keep='first')]
nelson_params = nelson_params.transpose().to_dict()

scalar_params = sqltodf('SELECT * FROM `eng-reactor-287421.ahmad_test.standardscaler_parameters_daily` order by date desc', BQ_CLIENT)
scalar_params.set_index('date', drop=True, inplace=True)
scalar_params = scalar_params[~scalar_params.index.duplicated(keep='first')]
scalar_params = scalar_params.transpose().to_dict()

shape_parameter = sqltodf('SELECT * FROM `eng-reactor-287421.ahmad_test.shape_parameters` order by Date desc', BQ_CLIENT)
shape_parameter.set_index('Date', drop=True, inplace=True)
shape_parameter = shape_parameter[~shape_parameter.index.duplicated(keep='first')]
shape_parameter = shape_parameter.transpose().to_dict()


def get_table_schema():
    '''Returns the schema required for the bigquery table storing the nelson siegel coefficients.'''
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
    job_config = bigquery.LoadJobConfig(schema=get_table_schema(), write_disposition='WRITE_APPEND')
    job = BQ_CLIENT.load_table_from_dataframe(data, HISTORICAL_PREDICTION_TABLE, job_config=job_config)
    try:
        job.result()
        print('Upload Successful')
    except Exception as e:
        print('Failed to Upload')
        raise e


def get_yield_for_last_duration(row):
    if pd.isnull(row['last_calc_date'])or pd.isnull(row['last_trade_date']):
        # if there is no last trade, we use the duration of the current bond
        duration =  diff_in_days_two_dates(row['maturity_date'], row['trade_date']) / NUM_OF_DAYS_IN_YEAR
        ycl = yield_curve_level(duration, row['trade_date'].date(), nelson_params, scalar_params, shape_parameter) / 100
        return ycl
    duration =  diff_in_days_two_dates(row['last_calc_date'], row['last_trade_date']) / NUM_OF_DAYS_IN_YEAR
    ycl = yield_curve_level(duration, row['trade_date'].date(), nelson_params, scalar_params, shape_parameter) / 100
    return ycl


def update_data() -> (pd.DataFrame, datetime):
    '''Updates the master data file that is used to train and deploy the model. NOTE: if any of the variables in 
    `process_data(...)` or `SEQUENCE_LENGTH_YIELD_SPREAD_MODEL` are changed, then we need to rebuild the entire `processed_data_test.pkl` 
    since that data is will have the old preferences; an easy way to do that is to manually set `last_trade_date` to a 
    date way in the past (the desired start date of the data).'''
    file_name = 'processed_data_test.pkl'

    data, last_trade_date = get_data_and_last_trade_date(BUCKET_NAME, file_name)
    print(f'last trade date: {last_trade_date}')
    DATA_QUERY = return_data_query(last_trade_date, QUERY_FEATURES, ADDITIONAL_QUERY_CONDITIONS_FOR_YIELD_SPREAD_MODEL + QUERY_CONDITIONS)
    file_timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')

    data_from_last_trade_date = process_data(DATA_QUERY, 
                                             BQ_CLIENT, 
                                             SEQUENCE_LENGTH_YIELD_SPREAD_MODEL, 
                                             NUM_FEATURES, 
                                             f'raw_data_{file_timestamp}.pkl', 
                                             **OPTIONAL_ARGUMENTS_FOR_PROCESS_DATA)

    if data_from_last_trade_date is not None:    # there is new data since `last_trade_date`
        print(f'Restricting history to {SEQUENCE_LENGTH_YIELD_SPREAD_MODEL} trades')
        data_from_last_trade_date.trade_history = data_from_last_trade_date.trade_history.apply(lambda x: x[:SEQUENCE_LENGTH_YIELD_SPREAD_MODEL])
        data.trade_history = data.trade_history.apply(lambda x: x[:SEQUENCE_LENGTH_YIELD_SPREAD_MODEL])

        data_from_last_trade_date = replace_ratings_by_standalone_rating(data_from_last_trade_date)
        data_from_last_trade_date['yield'] = data_from_last_trade_date['yield'] * 100
        data_from_last_trade_date['last_trade_date'] = data_from_last_trade_date['last_trade_datetime'].dt.date
        data_from_last_trade_date['new_ficc_ycl'] = data_from_last_trade_date[['last_calc_date',
                                                                            'last_settlement_date',
                                                                            'trade_date',
                                                                            'last_trade_date',
                                                                            'maturity_date']].parallel_apply(get_yield_for_last_duration, axis=1)

        data_from_last_trade_date['new_ficc_ycl'] = data_from_last_trade_date['new_ficc_ycl'] * 100
        data_from_last_trade_date['target_attention_features'] = data_from_last_trade_date.parallel_apply(target_trade_processing_for_attention, axis=1)

        #### removing missing data
        data_from_last_trade_date['trade_history_sum'] = data_from_last_trade_date.trade_history.parallel_apply(lambda x: np.sum(x))
        data_from_last_trade_date.issue_amount = data_from_last_trade_date.issue_amount.replace([np.inf, -np.inf], np.nan)
        data_from_last_trade_date.dropna(inplace=True, subset=PREDICTORS + ['trade_history_sum'])

        print('Adding new data to master file')
        data = pd.concat([data_from_last_trade_date, data])    # concatenating `data_from_last_trade_date` to the original `data` dataframe
        data['new_ys'] =  data['yield'] - data['new_ficc_ycl']

    ####### Adding trade history features to the data ###########
    print('Adding features from previous trade history')
    data.sort_values('trade_datetime', inplace=True)
    temp = data[['cusip', 'trade_history', 'quantity', 'trade_type']].parallel_apply(trade_history_derived_features_yield_spread, axis=1)
    YS_COLS = get_trade_history_columns('yield_spread')
    data[YS_COLS] = pd.DataFrame(temp.tolist(), index=data.index)
    del temp
            
    data.sort_values('trade_datetime', ascending=False, inplace=True)
    #############################################################
    data.dropna(inplace=True, subset=PREDICTORS)

    if SAVE_MODEL_AND_DATA:
        print(f'Saving data to pickle file with name {file_name}')
        data.to_pickle(file_name)  
        print(f'Uploading data to {BUCKET_NAME}/{file_name}')
        upload_data(STORAGE_CLIENT, BUCKET_NAME, file_name)
    return data, last_trade_date


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


def train_model(data, last_trade_date):
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
                               NUM_FEATURES,
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


def send_results_email_table(result_df, last_trade_date):
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


def main():
    print(f'\n\nFunction starting {datetime.now()}')

    print('Processing data')
    data, last_trade_date = update_data()
    print('Data processed')
    
    print('Training model')
    model, encoders, mae, result_df = train_model(data, last_trade_date)
    print('Training done')

    if SAVE_MODEL_AND_DATA:
        print('Saving model')
        save_model(model, encoders, STORAGE_CLIENT, dollar_price_model=False)
        print('Finished saving the model\n\n')

    print('sending email')
    # send_results_email(mae, last_trade_date)
    try:
        send_results_email_table(result_df, last_trade_date)
    except Exception as e:
        print(e)
    print(f'Function executed {datetime.now()}')


if __name__ == '__main__':
    main()
