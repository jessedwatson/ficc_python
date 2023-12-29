'''
 # @ Author: Mitas Ray
 # @ Create date: 2023-12-18
 # @ Modified by: Mitas Ray
 # @ Modified date: 2023-12-19
 '''
import os
import gcsfs
import shutil
import numpy as np
import pandas as pd
from sklearn import preprocessing
from pickle5 import pickle
from tensorflow import keras
from datetime import datetime

from google.cloud import bigquery
from google.cloud import storage

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from ficc.utils.gcp_storage_functions import upload_data


SAVE_MODEL_AND_DATA = True    # boolean indicating whether the trained model will be saved to google cloud storage; set to `False` if testing

EMAIL_RECIPIENTS = ['ahmad@ficc.ai', 'isaac@ficc.ai', 'jesse@ficc.ai', 'gil@ficc.ai', 'mitas@ficc.ai', 'myles@ficc.ai']    # recieve an email following a successful run of the training script; set to only your email if testing

SEQUENCE_LENGTH_YIELD_SPREAD_MODEL = 5
SEQUENCE_LENGTH_DOLLAR_PRICE_MODEL = 2
NUM_FEATURES = 6

CATEGORICAL_FEATURES_VALUES = {'purpose_class' : list(range(53 + 1)),    # possible values for `purpose_class` are 0 through 53
                               'rating' : ['A', 'A+', 'A-', 'AA', 'AA+', 'AA-', 'AAA', 'B', 'B+', 'B-', 'BB', 'BB+', 'BB-',
                                           'BBB', 'BBB+', 'BBB-', 'CC', 'CCC', 'CCC+', 'CCC-' , 'D', 'NR', 'MR'],
                               'trade_type' : ['D', 'S', 'P'],
                               'incorporated_state_code' : ['AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'GU',
                                                            'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
                                                            'MO', 'MP', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
                                                            'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'US', 'UT', 'VA', 'VI',
                                                            'VT', 'WA', 'WI', 'WV', 'WY'] }

TTYPE_DICT = { (0,0):'D', (0,1):'S', (1,0):'P' }

_VARIANTS = ['max_qty', 'min_ago', 'D_min_ago', 'P_min_ago', 'S_min_ago']
YS_VARIANTS = ['max_ys', 'min_ys'] + _VARIANTS
DP_VARIANTS = ['max_dp', 'min_dp'] + _VARIANTS
_FEATS = ['_ttypes', '_ago', '_qdiff']
YS_FEATS = ['_ys'] + _FEATS
DP_FEATS = ['_dp'] + _FEATS


def get_creds():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/ahmad/ahmad_creds.json'
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/shayaan/ficc/ahmad_creds.json'
    return None


def get_storage_client():
    get_creds()
    return storage.Client()


def get_bq_client():
    get_creds()
    return bigquery.Client()


def get_trade_history_columns(model):
    '''Creates a list of columns.'''
    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'
    if model == 'yield_spread':
        variants = YS_VARIANTS
        feats = YS_FEATS
    else:
        variants = DP_VARIANTS
        feats = DP_FEATS
    
    columns = []
    for prefix in variants:
        for suffix in feats:
            columns.append(prefix + suffix)
    return columns


def target_trade_processing_for_attention(row):
    trade_mapping = {'D': [0,0], 'S': [0,1], 'P':[1,0]}
    target_trade_features = []
    target_trade_features.append(row['quantity'])
    target_trade_features = target_trade_features + trade_mapping[row['trade_type']]
    return np.tile(target_trade_features, (1, 1))


def replace_ratings_by_standalone_rating(data):
    data.loc[data.sp_stand_alone.isna(), 'sp_stand_alone'] = 'NR'
    data.rating = data.rating.astype('str')
    data.sp_stand_alone = data.sp_stand_alone.astype('str')
    data.loc[(data.sp_stand_alone != 'NR'), 'rating'] = data[(data.sp_stand_alone != 'NR')]['sp_stand_alone'].loc[:]
    return data


def create_input(df, encoders, non_cat_features, binary_features, categorical_features):
    datalist = []
    datalist.append(np.stack(df['trade_history'].to_numpy()))
    datalist.append(np.stack(df['target_attention_features'].to_numpy()))

    noncat_and_binary = []
    for f in non_cat_features + binary_features:
        noncat_and_binary.append(np.expand_dims(df[f].to_numpy().astype('float32'), axis=1))
    datalist.append(np.concatenate(noncat_and_binary, axis=-1))
    
    for f in categorical_features:
        encoded = encoders[f].transform(df[f])
        datalist.append(encoded.astype('float32'))

    return datalist


def get_data_and_last_trade_date(bucket_name, file_name):
    '''Get the dataframe from `bucket_name/file_name` and the most recent trade date from this dataframe.'''
    print(f'Downloading data from {bucket_name}/{file_name}')
    fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
    with fs.open(f'{bucket_name}/{file_name}') as f:
        data = pd.read_pickle(f)
    print(f'Data downloaded from {bucket_name}/{file_name}')
    
    last_trade_date = data.trade_date.max().date().strftime('%Y-%m-%d')
    return data, last_trade_date


def fit_encoders(data:pd.DataFrame, categorical_features:list, model:str):
    '''Fits label encoders to categorical features in the data. For a few of the categorical features, the values 
    don't change for these features we use the pre-defined set of values specified in `CATEGORICAL_FEATURES_VALUES`. 
    Outputs a tuple of dictionaries where the first item is the encoders and the second item is the maximum value 
    for each class.'''
    assert model in ('yield_spread', 'dollar_price'), f'Model should be either yield_spread or dollar_price, but was instead: {model}'
    if model == 'yield_spread':
        filename = 'encoders.pkl'
    else:
        filename = 'encoders_dollar_price.pkl'
    
    encoders = {}
    fmax = {}
    for feature in categorical_features:
        if feature in CATEGORICAL_FEATURES_VALUES:
            fprep = preprocessing.LabelEncoder().fit(CATEGORICAL_FEATURES_VALUES[feature])
        else:
            fprep = preprocessing.LabelEncoder().fit(data[feature].drop_duplicates())
        fmax[feature] = np.max(fprep.transform(fprep.classes_))
        encoders[feature] = fprep
    
    with open(filename, 'wb') as file:
        pickle.dump(encoders, file)
    return encoders, fmax


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    fit_callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=20,
                                                   verbose=0,
                                                   mode='auto',
                                                   restore_best_weights=True)]

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss=keras.losses.MeanAbsoluteError(),
                  metrics=[keras.metrics.MeanAbsoluteError()])

    history = model.fit(x_train, 
                        y_train, 
                        epochs=100, 
                        batch_size=1000,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=fit_callbacks,
                        use_multiprocessing=True,
                        workers=8) 

    _, mae = model.evaluate(x_test, 
                            y_test, 
                            verbose=1, 
                            batch_size=1000)
    return model, mae, history

def save_model(model, encoders, storage_client, dollar_price_model):
    '''`dollar_price_model` is a boolean flag that indicates whether we are 
    working with the dollar price model instead of hte yield spread model.'''
    suffix = '_dollar_price' if dollar_price_model else ''
    suffix_wo_underscore = 'dollar_price' if dollar_price_model else ''    # need this variable as well since the model naming is missing an underscore

    file_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    print(f'file time stamp: {file_timestamp}')

    print('Saving encoders and uploading encoders')
    encoders_filename = f'encoders{suffix}.pkl'
    with open(encoders_filename, 'wb') as file:
        pickle.dump(encoders, file)    
    upload_data(storage_client, 'ahmad_data', encoders_filename)

    print('Saving and uploading model')
    model_filename = f'saved_model_{suffix_wo_underscore}{file_timestamp}'
    model.save(model_filename)
    
    model_zip_filename = f'model{suffix}'
    shutil.make_archive(model_zip_filename, 'zip', model_filename)
    # shutil.make_archive(f'saved_model_{file_timestamp}', 'zip', f'saved_model_{file_timestamp}')
    
    upload_data(storage_client, 'ahmad_data', model_zip_filename)
    # upload_data(storage_client, 'ahmad_data/yield_spread_models', f'saved_model_{file_timestamp}.zip')
    os.system(f'rm -r {model_filename}')


def send_results_email(mae, last_trade_date, recipients:list):
    sender_email = 'notifications@ficc.ai'
    
    msg = MIMEMultipart()
    msg['Subject'] = f'Mae for model trained till {last_trade_date}'
    msg['From'] = sender_email

    message = MIMEText(f'The MAE for the model on trades that occurred on {last_trade_date} is {np.round(mae,3)}.', 'plain')
    msg.attach(message)

    smtp_server = 'smtp.gmail.com'
    port = 587

    with smtplib.SMTP(smtp_server, port) as server:
        try:
            server.starttls()
            server.login(sender_email, 'ztwbwrzdqsucetbg')
            for receiver in recipients:
                server.sendmail(sender_email, receiver, msg.as_string())
        except Exception as e:
            print(e)
        finally:
            server.quit()
