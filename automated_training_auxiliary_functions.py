'''
 # @ Author: Mitas Ray
 # @ Create date: 2023-12-18
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-01-09
 '''
import os
import gcsfs
import shutil
import numpy as np
import pandas as pd
from sklearn import preprocessing
from pickle5 import pickle
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

from google.cloud import bigquery
from google.cloud import storage

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from ficc.utils.gcp_storage_functions import upload_data
from ficc.utils.auxiliary_functions import get_ys_trade_history_features, get_dp_trade_history_features


SAVE_MODEL_AND_DATA = True    # boolean indicating whether the trained model will be saved to google cloud storage; set to `False` if testing

EMAIL_RECIPIENTS = ['ahmad@ficc.ai', 'isaac@ficc.ai', 'jesse@ficc.ai', 'gil@ficc.ai', 'mitas@ficc.ai', 'myles@ficc.ai']    # recieve an email following a successful run of the training script; set to only your email if testing

BUCKET_NAME = 'automated_training'


def get_creds():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/mitas/ficc/mitas_creds.json'    # '/home/ahmad/ahmad_creds.json'
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/shayaan/ficc/ahmad_creds.json'
    return None


def get_storage_client():
    get_creds()
    return storage.Client()


def get_bq_client():
    get_creds()
    return bigquery.Client()


def setup_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) == 0:
        print('No GPUs')
    else:
        for gpu in gpus:    # https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
            tf.config.experimental.set_memory_growth(gpu, True)


SEQUENCE_LENGTH_YIELD_SPREAD_MODEL = 5
SEQUENCE_LENGTH_DOLLAR_PRICE_MODEL = 2

CATEGORICAL_FEATURES_VALUES = {'purpose_class' : list(range(53 + 1)),    # possible values for `purpose_class` are 0 through 53
                               'rating' : ['A', 'A+', 'A-', 'AA', 'AA+', 'AA-', 'AAA', 'B', 'B+', 'B-', 'BB', 'BB+', 'BB-',
                                           'BBB', 'BBB+', 'BBB-', 'CC', 'CCC', 'CCC+', 'CCC-' , 'D', 'NR', 'MR'],
                               'trade_type' : ['D', 'S', 'P'],
                               'incorporated_state_code' : ['AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'GU',
                                                            'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
                                                            'MO', 'MP', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
                                                            'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'US', 'UT', 'VA', 'VI',
                                                            'VT', 'WA', 'WI', 'WV', 'WY'] }

TTYPE_DICT = {(0, 0): 'D', (0, 1): 'S', (1, 0): 'P'}

_VARIANTS = ['max_qty', 'min_ago', 'D_min_ago', 'P_min_ago', 'S_min_ago']
YS_VARIANTS = ['max_ys', 'min_ys'] + _VARIANTS
DP_VARIANTS = ['max_dp', 'min_dp'] + _VARIANTS
_FEATS = ['_ttypes', '_ago', '_qdiff']
YS_FEATS = ['_ys'] + _FEATS
DP_FEATS = ['_dp'] + _FEATS

LONG_TIME_AGO_IN_NUM_SECONDS = 9    # default `num_seconds_ago` value to signify that the trade was a long time ago (9 is a large value since the `num_seconds_ago` is log10 transformed)

QUERY_FEATURES = ['rtrs_control_number',
                  'cusip',
                  'yield',
                  'is_callable',
                  'refund_date',
                  'accrual_date',
                  'dated_date',
                  'next_sink_date',
                  'coupon',
                  'delivery_date',
                  'trade_date',
                  'trade_datetime',
                  'par_call_date',
                  'interest_payment_frequency',
                  'is_called',
                  'is_non_transaction_based_compensation',
                  'is_general_obligation',
                  'callable_at_cav',
                  'extraordinary_make_whole_call',
                  'make_whole_call',
                  'has_unexpired_lines_of_credit',
                  'escrow_exists',
                  'incorporated_state_code',
                  'trade_type',
                  'par_traded',
                  'maturity_date',
                  'settlement_date',
                  'next_call_date',
                  'issue_amount',
                  'maturity_amount',
                  'issue_price',
                  'orig_principal_amount',
                  'max_amount_outstanding',
                  'recent',
                  'dollar_price',
                  'calc_date',
                  'purpose_sub_class',
                  'called_redemption_type',
                  'calc_day_cat',
                  'previous_coupon_payment_date',
                  'instrument_primary_name',
                  'purpose_class',
                  'call_timing',
                  'call_timing_in_part',
                  'sink_frequency',
                  'sink_amount_type',
                  'issue_text',
                  'state_tax_status',
                  'series_name',
                  'transaction_type',
                  'next_call_price',
                  'par_call_price',
                  'when_issued',
                  'min_amount_outstanding',
                  'original_yield',
                  'par_price',
                  'default_indicator',
                  'sp_stand_alone',
                  'sp_long',
                  'moodys_long',
                  'coupon_type',
                  'federal_tax_status',
                  'use_of_proceeds',
                  'muni_security_type',
                  'muni_issue_type',
                  'capital_type',
                  'other_enhancement_type',
                  'next_coupon_payment_date',
                  'first_coupon_date',
                  'last_period_accrues_from_date']
ADDITIONAL_QUERY_FEATURES_FOR_DOLLAR_PRICE_MODEL = ['refund_price', 'publish_datetime', 'maturity_description_code']    # these features were used for testing, but are not needed, nonetheless, we keep them since the previous data files have these fields and `pd.concat(...)` will fail if the column set is different

QUERY_CONDITIONS = ['par_traded >= 10000', 
                    'coupon_type in (8, 4, 10, 17)', 
                    'capital_type <> 10', 
                    'default_exists <> TRUE', 
                    'most_recent_default_event IS NULL', 
                    'default_indicator IS FALSE', 
                    'msrb_valid_to_date > current_date',    # condition to remove cancelled trades
                    'settlement_date IS NOT NULL']
ADDITIONAL_QUERY_CONDITIONS_FOR_YIELD_SPREAD_MODEL = ['yield IS NOT NULL', 'yield > 0']

NUM_EPOCHS = 100
BATCH_SIZE = 1000
DROPOUT = 0.01

TESTING = False
if TESTING:
    SAVE_MODEL_AND_DATA = False
    print('Check get_creds(...) to make sure the credentials filepath are correct')
    NUM_EPOCHS = 10


D_prev = dict()
P_prev = dict()
S_prev = dict()


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


def return_data_query(last_trade_date, features, conditions):
    features_as_string = ', '.join(features)
    conditions = conditions + [f'trade_date > "{last_trade_date}"']
    conditions_as_string = ' AND '.join(conditions)
    return f'''SELECT {features_as_string}
               FROM `eng-reactor-287421.auxiliary_views.materialized_trade_history`
               WHERE {conditions_as_string}
               ORDER BY trade_datetime desc'''


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


def _trade_history_derived_features(row, yield_spread_or_dollar_price, using_treasury_spread):
    assert yield_spread_or_dollar_price in ('yield_spread', 'dollar_price'), f'Invalid value for yield_spread_or_dollar_price: {yield_spread_or_dollar_price}'
    if yield_spread_or_dollar_price == 'yield_spread':
        variants = YS_VARIANTS
        trade_history_features = get_ys_trade_history_features(using_treasury_spread)
    else:
        variants = DP_VARIANTS
        trade_history_features = get_dp_trade_history_features()

    ys_or_dp_idx = trade_history_features.index(yield_spread_or_dollar_price)
    par_traded_idx = trade_history_features.index('par_traded')
    trade_type1_idx = trade_history_features.index('trade_type1')
    trade_type2_idx = trade_history_features.index('trade_type2')
    seconds_ago_idx = trade_history_features.index('seconds_ago')


    def extract_feature_from_trade(row, name, trade):    # `name` is used solely for debugging
        ys_or_dp = trade[ys_or_dp_idx]
        ttypes = TTYPE_DICT[(trade[trade_type1_idx], trade[trade_type2_idx])] + row.trade_type
        seconds_ago = trade[seconds_ago_idx]
        quantity_diff = np.log10(1 + np.abs(10**trade[par_traded_idx] - 10**row.quantity))
        return [ys_or_dp, ttypes, seconds_ago, quantity_diff]


    global D_prev
    global S_prev
    global P_prev
    
    trade_history = row.trade_history
    most_recent_trade = trade_history[0]
    
    D_min_ago_t = D_prev.get(row.cusip, most_recent_trade)
    D_min_ago = LONG_TIME_AGO_IN_NUM_SECONDS        

    P_min_ago_t = P_prev.get(row.cusip, most_recent_trade)
    P_min_ago = LONG_TIME_AGO_IN_NUM_SECONDS
    
    S_min_ago_t = S_prev.get(row.cusip, most_recent_trade)
    S_min_ago = LONG_TIME_AGO_IN_NUM_SECONDS
    
    max_ys_or_dp_t = most_recent_trade
    max_ys_or_dp = most_recent_trade[ys_or_dp_idx]
    min_ys_or_dp_t = most_recent_trade
    min_ys_or_dp = most_recent_trade[ys_or_dp_idx]
    max_qty_t = most_recent_trade
    max_qty = most_recent_trade[par_traded_idx]
    min_ago_t = most_recent_trade
    min_ago = most_recent_trade[seconds_ago_idx]
    
    for trade in trade_history:
        seconds_ago = trade[seconds_ago_idx]
        # Checking if the first trade in the history is from the same block; TODO: shouldn't this be checked for every trade?
        if seconds_ago == 0: continue

        ys_or_dp = trade[ys_or_dp_idx]
        if ys_or_dp > max_ys_or_dp: 
            max_ys_or_dp_t = trade
            max_ys_or_dp =ys_or_dp
        elif ys_or_dp < min_ys_or_dp: 
            min_ys_or_dp_t = trade
            min_ys_or_dp = ys_or_dp

        par_traded = trade[par_traded_idx]
        if par_traded > max_qty: 
            max_qty_t = trade 
            max_qty = par_traded

        if seconds_ago < min_ago:    # TODO: isn't this just the most recent trade not in the same block, and isn't this initialized above already?
            min_ago_t = trade
            min_ago = seconds_ago
            
        side = TTYPE_DICT[(trade[trade_type1_idx], trade[trade_type2_idx])]
        if side == 'D':
            if seconds_ago < D_min_ago: 
                D_min_ago_t = trade
                D_min_ago = seconds_ago
                D_prev[row.cusip] = trade
        elif side == 'P':
            if seconds_ago < P_min_ago: 
                P_min_ago_t = trade
                P_min_ago = seconds_ago
                P_prev[row.cusip] = trade
        elif side == 'S':
            if seconds_ago < S_min_ago: 
                S_min_ago_t = trade
                S_min_ago = seconds_ago
                S_prev[row.cusip] = trade
        else: 
            print('invalid side', trade)
    
    variant_trade_dict = dict(zip(variants, [max_ys_or_dp_t, min_ys_or_dp_t, max_qty_t, min_ago_t, D_min_ago_t, P_min_ago_t, S_min_ago_t]))
    variant_trade_list = []
    for variant_name, variant_trade in variant_trade_dict.items():
        feature_list = extract_feature_from_trade(row, variant_name, variant_trade)
        variant_trade_list += feature_list
    return variant_trade_list


def trade_history_derived_features_yield_spread(using_treasury_spread):
    return lambda row: _trade_history_derived_features(row, 'yield_spread', using_treasury_spread)
def trade_history_derived_features_dollar_price(row):
    return _trade_history_derived_features(row, 'dollar_price')


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
                        epochs=NUM_EPOCHS, 
                        batch_size=BATCH_SIZE,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=fit_callbacks,
                        use_multiprocessing=True,
                        workers=8) 

    _, mae = model.evaluate(x_test, 
                            y_test, 
                            verbose=1, 
                            batch_size=BATCH_SIZE)
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
