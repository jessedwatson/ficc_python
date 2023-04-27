'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2023-01-23 12:12:16
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2023-04-19 18:56:34
 # @ Description:
 '''

import os
import gcsfs
import shutil
import numpy as np
import pandas as pd
from tensorflow import keras
from google.cloud import bigquery
from google.cloud import storage
from sklearn import preprocessing
from pickle5 import pickle
from ficc.data.process_data import process_data
from ficc.utils.auxiliary_functions import sqltodf
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.auxiliary_variables import PREDICTORS, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES, IDENTIFIERS, NUM_OF_DAYS_IN_YEAR
from ficc.utils.nelson_siegel_model import yield_curve_level
from ficc.utils.gcp_storage_functions import upload_data
from datetime import datetime, timedelta
from model import yield_spread_model

import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ahmad/ahmad_creds.json"
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/shayaan/ficc/ahmad_creds.json"

SEQUENCE_LENGTH = 5
NUM_FEATURES = 6
PREDICTORS.append('target_attention_features')
PREDICTORS.append('ficc_treasury_spread')
NON_CAT_FEATURES.append('ficc_treasury_spread')

storage_client = storage.Client()
bq_client = bigquery.Client()


nelson_params = sqltodf("select * from `eng-reactor-287421.ahmad_test.nelson_siegel_coef_daily` order by date desc", bq_client)
nelson_params.set_index("date", drop=True, inplace=True)
nelson_params = nelson_params[~nelson_params.index.duplicated(keep='first')]
nelson_params = nelson_params.transpose().to_dict()

scalar_params = sqltodf("select * from`eng-reactor-287421.ahmad_test.standardscaler_parameters_daily` order by date desc", bq_client)
scalar_params.set_index("date", drop=True, inplace=True)
scalar_params = scalar_params[~scalar_params.index.duplicated(keep='first')]
scalar_params = scalar_params.transpose().to_dict()

shape_parameter  = sqltodf("SELECT *  FROM `eng-reactor-287421.ahmad_test.shape_parameters` order by Date desc", bq_client)
shape_parameter.set_index("Date", drop=True, inplace=True)
shape_parameter = shape_parameter[~shape_parameter.index.duplicated(keep='first')]
shape_parameter = shape_parameter.transpose().to_dict()

ttype_dict = { (0,0):'D', (0,1):'S', (1,0):'P' }

ys_variants = ["max_ys", "min_ys", "max_qty", "min_ago", "D_min_ago", "P_min_ago", "S_min_ago"]
ys_feats = ["_ys", "_ttypes", "_ago", "_qdiff"]
D_prev = dict()
P_prev = dict()
S_prev = dict()

def get_trade_history_columns():
    '''
    This function is used to create a list of columns
    '''
    global ys_variants
    global ys_feats
    YS_COLS = []
    for prefix in ys_variants:
        for suffix in ys_feats:
            YS_COLS.append(prefix + suffix)
    return YS_COLS

def extract_feature_from_trade(row, name, trade):
    global ttype_dict
    yield_spread = trade[0]
    ttypes = ttype_dict[(trade[3],trade[4])] + row.trade_type
    seconds_ago = trade[5]
    quantity_diff = np.log10(1 + np.abs(10**trade[2] - 10**row.quantity))
    return [yield_spread, ttypes,  seconds_ago, quantity_diff]

def trade_history_derived_features(row):
    global ttype_dict
    global D_prev
    global S_prev
    global P_prev
    global ys_feats
    global ys_variants
    
    trade_history = row.trade_history
    trade = trade_history[0]
    
    D_min_ago_t = D_prev.get(row.cusip,trade)
    D_min_ago = 9        

    P_min_ago_t = P_prev.get(row.cusip,trade)
    P_min_ago = 9
    
    S_min_ago_t = S_prev.get(row.cusip,trade)
    S_min_ago = 9
    
    max_ys_t = trade; max_ys = trade[0]
    min_ys_t = trade; min_ys = trade[0]
    max_qty_t = trade; max_qty = trade[2]
    min_ago_t = trade; min_ago = trade[5]
    
    for trade in trade_history[0:]:
        #Checking if the first trade in the history is from the same block
        if trade[5] == 0: 
            continue
 
        if trade[0] > max_ys: 
            max_ys_t = trade
            max_ys = trade[0]
        elif trade[0] < min_ys: 
            min_ys_t = trade; 
            min_ys = trade[0]

        if trade[2] > max_qty: 
            max_qty_t = trade 
            max_qty = trade[2]
        if trade[5] < min_ago: 
            min_ago_t = trade; 
            min_ago = trade[5]
            
        side = ttype_dict[(trade[3],trade[4])]
        if side == "D":
            if trade[5] < D_min_ago: 
                D_min_ago_t = trade; D_min_ago = trade[5]
                D_prev[row.cusip] = trade
        elif side == "P":
            if trade[5] < P_min_ago: 
                P_min_ago_t = trade; P_min_ago = trade[5]
                P_prev[row.cusip] = trade
        elif side == "S":
            if trade[5] < S_min_ago: 
                S_min_ago_t = trade; S_min_ago = trade[5]
                S_prev[row.cusip] = trade
        else: 
            print("invalid side", trade)
    
    trade_history_dict = {"max_ys":max_ys_t,
                          "min_ys":min_ys_t,
                          "max_qty":max_qty_t,
                          "min_ago":min_ago_t,
                          "D_min_ago":D_min_ago_t,
                          "P_min_ago":P_min_ago_t,
                          "S_min_ago":S_min_ago_t}

    return_list = []
    for variant in ys_variants:
        feature_list = extract_feature_from_trade(row,variant,trade_history_dict[variant])
        return_list += feature_list
    
    return return_list

def return_data_query(last_trade_date):
    return f'''SELECT
                 rtrs_control_number,
                 cusip,
                 yield,
                 is_callable,
                 refund_date,
                 accrual_date,
                 dated_date,
                 next_sink_date,
                 coupon,
                 delivery_date,
                 trade_date,
                 trade_datetime,
                 par_call_date,
                 interest_payment_frequency,
                 is_called,
                 is_non_transaction_based_compensation,
                 is_general_obligation,
                 callable_at_cav,
                 extraordinary_make_whole_call,
                 make_whole_call,
                 has_unexpired_lines_of_credit,
                 escrow_exists,
                 incorporated_state_code,
                 trade_type,
                 par_traded,
                 maturity_date,
                 settlement_date,
                 next_call_date,
                 issue_amount,
                 maturity_amount,
                 issue_price,
                 orig_principal_amount,
                 max_amount_outstanding,
                 recent,
                 dollar_price,
                 calc_date,
                 purpose_sub_class,
                 called_redemption_type,
                 calc_day_cat,
                 previous_coupon_payment_date,
                 instrument_primary_name,
                 purpose_class,
                 call_timing,
                 call_timing_in_part,
                 sink_frequency,
                 sink_amount_type,
                 issue_text,
                 state_tax_status,
                 series_name,
                 transaction_type,
                 next_call_price,
                 par_call_price,
                 when_issued,
                 min_amount_outstanding,
                 original_yield,
                 par_price,
                 default_indicator,
                 sp_stand_alone,
                 sp_long,
                 moodys_long,
                 coupon_type,
                 federal_tax_status,
                 use_of_proceeds,
                 muni_security_type,
                 muni_issue_type,
                 capital_type,
                 other_enhancement_type,
                 next_coupon_payment_date,
                 first_coupon_date,
                 last_period_accrues_from_date,
               FROM
                 `eng-reactor-287421.auxiliary_views.materialized_trade_history`
               WHERE
                 yield IS NOT NULL
                 AND yield > 0
                 AND par_traded >= 10000
                 AND trade_date > '{last_trade_date}'
                 AND coupon_type in (8, 4, 10, 17)
                 AND capital_type <> 10
                 AND default_exists <> TRUE
                 AND most_recent_default_event IS NULL
                 AND default_indicator IS FALSE
                 AND msrb_valid_to_date > current_date -- condition to remove cancelled trades
                 AND settlement_date is not null
               ORDER BY trade_datetime desc'''


def target_trade_processing_for_attention(row):
  trade_mapping = {'D':[0,0], 'S':[0,1], 'P':[1,0]}
  target_trade_features = []
  target_trade_features.append(row['quantity'])
  target_trade_features = target_trade_features + trade_mapping[row['trade_type']]
  return np.tile(target_trade_features, (SEQUENCE_LENGTH,1))

def replace_ratings_by_standalone_rating(data):
  data.loc[data.sp_stand_alone.isna(), 'sp_stand_alone'] = 'NR'
  data.rating = data.rating.astype('str')
  data.sp_stand_alone = data.sp_stand_alone.astype('str')
  data.loc[(data.sp_stand_alone != 'NR'),'rating'] = data[(data.sp_stand_alone != 'NR')]['sp_stand_alone'].loc[:]
  return data

def get_yield_for_last_duration(row):
    if row['last_calc_date'] is None or row['last_trade_date'] is None:
        return None
    duration =  diff_in_days_two_dates(row['last_calc_date'],row['last_trade_date'])/NUM_OF_DAYS_IN_YEAR
    ycl = yield_curve_level(duration, row['trade_date'].date(), nelson_params, scalar_params, shape_parameter)/100
    return ycl

def update_data():
  print("Downloading data")
  fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')

  with fs.open('automated_training/processed_data.pkl') as f:
      data = pd.read_pickle(f)
  print('Download data')
  
  last_trade_date = data.trade_date.max().date().strftime('%Y-%m-%d')
  print(f"last trade date : {last_trade_date}")
  DATA_QUERY = return_data_query(last_trade_date)
  file_timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')

  new_data = process_data(DATA_QUERY,
                      bq_client,
                      SEQUENCE_LENGTH,NUM_FEATURES,
                      f"raw_data_{file_timestamp}.pkl",
                      'FICC_NEW',
                      estimate_calc_date=False,
                      remove_short_maturity=True,
                      remove_non_transaction_based=False,
                      remove_trade_type = [],
                      trade_history_delay = 1,
                      min_trades_in_history = 0,
                      process_ratings=False,
                      treasury_spread = True,
                      add_previous_treasury_rate=True,
                      add_previous_treasury_difference=True,
                      add_flags=False,
                      add_related_trades_bool=False,
                      production_set=False,
                      add_rtrs_in_history=False)
  
  new_data['target_attention_features'] = new_data.parallel_apply(target_trade_processing_for_attention, axis = 1)
  new_data = replace_ratings_by_standalone_rating(new_data)
  new_data['yield'] = new_data['yield'] * 100
  new_data['last_trade_date'] = new_data['last_trade_datetime'].dt.date
  new_data['new_ficc_ycl'] = new_data[['last_calc_date',
                                       'last_settlement_date',
                                       'trade_date',
                                       'last_trade_date']].parallel_apply(get_yield_for_last_duration, axis=1)

  new_data['new_ficc_ycl'] = new_data['new_ficc_ycl'] * 100
  data = pd.concat([new_data, data])

  ####### Adding trade history features to the data ###########
  temp = data[['cusip','trade_history','quantity','trade_type']].parallel_apply(trade_history_derived_features, axis=1)
  YS_COLS = get_trade_history_columns()
  data[YS_COLS] = pd.DataFrame(temp.tolist(), index=data.index)
  del temp
  
  for col in YS_COLS:
    if 'ttypes' in col and col not in PREDICTORS:
        PREDICTORS.append(col)
        CATEGORICAL_FEATURES.append(col)
    elif col not in PREDICTORS:
        NON_CAT_FEATURES.append(col)
        PREDICTORS.append(col)
  #############################################################
  
  data['new_ys'] = data['new_ficc_ycl'] - data['yield']

  data['trade_history_sum'] = data.trade_history.parallel_apply(lambda x: np.sum(x))
  data.issue_amount = data.issue_amount.replace([np.inf, -np.inf], np.nan)
  data.dropna(inplace=True, subset=PREDICTORS+['trade_history_sum'])
  data.to_pickle('processed_data.pkl')
  
  print('Uploading data')
  upload_data(storage_client, 'automated_training', 'processed_data.pkl')

  return data, last_trade_date

def create_input(df, encoders):
    datalist = []
    datalist.append(np.stack(df['trade_history'].to_numpy()))
    datalist.append(np.stack(df['target_attention_features'].to_numpy()))

    noncat_and_binary = []
    for f in NON_CAT_FEATURES + BINARY:
        noncat_and_binary.append(np.expand_dims(df[f].to_numpy().astype('float32'), axis=1))
    datalist.append(np.concatenate(noncat_and_binary, axis=-1))
    
    for f in CATEGORICAL_FEATURES:
        encoded = encoders[f].transform(df[f])
        datalist.append(encoded.astype('float32'))
    
    return datalist

def fit_encoders(data):
  encoders = {}
  fmax = {}
  for f in CATEGORICAL_FEATURES:
      print(f)
      fprep = preprocessing.LabelEncoder().fit(data[f].drop_duplicates())
      fmax[f] = np.max(fprep.transform(fprep.classes_))
      encoders[f] = fprep
      
  with open('encoders.pkl','wb') as file:
      pickle.dump(encoders,file)
  return encoders, fmax

def train_model(data, last_trade_date):
  encoders, fmax  = fit_encoders(data)
  
  data = data[(data.days_to_call == 0) | (data.days_to_call > np.log10(400))]
  data = data[(data.days_to_refund == 0) | (data.days_to_refund > np.log10(400))]
  data = data[(data.days_to_maturity == 0) | (data.days_to_maturity > np.log10(400))]
  data = data[data.days_to_maturity < np.log10(30000)]

  train_data = data[data.trade_date < last_trade_date]
  test_data = data[data.trade_date >= last_trade_date]
  
  
  x_train = create_input(train_data, encoders)
  y_train = train_data.new_ys
  
  x_test = create_input(test_data, encoders)
  y_test = test_data.new_ys

  model = yield_spread_model(x_train, 
                             SEQUENCE_LENGTH, 
                             NUM_FEATURES, 
                             PREDICTORS, 
                             CATEGORICAL_FEATURES, 
                             NON_CAT_FEATURES, 
                             BINARY,
                             fmax)
  
  fit_callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss",
                                                 patience=10,
                                                 verbose=0,
                                                 mode="auto",
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
                          batch_size = 1000)
  

  return model, encoders, mae           
  


def save_model(model, encoders):
  file_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
  print(f"file time stamp : {file_timestamp}")

  print("Saving encoders and uploading encoders")
  with open(f"encoders.pkl",'wb') as file:
      pickle.dump(encoders,file)    
  upload_data(storage_client, 'automated_training', f"encoders.pkl")

  print("Saving and uploading model")
  model.save(f"saved_model_{file_timestamp}")
  shutil.make_archive(f"model", 'zip', f"saved_model_{file_timestamp}")
  upload_data(storage_client, 'ahmad_data', f"model.zip")
  os.system(f"rm -r saved_model_{file_timestamp}")


def send_results_email(mae, last_trade_date):
    receiver_email = ["ahmad@ficc.ai","gil@ficc.ai","jesse@ficc.ai", "gil@ficc.ai"]
    sender_email = "notifications@ficc.ai"
    
    msg = MIMEMultipart()
    msg['Subject'] = f"Mae for model trained till {last_trade_date}"
    msg['From'] = sender_email


    message = MIMEText(f"The MAE for the model on trades that occurred on {last_trade_date} is {mae}.", 'plain')
    msg.attach(message)

    smtp_server = "smtp.gmail.com"
    port = 587

    with smtplib.SMTP(smtp_server,port) as server:
        try:
            server.starttls()
            server.login(sender_email, 'ztwbwrzdqsucetbg')
            for receiver in receiver_email:
                server.sendmail(sender_email, receiver, msg.as_string())
        except Exception as e:
            print(e)
        finally:
            server.quit() 


def main():
  print('\n\nFunction starting')
  
  print('Processing data')
  data, last_trade_date = update_data()
  print('Data processed')

#   print('Training model')
#   model, encoders, mae = train_model(data, last_trade_date)
#   print('Training done')

#   print('Saving model')
#   save_model(model, encoders)
#   print('Finished Training\n\n')

#   print('sending email')
#   send_results_email(mae, last_trade_date)


if __name__ == '__main__':
    main()