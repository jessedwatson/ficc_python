'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2023-01-23 12:12:16
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2023-02-03 10:26:10
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
from ficc.utils.auxiliary_variables import PREDICTORS, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES, IDENTIFIERS
from ficc.utils.gcp_storage_functions import upload_data
from datetime import datetime, timedelta
from model import yield_spread_model

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ahmad/ahmad_creds.json"
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/shayaan/ficc/ahmad_creds.json"
SEQUENCE_LENGTH = 5
NUM_FEATURES = 6
PREDICTORS.append('target_attention_features')
PREDICTORS.append('ficc_treasury_spread')
NON_CAT_FEATURES.append('ficc_treasury_spread')

storage_client = storage.Client()
bq_client = bigquery.Client()


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

def update_data():
  print("Downloading data")
  fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
  with fs.open('automated_training/processed_data.pkl') as f:
      data = pd.read_pickle(f)
  print('Download data')
  
  last_trade_date = data.trade_date.max().date().strftime('%Y-%m-%d')
  
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
                      production_set=False)
  
  new_data['target_attention_features'] = new_data.parallel_apply(target_trade_processing_for_attention, axis = 1)
  new_data = replace_ratings_by_standalone_rating(new_data)
  new_data['yield'] = new_data['yield'] * 100

  data = pd.concat([new_data, data])
  data['trade_history_sum'] = data.trade_history.parallel_apply(lambda x: np.sum(x))
  data.issue_amount = data.issue_amount.replace([np.inf, -np.inf], np.nan)
  data.dropna(inplace=True, subset=PREDICTORS+['trade_history_sum'])
  data.to_pickle('processed_data.pkl')
  upload_data(storage_client, 'automated_training', 'processed_data.pkl')
  return data

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

def train_model(data):
  encoders, fmax  = fit_encoders(data)
  x_train = create_input(data, encoders)
  y_train = data.yield_spread
  
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
  return model, encoders           
  


def save_model(model, encoders):
  file_timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
  print(f"file time stamp : {file_timestamp}")

  print("Saving encoders and uploading encoders")
  with open(f"encoders_{file_timestamp}.pkl",'wb') as file:
      pickle.dump(encoders,file)    
  upload_data(storage_client, 'ahmad_data', f"encoders_{file_timestamp}.pkl")

  print("Saving and uploading model")
  model.save(f"saved_model_{file_timestamp}")
  shutil.make_archive(f"model", 'zip', f"saved_model_{file_timestamp}")
  upload_data(storage_client, 'ahmad_data', f"model.zip")
  os.system(f"rm -r saved_model_{file_timestamp}")

def main():
  print('\n\nFunction starting')
  
  print('Processing data')
  data = update_data()
  print('Data processed')
  
  print('Training model')
  model, encoders = train_model(data)
  print('Training done')

  print('Saving model')
  save_model(model, encoders)
  
  print('Finished Training\n\n')


if __name__ == '__main__':
    main()
