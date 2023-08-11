'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2023-01-23 12:12:16
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2023-08-11 23:14:13
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
from ficc.utils.auxiliary_variables import PREDICTORS_DOLLAR_PRICE, NON_CAT_FEATURES_DOLLAR_PRICE, BINARY_DOLLAR_PRICE, CATEGORICAL_FEATURES_DOLLAR_PRICE, IDENTIFIERS, NUM_OF_DAYS_IN_YEAR
from ficc.utils.nelson_siegel_model import yield_curve_level
from ficc.utils.gcp_storage_functions import upload_data
from datetime import datetime, timedelta
from dollar_model import dollar_price_model

import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ahmad/ahmad_creds.json"
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/shayaan/ficc/ahmad_creds.json"

SEQUENCE_LENGTH = 2
NUM_FEATURES = 6


categorical_feature_values = {'purpose_class' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                                 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                                                 47, 48, 49, 50, 51, 52, 53],
                              'rating' : ['A', 'A+', 'A-', 'AA', 'AA+', 'AA-', 'AAA', 'B', 'B+', 'B-', 'BB', 'BB+', 'BB-',
                                         'BBB', 'BBB+', 'BBB-', 'CC', 'CCC', 'CCC+', 'CCC-' , 'D', 'NR', 'MR'],
                              'trade_type' : ['D', 'S', 'P'],
                              'incorporated_state_code' : ['AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'GU',
                                                         'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
                                                         'MO', 'MP', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
                                                         'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'US', 'UT', 'VA', 'VI',
                                                         'VT', 'WA', 'WI', 'WV', 'WY'] }

storage_client = storage.Client()
bq_client = bigquery.Client()


ttype_dict = { (0,0):'D', (0,1):'S', (1,0):'P' }

dp_variants = ["max_dp", "min_dp", "max_qty", "min_ago", "D_min_ago", "P_min_ago", "S_min_ago"]
dp_feats = ["_dp", "_ttypes", "_ago", "_qdiff"]
D_prev = dict()
P_prev = dict()
S_prev = dict()

def get_trade_history_columns():
    '''
    This function is used to create a list of columns
    '''
    global dp_variants
    global dp_feats
    YS_COLS = []
    for prefix in dp_variants:
        for suffix in dp_feats:
            YS_COLS.append(prefix + suffix)
    return YS_COLS

def extract_feature_from_trade(row, name, trade):
    global ttype_dict
    dollar_price = trade[0]
    ttypes = ttype_dict[(trade[2],trade[3])] + row.trade_type
    seconds_ago = trade[4]
    quantity_diff = np.log10(1 + np.abs(10**trade[1] - 10**row.quantity))
    return [dollar_price, ttypes,  seconds_ago, quantity_diff]

def trade_history_derived_features(row):
    global ttype_dict
    global D_prev
    global S_prev
    global P_prev
    global dp_feats
    global dp_variants
    trade_history = row.trade_history
    trade = trade_history[0]
    
    D_min_ago_t = D_prev.get(row.cusip,trade)
    D_min_ago = 9        

    P_min_ago_t = P_prev.get(row.cusip,trade)
    P_min_ago = 9
    
    S_min_ago_t = S_prev.get(row.cusip,trade)
    S_min_ago = 9
    
    max_dp_t = trade; max_dp = trade[0]
    min_dp_t = trade; min_dp = trade[0]
    max_qty_t = trade; max_qty = trade[1]
    min_ago_t = trade; min_ago = trade[4]
    
    for trade in trade_history[0:]:
        #Checking if the first trade in the history is from the same block
        if trade[4] <= 0: 
            continue
 
        if trade[0] > max_dp: 
            max_dp_t = trade
            max_dp = trade[0]
        elif trade[0] < min_dp: 
            min_dp_t = trade; 
            min_dp = trade[0]

        if trade[1] > max_qty: 
            max_qty_t = trade 
            max_qty = trade[1]
        if trade[4] < min_ago: 
            min_ago_t = trade; 
            min_ago = trade[4]
            
        side = ttype_dict[(trade[2],trade[3])]
        if side == "D":
            if trade[4] < D_min_ago: 
                D_min_ago_t = trade
                D_min_ago = trade[4]
                D_prev[row.cusip] = trade
        elif side == "P":
            if trade[4] < P_min_ago: 
                P_min_ago_t = trade
                P_min_ago = trade[4]
                P_prev[row.cusip] = trade
        elif side == "S":
            if trade[4] < S_min_ago: 
                S_min_ago_t = trade
                S_min_ago = trade[4]
                S_prev[row.cusip] = trade
        else: 
            print("invalid side", trade)
    
    trade_history_dict = {"max_dp":max_dp_t,
                          "min_dp":min_dp_t,
                          "max_qty":max_qty_t,
                          "min_ago":min_ago_t,
                          "D_min_ago":D_min_ago_t,
                          "P_min_ago":P_min_ago_t,
                          "S_min_ago":S_min_ago_t}

    return_list = []
    for variant in dp_variants:
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
                refund_price,
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
                publish_datetime,
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
                maturity_description_code 
               FROM
                 `eng-reactor-287421.auxiliary_views.materialized_trade_history`
               WHERE
                 par_traded >= 10000
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
    return np.tile(target_trade_features, (1,1))

def replace_ratings_by_standalone_rating(data):
    data.loc[data.sp_stand_alone.isna(), 'sp_stand_alone'] = 'NR'
    data.rating = data.rating.astype('str')
    data.sp_stand_alone = data.sp_stand_alone.astype('str')
    data.loc[(data.sp_stand_alone != 'NR'),'rating'] = data[(data.sp_stand_alone != 'NR')]['sp_stand_alone'].loc[:]
    return data


def update_data():
  '''
  This function updates the master data file that is used to train and 
  deploy the model. 
  Input: None
  Output: dataframe, datetime  
  '''
  print("Downloading data")
  fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')

  with fs.open('automated_training/processed_data_dollar_price.pkl') as f:
      data = pd.read_pickle(f)

  print('Data downloaded')
  
  last_trade_date = data.trade_date.max().date().strftime('%Y-%m-%d')

  print(f"last trade date : {last_trade_date}")
  DATA_QUERY = return_data_query(last_trade_date)
  file_timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')

  new_data = process_data(DATA_QUERY,
                          bq_client,
                          SEQUENCE_LENGTH,
                          NUM_FEATURES,
                          f"raw_data_{file_timestamp}.pkl",
                          'FICC_NEW',
                          remove_short_maturity=False,
                          trade_history_delay = 0.2,
                          min_trades_in_history = 0,
                          treasury_spread = False,
                          add_flags=False,
                          add_previous_treasury_rate=True,
                          add_previous_treasury_difference=True,
                          add_related_trades_bool=False,
                          add_rtrs_in_history=False,
                          only_dollar_price_history=True,)
  
  # Restricting the trade history to length 2, at present 2 is the length of trade history
  # that gives the maximum accuracy
  print(f"Restricting history to {SEQUENCE_LENGTH} trades")
  new_data.trade_history = new_data.trade_history.apply(lambda x: x[:SEQUENCE_LENGTH])
  data.trade_history = data.trade_history.apply(lambda x: x[:SEQUENCE_LENGTH])

  new_data = replace_ratings_by_standalone_rating(new_data)
  new_data['yield'] = new_data['yield'] * 100
  new_data['target_attention_features'] = new_data.parallel_apply(target_trade_processing_for_attention, axis = 1)

  #### removing missing data
  print("Adding new data to master file")
  data = pd.concat([new_data, data])

  data['trade_history_sum'] = data.trade_history.parallel_apply(lambda x: np.sum(x))
  data.issue_amount = data.issue_amount.replace([np.inf, -np.inf], np.nan)

  ####### Adding trade history features to the data ###########
  print("Adding features from previous trade history")
  temp = data[['cusip','trade_history','quantity','trade_type']].parallel_apply(trade_history_derived_features, axis=1)
  YS_COLS = get_trade_history_columns()
  data[YS_COLS] = pd.DataFrame(temp.tolist(), index=data.index)
  del temp
        
  #############################################################
  data.dropna(inplace=True, subset=PREDICTORS_DOLLAR_PRICE+['trade_history_sum'])
  
  print("Saving data to pickle file")
  data.to_pickle('processed_data_dollar_price.pkl')  
  
  print("Uploading data")
  upload_data(storage_client, 'automated_training', 'processed_data_dollar_price.pkl')
  
  return data, last_trade_date

def create_input(df, encoders):
    datalist = []
    datalist.append(np.stack(df['trade_history'].to_numpy()))
    datalist.append(np.stack(df['target_attention_features'].to_numpy()))

    noncat_and_binary = []
    for f in NON_CAT_FEATURES_DOLLAR_PRICE + BINARY_DOLLAR_PRICE:
        noncat_and_binary.append(np.expand_dims(df[f].to_numpy().astype('float32'), axis=1))
    datalist.append(np.concatenate(noncat_and_binary, axis=-1))
    
    for f in CATEGORICAL_FEATURES_DOLLAR_PRICE:
        encoded = encoders[f].transform(df[f])
        datalist.append(encoded.astype('float32'))
    
    return datalist

def fit_encoders(data):
    '''
    This function fits label encoders to categorical features in the data.
    For a few of the categorical features, the values don't change for these features
    we use the pre-defined set of values.
    
    Input : data frame
    Output : Tuple of dictionaries 
    '''
    encoders = {}
    fmax = {}
    for f in CATEGORICAL_FEATURES_DOLLAR_PRICE:
        if f in ['rating', 'incorporated_state_code', 'trade_type', 'purpose_class']:
            fprep = preprocessing.LabelEncoder().fit(categorical_feature_values[f])
        else:
            fprep = preprocessing.LabelEncoder().fit(data[f].drop_duplicates())
        fmax[f] = np.max(fprep.transform(fprep.classes_))
        encoders[f] = fprep
    
    with open('encoders_dollar_price.pkl','wb') as file:
        pickle.dump(encoders,file)
    return encoders, fmax

def train_model(data, last_trade_date):
    
    encoders, fmax  = fit_encoders(data)

    train_data = data[data.trade_date <= last_trade_date]
    test_data = data[data.trade_date > last_trade_date]
    
    x_train = create_input(train_data, encoders)
    y_train = train_data.dollar_price

    x_test = create_input(test_data, encoders)
    y_test = test_data.dollar_price

    model = dollar_price_model(x_train, 
                               SEQUENCE_LENGTH, 
                               NUM_FEATURES - 1,
                               CATEGORICAL_FEATURES_DOLLAR_PRICE, 
                               NON_CAT_FEATURES_DOLLAR_PRICE, 
                               BINARY_DOLLAR_PRICE,
                               fmax)

    fit_callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    patience=20,
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
    with open(f"encoders_dollar_price.pkl",'wb') as file:
        pickle.dump(encoders,file)    
    upload_data(storage_client, 'ahmad_data', f"encoders_dollar_price.pkl")

    print("Saving and uploading model")
    model.save(f"saved_model_dollar_price{file_timestamp}")
    
    shutil.make_archive(f"model_dollar_price", 'zip', f"saved_model_dollar_price{file_timestamp}")
    # shutil.make_archive(f"saved_model_{file_timestamp}", 'zip', f"saved_model_{file_timestamp}")
    
    upload_data(storage_client, 'ahmad_data', f"model_dollar_price.zip")
    # upload_data(storage_client, 'ahmad_data/yield_spread_models', f"saved_model_{file_timestamp}.zip")
    os.system(f"rm -r saved_model_dollar_price{file_timestamp}")


def send_results_email(mae, last_trade_date):
    receiver_email = ["ahmad@ficc.ai"]
    sender_email = "notifications@ficc.ai"
    
    msg = MIMEMultipart()
    msg['Subject'] = f"Mae for dollar model trained till {last_trade_date}"
    msg['From'] = sender_email


    message = MIMEText(f"The MAE for the dollar model on trades that occurred on {last_trade_date} is {mae}.", 'plain')
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
    print(f'\n\nFunction starting {datetime.now()}')

    print('Processing data')
    data, last_trade_date = update_data()
    print('Data processed')
    
    print('Training model')
    model, encoders, mae = train_model(data, last_trade_date)
    print('Training done')

    print('Saving model')
    save_model(model, encoders)
    print('Finished Training\n\n')

    print('sending email')
    send_results_email(mae, last_trade_date)
    
    print(f'Funciton executed {datetime.now()}\n\n')

if __name__ == '__main__':
    main()