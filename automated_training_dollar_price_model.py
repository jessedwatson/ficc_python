'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2023-01-23 12:12:16
 # @ Modified by: Mitas Ray
 # @ Modified time: 2023-12-19
 '''
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import bigquery
from google.cloud import storage
from ficc.data.process_data import process_data
from ficc.utils.auxiliary_variables import PREDICTORS_DOLLAR_PRICE, NON_CAT_FEATURES_DOLLAR_PRICE, BINARY_DOLLAR_PRICE, CATEGORICAL_FEATURES_DOLLAR_PRICE
from ficc.utils.gcp_storage_functions import upload_data
from datetime import datetime
from dollar_model import dollar_price_model

from automated_training_auxiliary_functions import NUM_FEATURES, \
                                                   SEQUENCE_LENGTH_DOLLAR_PRICE_MODEL, \
                                                   TTYPE_DICT, \
                                                   DP_VARIANTS, \
                                                   get_trade_history_columns, \
                                                   target_trade_processing_for_attention, \
                                                   replace_ratings_by_standalone_rating, \
                                                   create_input, \
                                                   get_data_and_last_trade_date, \
                                                   fit_encoders, \
                                                   train_and_evaluate_model, \
                                                   save_model, \
                                                   send_results_email


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/ahmad/ahmad_creds.json'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/shayaan/ficc/ahmad_creds.json'

storage_client = storage.Client()
bq_client = bigquery.Client()

D_prev = dict()
P_prev = dict()
S_prev = dict()


def extract_feature_from_trade(row, name, trade):
    # global TTYPE_DICT
    dollar_price = trade[0]
    ttypes = TTYPE_DICT[(trade[2],trade[3])] + row.trade_type
    seconds_ago = trade[4]
    quantity_diff = np.log10(1 + np.abs(10**trade[1] - 10**row.quantity))
    return [dollar_price, ttypes,  seconds_ago, quantity_diff]


def trade_history_derived_features(row):
    # global TTYPE_DICT
    global D_prev
    global S_prev
    global P_prev
    # global DP_FEATS
    # global DP_VARIANTS
    trade_history = row.trade_history
    trade = trade_history[0]
    
    D_min_ago_t = D_prev.get(row.cusip,trade)
    D_min_ago = 9        

    P_min_ago_t = P_prev.get(row.cusip,trade)
    P_min_ago = 9
    
    S_min_ago_t = S_prev.get(row.cusip,trade)
    S_min_ago = 9
    
    max_dp_t = trade
    max_dp = trade[0]
    min_dp_t = trade
    min_dp = trade[0]
    max_qty_t = trade
    max_qty = trade[1]
    min_ago_t = trade
    min_ago = trade[4]
    
    for trade in trade_history[0:]:
        # Checking if the first trade in the history is from the same block
        if trade[4] <= 0: continue    # TODO: this is `==` in `automated_training_yield_spread_model.py`
 
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
            
        side = TTYPE_DICT[(trade[2], trade[3])]
        if side == 'D':
            if trade[4] < D_min_ago: 
                D_min_ago_t = trade
                D_min_ago = trade[4]
                D_prev[row.cusip] = trade
        elif side == 'P':
            if trade[4] < P_min_ago: 
                P_min_ago_t = trade
                P_min_ago = trade[4]
                P_prev[row.cusip] = trade
        elif side == 'S':
            if trade[4] < S_min_ago: 
                S_min_ago_t = trade
                S_min_ago = trade[4]
                S_prev[row.cusip] = trade
        else: 
            print('invalid side', trade)
    
    trade_history_dict = {'max_dp': max_dp_t,
                          'min_dp': min_dp_t,
                          'max_qty': max_qty_t,
                          'min_ago': min_ago_t,
                          'D_min_ago': D_min_ago_t,
                          'P_min_ago': P_min_ago_t,
                          'S_min_ago': S_min_ago_t}

    return_list = []
    for variant in DP_VARIANTS:
        feature_list = extract_feature_from_trade(row, variant, trade_history_dict[variant])
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


def update_data() -> (pd.DataFrame, datetime.datetime):
    '''Updates the master data file that is used to train and deploy the model. NOTE: if any of the variables in 
    `process_data(...)` or `SEQUENCE_LENGTH_DOLLAR_PRICE_MODEL` are changed, then we need to rebuild the entire 
    `processed_data_dollar_price.pkl` since that data is will have the old preferences; an easy way to do that 
    is to manually set `last_trade_date` to a date way in the past (the desired start date of the data).'''
    bucket_name = 'automated_training'
    file_name = 'processed_data_dollar_price.pkl'
    
    data, last_trade_date = get_data_and_last_trade_date(bucket_name, file_name)
    print(f'last trade date: {last_trade_date}')
    DATA_QUERY = return_data_query(last_trade_date)
    file_timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')

    data_from_last_trade_date = process_data(DATA_QUERY,
                                             bq_client,
                                             SEQUENCE_LENGTH_DOLLAR_PRICE_MODEL,
                                             NUM_FEATURES,
                                             f'raw_data_{file_timestamp}.pkl',
                                             'FICC_NEW',
                                             remove_short_maturity=False,
                                             trade_history_delay=0.2,
                                             min_trades_in_history=0,
                                             treasury_spread=False,
                                             add_flags=False,
                                             add_previous_treasury_rate=True,
                                             add_previous_treasury_difference=True,
                                             add_related_trades_bool=False,
                                             add_rtrs_in_history=False,
                                             only_dollar_price_history=True)
    
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
    temp = data[['cusip', 'trade_history', 'quantity', 'trade_type']].parallel_apply(trade_history_derived_features, axis=1)
    DP_COLS = get_trade_history_columns('dollar_price')
    data[DP_COLS] = pd.DataFrame(temp.tolist(), index=data.index)
    del temp

    data.sort_values('trade_datetime', ascending=False, inplace=True)
    #############################################################
    data.dropna(inplace=True, subset=PREDICTORS_DOLLAR_PRICE)
    
    print(f'Saving data to pickle file with name {file_name}')
    data.to_pickle(file_name)  
    print(f'Uploading data to {bucket_name}/{file_name}')
    upload_data(storage_client, bucket_name, file_name)
    return data, last_trade_date


def train_model(data, last_trade_date):
    encoders, fmax  = fit_encoders(data, CATEGORICAL_FEATURES_DOLLAR_PRICE, 'dollar_price')

    train_data = data[data.trade_date <= last_trade_date]
    test_data = data[data.trade_date > last_trade_date]
    
    x_train = create_input(train_data, encoders, NON_CAT_FEATURES_DOLLAR_PRICE, BINARY_DOLLAR_PRICE, CATEGORICAL_FEATURES_DOLLAR_PRICE)
    y_train = train_data.dollar_price

    x_test = create_input(test_data, encoders, NON_CAT_FEATURES_DOLLAR_PRICE, BINARY_DOLLAR_PRICE, CATEGORICAL_FEATURES_DOLLAR_PRICE)
    y_test = test_data.dollar_price

    model = dollar_price_model(x_train, 
                               SEQUENCE_LENGTH_DOLLAR_PRICE_MODEL, 
                               NUM_FEATURES - 1, 
                               CATEGORICAL_FEATURES_DOLLAR_PRICE, 
                               NON_CAT_FEATURES_DOLLAR_PRICE, 
                               BINARY_DOLLAR_PRICE, 
                               fmax)
    
    model, mae, history = train_and_evaluate_model(model, x_train, y_train, x_test, y_test)
    return model, encoders, mae


def main():
    print(f'\n\nFunction starting {datetime.now()}')

    print('Processing data')
    data, last_trade_date = update_data()
    print('Data processed')
    
    print('Training model')
    model, encoders, mae = train_model(data, last_trade_date)
    print('Training done')

    print('Saving model')
    save_model(model, encoders, storage_client, dollar_price_model=True)
    print('Finished Training\n\n')

    print('sending email')
    send_results_email(mae, last_trade_date, ['ahmad@ficc.ai', 'isaac@ficc.ai', 'jesse@ficc.ai', 'gil@ficc.ai', 'mitas@ficc.ai', 'myles@ficc.ai'])
    print(f'Function executed {datetime.now()}\n\n')


if __name__ == '__main__':
    main()
