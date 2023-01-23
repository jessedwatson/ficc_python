'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2023-01-23 12:12:16
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2023-01-23 15:52:42
 # @ Description:
 '''

import os
import gcsfs
import pandas as pd
from google.cloud import bigquery
from google.cloud import storage
from ficc.data.process_data import process_data
from datetime import datetime, timedelta

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/shayaan/ficc/ahmad_creds.json"
SEQUENCE_LENGTH = 5
NUM_FEATURES = 6

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
               ORDER BY trade_datetime desc limit 10'''

def main():
    fs = gcsfs.GCSFileSystem(project='eng-reactor-287421')
    with fs.open('automated_training/processed_data.pkl') as f:
        data = pd.read_pickle(f)
    
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
                        add_related_trades_bool=True)
    
    data = pd.concat([new_data, data])
    # data = data.sort_values('trade_datetime', ascending=False)
    data.to_pickle('processed_data.pkl')
    


if __name__ == '__main__':
    main()