'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 09:44:22
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-11-28 09:57:00
 # @ Description: This file is an example of how to call the ficc data package. 
 # The driver method for the package is the proces data function. 
 # The method takes the following arguments. 
 	#   1. A query that will be used to fetch data from BigQuery. 
	#   2. BigQuery client. 
 	#   3. The sequence length of the trade history can take 32 as its maximum value. 
	#   4. The number of features that the trade history contains. 
	#   5. Link to save the raw data grabbed from BigQuery. 
  #   6. The yield curve to use acceptable options are S&P, FICC, FICC_NEW, MMD and MSRB_YTW(to train estimating the yield). 
  #   7. remove_short_maturity flag to remove trades that mature within 400 days from trade date
  #   8. trade_history_delay flag to remove trades from history which occur within the specified minutes of the target trade
  #   9. min_trades_in_history the minimum number of trades allowed in the history
	#   10. A list containing the features that will be used for training. This is an optional parameter
 '''

import os
import time
from google.cloud import bigquery
from ficc.data.process_data import process_data

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/shayaan/ficc/ahmad_creds.json"
SEQUENCE_LENGTH = 5
NUM_FEATURES = 7

DATA_QUERY = '''SELECT
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
--`eng-reactor-287421.jesse_tests.trade_history_with_reference_data`
WHERE
  yield IS NOT NULL
  AND yield > 0
  AND par_traded >= 10000
  AND trade_date >= '2022-05-01'
  AND trade_date <= '2022-10-07'
  AND maturity_description_code = 2
  AND coupon_type in (8, 4, 10)
  AND capital_type <> 10
  AND default_exists <> TRUE
  AND sale_type <> 4
  AND sec_regulation IS NULL
  AND most_recent_default_event IS NULL
  AND default_indicator IS FALSE
--  AND DATETIME_DIFF(trade_datetime,recent[SAFE_OFFSET(0)].trade_datetime,SECOND) < 1000000 -- 12 days to the most recent trade
  AND msrb_valid_to_date > current_date -- condition to remove cancelled trades
  ORDER BY trade_datetime desc 
  limit 10000
  '''

# DATA_QUERY = '''
# SELECT
#     * except(most_recent_event)
#   FROM
#     `eng-reactor-287421.auxiliary_views.materialized_trade_history`
#   WHERE
#     msrb_valid_to_date > current_date -- condition to remove cancelled trades
#     AND rtrs_control_number = 2022051107687000
#   ORDER BY
#     trade_datetime desc
# '''


bq_client = bigquery.Client()

if __name__ == "__main__":
    start_time  = time.time()
    # trade_data = process_data(DATA_QUERY,
    #                           bq_client,
    #                           SEQUENCE_LENGTH,
    #                           NUM_FEATURES,
    #                           'data.pkl',
    #                           "FICC_NEW",
    #                           remove_short_maturity=False,
    #                           trade_history_delay = 1,
    #                           min_trades_in_history = 0,
    #                           process_ratings=False,
    #                           treasury_spread=True,
    #                           add_previous_treasury_rate=False,
    #                           add_previous_treasury_difference=False,
    #                           use_last_duration=True)
    trade_data = process_data(DATA_QUERY, 
                    bq_client,
                    SEQUENCE_LENGTH,
                    NUM_FEATURES,
                    'data.pkl',
                    'FICC_NEW',
                    estimate_calc_date=False,
                    remove_short_maturity=False,
                    remove_non_transaction_based=False,
                    remove_trade_type = [],
                    trade_history_delay = 0,
                    min_trades_in_history = 0,
                    process_ratings=False,
                    treasury_spread = True,
                    add_previous_treasury_rate=True,
                    add_previous_treasury_difference=True,
                    use_last_duration=False,
                    add_flags=False)
    
    end_time = time.time()
    print(f"time elapsed in seconds = {end_time - start_time}")
    print(trade_data)
