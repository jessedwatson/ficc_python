'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 09:44:22
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-07-19 21:45:42
 # @ Description: This file is an example of how to call the ficc data package. 
 # The driver method for the package is the proces data function. 
 # The method takes the following arguments. 
 	#   1. A query that will be used to fetch data from BigQuery. 
	#   2. BigQuery client. 
 	#   3. The sequence length of the trade history can take 32 as its maximum value. 
	#   4. The number of features that the trade history contains. 
	#   5. The yield curve to use acceptable options S&P or ficc. 
	#   6. Link to save the raw data grabbed from BigQuery. 
	#   7. A list containing the features that will be used for training. This is an optional parameter
 '''


import ficc.utils.globals as globals
import os
import pickle5 as pickle
from google.cloud import bigquery
from ficc.data.process_data import process_data
from ficc.utils.auxiliary_variables import PREDICTORS

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/shayaan/ficc/eng-reactor-287421-112eb767e1b3.json"
SEQUENCE_LENGTH = 5
NUM_FEATURES = 5

DATA_QUERY = ''' 
SELECT
  *
FROM
  `eng-reactor-287421.auxiliary_views.materialized_trade_history`
WHERE
  yield IS NOT NULL
  AND yield > 0
  AND par_traded >= 10000
  AND trade_date >= '2021-08-01'
  AND maturity_description_code = 2
  AND coupon_type in (8, 4, 10)
  AND capital_type <> 10
  AND default_exists <> TRUE
  AND sale_type <> 4
  AND sec_regulation IS NULL
  AND most_recent_default_event IS NULL
  AND default_indicator IS FALSE
  -- AND date_diff(calc_date, current_date(),YEAR) > 25
  -- AND DATETIME_DIFF(trade_datetime,recent[SAFE_OFFSET(0)].trade_datetime,SECOND) < 1000000 -- 12 days to the most recent trade
  AND msrb_valid_to_date > current_date -- condition to remove cancelled trades
ORDER BY
  trade_datetime DESC
  limit 1000
''' 

bq_client = bigquery.Client()

if __name__ == "__main__":
    trade_data = process_data(DATA_QUERY,
                              bq_client,
                              SEQUENCE_LENGTH,
                              NUM_FEATURES,
                              'data.pkl',
                              "MMD",
                              estimate_calc_date=False,
                              remove_short_maturity=True,
                              remove_non_transaction_based=False,
                              remove_trade_type = [],
                              trade_history_delay = 1,
                              min_trades_in_history = 0,
                              process_ratings=False)
    print(trade_data)
