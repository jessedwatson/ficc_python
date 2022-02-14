'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 09:44:22
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-02-14 12:40:47
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

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/shayaan/ficc/ml_models/sequence_predictors/eng-reactor-287421-112eb767e1b3.json"
SEQUENCE_LENGTH = 5
NUM_FEATURES = 5
DATA_QUERY = """ SELECT
  *
FROM
  `eng-reactor-287421.auxiliary_views.materialized_trade_history`
WHERE
  yield IS NOT NULL
  AND yield > 0 
  AND yield <= 3
  AND trade_type = 'D' 
  AND par_traded IS NOT NULL
  AND trade_date >= '2021-07-01' 
  AND trade_date <= '2021-10-01'
  AND maturity_description_code = 2
  AND incorporated_state_code <> 'US'
  AND coupon_type = 8
  AND msrb_valid_to_date > current_date -- condition to remove cancelled trades
ORDER BY
  trade_datetime DESC
LIMIT 100
            """

bq_client = bigquery.Client()

if __name__ == "__main__":
    trade_data = process_data(DATA_QUERY,
                              bq_client,
                              SEQUENCE_LENGTH,
                              NUM_FEATURES,
                              'data.pkl',
                              "MMD",
                              estimate_calc_date=True,
                              remove_short_maturity=True,
                              remove_non_transaction_based=False,
                              remove_trade_type = ['S','P'],
                              trade_history_delay = 1)
    
