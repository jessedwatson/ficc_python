'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 09:44:22
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-04-11 13:26:59
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
from ficc.utils.yield_curve import get_ficc_ycl
from ficc.utils.auxiliary_functions import sqltodf
from ficc.utils.auxiliary_functions import convert_dates

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/shayaan/ficc/eng-reactor-287421-112eb767e1b3.json"

bq_client = bigquery.Client()

if __name__ == "__main__":
	query = """ 
	SELECT
	*
	FROM
	`eng-reactor-287421.auxiliary_views.materialized_trade_history`
	WHERE
	yield IS NOT NULL
	AND yield > 0
	AND par_traded >= 10000
	AND maturity_description_code = 2
	AND coupon_type in (8, 4, 10)
	AND capital_type <> 10
	AND default_exists <> TRUE
	AND sale_type <> 4
	AND sec_regulation IS NULL
	AND most_recent_default_event IS NULL
	AND default_indicator IS FALSE
	AND DATETIME_DIFF(trade_datetime,recent[SAFE_OFFSET(0)].trade_datetime,SECOND) < 1000000 -- 12 days to the most recent trade
	AND msrb_valid_to_date > current_date -- condition to remove cancelled trades
	AND rtrs_control_number = 2022040108859100
	ORDER BY
	trade_datetime DESC 
	"""
	trade = sqltodf(query, bq_client)
	ficc_ycl = trade.apply(lambda x: get_ficc_ycl(x, client=bq_client), axis=1)
	spread = trade['yield'] * 100 - ficc_ycl
	print(spread)