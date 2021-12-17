
from ficc.utils.yield_curve import yield_curve_params
import ficc.utils.globals as globals
import os
from google.cloud import bigquery
from ficc.data.process_trade_history import process_trade_history

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/shayaan/ficc/eng-reactor-287421-112eb767e1b3.json"
SEQUENCE_LENGTH = 5
NUM_FEATURES = 5
DATA_QUERY = """ SELECT
  *
FROM
  `eng-reactor-287421.primary_views.speedy_trade_history` 
WHERE
  yield IS NOT NULL
  AND yield > 0 
  AND yield <= 3 
  AND par_traded IS NOT NULL
  AND sp_long IS NOT NULL
  AND trade_date >= '2021-07-01' 
  AND trade_date <= '2021-10-01'
  AND msrb_valid_to_date > current_date -- condition to remove cancelled trades
ORDER BY
  trade_date DESC
LIMIT 10
            """

bq_client = bigquery.Client()

if __name__ == "__main__":
    trade_data = process_trade_history(DATA_QUERY,bq_client,SEQUENCE_LENGTH,NUM_FEATURES,'tuning.pkl')
    print(trade_data.head())