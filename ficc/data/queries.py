
standard_training_query = """ SELECT
  *
FROM
  `eng-reactor-287421.auxiliary_views.materialized_trade_history`
WHERE
  yield IS NOT NULL
  AND yield > 0 
  AND yield <= 3 
  AND par_traded IS NOT NULL
  AND trade_date >= '2021-07-01' 
  AND trade_date <= '2021-10-01'
  AND maturity_description_code = 2
  AND incorporated_state_code <> 'US'
  AND coupon_type = 8
  AND msrb_valid_to_date > current_date -- condition to remove cancelled trades
ORDER BY
  trade_date DESC
"""
