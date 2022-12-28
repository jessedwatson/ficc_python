'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-17 12:09:34
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-12-28 01:21:34
 # @ Description:
 '''

import pandas as pd
import numpy as np
from ficc.utils.auxiliary_variables import COUPON_FREQUENCY_DICT
from ficc.utils.auxiliary_functions import get_latest_trade_feature
from ficc.utils.diff_in_days import diff_in_days
from ficc.utils.days_in_interest_payment import days_in_interest_payment
from ficc.utils.fill_missing_values import fill_missing_values
from ficc.utils.auxiliary_functions import calculate_a_over_e

def process_features(df, keep_nan):
    # Removing bonds from Puerto Rico
    df = df[df.incorporated_state_code != 'PR']

    df.interest_payment_frequency.fillna(0, inplace=True)
    df.loc[:,'interest_payment_frequency'] = df.interest_payment_frequency.apply(lambda x: COUPON_FREQUENCY_DICT[x])
    
    # Processing 
    df.loc[:,'quantity'] = np.log10(df.par_traded.astype(np.float32))
    df.coupon = df.coupon.astype(np.float32)
    df.issue_amount = np.log10(1 + df.issue_amount.astype(np.float32))
    df.maturity_amount = np.log10(1.0 + df.maturity_amount.astype(float))
    df.orig_principal_amount = np.log10(1.0 + df.orig_principal_amount.astype(float))
    #Check the outstanding_amount
    df.max_amount_outstanding = np.log10(1.0 + df.max_amount_outstanding.astype(float))
    
    # Creating Binary features
    df.loc[:,'callable'] = df.is_callable  
    df.loc[:,'called'] = df.is_called 
    df.loc[:,'zerocoupon'] = df.coupon == 0
    df.loc[:,'whenissued'] = df.delivery_date >= df.trade_date
    df.loc[:,'sinking'] = ~df.next_sink_date.isnull()
    df.loc[:,'deferred'] = (df.interest_payment_frequency == 'Unknown') | df.zerocoupon
    
    # Converting the dates to a number of days from the settlement date. 

    # We only consider trades to be reportedly correctly if the trades are settled within one month of the trade date. 
    df.loc[:,'days_to_settle'] = (df.settlement_date - df.trade_date).dt.days.fillna(0)
    print('Removing trades which are settled more than a month from trade date')
    df = df[df.days_to_settle < 30]

    df.loc[:, 'days_to_maturity'] =  np.log10(1 + (df.maturity_date - df.settlement_date).dt.days)
    df.loc[:, 'days_to_call'] = np.log10(1 + (df.next_call_date - df.settlement_date).dt.days)
    df.loc[:, 'days_to_refund'] = np.log10(1 + (df.refund_date - df.settlement_date).dt.days)
    df.loc[:, 'days_to_par'] = np.log10(1 + (df.par_call_date - df.settlement_date).dt.days)
    df.loc[:, 'call_to_maturity'] = np.log10(1 + (df.maturity_date - df.next_call_date).dt.days)

    df.days_to_maturity.replace(-np.inf, np.nan, inplace=True)
    df.days_to_call.replace(-np.inf, np.nan, inplace=True)
    df.days_to_refund.replace(-np.inf, np.nan, inplace=True)
    df.days_to_par.replace(-np.inf, np.nan, inplace=True)
    
    # Adding features of the last trade i.e the trade before the most recent trade
    # temp_df = df.trade_history.apply(get_latest_trade_feature)
    # df[['last_seconds_ago', 'last_yield_spread', 'last_size']] = pd.DataFrame(temp_df.tolist(), index=df.index)
    # del temp_df

    # Adding features from MSRB rule 33G
    df.loc[:, 'accrued_days'] = df.apply(diff_in_days, calc_type="accrual", axis=1)
    df.loc[:, 'days_in_interest_payment'] = df.apply(days_in_interest_payment, axis=1)
    df.loc[:, 'scaled_accrued_days'] = df['accrued_days'] / (360/df['days_in_interest_payment'])
    df.loc[:, 'A/E'] = df.apply(calculate_a_over_e, axis=1)
    df = fill_missing_values(df, keep_nan)

    return df