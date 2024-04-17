'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2021-12-17
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-04-17
 # @ Description:
 '''
import os
import pandas as pd
import pickle5 as pickle

from ficc.utils.auxiliary_functions import sqltodf, process_ratings
from ficc.utils.pad_trade_history import pad_trade_history
from ficc.utils.trade_list_to_array import trade_list_to_array


def fetch_trade_data(query, client, PATH='data.pkl', save_data=True):
    if os.path.isfile(PATH):
        print(f'Data file {PATH} found, reading data from it')
        with open(PATH, 'rb') as f: 
            (q, trades_df) = pickle.load(f)
        if q == query:
            return trades_df
        else:
            raise Exception (f'Saved query is incorrect:\n{q}')
    
    print(f'Grabbing data from BigQuery with query:')
    print(query)
    trades_df = sqltodf(query, client)

    if save_data:
        print(f'Saving query and data to {PATH}')
        ds = (query, trades_df)
        with open(PATH, 'wb') as f: 
            pickle.dump(ds, f)
    return trades_df


def restrict_number_of_trades(series: pd.Series, num_trades: int, processing_similar_trades: bool) -> pd.Series:
    '''`processing_similar_trades` is used solely for print output.'''
    trade_history_prefix = 'similar ' if processing_similar_trades else ''
    print(f'Restricting the {trade_history_prefix}trade history to the {num_trades} most recent trades')
    return series.trade_history.apply(lambda history: history[:num_trades])


def pad_trade_history_column(series: pd.Series, num_trades_in_history: int, min_trades_in_history: int, num_features_for_each_trade_in_history: int, processing_similar_trades: bool) -> pd.Series:
    '''`processing_similar_trades` is used solely for print output.'''
    trade_history_prefix = 'similar ' if processing_similar_trades else ''
    print(f'Padding {trade_history_prefix}trade history')
    print(f'Minimum number of trades required in the {trade_history_prefix}trade history: {min_trades_in_history}')
    return series.parallel_apply(pad_trade_history, args=[num_trades_in_history, 
                                                          num_features_for_each_trade_in_history,
                                                          min_trades_in_history])


def restrict_number_of_trades_and_pad_trade_history(df: pd.DataFrame, trade_history_column_name: str, num_trades_in_history: int, min_trades_in_history: int, num_features_for_each_trade_in_history: int, processing_similar_trades: bool = False) -> pd.DataFrame:
    '''`processing_similar_trades` is used solely for print output.'''
    df[trade_history_column_name] = restrict_number_of_trades(df[trade_history_column_name], num_trades_in_history, processing_similar_trades)
    df[trade_history_column_name] = pad_trade_history_column(df[trade_history_column_name], num_trades_in_history, min_trades_in_history, num_features_for_each_trade_in_history, processing_similar_trades)
    return df


def process_trade_history(query: str,
                          client, 
                          num_trades_in_history: int, 
                          num_features_for_each_trade_in_history: int, 
                          PATH: str,  
                          remove_short_maturity: bool,
                          trade_history_delay: int, 
                          min_trades_in_history: int, 
                          use_treasury_spread: bool,
                          add_rtrs_in_history: bool,
                          only_dollar_price_history: bool, 
                          yield_curve_to_use: str, 
                          treasury_rate_dict: dict = None, 
                          nelson_params: dict = None, 
                          scalar_params: dict = None, 
                          shape_parameter: dict = None, 
                          save_data: bool = True, 
                          process_similar_trades_history: bool = False):
    trades_df = fetch_trade_data(query, client, PATH, save_data)
    if len(trades_df) == 0:
        print('Raw data contains 0 trades')
        return None
    print(f'Raw data contains {len(trades_df)} trades ranging from trade datetimes of {trades_df.trade_datetime.min()} to {trades_df.trade_datetime.max()}')
    
    trades_df = process_ratings(trades_df)
    # trades_df = convert_object_to_category(trades_df)

    print('Creating trade history')
    if remove_short_maturity is True: print('Removing trades with shorter maturity')
    print(f'Removing trades less than {trade_history_delay} seconds in the history')
    
    processed_trade_history_column_name = 'trade_history'
    last_features_column_name = 'temp_last_features'
    temp = pd.DataFrame(data=None, index=trades_df.index, columns=[processed_trade_history_column_name, last_features_column_name])
    unprocessed_trade_history_column_name = 'recent'
    temp = trades_df[unprocessed_trade_history_column_name].parallel_apply(trade_list_to_array, args=([remove_short_maturity,
                                                                                                       trade_history_delay,
                                                                                                       use_treasury_spread,
                                                                                                       add_rtrs_in_history,
                                                                                                       only_dollar_price_history, 
                                                                                                       yield_curve_to_use, 
                                                                                                       treasury_rate_dict, 
                                                                                                       nelson_params, 
                                                                                                       scalar_params, 
                                                                                                       shape_parameter]))
                                                                        
    trades_df[[processed_trade_history_column_name, last_features_column_name]] = pd.DataFrame(temp.tolist(), index=trades_df.index)
    del temp
    print('Trade history created')
    print('Getting last trade features')
    trades_df[['last_yield_spread', 
               'last_ficc_ycl', 
               'last_rtrs_control_number', 
               'last_yield', 
               'last_dollar_price', 
               'last_seconds_ago', 
               'last_size', 
               'last_calc_date', 
               'last_maturity_date', 
               'last_next_call_date', 
               'last_par_call_date', 
               'last_refund_date', 
               'last_trade_datetime', 
               'last_calc_day_cat', 
               'last_settlement_date', 
               'last_trade_type']] = pd.DataFrame(trades_df[last_features_column_name].tolist(), index=trades_df.index)
    trades_df = trades_df.drop(columns=[last_features_column_name, unprocessed_trade_history_column_name])
    trades_df = restrict_number_of_trades_and_pad_trade_history(trades_df, processed_trade_history_column_name, num_trades_in_history, min_trades_in_history, num_features_for_each_trade_in_history)
    trade_history_features = [processed_trade_history_column_name]

    if process_similar_trades_history is False:
        print('Creating similar trade history')
        processed_trade_history_column_name = 'similar_trade_history'
        last_features_column_name = 'temp_last_similar_features'
        temp = pd.DataFrame(data=None, index=trades_df.index, columns=[processed_trade_history_column_name, last_features_column_name])
        unprocessed_trade_history_column_name = 'recent_5_year_mat'
        temp = trades_df[unprocessed_trade_history_column_name].parallel_apply(trade_list_to_array, args=([remove_short_maturity,
                                                                                                           trade_history_delay,
                                                                                                           use_treasury_spread,
                                                                                                           add_rtrs_in_history,
                                                                                                           only_dollar_price_history, 
                                                                                                           yield_curve_to_use, 
                                                                                                           treasury_rate_dict, 
                                                                                                           nelson_params, 
                                                                                                           scalar_params, 
                                                                                                           shape_parameter]))
        # TODO: speed the below line up by not storing the unnecessary information for the most recent trade (needed when processing same CUSIP trade history, but not for similar trade history)
        trades_df[[processed_trade_history_column_name, last_features_column_name]] = pd.DataFrame(temp.tolist(), index=trades_df.index)
        del temp
        print('Similar trade history created')
        trades_df = trades_df.drop(columns=[last_features_column_name, unprocessed_trade_history_column_name])
        trades_df = restrict_number_of_trades_and_pad_trade_history(trades_df, processed_trade_history_column_name, num_trades_in_history, min_trades_in_history, num_features_for_each_trade_in_history, True)
        trade_history_features.append(processed_trade_history_column_name)
    
    num_trades_before_removing_null_history = len(trades_df)
    trades_df.dropna(subset=trade_history_features, inplace=True)
    print(f'Processed trade history contains {len(trades_df)} trades. Prior to removing null histories (i.e., removing null values in {trade_history_features}), it contained {num_trades_before_removing_null_history} trades.')
    return trades_df
