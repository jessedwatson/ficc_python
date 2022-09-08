'''
 # @ Author: Mitas Ray
 # @ Create Time: 2022-08-08 12:11:00
 # @ Modified by: Mitas Ray
 # @ Modified time: 2022-09-06 15:00:00
 # @ Description: Adds flags to trades to provide additional features
 '''

import numpy as np
import pandas as pd

from ficc.utils.auxiliary_variables import IS_REPLICA, IS_BOOKKEEPING, IS_SAME_DAY, NTBC_PRECURSOR


def subarray_sum(lst, target_sum, indices):
    '''The goal is to find a sublist in `lst`, such that the sum of the sublist equals 
    `target_sum`. If such a sublist cannot be formed, then return an dempty list. 
    Otherwise return the indices that should be removed from `lst` so that summing the 
    remaining items equals `target_sum`. The sublist in `indices` is returned. 
    Reference: https://www.geeksforgeeks.org/find-subarray-with-given-sum/ '''
    len_lst = len(lst)
    current_sum = lst[0]
    start, end = 0, 1
    while end <= len_lst:
        while current_sum > target_sum and start < end - 1:    # remove items from beginning of current sublist if the current sum is larger than the target
            current_sum -= lst[start]
            start += 1
        if current_sum == target_sum:
            return indices[start:end]
        if end < len_lst:
            current_sum += lst[end]
        end += 1
    return []


def _add_same_day_flag_for_group(group_df):
    '''This flag denotes a trade where the dealer had the purchase and sell lined up 
    beforehand. We mark a trade as same day when:
    1. A group of dealer sell trades are considered same day if the total par_traded of the 
    dealer purchase trades for that day is equal to or greater than the total par_traded of the 
    dealer sell trades. In this case, a group of dealer purchases trades are considered 
    same day if there is a continuous (continuous defined as a dealer purchase trade not 
    skipped over chronologically) sequence of dealer purchase trades that equal the total 
    par_traded of the dealer sell trades.
    2. An inter-dealer trade is considered *same day* if the par_traded is equal to the total 
    par_traded of the dealer sell trades for that day and if the total par_traded of the dealer purchase 
    trades for that day is greater than or equal to the total par_traded of the dealer sell trades.'''

    group_df_by_trade_type = group_df.groupby('trade_type', observed=True)
    if not ({'S', 'P'} <= set(group_df_by_trade_type.groups.keys())): return []

    dealer_sold_indices, dealer_purchase_indices = group_df_by_trade_type.get_group('S').index, group_df_by_trade_type.get_group('P').index
    group_df_by_trade_type_sums = group_df_by_trade_type['par_traded'].sum()
    total_dealer_sold, total_dealer_purchased = group_df_by_trade_type_sums['S'], group_df_by_trade_type_sums['P']

    indices_to_mark = []
    if total_dealer_sold <= total_dealer_purchased:
        indices_to_mark.extend(dealer_sold_indices)
        dd_indices_to_mark = group_df[(group_df['trade_type'] == 'D') & (group_df['par_traded'] == total_dealer_sold)].index
        indices_to_mark.extend(dd_indices_to_mark)
        
        if total_dealer_sold == total_dealer_purchased:
            indices_to_mark.extend(dealer_purchase_indices)
        else:
            indices_to_mark_from_dealer_purchase_indices = subarray_sum(group_df_by_trade_type.get_group('P')['par_traded'].values, total_dealer_sold, dealer_purchase_indices)
            indices_to_mark.extend(indices_to_mark_from_dealer_purchase_indices)

    return indices_to_mark


def add_same_day_flag(df, flag_name=IS_SAME_DAY, use_parallel_apply=True):
    '''Call `_add_same_day_flag_for_group(...)` on each group as 
    specified in the `groupby`. Similar code structure to other 
    `add_*_flag(...)` functions.'''
    df = df.astype({'par_traded': np.float64})    # `par_traded` type is Category so need to change it order to sum up; chose float64 to prevent potential rounding errors

    df[flag_name] = False
    group_by_day_cusip = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'cusip'], observed=True)[['par_traded', 'trade_type']]    # only need the 'par_traded' and 'trade_type' columns in the helper function
    apply_func = pd.DataFrame.parallel_apply if use_parallel_apply else pd.DataFrame.apply    # choose between .apply(...) and .parallel_apply(...)
    day_cusip_to_indices_to_mark = apply_func(group_by_day_cusip, _add_same_day_flag_for_group)
    indices_to_mark = day_cusip_to_indices_to_mark.sum()
    df.loc[indices_to_mark, flag_name] = True
    return df


def _add_replica_flag_for_group(group_df):
    '''Mark a trade as a replica if there is a trade on the same 
    day with the same price, same direction, and same quantity. The idea 
    of marking these trades is to exclude them from the trade history, as 
    these trades are probably being sold in the same block, and so having 
    all of these trades in the trade history would be less economically 
    meaningful.'''
    return group_df.index.to_list() if len(group_df) >= 2 else []


def add_replica_flag(df, flag_name=IS_REPLICA, use_parallel_apply=True):
    '''Call `_add_replica_flag_for_group(...)` on each group as 
    specified in the `groupby`. Similar code structure to other 
    `add_*_flag(...)` functions.'''
    df[flag_name] = False
    group_by_day_cusip_quantity_price_tradetype = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'cusip', 'quantity', 'dollar_price', 'trade_type'], observed=True)
    apply_func = pd.DataFrame.parallel_apply if use_parallel_apply else pd.DataFrame.apply    # choose between .apply(...) and .parallel_apply(...)
    day_cusip_quantity_price_tradetype_to_indices_to_mark = apply_func(group_by_day_cusip_quantity_price_tradetype, _add_replica_flag_for_group)
    indices_to_mark = day_cusip_quantity_price_tradetype_to_indices_to_mark.sum()
    df.loc[indices_to_mark, flag_name] = True
    return df


def add_bookkeeping_flag(df, flag_name=IS_BOOKKEEPING, use_parallel_apply=True):
    '''Select only the inter-dealer trades and then call `_add_replica_flag_for_group(...)` 
    on each group as specified in the `groupby`. Similar code structure to other 
    `add_*_flag(...)` functions.'''
    df[flag_name] = False
    dd_group_by_day_cusip_quantity_price = df[df['trade_type'] == 'D'].groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'cusip', 'quantity', 'dollar_price'], observed=True)    # select only inter-dealer trades before performing .groupby since only inter-dealer trades can be bookkeeping
    apply_func = pd.DataFrame.parallel_apply if use_parallel_apply else pd.DataFrame.apply    # choose between .apply(...) and .parallel_apply(...)
    day_cusip_quantity_price_to_indices_to_mark = apply_func(dd_group_by_day_cusip_quantity_price, _add_replica_flag_for_group)
    indices_to_mark = day_cusip_quantity_price_to_indices_to_mark.sum()
    df.loc[indices_to_mark, flag_name] = True
    return df


def _add_ntbc_precursor_flag_for_group(group_df):
    '''This flag denotes an inter-dealer trade that occurs on the same day as 
    a non-transaction-based-compensation customer trade with the same price and 
    quantity. The idea for marking it is that this inter-dealer trade may not be 
    genuine (i.e., window-dressing). Note that we have a buffer of occurring on 
    the same day since we see examples in the data (e.g., cusip 549696RS3, 
    trade_datetime 2022-04-01) having the corresponding inter-dealer trade occurring 
    4 seconds before, instead of the exact same time, as the customer bought trade.'''
    is_dd_trade = group_df['trade_type'] == 'D'
    if (len(group_df) < 2) or (not group_df['is_non_transaction_based_compensation'].any()) or (not is_dd_trade.any()): return []
    return group_df[is_dd_trade].index.to_list()


def add_ntbc_precursor_flag(df, flag_name=NTBC_PRECURSOR, use_parallel_apply=True):
    '''Call `_add_ntbc_precursor_flag_for_group(...)` on each group as 
    specified in the `groupby`. Similar code structure to other 
    `add_*_flag(...)` functions.'''
    df[flag_name] = False
    group_by_day_cusip_quantity_price = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'cusip', 'quantity', 'dollar_price'], observed=True)[['is_non_transaction_based_compensation', 'trade_type']]    # only need the 'is_non_transaction_based_compensation' and 'trade_type' columns in the helper function
    apply_func = pd.DataFrame.parallel_apply if use_parallel_apply else pd.DataFrame.apply    # choose between .apply(...) and .parallel_apply(...)
    day_cusip_quantity_price_to_indices_to_mark = apply_func(group_by_day_cusip_quantity_price, _add_ntbc_precursor_flag_for_group)
    indices_to_mark = day_cusip_quantity_price_to_indices_to_mark.sum()
    df.loc[indices_to_mark, flag_name] = True
    return df
