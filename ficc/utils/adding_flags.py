'''
 # @ Author: Mitas Ray
 # @ Create Time: 2022-08-08 12:11:00
 # @ Modified by: 
 # @ Modified time: 
 # @ Description: Adds flags to trades to provide additional features
 '''

import numpy as np
import pandas as pd

# from ficc.utils.auxiliary_variables import SPECIAL_CONDITIONS
# SPECIAL_CONDITIONS_TO_FILTER_ON = [condition for condition in SPECIAL_CONDITIONS if condition != 'is_alternative_trading_system']    # this removes conditions that we do not want to group on


def get_most_recent_index_and_others(df, with_alternative_trading_system_flag=False, get_earliest_index=False):
    '''If `with_alternative_trading_system_flag` is `True`, then return the most recent 
    index with the alternative trading system flag. If no trades in `df` have the 
    flag, then behave as if `with_alternative_trading_system_flag` is `False`. Note: 
    only inter-dealer trades can have the alternative trading system flag. If 
    `get_earliest_index` is `True` instead of returning the most recent index, return 
    the earliest index (opposite of most recent).'''
    if with_alternative_trading_system_flag:    # check whether there is a most recent index with the alternative trading system flag
        assert 'is_alternative_trading_system' in df.columns
        df_with_alternative_trading_system_flag = df[df['is_alternative_trading_system']]
        indices = df_with_alternative_trading_system_flag.index.to_list()
    if not with_alternative_trading_system_flag or indices == []:    # if the alternative trading system flag was not desired or not found
        indices = df.index.to_list()
    trade_index = indices[0] if not get_earliest_index else indices[-1]   # since `df` is sorted in descending order of `trade_date`, the first item is the most recent and the last item is the earlest

    return trade_index, [index for index in df.index.to_list() if index != trade_index]


def indices_to_remove_from_beginning_or_end_to_reach_sum(lst, target_sum):
    '''The goal is to find a continuous stream of items in `lst` where at least one of the 
    endpoints of `lst`, such that the sum of this stream of items equals `target_sum`. If 
    such a sublist cannot be formed, then return `None`. Otherwise return the indices that 
    should be removed from `lst` so that summing the remaining items equals `target_sum`.'''
    # forward pass
    lst_total = sum(lst)
    assert lst_total > target_sum
    indices = []
    for index, item in enumerate(lst):
        lst_total -= item
        indices.append(index)
        if lst_total == target_sum:
            return indices
    
    # backward pass
    lst_total = sum(lst)
    indices = []
    for index in range(len(lst) - 1, -1, -1): 
        item = lst[index]
        lst_total -= item
        indices.append(index)
        if lst_total == target_sum:
            return indices


def _add_bookkeeping_flag_for_group(group_df, flag_name, orig_df=None):
    '''Mark an inter-dealer trade as bookkeeping if there are multiple 
    inter-dealer trades of the same quantity at the same price for a 
    particular day. The intuition here is that this bond is moving from 
    desk to desk. All except the most recent one in this group are 
    marked as bookkeeping.'''
    assert flag_name in group_df.columns, '`{flag_name}` must be a column in the dataframe in order to mark that this trade just switched desks'
    if orig_df is None: orig_df = group_df
    if set(group_df['trade_type']) != {'D'} or len(group_df) < 2: return orig_df    # dataframe has trades that are not inter-dealer or has a size less than 2
    orig_df.loc[group_df.index.to_list(), flag_name] = True    # mark all trades in the group
    # mark all but the most recent trade as bookkeeping
    # _, all_but_most_recent_index = get_most_recent_index_and_others(group_df)
    # orig_df.loc[all_but_most_recent_index, flag_name] = True
    return orig_df


def add_bookkeeping_flag(df, flag_name):
    '''Call `_add_bookkeeping_flag_for_group(...)` on each group as 
    specified in the `groupby`.'''
    if flag_name in df.columns and df[flag_name].any(): return df
    print(f'Adding {flag_name} flag to data')
    df = df.copy()
    if flag_name not in df.columns: df[flag_name] = False
    groups_same_day_quantity_price_tradetype_cusip = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'quantity', 'dollar_price', 'trade_type', 'cusip'])    # considered adding SPECIAL_CONDITIONS_TO_FILTER_ON in the groupby but it makes the condition too restrictive
    groups_same_day_quantity_price_tradetype_cusip_largerthan1_onlyDD = [group_df for _, group_df in groups_same_day_quantity_price_tradetype_cusip if set(group_df['trade_type']) == {'D'} and len(group_df) > 1]
    for group_df in groups_same_day_quantity_price_tradetype_cusip_largerthan1_onlyDD:
        df = _add_bookkeeping_flag_for_group(group_df, flag_name, df)
    return df


def _add_same_day_flag_for_group(group_df, flag_name, orig_df=None):
    '''This flag denotes a trade where the dealer had the purchase 
    and sell lined up beforehand. Our logic for identifying trades 
    that occur on the same day are as follows:
    1. A group of dealer sell trades are considered same day if the 
    total cost of the dealer purchase trades for that day is equal to 
    or greater than the total cost of the dealer sell trades. In this 
    case, a group of dealer purchases trades are considered same day 
    if there is a continuous (continuous defined as a dealer purchase 
    trade not skipped over chronologically) sequence of dealer purchase 
    trades that equal the total cost of the dealer sell trades. We 
    assume this sequence of dealer purchase trades includes either the 
    first dealer purchase trade of the day and/or the last dealer 
    purchase trade of the day. We may expand this criteria to not have 
    to include either the first and/or last dealer purchase trade.
    2. An inter-dealer trade is considered *same day* if the quantity is 
    equal to the total cost of the dealer sell trades for that day and 
    if the total cost of the dealer purchase trades for that day is 
    greater than or equal to the total cost of the dealer sell trades.'''
    assert flag_name in group_df.columns, '`{flag_name}` must be a column in the dataframe in order to mark that a trade was arranged so that a bond would not have to be held overnight'
    groups_by_trade_type = group_df.groupby('trade_type').sum()
    if orig_df is None: orig_df = group_df
    if 'S' not in groups_by_trade_type.index or 'P' not in groups_by_trade_type.index: return orig_df
    dealer_sold_indices = group_df[group_df['trade_type'] == 'S'].index.values
    dealer_purchase_indices = group_df[group_df['trade_type'] == 'P'].index.values
    total_dealer_sold = groups_by_trade_type.loc['S']['quantity']
    total_dealer_purchased = groups_by_trade_type.loc['P']['quantity']

    indices_to_mark = []
    if total_dealer_sold <= total_dealer_purchased:
        indices_to_mark.extend(dealer_sold_indices)
        for index, quantity in group_df[group_df['trade_type'] == 'D']['quantity'].iteritems():
            if quantity == total_dealer_sold:
                indices_to_mark.append(index)
        
        if total_dealer_sold == total_dealer_purchased:
            indices_to_mark.extend(dealer_purchase_indices)
        else:
            indices_to_remove_from_dealer_purchase_indices = indices_to_remove_from_beginning_or_end_to_reach_sum(group_df[group_df['trade_type'] == 'P']['quantity'].values, total_dealer_sold)
            if indices_to_remove_from_dealer_purchase_indices is not None:
                for index_to_remove in sorted(indices_to_remove_from_dealer_purchase_indices, reverse=True):    # need to sort in reverse order to make sure future indices are still valid after removing current index; e.g., cannot remove elements at index 0 and 1 of a two element list in that order (index 1 does not exist after removing index 0)
                    dealer_purchase_indices = np.delete(dealer_purchase_indices, index_to_remove, axis=0)
                indices_to_mark.extend(dealer_purchase_indices)
    
    orig_df.loc[indices_to_mark, flag_name] = True
    return orig_df


def add_same_day_flag(df, flag_name):
    '''Call `_add_bookkeeping_flag_for_group(...)` on each group as 
    specified in the `groupby`.'''
    if flag_name in df.columns and df[flag_name].any(): return df
    print(f'Adding {flag_name} flag to data')
    df = df.copy()
    if flag_name not in df.columns: df[flag_name] = False
    groups = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), 'cusip'])
    groups_with_sp = [group_df for _, group_df in groups if {'S', 'P'} <= set(group_df['trade_type'])]
    for group_df in groups_with_sp:
        df = _add_same_day_flag_for_group(group_df, flag_name, df)
    return df


def _add_replica_flag_for_group(group_df, flag_name, orig_df=None):
    '''Mark a trade as a replica if there is a previous trade on the same 
    day with the same price, same direction, and same quantity. The idea 
    of marking these trades is to exclude them from the trade history, as 
    these trades are probably being sold in the same block, and so having 
    all of these trades in the trade history would be less economically 
    meaningful in the trade history. All except the earliest trade in this 
    group are marked as a replica.'''
    assert flag_name in group_df.columns, '`{flag_name}` must be a column in the dataframe in order to mark that this trade is identical to another trade that occurs during the same day'
    if orig_df is None: orig_df = group_df
    if len(group_df) < 2: return orig_df    # dataframe has a size less than 2
    orig_df.loc[group_df.index.to_list(), flag_name] = True    # mark all trades in the group
    # mark all but the earliest trade as a replica
    # _, all_but_earliest_index = get_most_recent_index_and_others(group_df, get_earliest_index=True)
    # orig_df.loc[all_but_earliest_index, flag_name] = True    # orig_df[flag_name][all_but_earliest_index] = True
    return orig_df


def add_replica_flag(df, flag_name):
    '''Call `_add_replica_flag_for_group(...)` on each group as 
    specified in the `groupby`.'''
    if flag_name in df.columns and df[flag_name].any(): return df
    print(f'Adding {flag_name} flag to data')
    df = df.copy()
    if flag_name not in df.columns: df[flag_name] = False

    # Next 3 lines ensure that correct feature is used to find the quantity. When 
    # this function is run before `process_features(...)`, the quantity is 
    # represented as the feature `par_traded`. After running `process_features(...)`,
    # the quantity is represented as the feature `quantity`.
    columns_set = set(df.columns)
    quantity_feature = 'quantity' if 'quantity' in columns_set else 'par_traded'
    assert 'par_traded' in columns_set, 'Neither "quantity" nor "par_traded" exist in the dataframe'

    groups_same_day_quantity_price_tradetype_cusip = df.groupby([pd.Grouper(key='trade_datetime', freq='1D'), quantity_feature, 'dollar_price', 'trade_type', 'cusip'])    # considered adding SPECIAL_CONDITIONS_TO_FILTER_ON in the groupby but it makes the condition too restrictive
    groups_same_day_quantity_price_tradetype_cusip_largerthan1 = [group_df for _, group_df in groups_same_day_quantity_price_tradetype_cusip if len(group_df) > 1]
    for group_df in groups_same_day_quantity_price_tradetype_cusip_largerthan1:
        df = _add_replica_flag_for_group(group_df, flag_name, df)
    return df
