import os
import time

from datetime import datetime

import pandas as pd

from ficc.utils.auxiliary_variables import IS_BOOKKEEPING, IS_SAME_DAY, NTBC_PRECURSOR, IS_REPLICA
from ficc.utils.adding_flags import add_bookkeeping_flag, add_same_day_flag, add_ntbc_precursor_flag, add_replica_flag
from ficc.utils.adding_flags_v2 import add_bookkeeping_flag_v2, add_same_day_flag_v2, add_ntbc_precursor_flag_v2, add_replica_flag_v2

import sys
sys.path.insert(0, '/Users/mitas/ficc/ficc/ml_models/sequence_predictors/')

from rating_model_mitas.data_prep import read_processed_file_pickle

FILENAME = '/Users/mitas/ficc/ficc/ml_models/sequence_predictors/data/processed_data_ficc_ycl_2021-12-31-23-59.pkl'
FLAGS = [IS_BOOKKEEPING, IS_SAME_DAY, NTBC_PRECURSOR, IS_REPLICA]


def load_file_and_create_datasets(filename=FILENAME):
    three_months_data = read_processed_file_pickle(filename)
    one_week_data = three_months_data[three_months_data['trade_datetime'] <= datetime(2021, 10, 8)]
    one_month_data = three_months_data[three_months_data['trade_datetime'] <= datetime(2021, 10, 31)]
    return one_week_data, one_month_data, three_months_data


def add_flags(data, save_filename=None):
    start_time = time.time()
    data = add_same_day_flag(data)
    data = add_ntbc_precursor_flag(data)
    data = add_replica_flag(data)
    data = add_bookkeeping_flag(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time} seconds')
    if save_filename != None: data.to_pickle(f'./files/{save_filename}.pkl')
    return data, elapsed_time


def add_flags_v2(data, save_filename=None, compare_filename=None):
    if compare_filename != None: assert os.path.exists(f'./files/{compare_filename}.pkl'), 'No file to compare against'
    start_time = time.time()
    data = add_same_day_flag_v2(data)
    data = add_ntbc_precursor_flag_v2(data)
    data = add_replica_flag_v2(data)
    data = add_bookkeeping_flag_v2(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time} seconds')
    if save_filename != None: data.to_pickle(f'./files/{save_filename}_v2.pkl')
    if compare_filename != None:
        truth = pd.read_pickle(f'./files/{compare_filename}.pkl')
        for flag in FLAGS:
            assert data[flag].equals(truth[flag]), f'{flag} values are not equal'
        print(f'All flags match')
    return data, elapsed_time


def create_ground_truth_datasets():
    one_week_data, one_month_data, three_months_data = load_file_and_create_datasets()
    _, elapsed_time_one_week = add_flags(one_week_data, 'one_week')
    _, elapsed_time_one_month = add_flags(one_month_data, 'one_month')
    _, elapsed_time_three_months = add_flags(three_months_data, 'three_months')
    with open('times.txt', 'w') as f:
        f.write(f'One week: {elapsed_time_one_week}\nOne month: {elapsed_time_one_month}\nThree months: {elapsed_time_three_months}')


if __name__ == '__main__':
    one_week_data, one_month_data, three_months_data = load_file_and_create_datasets()
    # for column in one_week_data.columns:
    #     print(column, one_week_data[column].dtype)
    _, elapsed_time_one_week = add_flags(one_week_data, 'one_week')


# ##### Results #####
# Adding `observed=True` flag to the groupby commands does not cause a speed up; leaving for now, since Charles suggested it and noticed speed ups with it in his code for other situations