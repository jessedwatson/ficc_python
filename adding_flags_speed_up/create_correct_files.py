import time
from datetime import datetime

from ficc.utils.adding_flags import add_bookkeeping_flag, add_same_day_flag, add_ntbc_precursor_flag, add_replica_flag
from ficc.utils.adding_flags_v2 import add_bookkeeping_flag_v2, add_same_day_flag_v2, add_ntbc_precursor_flag_v2, add_replica_flag_v2

import sys
sys.path.insert(0, '/Users/mitas/ficc/ficc/ml_models/sequence_predictors/')

from rating_model_mitas.data_prep import read_processed_file_pickle


def add_flags(data, save_name=None):
    start_time = time.time()
    data = add_replica_flag(data)
    data = add_bookkeeping_flag(data)
    data = add_same_day_flag(data)
    data = add_ntbc_precursor_flag(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time} seconds')
    if save_name != None: data.to_pickle(f'./files/{save_name}.pkl')
    return data, elapsed_time


def add_flags_v2(data, save_name=None):
    start_time = time.time()
    data = add_replica_flag_v2(data)
    data = add_bookkeeping_flag_v2(data)
    data = add_same_day_flag_v2(data)
    data = add_ntbc_precursor_flag_v2(data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time} seconds')
    if save_name != None: data.to_pickle(f'./files/{save_name}_v2.pkl')
    return data, elapsed_time


if __name__ == '__main__':
    filename = '/Users/mitas/ficc/ficc/ml_models/sequence_predictors/data/processed_data_ficc_ycl_2021-12-31-23-59.pkl'
    three_months_data = read_processed_file_pickle(filename)
    one_week_data = three_months_data[three_months_data['trade_datetime'] <= datetime.date(2021, 10, 8)]
    one_month_data = three_months_data[three_months_data['trade_datetime'] <= datetime.date(2021, 10, 31)]
    _, elapsed_time_one_week = add_flags(one_week_data, 'one_week')
    _, elapsed_time_one_month = add_flags(one_month_data, 'one_month')
    _, elapsed_time_three_months = add_flags(three_months_data, 'three_months')
    with open('times.txt', 'w') as f:
        f.write(f'One week: {elapsed_time_one_week}\nOne month: {elapsed_time_one_month}\nThree months: {elapsed_time_three_months}')
