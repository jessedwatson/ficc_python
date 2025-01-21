'''
Author: Mitas Ray
Date: 2025-01-21
Last Editor: Mitas Ray
Last Edit Date: 2025-01-21
Description: Used to train a model with a processed data file. Heavily uses code from `automated_training/`.
'''
import os
import sys

import numpy as np
import pandas as pd


ficc_package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))    # get the directory containing the 'ficc_python/' package
sys.path.append(ficc_package_dir)    # add the directory to sys.path


import automated_training.automated_training_auxiliary_functions

from automated_training.automated_training_auxiliary_variables import MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME, BUCKET_NAME
automated_training.automated_training_auxiliary_functions.SAVE_MODEL_AND_DATA = False
from automated_training.automated_training_auxiliary_functions import train_model, get_optional_arguments_for_process_data, get_data_and_last_trade_datetime

from ficc.utils.auxiliary_functions import function_timer, get_ys_trade_history_features, get_dp_trade_history_features


MODEL = 'yield_spread_with_similar_trades'

TESTING = True
if TESTING: automated_training.automated_training_auxiliary_functions.NUM_EPOCHS = 5


@function_timer
def get_processed_data_pickle_file(model: str = MODEL) -> pd.DataFrame:
    file_name = MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME[model]
    data, most_recent_trade_datetime, _ = get_data_and_last_trade_datetime(BUCKET_NAME, file_name)
    print(f'Loaded data from gs://{BUCKET_NAME}/{file_name}. Most recent trade datetime: {most_recent_trade_datetime}')
    return data


def get_num_features_for_each_trade_in_history(model: str = MODEL) -> int:
    optional_arguments = get_optional_arguments_for_process_data(model)
    use_treasury_spread = optional_arguments.get('use_treasury_spread', False)    # from `automated_training_auxiliary_functions.py::update_data(...)`
    trade_history_features = get_ys_trade_history_features(use_treasury_spread) if 'yield_spread' in model else get_dp_trade_history_features()    # from `automated_training_auxiliary_functions.py::get_new_data(...)`
    return len(trade_history_features)    # from `automated_training_auxiliary_functions.py::get_new_data(...)`


def train_model_from_data_file(data_file_path: str, num_days: int, results_file_path: str):
    data = None    # TODO: load data from `file_path`
    # TODO: create `for` loop that iterates through `last_trade_date` options, and also truncates the DataFrame from the end
    most_recent_dates = np.sort(data['trade_date'].unique())[::-1]
    most_recent_dates = most_recent_dates[:num_days + 1]    # restrict to `num_days` most recent dates
    for day_idx in range(num_days):
        date_for_test_set, most_recent_date_for_training_set = most_recent_dates[day_idx], most_recent_dates[day_idx + 1]
        data = data[data['trade_date'] <= date_for_test_set]    # iteratively remove the last date from `data`
        model, _, _, _, _, mae, mae_df_list, _ = train_model(data, most_recent_date_for_training_set, MODEL, get_num_features_for_each_trade_in_history())
        

if __name__ == '__main__':
    os.makedirs('files', exist_ok=True)    # `os.makedirs(...)` creates directories along with any missing parent directories; `exist_ok=True` parameter ensures that no error is raised if the directory already exists
    data = get_processed_data_pickle_file(MODEL)
    train_model_from_data_file(data, 1, 'files/output.txt')
