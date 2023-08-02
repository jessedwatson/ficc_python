import os
import gcsfs
import shutil
import numpy as np
import pandas as pd
from tensorflow import keras
from google.cloud import bigquery
from google.cloud import storage
from sklearn import preprocessing
from pickle5 import pickle
from ficc.data.process_data import process_data
from ficc.utils.auxiliary_functions import sqltodf
from ficc.utils.diff_in_days import diff_in_days_two_dates
from ficc.utils.auxiliary_variables import PREDICTORS, NON_CAT_FEATURES, BINARY, CATEGORICAL_FEATURES, IDENTIFIERS, NUM_OF_DAYS_IN_YEAR
from ficc.utils.nelson_siegel_model import yield_curve_level
from ficc.utils.gcp_storage_functions import upload_data
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


def target_trade_processing_for_attention(row):
    trade_mapping = {'D':[0,0], 'S':[0,1], 'P':[1,0]}
    target_trade_features = []
    target_trade_features.append(row['quantity'])
    target_trade_features = target_trade_features + trade_mapping[row['trade_type']]
    return np.tile(target_trade_features, (1,1))


df = pd.read_pickle('processed_data.pkl')
print('File read')
print('adding target_attention features')
df['target_attention_features'] = df.parallel_apply(target_trade_processing_for_attention, axis = 1)
print('done')
print(df.target_attention_features.shape)
print('saving file')
df.to_pickle('processed_data.pkl')
