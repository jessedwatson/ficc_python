'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2023-01-23
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-03-28
 '''
import sys
import numpy as np
import pandas as pd
from ficc.utils.auxiliary_functions import function_timer

from automated_training_auxiliary_functions import setup_gpus, \
                                                   train_save_evaluate_model


setup_gpus()


def apply_exclusions(data: pd.DataFrame, dataset_name: str = None):
    from_dataset_name = f' from {dataset_name}' if dataset_name is not None else ''
    data_before_exclusions = data[:]
    
    previous_size = len(data)
    data = data[(data.days_to_call == 0) | (data.days_to_call > np.log10(400))]
    current_size = len(data)
    if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having 0 < days_to_call <= 400')
    
    previous_size = current_size
    data = data[(data.days_to_refund == 0) | (data.days_to_refund > np.log10(400))]
    current_size = len(data)
    if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having 0 < days_to_refund <= 400')
    
    previous_size = current_size
    data = data[(data.days_to_maturity == 0) | (data.days_to_maturity > np.log10(400))]
    current_size = len(data)
    if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having 0 < days_to_maturity <= 400')
    
    previous_size = current_size
    data = data[data.days_to_maturity < np.log10(30000)]
    current_size = len(data)
    if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having days_to_maturity >= 30000')
    
    ## null last_calc_date exclusion was removed on 2024-02-19
    # previous_size = current_size
    # data = data[~data.last_calc_date.isna()]
    # current_size = len(data)
    # if previous_size != current_size: print(f'Removed {previous_size - current_size} trades{from_dataset_name} for having a null value for last_calc_date')

    return data, data_before_exclusions


@function_timer
def main():
    current_date_passed_in = sys.argv[1] if len(sys.argv) == 2 else None
    train_save_evaluate_model('yield_spread', apply_exclusions, current_date_passed_in)


if __name__ == '__main__':
    main()
