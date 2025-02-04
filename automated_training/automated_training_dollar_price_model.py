'''
 # @ Author: Mitas Ray
 # @ Create date: 2024-01-24
 # @ Modified by: Mitas Ray
 # @ Modified date: 2025-01-10
 '''
import os
import sys

from auxiliary_functions import setup_gpus, train_save_evaluate_model


ficc_package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))    # get the directory containing the 'ficc_python/' package
sys.path.append(ficc_package_dir)    # add the directory to sys.path


from ficc.utils.auxiliary_functions import function_timer


@function_timer
def main():
    current_date_passed_in = sys.argv[1] if len(sys.argv) == 2 else None
    train_save_evaluate_model('dollar_price', current_date=current_date_passed_in)


if __name__ == '__main__':
    setup_gpus()
    main()
