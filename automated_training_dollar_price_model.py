'''
 # @ Author: Ahmad Shayaan
 # @ Create date: 2023-01-23
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-03-28
 '''
import sys
from ficc.utils.auxiliary_functions import function_timer

from automated_training_auxiliary_functions import setup_gpus, \
                                                   train_save_evaluate_model


setup_gpus()


@function_timer
def main():
    current_date_passed_in = sys.argv[1] if len(sys.argv) == 2 else None
    train_save_evaluate_model('dollar_price', current_date=current_date_passed_in)


if __name__ == '__main__':
    main()
