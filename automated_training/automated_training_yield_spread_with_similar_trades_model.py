'''
 # @ Author: Mitas Ray
 # @ Create date: 2024-04-15
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-08-26
 '''
import sys
from ficc.utils.auxiliary_functions import function_timer

from automated_training_auxiliary_functions import setup_gpus, train_save_evaluate_model, apply_exclusions


@function_timer
def main():
    current_date_passed_in = sys.argv[1] if len(sys.argv) == 2 else None
    train_save_evaluate_model('yield_spread_with_similar_trades', apply_exclusions, current_date_passed_in)


if __name__ == '__main__':
    setup_gpus()
    main()
