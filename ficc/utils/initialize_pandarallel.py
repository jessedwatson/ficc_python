'''
Author: Mitas Ray
Create Date: 2024-11-07
Last Editor: Mitas Ray
Last Edit Date: 2024-11-07
'''
import os


def initialize_pandarallel(num_cores_for_pandarallel: int = os.cpu_count() // 2):
    from pandarallel import pandarallel    # used to multi-thread df apply with `.parallel_apply(...)`
    
    try:
        pandarallel.initialize(progress_bar=False, nb_workers=num_cores_for_pandarallel)
        print(f'Initialized pandarallel with {num_cores_for_pandarallel} cores')
    except RuntimeError:
        print('pandarallel has already been initialized')
