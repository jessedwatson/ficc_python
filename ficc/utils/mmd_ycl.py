'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2022-02-03 19:38:22
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-02-09 13:26:38
 # @ Description:
 '''

import numpy as np
import ficc.utils.globals as globals

def mmd_ycl(target_date, maturity):
    maturity = int(np.round(maturity))
    if maturity == 0:
        maturity = 1
    mmd_column_name = f"Year_{maturity}"
    return globals.mmd_ycl.loc[target_date][mmd_column_name].values[0]
    