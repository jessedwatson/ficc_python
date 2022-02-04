'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2022-02-03 19:38:22
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-02-03 20:35:41
 # @ Description:
 '''

import ficc.utils.globals as globals

def get_mmd_level(target_date, maturity):
    if maturity < 5:
        return globals.mmd_ycl.loc[target_date]['MMD_5Y']
    else:
        return globals.mmd_ycl.loc[target_date]['MMD_10Y']
    