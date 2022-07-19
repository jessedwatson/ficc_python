'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2022-02-03 19:38:22
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-07-19 15:20:04
 # @ Description:
 '''

import numpy as np
import ficc.utils.globals as globals

def mmd_ycl(target_date, maturity):
    '''
    This function is used to return the MMD yield curve level
    for a particular maturity for the given date

    parameters
    target_date : datetime.date
    maturity: float
    '''
    maturity = int(np.round(maturity))
    
    if maturity == 0:
        maturity = 1
    
    elif maturity > 30:
        maturity = 30

    year_key = f"_{maturity}Y" 
    
    ycl =  globals.mmd_ycl[target_date][year_key]

    return ycl
    
    


