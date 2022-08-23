'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-15 14:05:16
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-08-22 16:01:36
 # @ Description: The file contains variables which are used across 
                  the data processing module
 '''

def init(): 
    global FICC_ERROR 
    FICC_ERROR = ""
    
    global nelson_params
    nelson_params = None
    
    global scalar_params
    scalar_params = None

    global YIELD_CURVE_TO_USE
    YIELD_CURVE_TO_USE = "FICC"

    global mmd_ycl
    mmd_ycl = None

    global shape_parameter
    shape_parameter = None