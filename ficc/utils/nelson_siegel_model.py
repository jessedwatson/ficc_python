'''
 # @ Author: Issac Lim
 # @ Create date: 2021-08-23
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-02-13
 # @ Description: This is an implementation of the Nelson Seigel intereset rate 
 # model to predic the yield curve. 
 # @ Modification: Nelson-Seigel coefficeints are used from a dataframe
 # instead of grabbing them from memory store
 '''
import copy
import numpy as np
from pandas.tseries.offsets import BDay
from functools import wraps
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

PROJECT_ID = 'eng-reactor-287421'


def cache(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        # creating a key using maturity and target date
        cache_key = str(args[0]) + str(args[1])
        if cache_key in wrapper.cache:
            output = wrapper.cache[cache_key]
        else:
            output = function(*args)
            wrapper.cache[cache_key] = output
        return output

    wrapper.cache = dict()
    return wrapper


def decay_transformation(t: np.array, L: float):
    '''This function takes a numpy array of maturities (or a single float) and a shape parameter. It returns the exponential function
    calculated from those values. This is the first feature of the nelson-siegel model.'''
    return L * (1 - np.exp(-t / L)) / t


def laguerre_transformation(t: np.array, L: float):
    '''This function takes a numpy array of maturities (or a single float) and a shape parameter. It returns the laguerre function
    calculated from those values. This is the second feature of the nelson-siegel model.'''
    return (L * (1 - np.exp(-t / L)) / t) - np.exp(-t / L)


def load_model_parameters(target_date, nelson_params, scalar_params, shape_parameter):
    '''This function grabs the nelson siegel and standard scalar coefficient from the dataframes.'''
    target_date = (target_date - BDay(1)).date()
    target_date_shape = copy.deepcopy(target_date)

    while target_date not in nelson_params.keys():
        target_date = (target_date - BDay(1)).date()

    nelson_coeff = nelson_params[target_date].values()
    scalar_coeff = scalar_params[target_date].values()

    while target_date_shape not in shape_parameter.keys():
        target_date_shape = (target_date_shape - BDay(1)).date()
    shape_param = shape_parameter[target_date_shape]['L']

    return nelson_coeff, scalar_coeff, shape_param


def get_scaled_features(t: np.array, exponential_mean: float, exponential_std: float, laguerre_mean: float, laguerre_std: float, shape_paramter: float):
    '''This function takes as input the parameters loaded from the scaler parameter table in bigquery on a given day, alongside an array (or a
    single float) value to be scaled as input to make predictions. It then manually recreate the transformations from the sklearn
    StandardScaler used to scale data in training by first creating the exponential and laguerre functions then scaling them.'''
    X1 = (decay_transformation(t, shape_paramter) - exponential_mean) / exponential_std
    X2 = (laguerre_transformation(t, shape_paramter) - laguerre_mean) / laguerre_std
    return X1, X2


def predict_ytw(maturity: np.array,
                const: float,
                exponential: float,
                laguerre: float,
                exponential_mean: float,
                exponential_std: float,
                laguerre_mean: float,
                laguerre_std: float,
                shape_parameter: float):
    '''This is a wrapper function that takes the prediction inputs, the scaler parameters and the model parameters from a given day. It then
    scales the input using the get_scaled_features function to obtain the model inputs, and predicts the yield-to-worst implied by the
    nelson-siegel model on that day. Because the nelson-siegel model is linear, we can do a simple calculation.'''
    X1, X2 = get_scaled_features(maturity, exponential_mean, exponential_std, laguerre_mean, laguerre_std, shape_parameter)
    return const + exponential * X1 + laguerre * X2


@cache
def yield_curve_level(maturity: float, target_date, nelson_params, scalar_params, shape_parameter):
    '''This is the main function takes as input a json containing two arguments: the maturity we want the yield-to-worst for and the target
    ate from which we want the yield curve used in the ytw calculations to be from. There are several conditional statements to deal with
    different types of exceptions.

    The cloud function returns a json containing the status (Failed or Success), the error message (if any)
    and the result (nan if calculation was unsuccessful).'''
    # if a `target_date` is provided but it is in an invalid format, then the correct values from the model and scaler parameters cannot be retrieved, and an error is also returned
    nelson_siegel_daily_coef, scaler_daily_parameters, shape_param = load_model_parameters(target_date, nelson_params, scalar_params, shape_parameter)
    const, exponential, laguerre = nelson_siegel_daily_coef
    exponential_mean, exponential_std, laguerre_mean, laguerre_std = scaler_daily_parameters

    # if the function gets this far without raising an error, then the values are correct and so, a prediction is returned
    prediction = predict_ytw(maturity, const, exponential, laguerre, exponential_mean, exponential_std, laguerre_mean, laguerre_std, shape_param)
    return prediction
