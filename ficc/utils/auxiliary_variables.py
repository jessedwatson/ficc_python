'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-17 12:07:51
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2021-12-17 12:45:24
 # @ Description:
 '''

COUPON_FREQUENCY_DICT = {0:"Unknown",
                        1:"Semiannually",
                        2:"Monthly",
                        3:"Annually",
                        4:"Weekly",
                        5:"Quarterly",
                        6:"Every 2 years",
                        7:"Every 3 years",
                        8:"Every 4 years",
                        9:"Every 5 years",
                        10:"Every 7 years",
                        11:"Every 8 years",
                        12:"Biweekly",
                        13:"Changeable",
                        14:"Daily",
                        15:"Term mode",
                        16:"Interest at maturity",
                        17:"Bimonthly",
                        18:"Every 13 weeks",
                        19:"Irregular",
                        20:"Every 28 days",
                        21:"Every 35 days",
                        22:"Every 26 weeks",
                        23:"Not Applicable",
                        24:"Tied to prime",
                        25:"One time",
                        26:"Every 10 years",
                        27:"Frequency to be determined",
                        28:"Mandatory put",
                        29:"Every 52 weeks",
                        30:"When interest adjusts-commercial paper",
                        31:"Zero coupon",
                        32:"Certain years only",
                        33:"Under certain circumstances",
                        34:"Every 15 years",
                        35:"Custom",
                        36:"Single Interest Payment"
                        }


IDENTIFIERS = ['rtrs_control_number', 'cusip']


BINARY = ['callable',
          'sinking',
          'zerocoupon',
          'is_non_transaction_based_compensation',
          'is_general_obligation',
          'callable_at_cav',           
          'extraordinary_make_whole_call', 
          'make_whole_call',
          'has_unexpired_lines_of_credit',
          'escrow_exists',
          ]

CATEGORICAL_FEATURES = ['rating',
                        'incorporated_state_code',
                        'trade_type',
                        'transaction_type',
                        'maturity_description_code',
                        'purpose_class']

NON_CAT_FEATURES = ['quantity',
                    'days_to_maturity',
                    'days_to_call',
                    'coupon',
                    'issue_amount',
                    'last_seconds_ago',
                    'last_yield_spread',
                    'days_to_settle',
                    'days_to_par',
                    'maturity_amount',
                    'issue_price', 
                    'orig_principal_amount',
                    'max_amount_outstanding']

TRADE_HISTORY = ['trade_history']
TARGET = ['yield_spread']

PREDICTORS = BINARY + CATEGORICAL_FEATURES + NON_CAT_FEATURES + TARGET + TRADE_HISTORY