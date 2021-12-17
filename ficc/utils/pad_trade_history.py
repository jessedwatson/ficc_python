'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-16 14:51:09
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2021-12-16 14:56:20
 # @ Description:The pad_trade_history function pads the trade historie with zeros, to make their
 #  length equal to the sequence length. The function pads the end of trade history and creates 
 #  a single sequence. The paddings are added after the most recent trades.
 # 
 #  If the length of the trade history is equal to the sequence length the function returns
 #  the list as is. As an initial step, we are only padding trades that have at least half the 
 #  sequence length number of trades in the sequence. We will expand the model to include comps for 
 #  CUSIPs which do not have sufficient history
 '''

import numpy as np

def pad_trade_history(x, SEQUENCE_LENGTH, NUM_FEATURES):
    
    if len(x) < SEQUENCE_LENGTH and len(x) > SEQUENCE_LENGTH//2: 
        temp = x.tolist()
        temp = temp + [[0]*NUM_FEATURES]*(SEQUENCE_LENGTH - len(x))
        try:
            return np.stack(temp)
        except Exception as e:
            print("Failed to pad trade history for")
            for i in temp:
                print(i)
    
    #returning none for data less than sequence length
    elif len(x) < SEQUENCE_LENGTH:
        return None

    else:
        return x
