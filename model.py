'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2023-02-01 10:38:48
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2023-02-01 14:56:46
 # @ Description:
 '''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers.experimental.preprocessing import Normalization

BATCH_SIZE = 1000
DROPOUT = 0.01
tf.keras.utils.set_random_seed(10)

class CustomAttention(tf.keras.layers.Layer):
    def __init__(self, depth):
        super(CustomAttention, self).__init__()
        self.depth = depth
        self.wq = layers.Dense(depth, name='weights_query') 
        self.wk = layers.Dense(depth, name='weights_key')
        self.wv = layers.Dense(depth, name='weights_value')

    def scaled_dot_product_attention(self, q, v, k):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        scaling = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(scaling)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=1) 
        output = tf.matmul(attention_weights, v)
        
        return output
    
    def call(self, q, v, k):
        q = self.wq(q)
        v = self.wv(v)
        k = self.wk(k)
        output = self.scaled_dot_product_attention(q, v, k)
        return output    


def model_definition(trade_history_normalizer,
                     noncat_binary_normalizer,
                     SEQUENCE_LENGTH, 
                     NUM_FEATURES, 
                     CATEGORICAL_FEATURES, 
                     NON_CAT_FEATURES, 
                     BINARY,
                     fmax):
    inputs = []
    layer = []

    ############## INPUT BLOCK ###################
    trade_history_input = layers.Input(name="trade_history_input", 
                                    shape=(SEQUENCE_LENGTH,NUM_FEATURES), 
                                    dtype = tf.float32) 

    target_attention_input = layers.Input(name="target_attention_input", 
                                    shape=(SEQUENCE_LENGTH, 3), 
                                    dtype = tf.float32) 


    inputs.append(trade_history_input)
    inputs.append(target_attention_input)

    inputs.append(layers.Input(
        name="NON_CAT_AND_BINARY_FEATURES",
        shape=(len(NON_CAT_FEATURES + BINARY),)
    ))


    layer.append(noncat_binary_normalizer(inputs[2]))
    ####################################################


    ############## TRADE HISTORY MODEL #################

    lstm_layer = layers.LSTM(50, 
                            activation='tanh',
                            input_shape=(SEQUENCE_LENGTH,NUM_FEATURES),
                            return_sequences = True,
                            name='LSTM')

    lstm_attention_layer = CustomAttention(50)

    lstm_layer_2 = layers.LSTM(100, 
                            activation='tanh',
                            input_shape=(SEQUENCE_LENGTH,50),
                            return_sequences = False,
                            name='LSTM_2')


    features = lstm_layer(trade_history_normalizer(inputs[0]))
    features = lstm_attention_layer(features, features, inputs[1])
    features = layers.BatchNormalization()(features)
    # features = layers.Dropout(DROPOUT)(features)

    features = lstm_layer_2(features)
    features = layers.BatchNormalization()(features)
    # features = layers.Dropout(DROPOUT)(features)

    trade_history_output = layers.Dense(100, 
                                        activation='relu')(features)

    ####################################################

    ############## REFERENCE DATA MODEL ################
    global encoders
    for f in CATEGORICAL_FEATURES:
        fin = layers.Input(shape=(1,), name = f)
        inputs.append(fin)
        embedded = layers.Flatten(name = f + "_flat")( layers.Embedding(input_dim = fmax[f]+1,
                                                                        output_dim = max(30,int(np.sqrt(fmax[f]))),
                                                                        input_length= 1,
                                                                        name = f + "_embed")(fin))
        layer.append(embedded)

        
    reference_hidden = layers.Dense(400,
                                    activation='relu',
                                    name='reference_hidden_1')(layers.concatenate(layer, axis=-1))

    reference_hidden = layers.BatchNormalization()(reference_hidden)
    reference_hidden = layers.Dropout(DROPOUT)(reference_hidden)

    reference_hidden2 = layers.Dense(200,activation='relu',name='reference_hidden_2')(reference_hidden)
    reference_hidden2 = layers.BatchNormalization()(reference_hidden2)
    reference_hidden2 = layers.Dropout(DROPOUT)(reference_hidden2)

    reference_output = layers.Dense(100,activation='tanh',name='reference_hidden_3')(reference_hidden2)

    ####################################################

    feed_forward_input = layers.concatenate([reference_output, trade_history_output])

    hidden = layers.Dense(300,activation='relu')(feed_forward_input)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Dropout(DROPOUT)(hidden)

    hidden2 = layers.Dense(100,activation='tanh')(hidden)
    hidden2 = layers.BatchNormalization()(hidden2)
    hidden2 = layers.Dropout(DROPOUT)(hidden2)

    final = layers.Dense(1)(hidden2)

    model = keras.Model(inputs=inputs, outputs=final)
    return model

def yield_spread_model(x_train, 
                       SEQUENCE_LENGTH, 
                       NUM_FEATURES, 
                       PREDICTORS, 
                       CATEGORICAL_FEATURES, 
                       NON_CAT_FEATURES, 
                       BINARY,
                       fmax):
    
    trade_history_normalizer = Normalization(name='Trade_history_normalizer')
    trade_history_normalizer.adapt(x_train[0],batch_size=BATCH_SIZE)

    noncat_binary_normalizer = Normalization(name='Numerical_binary_normalizer')
    noncat_binary_normalizer.adapt(x_train[2], batch_size = BATCH_SIZE)

    model = model_definition(trade_history_normalizer,
                             noncat_binary_normalizer, 
                             SEQUENCE_LENGTH, 
                             NUM_FEATURES, 
                             CATEGORICAL_FEATURES, 
                             NON_CAT_FEATURES, 
                             BINARY,
                             fmax)
    
    return model