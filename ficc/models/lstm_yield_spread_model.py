import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from ficc.models.registration import register_model


def yield_spread_model_v1(
    hp,
    SEQUENCE_LENGTH,
    NUM_FEATURES,
    NON_CAT_FEATURES,
    BINARY,
    CATEGORICAL_FEATURES,
    fmax,
    learning_rate,
    noncat_binary_normalizer,
    trade_history_normalizer,
    optimizer=keras.optimizers.Adam
):
    inputs = []
    layer = []

    ############## INPUT BLOCK ###################
    trade_history_input = layers.Input(name="trade_history_input", 
                                       shape=(SEQUENCE_LENGTH,NUM_FEATURES), 
                                       dtype = tf.float32) 

    inputs.append(trade_history_input)

    inputs.append(layers.Input(
        name="NON_CAT_AND_BINARY_FEATURES",
        shape=(len(NON_CAT_FEATURES + BINARY),)
    ))

    layer.append(noncat_binary_normalizer(inputs[1]) if noncat_binary_normalizer is not None else inputs[1])
    ####################################################


    ############## TRADE HISTORY MODEL #################

    # Adding the time2vec encoding to the input to transformer
    lstm_layer = layers.LSTM(hp.Int("lstm_layer_1_units", min_value=100, max_value=900, step=50, default=460), 
                             activation='tanh',
                             input_shape=(SEQUENCE_LENGTH,NUM_FEATURES),
                             return_sequences = True,
                             name='LSTM')

    lstm_layer_2 = layers.LSTM(hp.Int("lstm_layer_2_units", min_value=100, max_value=900, step=50, default=460), 
                               activation='tanh',
                               input_shape=(SEQUENCE_LENGTH,50),
                               return_sequences = False,
                               name='LSTM_2')

    features = lstm_layer(trade_history_normalizer(inputs[0]) if trade_history_normalizer is not None else inputs[0])
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(hp.Choice("dropout_1", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(features)

    features = lstm_layer_2(features)
    features = layers.Dropout(hp.Choice("dropout_2", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(features)
    features = layers.BatchNormalization()(features)


    trade_history_output = layers.Dense(hp.Int("trade_history_output_layer", min_value=100, max_value=900, step=50, default=460), 
                                        activation='relu')(features)

    ####################################################

    ############## REFERENCE DATA MODEL ################
    
    for f in CATEGORICAL_FEATURES:
        fin = layers.Input(shape=(1,), name = f)
        inputs.append(fin)
        embedded = layers.Flatten(name = f + "_flat")( layers.Embedding(input_dim = fmax[f]+1,
                                                                        output_dim = hp.Int("embedding_dim", min_value=1, max_value=100, step=5, default=10),
                                                                        input_length= 1,
                                                                        name = f + "_embed")(fin))
        layer.append(embedded)

    reference_hidden = layers.Dense(hp.Int("reference_hidden_1_units", min_value=100, max_value=900, step=50, default=260), 
                                    activation='relu',
                                    name='reference_hidden_1')(layers.concatenate(layer))
    reference_hidden = layers.BatchNormalization()(reference_hidden)
    reference_hidden = layers.Dropout(hp.Choice("dropout_3", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(reference_hidden)
    
    reference_hidden2 = layers.Dense(hp.Int("reference_hidden_2_units", min_value=100, max_value=900, step=50, default=10), 
                                     activation='relu',
                                     name='reference_hidden_2')(reference_hidden)
    reference_hidden2 = layers.BatchNormalization()(reference_hidden2)
    reference_hidden2 = layers.Dropout(hp.Choice("dropout_4", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(reference_hidden2)


    referenece_output = layers.Dense(hp.Int("reference_hidden_3_units", min_value=100, max_value=900, step=50, default=460), 
                                     activation='tanh',
                                     name='reference_hidden_3')(reference_hidden2)

    ####################################################


    feed_forward_input = layers.concatenate([referenece_output, trade_history_output])

    hidden = layers.Dense(hp.Int("output_block_1_units", min_value=100, max_value=900, step=50, default=250), 
                          activation='relu')(feed_forward_input)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Dropout(hp.Choice("dropout_5", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(hidden)
    
    hidden2 = layers.Dense(hp.Int("output_block_2_units", min_value=100, max_value=900, step=50, default=600), 
                           activation='tanh')(hidden)
    hidden2 = layers.BatchNormalization()(hidden2)
    hidden2 = layers.Dropout(hp.Choice("dropout_6", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(hidden2)

    final = layers.Dense(1)(hidden2)

    model = keras.Model(inputs=inputs, outputs=final)

    model.compile(optimizer=optimizer(learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5], default=learning_rate)),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.MeanAbsoluteError()])

    return model

register_model("lstm_yield_spread_model", 1, yield_spread_model_v1,
               "Initial LSTM-based yield spread model")

def flat_yield_spread_model_v1(
    hp,
    SEQUENCE_LENGTH,
    NUM_FEATURES,
    NON_CAT_FEATURES,
    BINARY,
    CATEGORICAL_FEATURES,
    fmax,
    learning_rate,
    noncat_binary_normalizer,
    trade_history_normalizer,
    optimizer=keras.optimizers.Adam
):
    inputs = []
    layer = []

    ############## INPUT BLOCK ###################
    inputs.append(None) # Not using the trade history

    inputs.append(layers.Input(
        name="NON_CAT_AND_BINARY_FEATURES",
        shape=(len(NON_CAT_FEATURES + BINARY),)
    ))

    layer.append(noncat_binary_normalizer(inputs[1]) if noncat_binary_normalizer is not None else inputs[1])
    ####################################################



    ############## REFERENCE DATA MODEL ################
    
    for f in CATEGORICAL_FEATURES:
        fin = layers.Input(shape=(1,), name = f)
        inputs.append(fin)
        embedded = layers.Flatten(name = f + "_flat")( layers.Embedding(input_dim = fmax[f]+1,
                                                                        output_dim = hp.Int("embedding_dim", min_value=1, max_value=100, step=5, default=10),
                                                                        input_length= 1,
                                                                        name = f + "_embed")(fin))
        layer.append(embedded)

    reference_hidden = layers.Dense(hp.Int("reference_hidden_1_units", min_value=100, max_value=900, step=50, default=260), 
                                    activation='relu',
                                    name='reference_hidden_1')(layers.concatenate(layer))
    reference_hidden = layers.BatchNormalization()(reference_hidden)
    reference_hidden = layers.Dropout(hp.Choice("dropout_3", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(reference_hidden)
    
    reference_hidden2 = layers.Dense(hp.Int("reference_hidden_2_units", min_value=100, max_value=900, step=50, default=10), 
                                     activation='relu',
                                     name='reference_hidden_2')(reference_hidden)
    reference_hidden2 = layers.BatchNormalization()(reference_hidden2)
    reference_hidden2 = layers.Dropout(hp.Choice("dropout_4", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(reference_hidden2)


    reference_output = layers.Dense(hp.Int("reference_hidden_3_units", min_value=100, max_value=900, step=50, default=460), 
                                     activation='tanh',
                                     name='reference_hidden_3')(reference_hidden2)

    ####################################################


    feed_forward_input = reference_output

    hidden = layers.Dense(hp.Int("output_block_1_units", min_value=100, max_value=900, step=50, default=250), 
                          activation='relu')(feed_forward_input)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Dropout(hp.Choice("dropout_5", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(hidden)
    
    hidden2 = layers.Dense(hp.Int("output_block_2_units", min_value=100, max_value=900, step=50, default=600), 
                           activation='tanh')(hidden)
    hidden2 = layers.BatchNormalization()(hidden2)
    hidden2 = layers.Dropout(hp.Choice("dropout_6", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(hidden2)

    final = layers.Dense(1)(hidden2)

    model = keras.Model(inputs=inputs, outputs=final)

    model.compile(optimizer=optimizer(learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5], default=learning_rate)),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.MeanAbsoluteError()])

    return model

register_model("flat_yield_spread_model", 1, flat_yield_spread_model_v1,
               "Initial yield spread model, without the trade history")



def deep_yield_spread_model_v1(
    hp,
    SEQUENCE_LENGTH,
    NUM_FEATURES,
    NON_CAT_FEATURES,
    BINARY,
    CATEGORICAL_FEATURES,
    fmax,
    learning_rate,
    noncat_binary_normalizer,
    trade_history_normalizer,
    depth=4,
    optimizer=keras.optimizers.Adam
):
    inputs = []
    layer = []

    ############## INPUT BLOCK ###################
    trade_history_input = layers.Input(name="trade_history_input", 
                                       shape=(SEQUENCE_LENGTH,NUM_FEATURES), 
                                       dtype = tf.float32) 

    inputs.append(trade_history_input)

    inputs.append(layers.Input(
        name="NON_CAT_AND_BINARY_FEATURES",
        shape=(len(NON_CAT_FEATURES + BINARY),)
    ))

    layer.append(noncat_binary_normalizer(inputs[1]) if noncat_binary_normalizer is not None else inputs[1])
    ####################################################


    ############## TRADE HISTORY MODEL #################

    # Adding the time2vec encoding to the input to transformer
    lstm_layer = layers.LSTM(hp.Int("lstm_layer_1_units", min_value=100, max_value=900, step=50, default=460), 
                             activation='tanh',
                             input_shape=(SEQUENCE_LENGTH,NUM_FEATURES),
                             return_sequences = True,
                             name='LSTM')

    lstm_layer_2 = layers.LSTM(hp.Int("lstm_layer_2_units", min_value=100, max_value=900, step=50, default=460), 
                               activation='tanh',
                               input_shape=(SEQUENCE_LENGTH,50),
                               return_sequences = False if depth == 1 else True,
                               name='LSTM_2')

    features = lstm_layer(trade_history_normalizer(inputs[0]) if trade_history_normalizer is not None else inputs[0])
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(hp.Choice("dropout_1", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(features)

    features = lstm_layer_2(features)
    features = layers.Dropout(hp.Choice("dropout_2", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(features)
    features = layers.BatchNormalization()(features)

    x = features
    for block in range(1, depth):
        block_lstm_layer_1 = layers.LSTM(hp.Int(f"block{block}_lstm_layer_1_units", min_value=100, max_value=900, step=50, default=460), 
                                activation='tanh',
                                input_shape=(SEQUENCE_LENGTH,NUM_FEATURES),
                                return_sequences = True,
                                name=f'LSTM_1_{block}')
        block_lstm_layer_2 = layers.LSTM(hp.Int(f"block{block}_lstm_layer_2_units", min_value=100, max_value=900, step=50, default=460), 
                                activation='tanh',
                                input_shape=(SEQUENCE_LENGTH,50),
                                return_sequences = False if block == (depth - 1) else True,
                                name=f'LSTM_2_{block}')

        features = block_lstm_layer_1(features)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(hp.Choice(f"dropout_1_{block}", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(features)

        features = block_lstm_layer_2(features)
        features = layers.Dropout(hp.Choice(f"dropout_2_{block}", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(features)
        features = layers.BatchNormalization()(features)

        if block != (depth - 1):
            features = features + x


    trade_history_output = layers.Dense(hp.Int("trade_history_output_layer", min_value=100, max_value=900, step=50, default=460), 
                                        activation='relu')(features)

    ####################################################

    ############## REFERENCE DATA MODEL ################
    
    for f in CATEGORICAL_FEATURES:
        fin = layers.Input(shape=(1,), name = f)
        inputs.append(fin)
        embedded = layers.Flatten(name = f + "_flat")( layers.Embedding(input_dim = fmax[f]+1,
                                                                        output_dim = hp.Int("embedding_dim", min_value=1, max_value=100, step=5, default=10),
                                                                        input_length= 1,
                                                                        name = f + "_embed")(fin))
        layer.append(embedded)

    reference_hidden = layers.Dense(hp.Int("reference_hidden_1_units", min_value=100, max_value=900, step=50, default=260), 
                                    activation='relu',
                                    name='reference_hidden_1')(layers.concatenate(layer))
    reference_hidden = layers.BatchNormalization()(reference_hidden)
    reference_hidden = layers.Dropout(hp.Choice("dropout_3", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(reference_hidden)
    
    reference_hidden2 = layers.Dense(hp.Int("reference_hidden_2_units", min_value=100, max_value=900, step=50, default=10), 
                                     activation='relu',
                                     name='reference_hidden_2')(reference_hidden)
    reference_hidden2 = layers.BatchNormalization()(reference_hidden2)
    reference_hidden2 = layers.Dropout(hp.Choice("dropout_4", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(reference_hidden2)


    referenece_output = layers.Dense(hp.Int("reference_hidden_3_units", min_value=100, max_value=900, step=50, default=460), 
                                     activation='tanh',
                                     name='reference_hidden_3')(reference_hidden2)

    ####################################################


    feed_forward_input = layers.concatenate([referenece_output, trade_history_output])

    hidden = layers.Dense(hp.Int("output_block_1_units", min_value=100, max_value=900, step=50, default=250), 
                          activation='relu')(feed_forward_input)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Dropout(hp.Choice("dropout_5", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(hidden)
    
    hidden2 = layers.Dense(hp.Int("output_block_2_units", min_value=100, max_value=900, step=50, default=600), 
                           activation='tanh')(hidden)
    hidden2 = layers.BatchNormalization()(hidden2)
    hidden2 = layers.Dropout(hp.Choice("dropout_6", values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(hidden2)

    final = layers.Dense(1)(hidden2)

    model = keras.Model(inputs=inputs, outputs=final)

    model.compile(optimizer=optimizer(learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5], default=learning_rate)),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.MeanAbsoluteError()])

    return model

register_model("deep_yield_spread_model", 1, deep_yield_spread_model_v1,
               "Initial LSTM-based yield spread model, but with deeper LSTM component")
