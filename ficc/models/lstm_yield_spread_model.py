import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from . import register_model


def yield_spread_model_v1(
    hp,
    SEQUENCE_LENGTH,
    NUM_FEATURES,
    NON_CAT_FEATURES,
    BINARY,
    CATEGORICAL_FEATURES,
    fmax,
    learning_rate,
    optimizer=keras.optimizers.Adam
):
    inputs = []
    layer = []

    ############## INPUT BLOCK ###################
    trade_history_input = layers.Input(name="trade_history_input",
                                       shape=(SEQUENCE_LENGTH, NUM_FEATURES),
                                       dtype=tf.float32)

    inputs.append(trade_history_input)

    for i in NON_CAT_FEATURES + BINARY:
        inputs.append(layers.Input(shape=(1,), name=f"{i}"))

    for i in inputs[1:]:
        layer.append(layers.Normalization()(i))
    ####################################################

    ############## TRADE HISTORY MODEL #################

    # Adding the time2vec encoding to the input to transformer
    lstm_layer = layers.LSTM(hp.Int("lstm_layer_1_units", min_value=30, max_value=500, step=5, default=460),
                             activation='tanh',
                             input_shape=(SEQUENCE_LENGTH, NUM_FEATURES),
                             return_sequences=True,
                             name='LSTM')

    lstm_layer_2 = layers.LSTM(hp.Int("lstm_layer_2_units", min_value=90, max_value=500, step=5, default=460),
                               activation='tanh',
                               input_shape=(SEQUENCE_LENGTH, 50),
                               return_sequences=False,
                               name='LSTM_2')

    lstm_dropout = hp.Float("lstm_dropout", min_value=0.0,
                            max_value=0.2, step=0.05, default=0.0)

    features = lstm_layer(inputs[0])
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(lstm_dropout)(features)

    features = lstm_layer_2(features)
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(lstm_dropout)(features)

    trade_history_output = layers.Dense(hp.Int("trade_history_output_layer", min_value=50, max_value=500, step=10, default=460),
                                        activation='relu')(features)

    ####################################################

    ############## REFERENCE DATA MODEL ################
    for f in CATEGORICAL_FEATURES:
        fin = layers.Input(shape=(1,), name=f)
        inputs.append(fin)
        embedded = layers.Flatten(name=f + "_flat")(layers.Embedding(input_dim=fmax[f]+1,
                                                                     output_dim=hp.Int(
                                                                         "embedding_dim", min_value=10, max_value=50, step=5, default=10),
                                                                     input_length=1,
                                                                     name=f + "_embed")(fin))
        layer.append(embedded)

    ref_dropout = hp.Float("ref_dropout", min_value=0.0,
                           max_value=0.2, step=0.05, default=0.0)

    reference_hidden = layers.Dense(hp.Int("reference_hidden_1_units", min_value=250, max_value=450, step=10, default=260),
                                    activation='relu',
                                    name='reference_hidden_1')(layers.concatenate(layer))

    reference_hidden = layers.BatchNormalization()(reference_hidden)
    reference_hidden = layers.Dropout(ref_dropout)(reference_hidden)

    reference_hidden2 = layers.Dense(hp.Int("reference_hidden_2_units", min_value=10, max_value=350, step=10, default=10),
                                     activation='relu',
                                     name='reference_hidden_2')(reference_hidden)

    reference_hidden2 = layers.BatchNormalization()(reference_hidden2)
    reference_hidden2 = layers.Dropout(ref_dropout)(reference_hidden2)

    referenece_output = layers.Dense(hp.Int("reference_hidden_3_units", min_value=50, max_value=500, step=10, default=460),
                                     activation='tanh',
                                     name='reference_hidden_3')(reference_hidden2)

    ####################################################

    feed_forward_input = layers.concatenate(
        [referenece_output, trade_history_output])

    final_dropout = hp.Float("final_dropout", min_value=0.0,
                             max_value=0.2, step=0.05, default=0.0)

    hidden = layers.Dense(hp.Int("output_block_1_units", min_value=250, max_value=350, step=10, default=250),
                          activation='relu')(feed_forward_input)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Dropout(final_dropout)(hidden)

    hidden2 = layers.Dense(hp.Int("output_block_2_units", min_value=100, max_value=650, step=10, default=600),
                           activation='tanh')(hidden)
    hidden2 = layers.BatchNormalization()(hidden2)
    hidden2 = layers.Dropout(final_dropout)(hidden2)

    final = layers.Dense(1)(hidden2)

    model = keras.Model(inputs=inputs, outputs=final)

    model.compile(optimizer=optimizer(learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5], default=learning_rate)),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.MeanAbsoluteError()])

    return model


register_model("lstm_yield_spread_model", 1, yield_spread_model_v1,
               "Initial LSTM-based yield spread model")
