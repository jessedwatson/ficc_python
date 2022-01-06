import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from ficc.models.registration import register_model


def gnn_yield_spread_model_v1(
    hp,
    SEQUENCE_LENGTH,
    NUM_FEATURES,
    NON_CAT_FEATURES,
    BINARY,
    CATEGORICAL_FEATURES,
    N,
    fmax,
    learning_rate,
    noncat_binary_normalizer,
    trade_history_normalizer,
    d_model=512,
    dff=2048,
    optimizer=keras.optimizers.Adam
):
    inputs = []
    layer = []

    ############## INPUT BLOCK ###################
    trade_history_input = layers.Input(name="trade_history_input",
                                       shape=(SEQUENCE_LENGTH, NUM_FEATURES),
                                       dtype=tf.float32)

    inputs.append(trade_history_input)

    inputs.append(layers.Input(
        name="NON_CAT_AND_BINARY_FEATURES",
        shape=(len(NON_CAT_FEATURES + BINARY),)
    ))

    layer.append(noncat_binary_normalizer(
        inputs[1]) if noncat_binary_normalizer is not None else inputs[1])
    ####################################################

    ############## TRADE HISTORY MODEL #################

    # Adding the time2vec encoding to the input to transformer
    lstm_layer = layers.LSTM(hp.Int("lstm_layer_1_units", min_value=100, max_value=900, step=50, default=460),
                             activation='tanh',
                             input_shape=(SEQUENCE_LENGTH, NUM_FEATURES),
                             return_sequences=True,
                             name='LSTM')

    lstm_layer_2 = layers.LSTM(hp.Int("lstm_layer_2_units", min_value=100, max_value=900, step=50, default=460),
                               activation='tanh',
                               input_shape=(SEQUENCE_LENGTH, 50),
                               return_sequences=False,
                               name='LSTM_2')

    features = lstm_layer(trade_history_normalizer(
        inputs[0]) if trade_history_normalizer is not None else inputs[0])
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(hp.Choice("dropout_1", values=[
                              0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(features)

    features = lstm_layer_2(features)
    features = layers.Dropout(hp.Choice("dropout_2", values=[
                              0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(features)
    features = layers.BatchNormalization()(features)

    trade_history_output = layers.Dense(hp.Int("trade_history_output_layer", min_value=100, max_value=900, step=50, default=460),
                                        activation='relu')(features)

    ####################################################

    ############## REFERENCE DATA MODEL ################

    for f in CATEGORICAL_FEATURES:
        fin = layers.Input(shape=(1,), name=f)
        inputs.append(fin)
        embedded = layers.Flatten(name=f + "_flat")(layers.Embedding(input_dim=fmax[f]+1,
                                                                     output_dim=hp.Int(
                                                                         "embedding_dim", min_value=1, max_value=100, step=5, default=10),
                                                                     input_length=1,
                                                                     name=f + "_embed")(fin))
        layer.append(embedded)

    reference_hidden = layers.Dense(hp.Int("reference_hidden_1_units", min_value=100, max_value=900, step=50, default=260),
                                    activation='relu',
                                    name='reference_hidden_1')(layers.concatenate(layer))
    reference_hidden = layers.BatchNormalization()(reference_hidden)
    reference_hidden = layers.Dropout(hp.Choice("dropout_3", values=[
                                      0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(reference_hidden)

    reference_hidden2 = layers.Dense(hp.Int("reference_hidden_2_units", min_value=100, max_value=900, step=50, default=10),
                                     activation='relu',
                                     name='reference_hidden_2')(reference_hidden)
    reference_hidden2 = layers.BatchNormalization()(reference_hidden2)
    reference_hidden2 = layers.Dropout(hp.Choice("dropout_4", values=[
                                       0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(reference_hidden2)

    reference_output = layers.Dense(hp.Int("reference_hidden_3_units", min_value=100, max_value=900, step=50, default=460),
                                    activation='tanh',
                                    name='reference_hidden_3')(reference_hidden2)

    ####################################################

    feed_forward_input = layers.concatenate(
        [reference_output, trade_history_output])

    hidden = layers.Dense(hp.Int("output_block_1_units", min_value=100, max_value=900, step=50, default=250),
                          activation='relu')(feed_forward_input)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Dropout(hp.Choice("dropout_5", values=[
                            0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(hidden)

    hidden2 = layers.Dense(hp.Int("output_block_2_units", min_value=100, max_value=900, step=50, default=600),
                           activation='tanh')(hidden)
    hidden2 = layers.BatchNormalization()(hidden2)
    xformed_nodes = layers.Dropout(hp.Choice("dropout_6", values=[
                                   0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], default=0.0))(hidden2)

    adj = layers.Input(name="adjacency",
                       shape=(None,),
                       dtype=tf.int64,
                       ragged=True)
    inputs.append(adj)

    neighbors = tf.gather(xformed_nodes, adj).to_tensor()

    neighborhoods = tf.concat([tf.broadcast_to(tf.expand_dims(
        xformed_nodes, axis=1), [tf.keras.backend.shape(neighbors)[0], tf.keras.backend.shape(neighbors)[1], 600]), neighbors], axis=-1)
    neighborhoods = layers.Dense(hp.Int("edge_units", min_value=100, max_value=900, step=50, default=1200),
                                 activation='relu')(neighborhoods)
    neighborhoods = layers.BatchNormalization()(neighborhoods)
    neighborhoods = layers.Dense(hp.Int("edge_units2", min_value=100, max_value=900, step=50, default=600),
                                 activation='relu')(neighborhoods)
    neighborhoods = layers.BatchNormalization()(neighborhoods)

    query = layers.Dense(d_model)(neighborhoods)
    value = layers.Dense(d_model)(neighborhoods)
    key = layers.Dense(d_model)(neighborhoods)
    attention = layers.Attention()([query, value, key])
    attention = layers.Dense(600)(attention)
    x = layers.Add()([neighborhoods, attention])  # residual connection
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Feed Forward
    dense = layers.Dense(dff, activation='relu')(x)
    dense = layers.Dense(600)(dense)
    x = layers.Add()([x, dense])     # residual connection
    encoder = layers.LayerNormalization(epsilon=1e-6)(x)
    encoder = encoder[:, 0, :]

    final = layers.Dense(1)(encoder)

    model = keras.Model(inputs=inputs, outputs=final)

    model.compile(optimizer=optimizer(learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4, 1e-5], default=learning_rate)),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.MeanAbsoluteError()])

    return model


register_model("gnn_yield_spread_model", 1, gnn_yield_spread_model_v1,
               "Initial GNN-based yield spread model")
