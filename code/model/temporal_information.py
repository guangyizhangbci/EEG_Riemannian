import tensorflow as tf
import yaml
import os,sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import utils



def load_config(name):
    with open(os.path.join(sys.path[0], name)) as file:
        config = yaml.safe_load(file)

    return config

config = load_config('dataset_params.yaml')

def to_categorical(y):
    """ 1-hot encodes a tensor """
    num_classes = len(np.unique(y))
    return np.eye(num_classes, dtype='uint8')[y.astype(int)]

def attention_3d_block(inputs, timestep):
    input_dim = int(inputs.shape[2])
    a = tf.keras.layers.Permute((2, 1))(inputs)
    a = tf.keras.layers.Dense(timestep, activation='tanh')(a)
    a = tf.keras.layers.Dense(timestep, activation='softmax')(a)
    #a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    #a = RepeatVector(input_dim)(a)_train, X_test, Y_train, Y_test)
    a_probs = tf.keras.layers.Permute((2, 1))(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul')
    #a_probs     = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = tf.keras.layers.Multiply()([inputs, a_probs])
    output_attention_mul = tf.keras.layers.Lambda(lambda z: tf.math.reduce_sum(z, axis=1))(output_attention_mul)

    return output_attention_mul



def corcoeff(y_true, y_pred):
    num = tf.reduce_sum((y_true-tf.reduce_mean(y_true))*(y_pred-tf.reduce_mean(y_pred)))
    den = tf.square(tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) * tf.reduce_sum(tf.square(y_pred - tf.reduce_mean(y_pred))))
    return num/den


def temporal_info_stream(X_train, X_val, X_test, Y_train, Y_val, Y_test, dataset, net_params):

    X = np.vstack((X_train, X_val, X_test))
    X = np.transpose(X, (0,2,1,3)) #trial, timestep, channel, features

    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]*X.shape[3])) #trial, timestep, channel*features
    X_new = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))

    scalar = MinMaxScaler(feature_range=(-1, 1),  copy=True)
    scalar.fit(X_new)
    X_new  = scalar.transform(X_new)

    X = np.reshape(X_new, (X.shape[0], X.shape[1], X.shape[2]))

    X_train = X[0:X_train.shape[0]]
    X_val   = X[X_train.shape[0]:-X_test.shape[0]]
    X_test  = X[-X_test.shape[0]:]



    if dataset =='SEED' or dataset =='BCI_IV_2a':
        Y_train = to_categorical(Y_train)
        Y_val   = to_categorical(Y_val)
        Y_test  = to_categorical(Y_test)

    Y_train = Y_train.squeeze()
    Y_val   = Y_val.squeeze()
    Y_test  = Y_test.squeeze()

    batch_norm_module = tf.keras.Sequential(
                        [
                            tf.keras.layers.BatchNormalization(momentum=0.99,epsilon=0.001,
                                                              beta_initializer="zeros",
                                                              gamma_initializer="ones"),
                            tf.keras.layers.LeakyReLU(alpha=0.3),
                        ]
    )



    inputs       = tf.keras.Input(shape=(X.shape[1], X.shape[2]))

    print(inputs.shape)
    for layer_num in range(1, net_params['layer_num']+1):
        if net_params['bidirectional_flag'] ==False:
            if layer_num == 1:
                lstm_out = tf.keras.layers.LSTM(units=256, return_sequences=True, activation="tanh",
                                                recurrent_activation="sigmoid", use_bias=True,
                                                kernel_initializer="glorot_uniform",
                                                recurrent_initializer="orthogonal",
                                                bias_initializer="zeros")(inputs)
                if dataset =='BCI_IV_2a':
                    lstm_out = tf.keras.layers.Dropout(0.0)(lstm_out)
                else:
                    lstm_out = batch_norm_module(lstm_out)


            else:
                lstm_out = tf.keras.layers.LSTM(units=256, return_sequences=True, activation="tanh",
                                                recurrent_activation="sigmoid", use_bias=True,
                                                kernel_initializer="glorot_uniform",
                                                recurrent_initializer="orthogonal",
                                                bias_initializer="zeros")(lstm_out)
                if dataset =='BCI_IV_2a':
                    lstm_out = tf.keras.layers.Dropout(0.0)(lstm_out)
                else:
                    lstm_out = batch_norm_module(lstm_out)
        else:

            if layer_num == 1:
                lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True, activation="tanh",
                                                recurrent_activation="sigmoid", use_bias=True,
                                                kernel_initializer="glorot_uniform",
                                                recurrent_initializer="orthogonal",
                                                bias_initializer="zeros"))(inputs)
                if dataset =='BCI_IV_2a':
                    lstm_out = tf.keras.layers.Dropout(0.0)(lstm_out)
                else:
                    lstm_out = batch_norm_module(lstm_out)


            else:
                lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True, activation="tanh",
                                                recurrent_activation="sigmoid", use_bias=True,
                                                kernel_initializer="glorot_uniform",
                                                recurrent_initializer="orthogonal",
                                                bias_initializer="zeros"))(lstm_out)
                if dataset =='BCI_IV_2a':
                    lstm_out = tf.keras.layers.Dropout(0.0)(lstm_out)
                else:
                    lstm_out = batch_norm_module(lstm_out)



    attention_mul  = attention_3d_block(lstm_out, timestep=config[dataset]['timestep'])
    attention_mul  = tf.keras.layers.Dense(64, activation='relu')(attention_mul)


    output =tf.keras.layers.Dense(config[dataset]['output_unit'],  activation =config[dataset]['act_func'])(attention_mul)
    model  = tf.keras.Model(inputs=inputs, outputs=output)

    if dataset =='SEED' or dataset =='BCI_IV_2a':
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
    elif dataset =='BCI_IV_2b':
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])
    elif dataset =='SEED_VIG':
        model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[corcoeff])
    else:
        raise Exception('Datasets Name Error')

    print(model.summary())


    checkpoint_path = os.path.join(config[dataset]['PATH'], 'ckpt/val_temporal_model.ckpt')


    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = 'auto', patience=net_params['early_stopping']),
                  tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', mode = 'auto', save_weights_only=True, save_best_only=True)]



    history = model.fit(X_train, Y_train, epochs=net_params['epochs'], batch_size=net_params['batch_size'], verbose=1, shuffle=True,
                         validation_data=(X_val, Y_val), callbacks=callbacks)


    model.load_weights(checkpoint_path)
    Y_pred = model.predict(X_val)


    return Y_pred








#
