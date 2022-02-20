import tensorflow as tf
import yaml
import os,sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# sys.path.append(os.path.abspath('/EEG_Riemannian'))
# print(sys.path[0])

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


    b = tf.keras.layers.Dense(timestep, activation='tanh')(inputs)
    b = tf.keras.layers.Dense(timestep, activation='softmax')(b)

    #a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    #a = RepeatVector(input_dim)(a)_train, X_test, Y_train, Y_test)
    a_probs = tf.keras.layers.Permute((2, 1))(a)
    b_probs = tf.keras.layers.Permute((2, 1), name='prob')(b)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul')
    #a_probs     = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = tf.keras.layers.Multiply()([inputs, a_probs])
    output_attention_mul = tf.keras.layers.Lambda(lambda z: tf.math.reduce_sum(z, axis=1))(output_attention_mul)

    return b_probs, output_attention_mul

def corcoeff(y_true, y_pred):
    num = tf.reduce_sum((y_true-tf.reduce_mean(y_true))*(y_pred-tf.reduce_mean(y_pred)))
    den = tf.square(tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) * tf.reduce_sum(tf.square(y_pred - tf.reduce_mean(y_pred))))
    return num/den





def spatial_temporal_info_stream(train_embed, test_embed, X_train_features, X_test_features, Y_train, Y_test, dataset, net_params):


    '''
    Spatal Info Stream
    '''

    embed = np.vstack((train_embed, test_embed))
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    embed = embed - np.mean(embed, keepdims=True)
    embed = 2*(embed /np.std(embed, keepdims=True))-1
    # X = np.expand_dims(X, axis=3)
    train_embed  = embed[0:train_embed.shape[0]]
    test_embed  = embed[-test_embed.shape[0]:]


    if dataset =='SEED' or dataset =='BCI_IV_2a':
        Y_train = to_categorical(Y_train)
        Y_test  = to_categorical(Y_test)

    Y_train = Y_train.squeeze()
    Y_test  = Y_test.squeeze()



    inputs_spatial = tf.keras.Input(shape=(embed.shape[1],))

    MLP      =  tf.keras.layers.Dense(512, activation='relu')(inputs_spatial)
    MLP      =  tf.keras.layers.Dropout(0.5)(MLP)
    MLP      =  tf.keras.layers.Dense(64, activation='relu')(MLP)
    MLP      =  tf.keras.layers.Dropout(0.5)(MLP)

    MLP_F = tf.keras.layers.Dense(1, activation='relu')(MLP)


    '''
    Temporal Info Stream
    '''

    X = np.vstack((X_train_features, X_test_features))
    X = np.transpose(X, (0,2,1,3)) #trial, timestep, channel, features

    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]*X.shape[3])) #trial, timestep, channel*features
    X_new = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))

    if dataset =='BCI_IV_2a':
        X_new = (X_new - np.mean(X_new))/np.std(X_new)
    else:
        scalar = MinMaxScaler(feature_range=(-1, 1),  copy=True)
        scalar.fit(X_new)
        X_new  = scalar.transform(X_new)

    X = np.reshape(X_new, (X.shape[0], X.shape[1], X.shape[2]))

    X_train_features = X[0:X_train_features.shape[0]]
    X_test_features  = X[-X_test_features.shape[0]:]


    batch_norm_module = tf.keras.Sequential(
                        [
                            tf.keras.layers.BatchNormalization(momentum=0.99,epsilon=0.001,
                                                              beta_initializer="zeros",
                                                              gamma_initializer="ones",name='batchnorm'),
                            tf.keras.layers.LeakyReLU(alpha=0.3),
                        ]
    )


    inputs_temporal       = tf.keras.Input(shape=(X.shape[1], X.shape[2]))

    print(dataset, config[dataset]['params']['LSTM_Layer_No'])
    dropout = [0.2, 0.1, 0.1]
    for layer_num in range(1, config[dataset]['params']['LSTM_Layer_No']+1):

        if config[dataset]['params']['Bi_LSTM'] ==False:
            if layer_num == 1:
                lstm_out = tf.keras.layers.LSTM(units=256, return_sequences=True, activation="tanh",
                                                recurrent_activation="sigmoid", use_bias=True,
                                                kernel_initializer="glorot_uniform",
                                                recurrent_initializer="orthogonal",
                                                bias_initializer="zeros")(inputs_temporal)
                if dataset =='BCI_IV_2a':
                    lstm_out = tf.keras.layers.Dropout(dropout[layer_num-1])(lstm_out)
                else:
                    lstm_out = batch_norm_module(lstm_out)


            else:
                lstm_out = tf.keras.layers.LSTM(units=256, return_sequences=True, activation="tanh",
                                                recurrent_activation="sigmoid", use_bias=True,
                                                kernel_initializer="glorot_uniform",
                                                recurrent_initializer="orthogonal",
                                                bias_initializer="zeros")(lstm_out)
                if dataset =='BCI_IV_2a':
                    lstm_out = tf.keras.layers.Dropout(dropout[layer_num-1])(lstm_out)
                else:
                    lstm_out = batch_norm_module(lstm_out)

        else:

            if layer_num == 1:
                lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True, activation="tanh",
                                                recurrent_activation="sigmoid", use_bias=True,
                                                kernel_initializer="glorot_uniform",
                                                recurrent_initializer="orthogonal",
                                                bias_initializer="zeros"))(inputs_temporal)
                if dataset =='BCI_IV_2a':
                    lstm_out = tf.keras.layers.Dropout(dropout[layer_num-1])(lstm_out)
                else:
                    lstm_out = batch_norm_module(lstm_out)


            else:
                lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True, activation="tanh",
                                                recurrent_activation="sigmoid", use_bias=True,
                                                kernel_initializer="glorot_uniform",
                                                recurrent_initializer="orthogonal",
                                                bias_initializer="zeros"))(lstm_out)
                if dataset =='BCI_IV_2a':
                    lstm_out = tf.keras.layers.Dropout(dropout[layer_num-1])(lstm_out)
                else:
                    lstm_out = batch_norm_module(lstm_out)




    _, attention_mul  = attention_3d_block(lstm_out, timestep=config[dataset]['timestep'])
    attention_mul  = tf.keras.layers.Dense(64, activation='relu')(attention_mul)


    attention_mul_F = tf.keras.layers.Dense(1, activation='relu')(attention_mul)


    '''
    Feature usion with attention mechanism
    '''

    attention_score  = tf.keras.layers.concatenate([MLP_F, attention_mul_F])
    attention_score  = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2), name='dim_expansion')(attention_score)
    attention_score  = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1)), name='dim_permute')(attention_score)
    attention_score,_  = attention_3d_block(attention_score, 2)
    # attention_score  = Lambda(lambda x: x, name='weight_score')(attention_score)

    attention_score_MLP  = tf.keras.layers.Lambda(lambda x: x[:,0]+1, name='value_fetch_1')(attention_score)
    attention_score_att  = tf.keras.layers.Lambda(lambda x: x[:,1]+1, name='value_fetch_2')(attention_score)
    attention_score_MLP  = tf.keras.layers.Multiply()([attention_score_MLP, MLP])
    attention_score_att  = tf.keras.layers.Multiply()([attention_score_att, attention_mul])
    attention_score_final= tf.keras.layers.concatenate([attention_score_MLP,attention_score_att])



    output =tf.keras.layers.Dense(config[dataset]['output_unit'],  activation =config[dataset]['act_func'])(attention_score_final)
    model  = tf.keras.Model(inputs=[inputs_spatial, inputs_temporal], outputs=output)

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


    if net_params['saved_ckpt_flag']==True:
        checkpoint_path = os.path.join(config[dataset]['PATH'], 'saved_ckpt/test_spatial_temporal_model.ckpt')

    else:

        checkpoint_path = os.path.join(config[dataset]['PATH'], 'ckpt/test_spatial_temporal_model.ckpt')

        '''
        Early stopping strategy based on training loss
        '''
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', mode = 'auto', patience=net_params['early_stopping']),
                      tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='loss', mode = 'auto', save_weights_only=True, save_best_only=True)]



        history = model.fit([train_embed, X_train_features], Y_train, epochs=net_params['epochs'], batch_size=net_params['batch_size'], verbose=1, shuffle=True,
                             validation_data=([test_embed, X_test_features], Y_test), callbacks=callbacks)




    model.load_weights(checkpoint_path)

    Y_pred = model.predict([test_embed, X_test_features])



    return Y_pred
























#
