import tensorflow as tf
import yaml
import os,sys
import numpy as np
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



def corcoeff(y_true, y_pred): # tensor
    num = tf.reduce_sum((y_true-tf.reduce_mean(y_true))*(y_pred-tf.reduce_mean(y_pred)))
    den = tf.square(tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) * tf.reduce_sum(tf.square(y_pred - tf.reduce_mean(y_pred))))
    return num/den


def spatial_info_stream(X_train, X_val, X_test, Y_train, Y_val, Y_test, dataset, net_params):

    X = np.vstack((X_train, X_val, X_test))
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    X = X - np.mean(X, keepdims=True)
    X = 2*(X /np.std(X, keepdims=True))-1
    # X = np.expand_dims(X, axis=3)
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

    inputs = tf.keras.Input(shape=(X_train.shape[1],))

    MLP      =  tf.keras.layers.Dense(512, activation='relu')(inputs)
    MLP      =  tf.keras.layers.Dropout(0.5)(MLP)
    MLP      =  tf.keras.layers.Dense(64, activation='relu')(MLP)
    MLP      =  tf.keras.layers.Dropout(0.5)(MLP)

    output =tf.keras.layers.Dense(config[dataset]['output_unit'],  activation =config[dataset]['act_func'])(MLP)
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


    checkpoint_path = os.path.join(config[dataset]['PATH'], 'ckpt/val_spatial_model.ckpt')

    '''
    Early stopping strategy based on validation loss
    '''
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = 'auto', patience=net_params['early_stopping']),
                  tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', mode = 'auto', save_weights_only=True, save_best_only=True)]



    history = model.fit(X_train, Y_train, epochs=net_params['epochs'], batch_size=net_params['batch_size'], verbose=0, shuffle=True,
                         validation_data=(X_val, Y_val), callbacks=callbacks)


    model.load_weights(checkpoint_path)
    Y_pred = model.predict(X_val)



    return Y_pred
