import os,sys
import numpy as np
import yaml


def load_config(name):
    with open(os.path.join(sys.path[0], name)) as file:
        config = yaml.safe_load(file)

    return config

config = load_config('dataset_params.yaml')


def root_mean_squared_error_numpy(y_true, y_pred): # numpy not tensor, used for EMSE calculation in SEED_VIG dataset
    return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))



def load_dataset_signal_addr(dataset):
    '''
    Load address of processed EEG signal from filter banks into dictionary
    '''

    if 'VIG' not in dataset:
        if 'BCI' in dataset:
            PATH = config[dataset]['PATH']

            data_train_addr  = os.path.join(PATH,'train/' + 'EEG/filter_data_{}.npy') # subject
            data_test_addr   = os.path.join(PATH,'test/'  + 'EEG/filter_data_{}.npy') # subject
            label_train_addr = os.path.join(PATH,'train/label_{}.npy') # subject
            label_test_addr  = os.path.join(PATH,'test/label_{}.npy') # subject

        elif 'SEED' in dataset:
            PATH = config[dataset]['PATH']

            data_train_addr  = os.path.join(PATH,'train/' + 'EEG/filter_data_{}_{}.npy') # subject, session
            data_test_addr   = os.path.join(PATH,'test/'  + 'EEG/filter_data_{}_{}.npy') # subject, session
            label_train_addr = os.path.join(PATH,'train/label_{}_{}.npy') # subject, session
            label_test_addr  = os.path.join(PATH,'test/label_{}_{}.npy') # subject, session

        else:
            raise Exception('Datasets Name Error')

        addr_dict = {str(data_train_addr):  data_train_addr,
                     str(data_test_addr):   data_test_addr,
                     str(label_train_addr): label_train_addr,
                     str(label_test_addr):  label_test_addr}

    elif dataset=='SEED_VIG':
        PATH = config[dataset]['PATH']

        data_addr    = os.path.join(PATH, 'EEG/filter_data_{}.npy') # subject
        # data_addr    = os.path.join(PATH, 'New/filtered_signal/data_{}.npy') # subject

        label_addr   = os.path.join(PATH,'Labels/{}.npy') # subject

        addr_dict = {str(data_addr):    data_addr,
                     str(label_addr):   label_addr}

    else:
        raise Exception('Datasets Name Error')

    return addr_dict


def load_dataset_feature_addr(dataset):
    '''
    Load address of extracted features (DE and PSD) into dictionary
    '''

    if 'VIG' not in dataset:
        if 'BCI' in dataset:
            PATH = config[dataset]['PATH']

            data_train_addr  = os.path.join(PATH,'train/' + 'Extracted Features/features_{}.npy') # subject
            data_test_addr   = os.path.join(PATH,'test/'  + 'Extracted Features/features_{}.npy') # subject
            label_train_addr = os.path.join(PATH,'train/label_{}.npy') # subject
            label_test_addr  = os.path.join(PATH,'test/label_{}.npy') # subject

        elif 'SEED' in dataset:
            PATH = config[dataset]['PATH']

            data_train_addr  = os.path.join(PATH,'train/' + 'Extracted Features/features_{}_{}.npy') # subject, session
            data_test_addr   = os.path.join(PATH,'test/' + 'Extracted Features/features_{}_{}.npy') # subject, session
            label_train_addr = os.path.join(PATH,'train/label_{}_{}.npy') # subject, session
            label_test_addr  = os.path.join(PATH,'test/label_{}_{}.npy') # subject, session

        else:
            raise Exception('Datasets Name Error')

        addr_dict = {str(data_train_addr):  data_train_addr,
                     str(data_test_addr):   data_test_addr,
                     str(label_train_addr): label_train_addr,
                     str(label_test_addr):  label_test_addr}

    elif dataset=='SEED_VIG':
        PATH = config[dataset]['PATH']

        data_addr    = os.path.join(PATH, 'Extracted Features/features_{}.npy') # subject
        label_addr   = os.path.join(PATH, 'Labels/{}.npy') # subject

        addr_dict = {str(data_addr):    data_addr,
                     str(label_addr):   label_addr}

    else:
        raise Exception('Datasets Name Error')

    return addr_dict




def parse_valid_data(x, y):
    '''
    check if any data trial contains nan of EEG signal
    '''
    if np.any(np.isnan(x))==True:
        row =np.argwhere(np.isnan(x))[:,0]
        row = np.unique(row)
        x = np.delete(x, row, axis=0)
        y = np.delete(y, row, axis=0)

    return x,y



def parse_valid_data_all(x_eeg, x_features, y):
    '''
    Remove any data trial contains nan of EEG signal
    and synchronize the index of new EEG signal and extracted features
    '''
    if np.any(np.isnan(x_eeg))==True:
        row =np.argwhere(np.isnan(x_eeg))[:,0]
        row = np.unique(row)

        x_eeg       = np.delete(x_eeg,      row, axis=0)
        x_features  = np.delete(x_features, row, axis=0)
        y           = np.delete(y,          row, axis=0)

    return x_eeg, x_features, y




def save_spatial_val_result(dataset, metrics_1_value, metrics_2_value, rank_num):
    '''
    Save the validation result using only spatial stream of RFNet
    '''
    if 'VIG' in dataset:
        metrics_1_name = 'rmse'
        metrics_2_name = 'corr'
    else:
        metrics_1_name = 'acc'
        metrics_2_name = 'kap'



    result_path = './' + dataset + '_result/spatial/'
    checkpoint_save_path = os.path.join(result_path, 'rank_{}/'.format(rank_num))

    if not os.path.exists(checkpoint_save_path):
        try:
            os.makedirs(checkpoint_save_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            pass


    np.savetxt(os.path.join(checkpoint_save_path, metrics_1_name + '.csv'), metrics_1_value, delimiter=",")
    np.savetxt(os.path.join(checkpoint_save_path, metrics_2_name + '.csv'), metrics_2_value, delimiter=",")



def save_temporal_val_result(dataset, metrics_1_value, metrics_2_value, bidirectional_flag, layer_num):
    '''
    Save the validation result using only temporal stream of RFNet
    '''
    if 'VIG' in dataset:
        metrics_1_name = 'rmse'
        metrics_2_name = 'corr'
    else:
        metrics_1_name = 'acc'
        metrics_2_name = 'kap'


    result_path = './' + dataset + '_result/temporal/'
    if bidirectional_flag == False:
        result_path = result_path + 'LSTM/'
    else:
        result_path = result_path + 'BiLSTM/'

    checkpoint_save_path = os.path.join(result_path, 'layer_{}/'.format(layer_num))

    if not os.path.exists(checkpoint_save_path):
        try:
            os.makedirs(checkpoint_save_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            pass


    np.savetxt(os.path.join(checkpoint_save_path, metrics_1_name + '.csv'), metrics_1_value, delimiter=",")
    np.savetxt(os.path.join(checkpoint_save_path, metrics_2_name + '.csv'), metrics_2_value, delimiter=",")





def save_test_result(dataset, metrics_1_value, metrics_2_value):
    '''
    Save the test result using RFNet
    '''
    if 'VIG' in dataset:
        metrics_1_name = 'rmse'
        metrics_2_name = 'corr'
    else:
        metrics_1_name = 'acc'
        metrics_2_name = 'kap'

    result_path = './' + dataset + '_result/test/'



    if not os.path.exists(result_path):
        try:
            os.makedirs(result_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            pass


    np.savetxt(os.path.join(result_path, metrics_1_name + '.csv'), metrics_1_value, delimiter=",")
    np.savetxt(os.path.join(result_path, metrics_2_name + '.csv'), metrics_2_value, delimiter=",")




#
