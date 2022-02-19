from __future__ import print_function, division
import tensorflow as tf
import matplotlib.pyplot as plt
import sys, os
import numpy as np
from tqdm import tqdm
from rich.progress import track
from time import time
import pyriemann
import yaml
import argparse
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score,accuracy_score
from sklearn.model_selection import KFold
from spatial_embedding import spatial_features
from model.spatial_temporal_information import spatial_temporal_info_stream
from utils import root_mean_squared_error_numpy, load_dataset_signal_addr, load_dataset_feature_addr, parse_valid_data_all, save_test_result

print('ready')


parser = argparse.ArgumentParser(description='Spatial Temporal_Info')
parser.add_argument('--dataset', default='BCI_IV_2b', type=str,
                    help='learning rate')
parser.add_argument('--cpu-seed', default=0, type=int, metavar='N',
                    help='cpu seed')
parser.add_argument('--gpu-seed', default=12345, type=int, metavar='N',
                    help='gpu seed')
parser.add_argument('--lr', default=0.001, type=float, metavar='N',
                    help='learning rate')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='training epochs')
parser.add_argument('--early-stopping', default=200, type=int, metavar='N',
                    help='EarlyStopping')
parser.add_argument('--riemannian_dist', default=True,  action='store_false')
parser.add_argument('--saved-ckpt',      default=False, action='store_false')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}



def load_config(name):
    with open(os.path.join(sys.path[0], name)) as file:
        config = yaml.safe_load(file)

    return config

config = load_config('dataset_params.yaml')

net_params = {'epochs': args.epochs, 'batch_size': args.batch_size, 'early_stopping': args.early_stopping, 'saved_ckpt_flag': args.saved_ckpt}



class experiments():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    '''
    The main file of experiments, the training loop depends on each dataset
    (e.g., train-test direct split or using k-Fold, session-dependent or not)
    '''

    def run_seed(self):
        '''
        Address of filtered EEG in each frequency band and extracted features (DE, PSD)
        '''
        addr_dict = load_dataset_signal_addr(self.dataset_name)
        signal_train_addr, signal_test_addr, label_train_addr, label_test_addr = list(addr_dict.values())

        addr_dict = load_dataset_feature_addr(self.dataset_name)
        features_train_addr, features_test_addr, _, _ = list(addr_dict.values())


        test_acc_result = np.zeros((config[self.dataset_name]['Subject_No'], config[self.dataset_name]['Session_No']))
        test_kap_result = np.zeros((config[self.dataset_name]['Subject_No'], config[self.dataset_name]['Session_No']))


        for subject_num in track(range(1, config[self.dataset_name]['Subject_No'] +1)):
            for session_num in range(1,config[self.dataset_name]['Session_No']+1):
                #____________________LOAD DATA____________________#
                X_train_signal   = np.load(signal_train_addr.format(subject_num, session_num))
                X_test_signal    = np.load(signal_test_addr.format(subject_num, session_num))
                X_train_features = np.load(features_train_addr.format(subject_num, session_num))
                X_test_features  = np.load(features_test_addr.format(subject_num, session_num))
                Y_train          = np.load(label_train_addr.format(subject_num, session_num))
                Y_test           = np.load(label_test_addr.format(subject_num, session_num))
                ####################################################


                train_embed, test_embed = spatial_features(config, self.dataset_name, args.riemannian_dist, config[self.dataset_name]['params']['Rank_No']).embedding(X_train_signal, X_test_signal)

                Y_pred =  spatial_temporal_info_stream(train_embed, test_embed, X_train_features, X_test_features, Y_train, Y_test, seld.dataset_name, net_params)

                test_acc_mlp = np.mean(accuracy_score(Y_test, np.argmax(Y_pred, axis=-1)))
                test_kap_mlp = np.mean(cohen_kappa_score(Y_test, np.argmax(Y_pred, axis=-1)))

                test_acc_result[subject_num-1,session_num-1] = test_acc_mlp
                test_kap_result[subject_num-1,session_num-1] = test_kap_mlp


        test_acc_result = np.mean(test_acc_result, axis=2)
        test_kap_result = np.mean(test_kap_result, axis=2)

        save_test_result(self.dataset_name, test_acc_result, test_kap_result)


    def run_seed_vig(self):
        '''
        Address of filtered EEG in each frequency band and extracted features (DE, PSD)
        '''
        addr_dict = load_dataset_signal_addr(self.dataset_name)
        signal_addr, label_addr = list(addr_dict.values())

        addr_dict = load_dataset_feature_addr(self.dataset_name)
        features_addr, _ = list(addr_dict.values())


        test_Fold_No = config[self.dataset_name]['Fold_No']
        test_rmse_result = np.zeros((config[self.dataset_name]['Subject_No'], test_Fold_No))
        test_corr_result = np.zeros((config[self.dataset_name]['Subject_No'], test_Fold_No))


        for subject_num in track(range(1, config[self.dataset_name]['Subject_No'] +1)):
            #____________________LOAD DATA____________________#
            X_signal     = np.load(signal_addr.format(subject_num))
            X_features   = np.load(features_addr.format(subject_num))
            Y            = np.load(label_addr.format(subject_num))
            ####################################################

            test_Fold_count=1

            Y_test_total   = np.zeros((1,1))
            Y_pred_total   = np.zeros((1,1))


            rmse_array = np.zeros(([config[self.dataset_name]['Subject_No'], test_Fold_No]))
            corr_array = np.zeros(([config[self.dataset_name]['Subject_No'], test_Fold_No]))


            kfold_test = KFold(test_Fold_No, True, 1)
            # kfold_test = KFold(Fold_No_test, False, None)


            Y_test_total   = np.zeros((0,1))
            Y_pred_total   = np.zeros((0,1))

            for train_index, test_index in kfold_test.split(X_signal):

                print("KFold No.", test_Fold_count)

                X_train_signal, X_test_signal, X_train_features, X_test_features, Y_train, Y_test = X_signal[train_index], X_signal[test_index], X_features[train_index], X_features[test_index], Y[train_index], Y[test_index]


                train_embed, test_embed = spatial_features(config, self.dataset_name, args.riemannian_dist, config[self.dataset_name]['params']['Rank_No']).embedding(X_train_signal, X_test_signal)


                Y_pred =  spatial_temporal_info_stream(train_embed, test_embed, X_train_features, X_test_features, Y_train, Y_test, self.dataset_name, net_params)


                temp_Y_test  = Y_test
                temp_Y_pred  = Y_pred

                Y_test_total  = np.vstack((Y_test_total, temp_Y_test))
                Y_pred_total  = np.vstack((Y_pred_total, temp_Y_pred))

            Y_test_total  = np.ravel(Y_test_total)
            Y_pred_total  = np.ravel(Y_pred_total)
            print(Y_test_total.shape, Y_pred_total.shape)
            test_Fold_count += 1


            rmse_value   =  root_mean_squared_error_numpy(Y_test_total, Y_pred_total) # RMSE value for all 885 samples
            corcoeff_value, _ = pearsonr(Y_test_total, Y_pred_total)


            rmse_array[subject_num-1, test_Fold_No-1]   = rmse_value
            corr_array[subject_num-1, test_Fold_No-1]   = corcoeff_value

            save_test_result(self.dataset_name, test_acc_result, test_kap_result)



    def run_bci(self):

        '''
        Address of filtered EEG in each frequency band and extracted features (DE, PSD)
        '''
        addr_dict = load_dataset_signal_addr(self.dataset_name)
        signal_train_addr, signal_test_addr, label_train_addr, label_test_addr = list(addr_dict.values())

        addr_dict = load_dataset_feature_addr(self.dataset_name)
        features_train_addr, features_test_addr, _, _ = list(addr_dict.values())


        test_acc_result = np.zeros((config[self.dataset_name]['Subject_No']))
        test_kap_result = np.zeros((config[self.dataset_name]['Subject_No']))



        for subject_num in track(range(1, config[self.dataset_name]['Subject_No']+1)):
        # for subject_num in track(range(1, 2)):

            #____________________LOAD DATA____________________#
            X_train_signal   = np.load(signal_train_addr.format(subject_num))
            X_test_signal    = np.load(signal_test_addr.format(subject_num))
            X_train_features = np.load(features_train_addr.format(subject_num))
            X_test_features  = np.load(features_test_addr.format(subject_num))
            Y_train          = np.load(label_train_addr.format(subject_num))
            Y_test           = np.load(label_test_addr.format(subject_num))
            Y_train          = np.expand_dims(Y_train, axis=1) -1  #1,2,3,4 ---> 0,1,2,3
            Y_test           = np.expand_dims(Y_test, axis=1) -1  #1,2,3,4 ---> 0,1,2,3
            ####################################################

            X_train_signal, X_train_features, Y_train = parse_valid_data_all(X_train_signal, X_train_features, Y_train)
            X_test_signal, X_test_features, Y_test    = parse_valid_data_all(X_test_signal, X_test_features, Y_test)

            train_embed, test_embed = spatial_features(config, self.dataset_name, args.riemannian_dist, config[self.dataset_name]['params']['Rank_No']).embedding(X_train_signal, X_test_signal)


            Y_pred =  spatial_temporal_info_stream(train_embed, test_embed, X_train_features, X_test_features, Y_train, Y_test, self.dataset_name, net_params)


            '''
            2a output label in one-hot form, 2b output label in range (0,1)
            '''
            if '2a' in self.dataset_name:
                Y_pred = np.argmax(Y_pred, axis=-1)
            else:
                Y_pred = np.round(Y_pred)
                Y_test  = Y_test.squeeze(1)

            test_acc_mlp = np.mean(accuracy_score(Y_test, Y_pred))
            test_kap_mlp = np.mean(cohen_kappa_score(Y_test, Y_pred))


            test_acc_result[subject_num-1] = test_acc_mlp
            test_kap_result[subject_num-1] = test_kap_mlp

            save_test_result(self.dataset_name, test_acc_result, test_kap_result)


    def run(self):

        if 'BCI' in self.dataset_name:
            self.run_bci()
        elif self.dataset_name=='SEED':
            self.run_seed()
        elif self.dataset_name=='SEED_VIG':
            self.run_seed_vig()
        else:
            raise Exception('Datasets Name Error')




if __name__ == '__main__':


    config = load_config('dataset_params.yaml')
    with tf.device("gpu:0"):
        np.random.seed(args.cpu_seed)
        tf.random.set_random_seed(args.gpu_seed)
        experiments(args.dataset).run()





#
