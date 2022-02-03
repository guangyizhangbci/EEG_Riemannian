from __future__ import print_function, division
import tensorflow as tf
import matplotlib.pyplot as plt
import sys, os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from rich.progress import track
from time import time
import pyriemann
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score,accuracy_score
from sklearn.model_selection import KFold
import yaml
import umap
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from spatial_embedding import spatial_features
from model.spatial_information import spatial_info_stream
from utils import root_mean_squared_error_numpy, load_dataset_signal_addr, parse_valid_data, save_spatial_val_result
import argparse


print('ready')

parser = argparse.ArgumentParser(description='Spatial Info')
parser.add_argument('--dataset', default='BCI_IV_2b', type=str,
                    help='learning rate')
parser.add_argument('--lr', default=0.001, type=float, metavar='N',
                    help='learning rate')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='training epochs')
parser.add_argument('--early-stopping', default=20, type=int, metavar='N',
                    help='EarlyStopping')

parser.add_argument('--riemannian_dist', default=True, action='store_false')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


def load_config(name):
    with open(os.path.join(sys.path[0], name)) as file:
        config = yaml.safe_load(file)

    return config

config = load_config('dataset_params.yaml')

net_params = {'epochs': args.epochs, 'batch_size': args.batch_size, 'early_stopping': args.early_stopping}

class experiments():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def run_seed(self):

        addr_dict = load_dataset_signal_addr(self.dataset_name)
        data_train_addr, data_test_addr, label_train_addr, label_test_addr = list(addr_dict.values())
        '''
        Parameters search: rank of EEG
        '''
        for rank_num in tqdm(range(1, config[self.dataset_name]['Channel_No'])):
            Fold_No=3
            val_acc_result = np.zeros((config[self.dataset_name]['Subject_No'], config[self.dataset_name]['Session_No'], Fold_No))
            val_kap_result = np.zeros((config[self.dataset_name]['Subject_No'], config[self.dataset_name]['Session_No'], Fold_No))

            for subject_num in track(range(1, config[self.dataset_name]['Subject_No'] +1)):
                for session_num in range(1,config[self.dataset_name]['Session_No']+1):
                    #____________________LOAD DATA____________________#
                    X_train = np.load(data_train_addr.format(subject_num, session_num))
                    X_test  = np.load(data_test_addr.format(subject_num, session_num))
                    Y_train = np.load(label_train_addr.format(subject_num, session_num))
                    Y_test  = np.load(label_test_addr.format(subject_num, session_num))
                    ####################################################
                    train_embed, test_embed = spatial_features(config, self.dataset_name, args.riemannian_dist, rank_num).embedding(X_train, X_test)

                    kfold = KFold(Fold_No, True, 1)


                    Fold_count = 1
                    train_label = Y_train

                    for train_index, test_index in kfold.split(train_embed):

                        X_train, X_val, Y_train, Y_val =  train_embed[train_index], train_embed[test_index], train_label[train_index], train_label[test_index]

                        X_test = test_embed


                        Y_pred = spatial_info_stream(X_train, X_val, X_test, Y_train, Y_val, Y_test, self.dataset_name, net_params)

                        val_acc_mlp = np.mean(accuracy_score(Y_val, np.argmax(Y_pred, axis=-1)))
                        val_kap_mlp = np.mean(cohen_kappa_score(Y_val, np.argmax(Y_pred, axis=-1)))

                        val_acc_result[subject_num-1,session_num-1, Fold_No-1] = val_acc_mlp
                        val_kap_result[subject_num-1,session_num-1, Fold_No-1] = val_kap_mlp

                        Fold_count += 1

            val_acc_result = np.mean(val_acc_result, axis=2)
            val_kap_result = np.mean(val_kap_result, axis=2)
            save_spatial_val_result(self.dataset_name, val_acc_result, val_kap_result, rank_num)


    def run_seed_vig(self):

        addr_dict = load_dataset_signal_addr(self.dataset_name)
        load_data_addr, load_label_addr = list(addr_dict.values())

        val_Fold_No  = 3
        test_Fold_No = config[self.dataset_name]['Fold_No']

        for rank_num in tqdm(range(1, config[self.dataset_name]['Channel_No'])):
            val_rmse_result = np.zeros((config[self.dataset_name]['Subject_No'], test_Fold_No, val_Fold_No))
            val_corr_result = np.zeros((config[self.dataset_name]['Subject_No'], test_Fold_No, val_Fold_No))


            for subject_num in track(range(1, config[self.dataset_name]['Subject_No'] +1)):
                #____________________LOAD DATA____________________#
                X     = np.load(load_data_addr.format(subject_num))
                Y     = np.load(load_label_addr.format(subject_num))
                ####################################################


                test_Fold_count=1

                Y_test_total   = np.zeros((1,1))
                Y_pred_total   = np.zeros((1,1))


                rmse_array = np.zeros(([config[self.dataset_name]['Subject_No'], test_Fold_No]))
                corr_array = np.zeros(([config[self.dataset_name]['Subject_No'], test_Fold_No]))


                # kfold_test = KFold(test_Fold_No, True, 1)
                kfold_test = KFold(test_Fold_No, False, None)


                for train_index, test_index in kfold_test.split(X):

                    print("KFold No.", test_Fold_count)

                    X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]


                    kfold_val = KFold(val_Fold_No, True, 1)

                    val_Fold_count = 1
                    train_label = Y_train

                    train_embed, test_embed = spatial_features(config, self.dataset_name, args.riemannian_dist, rank_num).embedding(X_train, X_test)


                    Y_val_total   = np.zeros((0,1))
                    Y_pred_total   = np.zeros((0,1))

                    for train_index, test_index in kfold_val.split(train_embed):

                        X_train, X_val, Y_train, Y_val =  train_embed[train_index], train_embed[test_index], train_label[train_index], train_label[test_index]

                        X_test = test_embed

                        Y_pred = spatial_info_stream(X_train, X_val, X_test, Y_train, Y_val, Y_test, self.dataset_name, net_params)


                        temp_Y_val  = Y_val
                        temp_Y_pred = Y_pred

                        Y_val_total  = np.vstack((Y_val_total, temp_Y_val))
                        Y_pred_total = np.vstack((Y_pred_total, temp_Y_pred))

                        val_Fold_count += 1


                    Y_val_total  = np.ravel(Y_val_total)
                    Y_pred_total = np.ravel(Y_pred_total)

                    rmse_value   =  root_mean_squared_error_numpy(Y_val_total, Y_pred_total) # RMSE value for all 885 samples
                    corcoeff_value, _ = pearsonr(Y_val_total, Y_pred_total)

                    test_Fold_count += 1

            rmse_array[subject_num-1, test_Fold_No-1]   = rmse_value
            corr_array[subject_num-1, test_Fold_No-1]   = corcoeff_value

            save_spatial_val_result(self.dataset_name, val_acc_result, val_kap_result, rank_num)



    def run_bci(self):

        addr_dict = load_dataset_signal_addr(self.dataset_name)

        data_train_addr, data_test_addr, label_train_addr, label_test_addr = list(addr_dict.values())

        '''
        Parameters search: rank of EEG
        '''

        for rank_num in tqdm(range(1, config[self.dataset_name]['Channel_No']+1)):

            Fold_No=5
            val_acc_result = np.zeros((config[self.dataset_name]['Subject_No'], Fold_No))
            val_kap_result = np.zeros((config[self.dataset_name]['Subject_No'], Fold_No))

            for subject_num in track(range(1, config[self.dataset_name]['Subject_No']+1)):
                #____________________LOAD DATA____________________#
                X_train = np.load(data_train_addr.format(subject_num)) #  trials, frequency_band, channel, signal
                Y_train = np.load(label_train_addr.format(subject_num))
                X_test  = np.load(data_test_addr.format(subject_num))
                Y_test  = np.load(label_test_addr.format(subject_num))
                Y_train = np.expand_dims(Y_train, axis=1) -1  #1,2,3,4 ---> 0,1,2,3
                Y_test  = np.expand_dims(Y_test, axis=1) -1  #1,2,3,4 ---> 0,1,2,3
                ####################################################

                X_train, Y_train = parse_valid_data(X_train, Y_train)
                X_test,  Y_test  = parse_valid_data(X_test,  Y_test)

                train_embed, test_embed = spatial_features(config, self.dataset_name, args.riemannian_dist, rank_num).embedding(X_train, X_test)


                kfold = KFold(Fold_No, False, None)


                Fold_count = 1
                train_label = Y_train

                for train_index, test_index in kfold.split(train_embed):

                    X_train, X_val, Y_train, Y_val =  train_embed[train_index], train_embed[test_index], train_label[train_index], train_label[test_index]

                    X_test = test_embed

                    Y_pred = spatial_info_stream(X_train, X_val, X_test, Y_train, Y_val, Y_test, self.dataset_name, net_params)

                    '''
                    2a output label in one-hot form, 2b output label in range (0,1)
                    '''
                    if '2a' in self.dataset_name:
                        Y_pred = np.argmax(Y_pred, axis=-1)
                    else:
                        Y_pred = np.round(Y_pred)
                        Y_val  = Y_val.squeeze(1)

                    val_acc_mlp = np.mean(accuracy_score(Y_val, Y_pred))
                    val_kap_mlp = np.mean(cohen_kappa_score(Y_val, Y_pred))


                    val_acc_result[subject_num-1,Fold_count-1] = val_acc_mlp
                    val_kap_result[subject_num-1,Fold_count-1] = val_kap_mlp

                    Fold_count += 1

            save_spatial_val_result(self.dataset_name, val_acc_result, val_kap_result, rank_num)


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
        experiments(args.dataset).run()



#
