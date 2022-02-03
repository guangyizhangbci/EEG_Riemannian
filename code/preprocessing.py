import numpy as np
from tqdm import tqdm
from load_data import load_data
from library.signal_filtering import signal_filtering
import os



fs = 250

'''
This pre-processing file is for BCI_IV_2a and BCI_IV_2b datasets

'''

def bci_iv_2a(self):
    PATH = '/media/patrick/DATA/BCI_Competition/'
    data_train_addr  = os.path.join(PATH,'train/data_{}') # subject
    data_test_addr   = os.path.join(PATH,'test/data_{}') # subject

    label_train_addr = os.path.join(PATH,'train/label_{}') # subject
    label_test_addr  = os.path.join(PATH,'test/label_{}') # subject

    data_train_filter_addr = os.path.join(PATH,'train/filter_data_{}') # subject
    data_test_filter_addr  = os.path.join(PATH,'test/filter_data_{}') # subject

    for subject_No in (range(1, 10)):

        #_________________training_data_________________________#
        data, label = load_data(subject_No, True, PATH)

        filter_data = []

        for trial_No in range(data.shape[0]):
            data_trial  = data[trial_No]
            filter_data.append(signal_filtering(data_trial))

        filter_data = np.array(filter_data)
        np.save(data_train_filter_addr.format(subject_No), filter_data)

        np.save(data_train_addr.format(subject_No),  data)
        np.save(label_train_addr.format(subject_No), label)


        #_________________testing_data_________________________#
        data, label = load_data(subject_No, False, PATH)
        filter_data = []
        for trial_No in range(data.shape[0]):
            data_trial  = data[trial_No]
            filter_data.append(signal_filtering(data_trial))

        filter_data = np.array(filter_data)
        np.save(data_test_filter_addr.format(subject_No), filter_data)

        np.save(data_test_addr.format(subject_No),  data)
        np.save(label_test_addr.format(subject_No), label)




def bci_iv_2b(self):

    D_PATH = '/media/patrick/DATA/BCICIV_2b/'
    L_PATH = os.path.join(D_PATH,'true_label/')
    fs = 250
    data_train_addr  = os.path.join(D_PATH,'train/data_{}') # subject
    data_test_addr   = os.path.join(D_PATH,'test/data_{}') # subject

    label_train_addr = os.path.join(D_PATH,'train/label_{}') # subject
    label_test_addr  = os.path.join(D_PATH,'test/label_{}') # subject

    data_train_filter_addr = os.path.join(D_PATH,'train/filter_data_{}') # subject
    data_test_filter_addr  = os.path.join(D_PATH,'test/filter_data_{}') # subject

    for subject_No in (range(1, 10)):

        #_________________training_data_________________________#

        data, label = load_data(subject_No, True, D_PATH, L_PATH)
        filter_data = []
        for trial_No in range(data.shape[0]):
            data_trial  = data[trial_No]
            filter_data.append(signal_filtering(data_trial))

        filter_data = np.array(filter_data)
        np.save(data_train_filter_addr.format(subject_No), filter_data)

        np.save(data_train_addr.format(subject_No),  data)
        np.save(label_train_addr.format(subject_No), label)


        #_________________testing_data_________________________#
        data, label = load_data(subject_No, False, D_PATH, L_PATH)
        filter_data = []
        for trial_No in range(data.shape[0]):
            data_trial  = data[trial_No]
            filter_data.append(signal_filtering(data_trial))

        filter_data = np.array(filter_data)
        np.save(data_test_filter_addr.format(subject_No), filter_data)

        np.save(data_test_addr.format(subject_No),  data) #Use Only at the first run of this script
        np.save(label_test_addr.format(subject_No), label)















    #
