import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal
import itertools
from scipy.stats import kurtosis, skew,entropy
from math import log, e
from sklearn.utils import shuffle
from tqdm import tqdm
import os
from scipy.io import loadmat
from os import listdir
from os.path import isfile, join
from scipy import stats
from library.signal_filtering import butter_bandpass,butter_bandpass_filter
from scipy.signal import welch
from scipy.integrate import simps
from pywt import wavedec, upcoef
from scipy.integrate import simps
fs = 250

# ___________________________________Features Extraction___________________________________#

def feature_extraction(data):  # data is current channel, temp_data are all 23 channels under the same frame

    sf = fs
    win = sf
    # freqs, psd = signal.welch(data, sf, nperseg=win, scaling='density')
    freqs, psd = signal.periodogram(data, sf, window= 'hann',scaling='density', detrend='constant')


    psd_all = np.zeros((25, ))
    for i in range(0, 25):
        low, high = i*2+0.5, (i+1)*2+0.5
        idx_min = np.argmax(freqs > low) - 1
        idx_max = np.argmax(freqs > high) - 1
        # print(freqs)
        idx = np.zeros(dtype=bool, shape=freqs.shape)
        idx[idx_min:idx_max] = True
        # print('idx_max', idx_min, idx_max)
        psd_all[i] = simps(psd[idx], freqs[idx])


    DE_all = np.zeros((25, ))

    for m in range(0, 25): #frequency band
            new_data = butter_bandpass_filter(data, lowcut= 0.5 + m*2 , highcut= 0.5 +(m+1)*2, fs=250, order=4)

            DE_all[m] = 0.5*np.log(2*np.pi*np.exp(1)*np.var(new_data))

    features_1 = psd_all.tolist()
    features_1 = np.log10(features_1)

    features_2 = DE_all

    features = np.hstack((features_1, features_2))

    return features



































###
