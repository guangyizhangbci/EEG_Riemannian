from __future__ import print_function, division
import numpy as np
from sklearn.preprocessing import StandardScaler
import pyriemann
from pyriemann.estimation import Covariances
from library.spfiltering import ProjCommonSpace
from library.featuring import Riemann



class spatial_features():
    def __init__(self, config, dataset, rieman_flag, rank_num):
        self.rieman_flag = rieman_flag  # Use Riemannian or Not
        self.rank_num = rank_num        # rank of a convariance matrix
        self.dataset = dataset          # Dataset name
        self.config = config            # Dataset specfic configuration dictionary

    def tangentspace_learning(self, spoc):

        ''' Learning Processing: from Riemannian Space to Tangent Space '''

        geom   = Riemann(n_fb=1, metric='riemann').transform(spoc)
        scaler = StandardScaler()
        scaler.fit(geom)
        sc = scaler.transform(geom)
        return sc

    def projection(self, X_train, X_test, rieman_flag, rank_num):


        '''Estimation of covariance matrix'''

        cov_train = Covariances('oas').transform(X_train)
        cov_train = cov_train[:, None, :, :]
        cov_test  = Covariances('oas').transform(X_test)
        cov_test  = cov_test[:, None, :, :]

        if self.rieman_flag==False:
            ''' Direct vectorization of spatial covariance matrices without Riemannian '''

            sc_train  = NaiveVec(method='upper').transform(cov_train)
            sc_test   = NaiveVec(method='upper').transform(cov_test)
        else:
            # spoc = ProjSPoCSpace(n_compo=n_compo, scale='auto')
            '''Dimensionality Reduction'''

            spoc = ProjCommonSpace(rank_num=rank_num)
            # spoc = ProjSPoCSpace(rank_num=rank_num, scale='auto')

            spoc_train = spoc.fit(cov_train).transform(cov_train)
            spoc_test  = spoc.fit(cov_train).transform(cov_test)

            # sc_train = spoc_train[:,0,:,:]
            # sc_test  = spoc_test[:,0,:,:]
            '''Dimensionality Reduction'''
            sc_train = self.tangentspace_learning(spoc_train)
            sc_test  = self.tangentspace_learning(spoc_test)

        return sc_train, sc_test


    def embedding(self, X_train, X_test):
        train_embed = []
        test_embed  = []

        '''
        Concatenate spatial embeddings from each frequency band
        '''

        for freqband_band in range(0, self.config[self.dataset]['Band_No']):

            X_training    =  X_train[:,freqband_band,:,:]
            X_testing     =  X_test[:,freqband_band,:,:]

            train_embedding, test_embedding = self.projection(X_training, X_testing, self.rieman_flag, self.rank_num)

            train_embed.append(train_embedding)
            test_embed.append(test_embedding)

        train_embed = np.asarray(train_embed)
        test_embed  = np.asarray(test_embed)
        train_embed = np.transpose(train_embed, [1, 0, 2])
        test_embed  = np.transpose(test_embed,  [1, 0, 2])
        train_embed = np.reshape(train_embed, (train_embed.shape[0],train_embed.shape[1]*train_embed.shape[2]))# concatenate feature from different EEG bandds
        test_embed  = np.reshape(test_embed,  (test_embed.shape[0], test_embed.shape[1]*test_embed.shape[2]))

        return train_embed, test_embed






#
