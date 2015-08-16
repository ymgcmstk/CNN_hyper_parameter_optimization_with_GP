#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cPickle as pickle
import numpy as np
import os

class Cifar10Fetcher:
    ds_path     = '/path/to/dataset'
    x_train     = None
    y_train     = []
    x_test      = None
    y_test      = []
    N_train     = 0
    N_test      = 0
    train_count = 0
    test_count  = 0
    iter_count  = 0
    used_count  = 0
    epoch_count = 0

    def __init__(self, minus_mean=True, norm=True):
        self.load_data()
        if minus_mean:
            self.minus_mean = minus_mean
            self.minus_mean_from_data()
        if norm:
            self.norm = norm
            self.norm_from_data()
        self.perm = np.arange(self.N_train)

    def load_data(self):
        for i in xrange(1,6):
            data_dictionary = pickle.load(open(os.path.join(self.ds_path, 'data_batch_%d') % i, 'rb'))
            if self.x_train is None:
                self.x_train = data_dictionary['data']
                self.y_train = data_dictionary['labels']
            else:
                self.x_train = np.vstack((self.x_train, data_dictionary['data']))
                self.y_train = self.y_train + data_dictionary['labels']
        self.x_train = self.x_train.reshape((len(self.x_train), 3, 32, 32)).astype(np.float32)
        self.y_train = np.array(self.y_train).astype(np.int32)

        test_data_dictionary = pickle.load(open(os.path.join(self.ds_path, 'test_batch'), 'rb'))
        self.x_test  = test_data_dictionary['data'].reshape(len(test_data_dictionary['data']), 3, 32, 32).astype(np.float32)
        self.y_test  = np.array(test_data_dictionary['labels']).astype(np.int32)
        self.N_train = len(self.x_train)
        self.N_test  = len(self.x_test)

    def fetch_train_data(self, batch_size, permutate=True, augment=False):
        self.iter_count += 1
        self.used_count += batch_size
        if self.N_train == 0:
            return
        if self.train_count == 0 and permutate:
            self.perm = np.random.permutation(self.N_train)
        start_ind = self.train_count
        if self.train_count >= self.N_train - batch_size:
            end_ind          = self.N_train
            self.train_count = 0
            self.epoch_count += 1
        else:
            end_ind          = self.train_count + batch_size
            self.train_count += batch_size
        return self.x_train[self.perm[start_ind:end_ind]], self.y_train[self.perm[start_ind:end_ind]]

    def fetch_test_data(self, batch_size, augment=False):
        if self.N_test == 0:
            return
        start_ind = self.test_count
        if self.test_count >= self.N_test - batch_size:
            end_ind         = self.N_test
            self.test_count = 0
        else:
            end_ind         = self.test_count + batch_size
            self.test_count += batch_size
        return self.x_test[start_ind:end_ind], self.y_test[start_ind:end_ind]

    def minus_mean_from_data(self):
        self.mean_data = self.x_train.mean(axis=0)
        self.x_train   -= self.mean_data
        self.x_test    -= self.mean_data
        assert np.abs(self.x_train.mean()) < 1e-5

    def norm_from_data(self):
        self.std     = self.x_train.std()
        self.x_train /= self.std
        self.x_test  /= self.std
        assert np.abs(self.x_train.std() - 1.0) < 1e-5

    @property
    def end_of_epoch_train(self):
        if self.train_count == 0:
            return True
        else:
            return False

    @property
    def end_of_epoch_test(self):
        if self.test_count == 0:
            return True
        else:
            return False
