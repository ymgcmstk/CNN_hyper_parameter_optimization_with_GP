#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import math
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import cPickle as pickle
import numpy as np
from cifar10_fetcher import Cifar10Fetcher as cifar
from easydict import EasyDict as edict

params            = edict({})
params.n_epoch    = 25
params.batchsize  = 100
params.gpu_flag   = 1 # gpu id or False
params.model_dir  = 'models'
params.model_name = False # model name or False
params.prefix     = 'cifar'

params.lr         = 0.001
params.momentum   = 0.9
params.decay      = 0.004

def init_model():
    wscale1 = math.sqrt(5 * 5 * 3) * 0.0001
    wscale2 = math.sqrt(5 * 5 * 32) * 0.01
    wscale3 = math.sqrt(5 * 5 * 32) * 0.01
    wscale4 = math.sqrt(576) * 0.1
    wscale5 = math.sqrt(64) * 0.1
    # wscale1, wscale2, wscale3, wscale4, wscale5 = [math.sqrt(2)] * 5

    model = FunctionSet(conv1=F.Convolution2D(3, 32, 5, wscale=wscale1, stride=1, pad=2),
                        conv2=F.Convolution2D(32, 32, 5, wscale=wscale2, stride=1, pad=2),
                        conv3=F.Convolution2D(32, 64, 5, wscale=wscale3, stride=1, pad=2),
                        fl4=F.Linear(576, 64, wscale=wscale4),
                        fl5=F.Linear(64, 10, wscale=wscale5))
    if params.gpu_flag:
        model.to_gpu()
    return model

def init_optimizer(model):
    optimizer = optimizers.MomentumSGD(lr=params.lr, momentum=params.momentum)
    optimizer.setup(model.collect_parameters())
    return optimizer

def forward(model, x_data, y_data, train=True):
    x, t = Variable(x_data, volatile=not train), Variable(y_data, volatile=not train)
    h = F.relu(F.max_pooling_2d(model.conv1(x), 3, stride=2))
    h = F.relu(F.average_pooling_2d(model.conv2(h), 3, stride=2))
    h = F.relu(F.average_pooling_2d(model.conv3(h), 3, stride=2))
    h = model.fl4(h)  # <- cifar10_quick.prototxt in caffe, instead of below line
    # h = F.relu(model.fl4(h))
    y = model.fl5(h)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

def save_model(model, model_name):
    if not os.path.exists(params.model_dir):
        os.mkdir(params.model_dir)
    save_path = os.path.join(params.model_dir, params.prefix + '_' + model_name + '.p')
    pickle.dump(model, open(save_path, 'wb'), -1)
    print 'Current model has been saved as ' + save_path + '.'

def load_model(model_name):
    model_path = os.path.join(os.model_dir, model_name)
    model = pickle.load(open(model_path, 'rb'))
    if params.gpu_flag:
        model.to_gpu()
    return model

def train_and_val(model, optimizer, fetcher):
    while True:
        if fetcher.epoch_count > params.n_epoch:
            break
        if fetcher.epoch_count == 8:
            optimizer.lr *= 0.1
        print 'epoch', fetcher.epoch_count

        # training
        sum_accuracy = 0
        sum_loss     = 0
        while True:
            x_batch, y_batch = fetcher.fetch_train_data(params.batchsize)
            if params.gpu_flag is not False:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)
            optimizer.zero_grads()
            loss, acc = forward(model, x_batch, y_batch)
            optimizer.weight_decay(params.decay)
            loss.backward()
            optimizer.update()

            sum_loss     += float(cuda.to_cpu(loss.data)) * len(y_batch)
            sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)
            if fetcher.end_of_epoch_train:
                break
        print 'train mean loss={}, accuracy={}'.format(
            sum_loss / fetcher.N_train, sum_accuracy / fetcher.N_train)

        # evaluation
        sum_accuracy = 0
        sum_loss     = 0
        while True:
            x_batch, y_batch = fetcher.fetch_test_data(params.batchsize)
            if params.gpu_flag is not False:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)

            loss, acc = forward(model, x_batch, y_batch, train=False)

            sum_loss     += float(cuda.to_cpu(loss.data)) * len(y_batch)
            sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)
            if fetcher.end_of_epoch_test:
                break
        print 'test  mean loss={}, accuracy={}'.format(
            sum_loss / fetcher.N_test, sum_accuracy / fetcher.N_test)

def main():
    if params.gpu_flag is not False:
        cuda.init(params.gpu_flag)
    if params.model_name is False:
        model = init_model()
    else:
        model = load_model(params.model_name)
    optimizer = init_optimizer(model)
    print 'fetching data ...'
    fetcher = cifar(norm=False)
    print 'done'
    train_and_val(model, optimizer, fetcher)

if __name__ == '__main__':
    main()
