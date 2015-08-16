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
from modified_bayesian_optimization import ModBayesianOptimization as BO

params                 = edict({})
params.n_epoch         = 1
params.gpu_flag        = 1 # gpu id or False
params.max_bo_iter     = 100
params.model_dir       = 'models'
params.model_name      = False # model name or False
params.prefix          = 'cifar'
params.opt_iter        = 100
params.opt_init_points = 15

def init_model(model_params):
    wscale1 = model_params.wscale1 # math.sqrt(5 * 5 * 3) * 0.0001
    wscale2 = model_params.wscale2 # math.sqrt(5 * 5 * 32) * 0.01
    wscale3 = model_params.wscale3 # math.sqrt(5 * 5 * 32) * 0.01
    wscale4 = model_params.wscale4 # math.sqrt(576) * 0.1
    wscale5 = model_params.wscale5 # math.sqrt(64) * 0.1
    # wscale1, wscale2, wscale3, wscale4, wscale5 = [math.sqrt(2)] * 5
    model = FunctionSet(conv1=F.Convolution2D(3, 32, 5, wscale=wscale1, stride=1, pad=2),
                        conv2=F.Convolution2D(32, 32, 5, wscale=wscale2, stride=1, pad=2),
                        conv3=F.Convolution2D(32, 64, 5, wscale=wscale3, stride=1, pad=2),
                        fl4=F.Linear(576, 64, wscale=wscale4),
                        fl5=F.Linear(64, 10, wscale=wscale5))
    if params.gpu_flag:
        model.to_gpu()
    return model

def init_optimizer(model, model_params):
    optimizer = optimizers.MomentumSGD(lr=model_params.lr, momentum=model_params.momentum)
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

def train_and_val(model, optimizer, fetcher, model_params):
    sum_accuracy_val = 0
    sum_loss_val     = 0
    while True:
        if fetcher.epoch_count == params.n_epoch:
            break
        # training
        sum_accuracy = 0
        sum_loss     = 0
        while True:
            x_batch, y_batch = fetcher.fetch_train_data(model_params.batchsize, permutate=False)
            if params.gpu_flag is not False:
                x_batch = cuda.to_gpu(x_batch)
                y_batch = cuda.to_gpu(y_batch)
            optimizer.zero_grads()
            loss, acc = forward(model, x_batch, y_batch)
            optimizer.weight_decay(model_params.decay)
            loss.backward()
            optimizer.update()

            if np.isinf(float(cuda.to_cpu(loss.data))) or np.isnan(float(cuda.to_cpu(loss.data))):
                return 0.1, float('inf')

            sum_loss     += float(cuda.to_cpu(loss.data)) * len(y_batch)
            sum_accuracy += float(cuda.to_cpu(acc.data)) * len(y_batch)

            if fetcher.end_of_epoch_train:
                break

    # evaluation
    while True:
        x_batch, y_batch = fetcher.fetch_test_data(model_params.batchsize)
        if params.gpu_flag is not False:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        loss, acc = forward(model, x_batch, y_batch, train=False)
        sum_loss_val     += float(cuda.to_cpu(loss.data)) * len(y_batch)
        sum_accuracy_val += float(cuda.to_cpu(acc.data)) * len(y_batch)
        if fetcher.end_of_epoch_test:
            break
    mean_loss_val     = sum_loss_val / fetcher.N_test
    mean_accuracy_val = sum_accuracy_val / fetcher.N_test
    return mean_accuracy_val, mean_loss_val

def train_model(wscale1, wscale2, wscale3, wscale4, wscale5, lr, batchsize, momentum, decay, fetcher):
    model_params           = edict({})
    model_params.wscale1   = math.sqrt(5 * 5 * 3) * 10**wscale1 # math.sqrt(5 * 5 * 3) * 0.0001
    model_params.wscale2   = math.sqrt(5 * 5 * 32) * 10**wscale2 # math.sqrt(5 * 5 * 32) * 0.01
    model_params.wscale3   = math.sqrt(5 * 5 * 32) * 10**wscale3 # math.sqrt(5 * 5 * 32) * 0.01
    model_params.wscale4   = math.sqrt(576) * 10**wscale4 # math.sqrt(576) * 0.1
    model_params.wscale5   = math.sqrt(64) * 10**wscale5 # math.sqrt(64) * 0.1
    model_params.lr        = 10**lr # 0.001
    model_params.batchsize = int(batchsize) # 100
    model_params.momentum  = momentum # 0.9
    model_params.decay     = 10**decay # 0.004

    model = init_model(model_params)
    optimizer = init_optimizer(model, model_params)
    sum_accuracy_val, _ = train_and_val(model, optimizer, fetcher, model_params)

    fetcher.iter_count  = 0
    fetcher.epoch_count = 0

    return sum_accuracy_val

def main():
    if params.gpu_flag is not False:
        cuda.init(params.gpu_flag)
    print 'fetching data ...'
    fetcher = cifar(norm=False)

    bo = BO(train_model,
            {'wscale1'   : (-5, 0),
             'wscale2'   : (-5, 0),
             'wscale3'   : (-5, 0),
             'wscale4'   : (-5, 0),
             'wscale5'   : (-5, 0),
             'lr'        : (-4, -2),
             'batchsize' : (30, 300),
             'momentum'  : (0.5, 1.0),
             'decay'     : (-4, -2)
            })

    """
    bo.explore({'wscale1'   : [-4],
                'wscale2'   : [-2],
                'wscale3'   : [-2],
                'wscale4'   : [-1],
                'wscale5'   : [-1],
                'lr'        : [-3],
                'batchsize' : [100],
                'momentum'  : [0.9],
                'decay'     : [-3]
                })
    """

    bo.add_f_args('fetcher', fetcher)
    bo.maximize(init_points=params.opt_init_points, n_iter=params.opt_iter)
    print bo.res['max']

if __name__ == '__main__':
    main()
