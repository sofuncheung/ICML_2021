# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import os
from GP_prob.GP_prob_gpy import GP_prob
from fc_kernel import kernel_matrix
import gc
import tensorflow as tf


train_size = 500

T1, T2, T3, T4 = 784, 40, 40, 1

DATAPATH = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))),
            'data')
if not os.path.exists(DATAPATH):
    DATAPATH = os.path.join(os.path.dirname(
            os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))),
            'data')

x_train_genuine = np.load(os.path.join(DATAPATH,'train_x_20000.npy'))[:train_size]
x_test = np.load(os.path.join(DATAPATH,'test_x_1000.npy'))[:100]

xs = np.concatenate((x_train_genuine, x_test), axis = 0)
K = kernel_matrix(xs, number_layers=2, sigmaw=np.sqrt(2), sigmab=1, n_gpus=0)

volume = np.load('volume_ys_list_1.npy')
volume_list = []
for i in range(len(volume)):
    tf.reset_default_graph()
    print('Here we are going with %04d epoch'%i)
    ys = volume[i]
    logPU = GP_prob(K, xs, ys)
    volume_list.append(logPU)
np.save('volume_list.npy', volume_list)

