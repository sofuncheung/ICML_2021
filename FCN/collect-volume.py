# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import os
from GP_prob.GP_prob_gpy import GP_prob
from fc_kernel_he_normal import kernel_matrix
import gc
import tensorflow as tf


train_size = 500

T1, T2, T3, T4 = 784, 40, 40, 1

sigmab = np.array([1.,1,1])

volume_list_no_attack = []

path = os.getcwd()
dir_list = []
for i in os.listdir(path):
    if os.path.isdir(i) and 'attack_size' in i:
        dir_list.append(i)
dir_list.sort()

DATAPATH = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))),
            'data')
if not os.path.exists(DATAPATH):
    DATAPATH = os.path.join(os.path.dirname(
            os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))),
            'data')

x_train_genuine = np.load(os.path.join(DATAPATH,'train_x_20000.npy'))[:train_size]
x_test = np.load(os.path.join(DATAPATH,'test_x_1000.npy'))

xs = np.concatenate((x_train_genuine, x_test), axis = 0)
K = kernel_matrix(xs, number_layers=2, sigmaw=np.sqrt(2), sigmab=sigmab, n_gpus=0)

for dirs in dir_list:
    current_path = os.path.join(path, dirs)
    os.chdir(current_path)
    print('Current dir:', dirs)
    for sample in range(5):
        tf.reset_default_graph()
        print('Calculating Volume for sample %d:' %(sample+1 ) )
        ys = np.load('ys_no_attack_%d.npy'%(sample+1))
        logPU = GP_prob(K, xs, ys)
        volume_list_no_attack.append(logPU)
    os.chdir(path)

np.save('volume_list_no_attack.npy', volume_list_no_attack)







